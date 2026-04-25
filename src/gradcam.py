import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from src.uncertainty import enable_mc_dropout


def compute_um_gradcam(model, img_tensor: torch.Tensor,
                        target_grade: int, n_passes: int = 10,
                        device: str = 'cpu') -> dict:
    """
    Uncertainty-Modulated Grad-CAM (UM-GradCAM).

    Returns three heatmaps (np.ndarray, 0-1 normalised):
      mean_attention    — where model reliably focuses
      uncertainty_map   — where attention fluctuates across MC passes
      certain_attention — high-confidence lesion regions (mean × (1-var))
    """
    enable_mc_dropout(model)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    target_layers = [model.backbone.blocks[-1][-1]]

    class OrdinalTarget:
        def __init__(self, grade):
            self.grade = int(grade)
        def __call__(self, model_output):
            # pytorch-grad-cam may pass a per-sample 1D tensor during loss
            # construction. Normalize outputs before selecting CORN threshold.
            if isinstance(model_output, (tuple, list)):
                model_output = model_output[0]
            if not torch.is_tensor(model_output):
                model_output = torch.as_tensor(model_output)

            if model_output.ndim == 0:
                return model_output

            # CORN uses K-1 threshold nodes for K grades; map grade g to node g-1.
            target_idx = max(0, min(self.grade - 1, model_output.shape[-1] - 1))

            if model_output.ndim == 1:
                return model_output[target_idx]

            return model_output[:, target_idx].sum()

    maps = []
    for _ in range(n_passes):
        with GradCAM(model=model, target_layers=target_layers) as cam:
            gmap = cam(input_tensor=img_tensor.float(),
                       targets=[OrdinalTarget(target_grade)])
        maps.append(gmap[0])

    maps = np.stack(maps, axis=0)          # (n_passes, H, W)
    mean_map = maps.mean(axis=0)
    var_map  = maps.var(axis=0)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    mean_norm = norm(mean_map)
    var_norm  = norm(var_map)

    return {
        'mean_attention':   mean_norm,
        'uncertainty_map':  var_norm,
        'certain_attention': mean_norm * (1 - var_norm),
    }


def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray,
                     colormap=cv2.COLORMAP_JET, alpha: float = 0.45) -> np.ndarray:
    h8 = np.uint8(255 * heatmap)
    colored = cv2.cvtColor(cv2.applyColorMap(h8, colormap), cv2.COLOR_BGR2RGB)
    colored = cv2.resize(colored, (img_np.shape[1], img_np.shape[0]))
    return cv2.addWeighted(img_np, 1 - alpha, colored, alpha, 0)