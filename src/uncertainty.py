import torch
import torch.nn as nn
import numpy as np
from src.model import get_grade_from_logits


def enable_mc_dropout(model: nn.Module) -> None:
    """Enables both standard Dropout and timm's DropPath for MC inference."""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__ == 'DropPath':
            m.train()


def mc_predict_with_dud(model: nn.Module, img_tensor: torch.Tensor,
                         n_passes: int = 20, device: str = 'cpu') -> dict:
    """
    MC Dropout inference with Directional Uncertainty Decomposition (DUD).

    Theoretical basis (semi-variance, Sortino 1991):
        U↑ = E[(g_t - ḡ)² | g_t > ḡ]   upside / severity-increasing risk
        U↓ = E[(ḡ - g_t)² | g_t < ḡ]   downside / over-treatment risk
        CAS = U↑ / (U↑ + U↓ + ε)        Clinical Asymmetry Score ∈ [0,1]

    Returns:
        dict with grade, uncertainty components, CAS, CRPS, urgency
    """
    enable_mc_dropout(model)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    grades = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(img_tensor)
            grades.append(get_grade_from_logits(logits).item())

    g         = np.array(grades, dtype=float)
    mean_g    = g.mean()
    final_g   = int(np.clip(round(mean_g), 0, 4))
    total_var = float(g.var())

    upper = g[g > mean_g]
    lower = g[g < mean_g]
    U_up   = float(((upper - mean_g) ** 2).mean()) if len(upper) > 0 else 0.0
    U_down = float(((mean_g - lower) ** 2).mean()) if len(lower) > 0 else 0.0

    eps  = 1e-8
    CAS  = U_up / (U_up + U_down + eps)
    CRPS = float(np.clip((final_g / 4) * 0.4 + total_var * 0.35 + CAS * 0.25, 0.0, 1.0))

    return {
        'final_grade':    final_g,
        'mean_grade':     round(mean_g, 2),
        'total_variance': round(total_var, 4),
        'U_up':           round(U_up, 4),
        'U_down':         round(U_down, 4),
        'CAS':            round(CAS, 4),
        'CRPS':           round(CRPS, 4),
        'all_grades':     grades,
        'referral_urgency': _get_urgency(final_g, CRPS, CAS),
    }


def _get_urgency(grade: int, crps: float, cas: float) -> str:
    if grade >= 3 or crps > 0.75:
        return 'URGENT'
    elif grade == 2 or crps > 0.50:
        return 'PRIORITY' if cas > 0.6 else 'ROUTINE'
    elif grade == 1 or crps > 0.30:
        return 'MONITOR'
    return 'CLEAR'