import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os

IMG_SIZE = 380


class RetinaScreenModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.4, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained,
            num_classes=0, global_pool='avg'
        )
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes - 1)

    def forward(self, x):
        # x: (batch, 3, 380, 380)
        feat = self.backbone(x)    # (batch, 1792)
        feat = self.dropout(feat)  # (batch, 1792)
        return self.classifier(feat)  # (batch, 4)


def get_grade_from_logits(logits):
    """Convert CORN logits → ordinal grade (0–4)."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def load_model(weights_path: str, device: str = 'cpu') -> RetinaScreenModel:
    model = RetinaScreenModel(pretrained=False)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_model_from_hub(repo_id: str, filename: str = 'weights/best_model.pth',
                        device: str = 'cpu') -> RetinaScreenModel:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_model(path, device=device)


def preprocess_image(img_np: np.ndarray) -> torch.Tensor:
    """Ben Graham preprocessing + albumentations pipeline."""
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ben Graham contrast enhancement
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 30), -4, 128)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform(image=img)['image']