"""Frozen ResNet50 patch-feature extractor shared by PatchCore and Trafo-AE.

They differ only in the final shape -- PatchCore wants one row per patch,
Trafo-AE the spatial map -- so that reshape stays at the call site.

Attribute names (``model``, ``avg``) are load-bearing: ``TransformerAE``
saves this module as ``backbone``, so renaming them invalidates every
existing ``autoencoder_weights.pth``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ResNetPatchFeatures(nn.Module):
    """Frozen ResNet50; concatenates hooked layer activations into one
    spatial feature map per image, aligned to the highest-resolution hook."""

    def __init__(self) -> None:
        super().__init__()
        # DEFAULT resolves to IMAGENET1K_V2 for resnet50; named explicitly so
        # a future torchvision default change can't silently swap the weights.
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.features: list[torch.Tensor] = []

        def hook(_model, _input, output):
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        # Built once, not per forward pass.
        self.avg = nn.AvgPool2d(3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the concatenated feature map, shape (B, C, H, W)."""
        self.features = []

        with torch.no_grad():
            _ = self.model(x)

        # Align every activation to the first (highest-resolution) map.
        target_size = self.features[0].shape[-2]
        return torch.cat(
            [F.adaptive_avg_pool2d(self.avg(f), target_size) for f in self.features], dim=1
        )
