from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anom_detect.deep_feature_ad.deep_feature_autoencoder_model import DeepFeatureAutoEncoder


class DeepFeatureAnomalyDetector(nn.Module):
    """Wraps :class:`DeepFeatureAutoEncoder` with scoring and segmentation.

    Turns the feature reconstruction error into a per-image anomaly score and
    an upsampled segmentation map.
    """

    def __init__(
        self,
        layer_hooks: Optional[list] = None,
        latent_dim: int = 100,
        smooth: bool = True,
        is_bn: bool = True,
    ) -> None:
        super().__init__()
        if layer_hooks is None:
            layer_hooks = ["layer2", "layer3"]

        self.autoencoder = DeepFeatureAutoEncoder(
            layer_hooks=layer_hooks, latent_dim=latent_dim, smooth=smooth, is_bn=is_bn
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(x)

    def compute_reconstruction_error(
        self, features: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Per-location L2 error between features and their reconstruction."""
        return torch.norm(features - reconstructed, p=2, dim=1)

    def compute_anomaly_score(self, error_map: torch.Tensor, k: int = 10) -> torch.Tensor:
        """One score per image: the mean of its top-k errors.

        Top-k rather than the full mean, because anomalies are localised and
        averaging every location dilutes the signal.
        """
        return torch.stack([
            torch.mean(torch.topk(error_map[i].flatten(), k)[0])
            for i in range(error_map.shape[0])
        ])

    def get_segmentation_map(
        self, error_map: torch.Tensor, target_size: tuple = (224, 224)
    ) -> torch.Tensor:
        """Upsample the error map to the input image size."""
        return F.interpolate(error_map.unsqueeze(1), size=target_size, mode="bilinear").squeeze(1)

    def get_stats(self) -> dict:
        return self.autoencoder.get_stats()
