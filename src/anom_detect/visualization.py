"""Shared plotting helper for segmentation-vs-ground-truth comparison images.

Every model's ``test()`` saves some of its test images as a 3-panel figure.
The title reports image-level correctness, which is comparable across models
even where the pixel-level maps are not.
"""
import os
from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt


def resolve_sample_count(num_seg_samples: Optional[int], dataset_size: int) -> int:
    """Number of test images to save comparisons for. ``None`` means "all"."""
    if num_seg_samples is None:
        return dataset_size
    return min(num_seg_samples, dataset_size)


def save_segmentation_comparison(
    original: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_map: torch.Tensor,
    out_path: str,
    true_label: str,
    pred_label: str,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> None:
    """Save an original / ground-truth / predicted-anomaly-map figure to ``out_path``.

    ``original`` is (C, H, W); the masks need only squeeze to (H, W). Pass
    ``mean``/``std`` to undo a normalisation before display; omit them for
    images that only went through ``ToTensor()``.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    image = original.detach().cpu().float()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
        image = image * std_t + mean_t
    image = image.permute(1, 2, 0).clamp(0, 1).numpy()

    gt = np.asarray(gt_mask.detach().cpu()).squeeze() if isinstance(gt_mask, torch.Tensor) else np.asarray(gt_mask).squeeze()
    pred = np.asarray(pred_map.detach().cpu().float()).squeeze() if isinstance(pred_map, torch.Tensor) else np.asarray(pred_map).squeeze()

    correct = pred_label == true_label
    verdict = "Correct" if correct else "Incorrect"

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    im = axs[2].imshow(pred, cmap="jet")
    axs[2].set_title("Predicted Anomaly Map")
    axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"True: {true_label} | Predicted: {pred_label} ({verdict})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
