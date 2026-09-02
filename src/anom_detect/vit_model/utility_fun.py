"""Score-map post-processing helpers for the ViT pipeline."""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

GAUSSIAN = 0
MEDIAN = 1


def Filter(score_map, filter_type: int = GAUSSIAN):
    """Smooth an anomaly score map with ``GAUSSIAN`` (0) or ``MEDIAN`` (1)."""
    if filter_type == GAUSSIAN:
        return gaussian_filter(score_map, sigma=4)
    if filter_type == MEDIAN:
        return median_filter(score_map, size=3)
    raise ValueError(f"Unsupported filter_type: {filter_type!r} (expected 0 or 1).")


def Binarization(mask, thres: float = 0.0, keep_values: bool = False):
    """Threshold a mask to 0/1, or to its original values when ``keep_values``."""
    if keep_values:
        return np.where(mask > thres, mask, 0.0)
    return np.where(mask > thres, 1.0, 0.0)
