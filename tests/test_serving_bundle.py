"""Tests for the committed serving bundle under models/dfr_hazelnut.

Guards both halves of "serve works on a fresh clone": the files are shaped the
way `serve` expects, and the trimmed checkpoint scores like the full one.
"""
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE = REPO_ROOT / "models" / "dfr_hazelnut"
PRODUCT_CLASS = "hazelnut"
SAMPLES = REPO_ROOT / "examples" / "sample_images" / PRODUCT_CLASS


def test_bundle_has_what_serve_loads():
    weights = BUNDLE / "checkpoints" / f"{PRODUCT_CLASS}_dfad_weights.pth"
    thresholds = BUNDLE / f"{PRODUCT_CLASS}_thresholds.yaml"
    assert weights.is_file(), "committed serving weights are missing"
    assert thresholds.is_file(), "committed thresholds are missing"


def test_bundle_carries_no_backbone_parameters():
    """The 100 MB of frozen ImageNet weights must stay out of the repo.

    Only the backbone's BatchNorm buffers belong in the bundle: they change
    during training (see scripts/export_serving_model.py), the rest does not.
    """
    state_dict = torch.load(
        BUNDLE / "checkpoints" / f"{PRODUCT_CLASS}_dfad_weights.pth",
        map_location="cpu",
        weights_only=True,
    )
    backbone = [key for key in state_dict if "feature_extractor.backbone." in key]
    assert backbone, "BatchNorm buffers should have been kept"
    assert all(
        key.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for key in backbone
    ), "bundle still carries frozen backbone parameters"


def test_sample_images_are_present():
    """The demo images `serve` is documented with must ship alongside it."""
    images = sorted(SAMPLES.glob("*.png"))
    assert [p.name for p in images] == [
        "crack_000.png", "good_000.png", "good_001.png", "hole_000.png", "print_000.png"
    ]


@pytest.mark.slow
def test_bundle_scores_samples_the_way_the_readme_says():
    """Good below threshold, the three defects above -- no training first."""
    pytest.importorskip("fastapi", reason="serving extras not installed")
    from PIL import Image

    from anom_detect.serve import DeepFeatureScorer

    scorer = DeepFeatureScorer(
        str(REPO_ROOT / "config.subset.yaml"), PRODUCT_CLASS, str(BUNDLE)
    )
    verdicts = {
        path.name: scorer.score(Image.open(path))[1] for path in sorted(SAMPLES.glob("*.png"))
    }
    assert verdicts == {
        "crack_000.png": True,
        "good_000.png": False,
        "good_001.png": False,
        "hole_000.png": True,
        "print_000.png": True,
    }
