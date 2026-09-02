"""Shared fixtures for smoke-testing the pipelines.

The ``tiny_*`` fixtures build a synthetic MVTec-AD-style dataset of
random-noise PNGs, so every model's train/threshold/test path runs on CPU with
no real dataset. They prove the pipelines run, nothing about model quality.
"""
import random
from pathlib import Path

import matplotlib

# Nothing here calls plt.show(), so force the non-interactive backend.
matplotlib.use("Agg")

import numpy as np
import pytest
import yaml
from PIL import Image

PRODUCT_CLASS = "toy"
IMAGE_SIZE = 64

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBSET_CONFIG = REPO_ROOT / "config.subset.yaml"
SUBSET_ROOT = REPO_ROOT / "data" / "mvtec_subset"


def _write_random_image(path, size=IMAGE_SIZE, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask_image(path, size=IMAGE_SIZE, seed=0):
    mask = np.zeros((size, size), dtype=np.uint8)
    # An anomalous square, so PRO/AUC sees a mix of mask values.
    quarter = size // 4
    mask[quarter : quarter * 3, quarter : quarter * 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


@pytest.fixture(scope="session")
def subset_config_path():
    """``config.subset.yaml``: real (downscaled) screw and wood imagery.

    Skips rather than fails when the subset is absent, so a sparse checkout
    still runs the rest of the suite.
    """
    if not SUBSET_CONFIG.is_file() or not SUBSET_ROOT.is_dir():
        pytest.skip(
            f"Dataset subset not found at {SUBSET_ROOT}. "
            "Build it with: python scripts/make_subset.py"
        )
    return SUBSET_CONFIG


@pytest.fixture
def tiny_dataset(tmp_path):
    """Build a minimal MVTecAD2-compatible dataset for one product class."""
    random.seed(0)
    dataset_root = tmp_path / "dataset"
    product_dir = dataset_root / PRODUCT_CLASS

    for i in range(4):
        _write_random_image(product_dir / "train" / "good" / f"train_{i}.png", seed=i)

    for i in range(2):
        _write_random_image(product_dir / "test" / "good" / f"good_{i}.png", seed=10 + i)

    for i in range(2):
        _write_random_image(product_dir / "test" / "bad" / f"bad_{i}.png", seed=20 + i)
        _write_mask_image(
            product_dir / "ground_truth" / "bad" / f"bad_{i}_mask.png", seed=20 + i
        )

    return dataset_root


@pytest.fixture
def classic_layout_dataset(tmp_path):
    """A *classic* MVTec AD layout: anomaly folders named per defect type
    (``crack``/``cut``) rather than the single ``bad`` folder MVTec AD 2 uses."""
    random.seed(0)
    dataset_root = tmp_path / "classic_dataset"
    product_dir = dataset_root / PRODUCT_CLASS

    for i in range(4):
        _write_random_image(product_dir / "train" / "good" / f"train_{i}.png", seed=i)

    for i in range(2):
        _write_random_image(product_dir / "test" / "good" / f"good_{i}.png", seed=10 + i)

    for offset, defect in enumerate(("crack", "cut")):
        _write_random_image(
            product_dir / "test" / defect / f"{defect}_000.png", seed=30 + offset
        )
        _write_mask_image(
            product_dir / "ground_truth" / defect / f"{defect}_000_mask.png",
            seed=30 + offset,
        )

    return dataset_root


@pytest.fixture
def classic_layout_config_path(tmp_path, classic_layout_dataset):
    """A config.yaml pointing at :func:`classic_layout_dataset`."""
    config_path = tmp_path / "classic_config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "DATASET_PATH": str(classic_layout_dataset),
                "SEED": 42,
                "DATASET_OBJECTS": [PRODUCT_CLASS],
                "FOUNDATIONAL_OBJECTS": [PRODUCT_CLASS],
                "MODELS_CONFIG": {},
            },
            fh,
        )
    return config_path


@pytest.fixture
def tiny_config_path(tmp_path, tiny_dataset):
    """Write a config.yaml with tiny hyperparameters pointing at tiny_dataset."""
    config = {
        "DATASET_PATH": str(tiny_dataset),
        "SEED": 42,
        "DATASET_OBJECTS": [PRODUCT_CLASS],
        "FOUNDATIONAL_OBJECTS": [PRODUCT_CLASS],
        "MODELS_CONFIG": {
            "base_autoencoder": {
                "batch_size": 2,
                "num_epochs": 1,
                "validation_split": 0.25,
                "learning_rate": 1e-3,
                "num_workers": 0,
            },
            "trafo_autoencoder": {
                "batch_size": 2,
                "num_epochs": 1,
                "validation_split": 0.25,
                "learning_rate": 1e-3,
                "num_workers": 0,
                "min_lr": 1e-5,
                "patience": 5,
                "delta": 1e-3,
            },
            "vit_autoencoder": {
                "batch_size": 2,
                "num_epochs": 1,
                "validation_split": 0.25,
                "learning_rate": 1e-4,
                "num_workers": 0,
                "min_lr": 1e-5,
                "patience": 5,
                "delta": 1e-3,
                "threshold_quantile": 0.9,
            },
            "DeepFeatureAE": {
                "batch_size": 2,
                "num_epochs": 1,
                "validation_split": 0.25,
                "learning_rate": 1e-3,
                "input_size": 224,
                "smooth": True,
                "layer_hooks": ["layer2", "layer3"],
                "latent_dim": 8,
                "is_bn": True,
                "threshold_computation_mode": "all",
            },
            "patchcore": {
                "batch_size": 1,
                "memory_bank_subsample_ratio": 0.1,
                "threshold_multiplier": 2.0,
                "distance_chunk_size": 10000,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    return config_path
