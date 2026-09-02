"""Tests against the real MVTec AD imagery in ``data/mvtec_subset``.

Covers what synthetic noise cannot: multiple defect folders per class,
``<stem>_mask.png`` ground truth, real image dimensions. Not a quality
benchmark -- 20 images and 3 epochs give no meaningful AUC.
"""
import argparse

import pytest
import torch

from anom_detect.base_model.base_autoencoder import BaseAEManager
from anom_detect.dataset_preprocessor import MVTecAD2, load_dataset_config, resolve_dataset_path
from anom_detect.patchcore.patchcore_class import PatchCoreManager

# Must match config.subset.yaml's DATASET_OBJECTS and scripts/make_subset.py's
# --train-images default; asserted below so a truncated subset fails loudly.
SUBSET_CLASSES = ("screw", "wood")
SUBSET_TRAIN_IMAGES = 20


def test_relative_dataset_path_resolves_against_config_dir(subset_config_path):
    """``DATASET_PATH: "data/mvtec_subset"`` must not depend on the CWD."""
    config = load_dataset_config(subset_config_path)
    resolved = resolve_dataset_path(config, subset_config_path)

    assert resolved.is_absolute()
    assert resolved.is_dir()
    assert {d.name for d in resolved.iterdir() if d.is_dir()} == set(SUBSET_CLASSES)


@pytest.mark.parametrize("product_class", SUBSET_CLASSES)
def test_train_split_is_all_good_images(subset_config_path, product_class, tmp_path):
    dataset = MVTecAD2(
        product_class, "train", config_path=str(subset_config_path)
    )

    assert len(dataset) == SUBSET_TRAIN_IMAGES
    assert dataset.has_segmentation_gt is False
    assert all("good" in path.replace("\\", "/") for path in dataset.image_paths)

    sample = dataset[0]["sample"]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape[0] == 3  # RGB


@pytest.mark.parametrize("product_class", SUBSET_CLASSES)
def test_test_split_spans_every_defect_folder(subset_config_path, product_class, tmp_path):
    """Classic MVTec AD splits anomalies across per-defect folders rather than
    a single ``bad`` folder; all of them have to be picked up."""
    dataset = MVTecAD2(
        product_class, "test", config_path=str(subset_config_path)
    )
    assert dataset.has_segmentation_gt is True

    dataset_root = resolve_dataset_path(
        load_dataset_config(subset_config_path), subset_config_path
    )
    expected_folders = {d.name for d in (dataset_root / product_class / "test").iterdir() if d.is_dir()}
    found_folders = {path.replace("\\", "/").rsplit("/", 2)[-2] for path in dataset.image_paths}

    assert found_folders == expected_folders
    assert len(expected_folders) > 2  # good + several defect types


@pytest.mark.parametrize("product_class", SUBSET_CLASSES)
def test_ground_truth_masks_match_the_defect_images(subset_config_path, product_class, tmp_path):
    dataset = MVTecAD2(
        product_class, "test", config_path=str(subset_config_path)
    )

    saw_defect_mask = False
    for idx, path in enumerate(dataset.image_paths):
        gt = dataset.get_gt_image(idx)
        if "good" in path.replace("\\", "/").rsplit("/", 2)[-2]:
            assert gt.max().item() == 0, f"good image has a non-empty mask: {path}"
        else:
            # A real defect mask marks at least some pixels; an all-zero one
            # means the wrong mask (or a resized-away one) got loaded.
            assert gt.max().item() > 0, f"defect image has an empty mask: {path}"
            saw_defect_mask = True

    assert saw_defect_mask


def test_all_concatenates_every_subset_class(subset_config_path, tmp_path):
    combined = MVTecAD2(
        "all", "train", config_path=str(subset_config_path)
    )
    assert len(combined) == SUBSET_TRAIN_IMAGES * len(SUBSET_CLASSES)


def test_base_autoencoder_end_to_end_on_wood(tmp_path, subset_config_path):
    """Full train -> threshold -> save -> test cycle on real wood images.

    The base autoencoder trains from scratch, so unlike the PatchCore test
    below this needs no pretrained-weight download.
    """
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    manager = BaseAEManager("wood", str(subset_config_path), str(train_path), str(test_path))

    manager.train()
    mean_error, std_error, threshold = manager.compute_thresh()
    manager.save_model(
        argparse.Namespace(
            config=str(subset_config_path),
            product_class="wood",
            model_name="base",
            train_path=str(train_path),
            test_path=str(test_path),
            mode="train",
        ),
        mean_error,
        std_error,
        threshold,
    )

    assert (train_path / "autoencoder_weights.pth").is_file()
    assert (train_path / "training_statistics.yaml").is_file()
    assert threshold > 0

    manager.test()  # must not raise
    assert any(test_path.rglob("*.png"))


@pytest.mark.slow
def test_patchcore_end_to_end_on_screw(tmp_path, subset_config_path):
    """PatchCore on real screw images: pulls the ImageNet ResNet50 weights."""
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    manager = PatchCoreManager("screw", str(subset_config_path), str(train_path), str(test_path))

    manager.train()
    mean_error, std_error, threshold = manager.compute_thresh()
    manager.save_model(
        argparse.Namespace(
            config=str(subset_config_path),
            product_class="screw",
            model_name="patchcore",
            train_path=str(train_path),
            test_path=str(test_path),
            mode="train",
        ),
        mean_error,
        std_error,
        threshold,
    )

    metrics = manager.test()
    assert 0.0 <= metrics["roc_auc_image"] <= 1.0
    assert 0.0 <= metrics["accuracy_at_threshold"] <= 1.0
