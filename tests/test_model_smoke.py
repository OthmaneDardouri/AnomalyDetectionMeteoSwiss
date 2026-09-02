"""One full train -> threshold -> save -> test cycle per model manager, on CPU
with one epoch of synthetic images. Proves the pipelines run, not that they
detect anything."""
import argparse

import pytest

from anom_detect.base_model.base_autoencoder import BaseAEManager
from anom_detect.patchcore.patchcore_class import PatchCoreManager
from anom_detect.trafo_model.trafo_autoencoder import TransAEManager
from anom_detect.vit_model.ViT import ViTManager

PRODUCT_CLASS = "toy"


def _args(config_path, train_path, test_path):
    return argparse.Namespace(
        config=str(config_path),
        product_class=PRODUCT_CLASS,
        model_name="smoke",
        train_path=str(train_path),
        test_path=str(test_path),
        mode="train",
    )


def test_base_autoencoder_train_and_test(tmp_path, tiny_config_path):
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    manager = BaseAEManager(PRODUCT_CLASS, str(tiny_config_path), str(train_path), str(test_path))

    manager.train()
    mean_error, std_error, threshold = manager.compute_thresh()
    manager.save_model(_args(tiny_config_path, train_path, test_path), mean_error, std_error, threshold)

    assert (train_path / "autoencoder_weights.pth").is_file()
    assert (train_path / "training_statistics.yaml").is_file()

    manager.test()  # must not raise


def test_trafo_autoencoder_train_and_test(tmp_path, tiny_config_path):
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    manager = TransAEManager(PRODUCT_CLASS, str(tiny_config_path), str(train_path), str(test_path))

    manager.train()
    mean_error, std_error, threshold = manager.compute_thresh()
    manager.save_model(_args(tiny_config_path, train_path, test_path), mean_error, std_error, threshold)

    assert (train_path / "autoencoder_weights.pth").is_file()

    manager.test()  # must not raise


def test_vit_manager_train_and_test(tmp_path, tiny_config_path):
    """Regression test: this used to crash on CPU-only machines because
    vit_model/spatial.py and vit_model/mdn1.py hardcoded `.cuda()`.
    """
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    manager = ViTManager(PRODUCT_CLASS, str(tiny_config_path), str(train_path), str(test_path))

    manager.train()
    threshold = manager.training_threshold()
    manager.save_model(_args(tiny_config_path, train_path, test_path), threshold)

    assert (train_path / "vit_weights.pth").is_file()
    assert (train_path / "g_weights.pth").is_file()

    pro_score, auc_score, auc_pr = manager.test()
    assert 0.0 <= pro_score <= 1.0
    assert 0.0 <= auc_score <= 1.0
    assert 0.0 <= auc_pr <= 1.0


@pytest.mark.slow
def test_patchcore_train_and_test(tmp_path, tiny_config_path):
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    manager = PatchCoreManager(PRODUCT_CLASS, str(tiny_config_path), str(train_path), str(test_path))

    manager.train()
    mean_error, std_error, threshold = manager.compute_thresh()
    manager.save_model(_args(tiny_config_path, train_path, test_path), mean_error, std_error, threshold)

    metrics = manager.test()
    assert 0.0 <= metrics["roc_auc_image"] <= 1.0
    assert 0.0 <= metrics["accuracy_at_threshold"] <= 1.0
