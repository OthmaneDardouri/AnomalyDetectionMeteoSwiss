"""Tests train_test.py's own argument parsing and branching, not just the
manager classes -- the way ``python train_test.py --model_name ...`` runs."""
import argparse
import sys

import pytest

import train_test
from anom_detect import cli

PRODUCT_CLASS = "toy"


def test_mode_all_runs_train_then_test(tiny_config_path, tmp_path, monkeypatch):
    """--mode all must dispatch a train step and then a test step."""
    steps = []
    monkeypatch.setattr(cli, "_run_step", lambda args: steps.append(args.mode))

    cli.run(
        argparse.Namespace(
            config=str(tiny_config_path),
            product_class=PRODUCT_CLASS,
            model_name="patchcore",
            train_path=str(tmp_path / "train"),
            test_path=str(tmp_path / "test"),
            mode="all",
            num_seg_samples="5",
        )
    )

    assert steps == ["train", "test"]


@pytest.mark.slow
def test_deep_feature_ad_train_and_test_via_cli(tmp_path, tiny_config_path, monkeypatch):
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_test.py",
            "--config", str(tiny_config_path),
            "--product_class", PRODUCT_CLASS,
            "--model_name", "deep_feature_ad",
            "--train_path", str(train_path),
            "--test_path", str(test_path),
            "--mode", "train",
        ],
    )
    train_test.main()

    assert (train_path / "checkpoints" / f"{PRODUCT_CLASS}_dfad_weights.pth").is_file()
    assert (train_path / f"{PRODUCT_CLASS}_thresholds.yaml").is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_test.py",
            "--config", str(tiny_config_path),
            "--product_class", PRODUCT_CLASS,
            "--model_name", "deep_feature_ad",
            "--train_path", str(train_path),
            "--test_path", str(test_path),
            "--mode", "test",
        ],
    )
    train_test.main()  # must not raise

    assert (test_path / "roc_curve.png").is_file()
