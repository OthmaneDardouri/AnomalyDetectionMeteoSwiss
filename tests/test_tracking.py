"""Tests for the optional MLflow tracking layer (anom_detect.tracking).

These never import ``mlflow`` itself: the point is that tracking is opt-in,
that a missing/failing MLflow degrades to a no-op, and that manager return
values map to the right metric names.
"""
import argparse

import pytest

from anom_detect import tracking


def _args(**overrides):
    base = dict(
        model_name="patchcore",
        product_class="toy",
        mode="test",
        config="config.yaml",
        train_path="train",
        test_path="test",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ANOM_DETECT_MLFLOW", raising=False)
    assert tracking.is_enabled() is False
    with tracking.track_run(_args()) as run:
        assert run is None


def test_enabled_by_env_var(monkeypatch):
    monkeypatch.setenv("ANOM_DETECT_MLFLOW", "1")
    assert tracking.is_enabled() is True


def test_missing_mlflow_degrades_to_noop(monkeypatch):
    """--mlflow without the package installed must not break the run."""
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)
    real_import = __import__("builtins").__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(__import__("builtins"), "__import__", fake_import)
    with tracking.track_run(_args(), enabled=True) as run:
        assert run is None


@pytest.mark.parametrize(
    ("model_name", "result", "expected"),
    [
        ("patchcore", 0.93, {"roc_auc_image": 0.93}),
        ("deep_feature_ad", 0.5, {"roc_auc_image": 0.5}),
        ("base", None, {}),
        ("vit", (0.1, 0.2, 0.3), {"pro_score": 0.1, "roc_auc_image": 0.2, "auc_pr": 0.3}),
        ("base", {"accuracy": 0.8, "note": "x"}, {"accuracy": 0.8}),
    ],
)
def test_as_metrics(model_name, result, expected):
    assert tracking._as_metrics(model_name, result) == expected


def test_hyperparameters_read_the_model_section(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "SEED: 7\nMODELS_CONFIG:\n  patchcore:\n    batch_size: 1\n", encoding="utf-8"
    )
    params = tracking._hyperparameters(_args(config=str(config)))
    assert params == {"batch_size": "1", "seed": "7"}


def test_hyperparameters_missing_config_is_empty():
    assert tracking._hyperparameters(_args(config="does-not-exist.yaml")) == {}


def test_mlflow_failures_never_propagate():
    """Every logging call is best-effort; an exploding client must not raise."""

    class ExplodingMlflow:
        def log_metrics(self, *args, **kwargs):
            raise RuntimeError("tracking server down")

        def log_artifact(self, *args, **kwargs):
            raise RuntimeError("tracking server down")

        def end_run(self):
            raise RuntimeError("tracking server down")

    run = tracking.MLflowRun(ExplodingMlflow(), _args())
    run._active = True
    run.log_result(0.9)
    run.log_metrics({"threshold": 1.0})
    run.close()
