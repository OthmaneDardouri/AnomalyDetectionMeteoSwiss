"""Optional MLflow tracking for one CLI train/test step.

Opt-in via ``--mlflow`` or ``ANOM_DETECT_MLFLOW=1``. When off -- or when
``mlflow`` isn't installed -- :func:`track_run` is a no-op context manager, so
``mlflow`` stays an optional dependency.

It lives here rather than in the five managers because the CLI already knows
the model, config and output directory, and each ``test()`` returns its
metrics. No manager had to change.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import yaml

from anom_detect.logging_utils import get_logger

logger = get_logger(__name__)

# ``test()`` returns a plain ROC-AUC float for most models; ViT returns a
# (PRO, image ROC-AUC, PR-AUC) triple instead. Anything else is skipped.
_VIT_METRIC_NAMES = ("pro_score", "roc_auc_image", "auc_pr")

# Config key holding the per-model hyperparameters, and the CLI model name ->
# config section mapping (they differ for historical reasons).
_CONFIG_SECTION = {
    "base": "base_autoencoder",
    "trafo": "trafo_autoencoder",
    "vit": "vit_autoencoder",
    "patchcore": "patchcore",
    "deep_feature_ad": "DeepFeatureAE",
}

# Small text outputs worth keeping with the run; model weights and the
# segmentation PNGs are deliberately left on disk.
_ARTIFACT_SUFFIXES = (".json", ".txt", ".yaml", ".yml")
# Relative on purpose -- see the comment in MLflowRun.start().
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_MAX_ARTIFACTS = 50


def is_enabled(flag: bool = False) -> bool:
    """True if MLflow logging was requested via ``--mlflow`` or the env var."""
    return bool(flag) or os.environ.get("ANOM_DETECT_MLFLOW", "").lower() in ("1", "true", "yes")


@contextmanager
def track_run(args: Any, enabled: bool = False) -> Iterator[Optional["MLflowRun"]]:
    """Yield an :class:`MLflowRun` for ``args``, or ``None`` when disabled.

    Never raises because of MLflow: a missing package or unreachable server
    warns and degrades to the no-op path. A run is never lost to tracking.
    """
    if not is_enabled(enabled):
        yield None
        return

    try:
        import mlflow
    except ImportError:
        logger.warning("--mlflow was requested but the 'mlflow' package is not installed; skipping.")
        yield None
        return

    run = MLflowRun(mlflow, args)
    try:
        run.start()
    except Exception as exc:  # noqa: BLE001 - tracking must never break a run
        logger.warning("Could not start MLflow run (%s); continuing without tracking.", exc)
        yield None
        return

    try:
        yield run
    finally:
        run.close()


class MLflowRun:
    """Thin wrapper around the handful of ``mlflow`` calls the CLI makes."""

    def __init__(self, mlflow_module: Any, args: Any) -> None:
        self._mlflow = mlflow_module
        self._args = args
        self._active = False

    def start(self) -> None:
        # MLflow 3 refuses the plain ./mlruns file store unless asked.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            # MLflow's default builds an absolute sqlite URI from the cwd and
            # never decodes it, so a path with spaces or parentheses writes to
            # a literal "...%20%281%29..." directory. A relative URI avoids it.
            self._mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        self._mlflow.set_experiment(
            os.environ.get("ANOM_DETECT_MLFLOW_EXPERIMENT", "anom-detect")
        )
        self._mlflow.start_run(
            run_name=f"{self._args.model_name}-{self._args.product_class}-{self._args.mode}"
        )
        self._active = True
        self._log_params()
        logger.info(
            "MLflow tracking to %s (experiment 'anom-detect').", self._mlflow.get_tracking_uri()
        )

    def _log_params(self) -> None:
        params = {
            "model_name": self._args.model_name,
            "product_class": self._args.product_class,
            "mode": self._args.mode,
        }
        params.update(_hyperparameters(self._args))
        self._safe(self._mlflow.log_params, params)

    def log_result(self, result: Any) -> None:
        """Log whatever ``train()``/``test()`` returned, if it looks numeric."""
        metrics = _as_metrics(self._args.model_name, result)
        if metrics:
            self._safe(self._mlflow.log_metrics, metrics)

    def log_metrics(self, metrics: dict) -> None:
        self._safe(self._mlflow.log_metrics, {k: float(v) for k, v in metrics.items()})

    def log_output_files(self, directory: str) -> None:
        """Attach the small text/JSON outputs written under ``directory``."""
        root = Path(directory)
        if not root.is_dir():
            return
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _ARTIFACT_SUFFIXES
        )[:_MAX_ARTIFACTS]
        for path in files:
            # MLflow rejects an artifact_path of "."; root files need no sub-path.
            relative = path.parent.relative_to(root).as_posix()
            self._safe(
                self._mlflow.log_artifact,
                str(path),
                artifact_path=None if relative == "." else relative,
            )

    def close(self) -> None:
        if self._active:
            self._safe(self._mlflow.end_run)
            self._active = False

    def _safe(self, func: Any, *call_args: Any, **kwargs: Any) -> None:
        try:
            func(*call_args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tracking must never break a run
            logger.warning("MLflow call %s failed: %s", getattr(func, "__name__", func), exc)


def _hyperparameters(args: Any) -> dict:
    """Read the model's section of the YAML config, flattened into params."""
    try:
        with open(args.config) as file:
            config = yaml.safe_load(file) or {}
    except OSError:
        return {}
    section = _CONFIG_SECTION.get(args.model_name, args.model_name)
    params = dict(config.get("MODELS_CONFIG", {}).get(section, {}))
    params["seed"] = config.get("SEED", 42)
    return {str(key): str(value) for key, value in params.items()}


def _as_metrics(model_name: str, result: Any) -> dict:
    """Turn a manager's return value into a ``{name: float}`` metric mapping."""
    if isinstance(result, bool) or result is None:
        return {}
    if isinstance(result, (int, float)):
        return {"roc_auc_image": float(result)}
    if model_name == "vit" and isinstance(result, tuple) and len(result) == len(_VIT_METRIC_NAMES):
        return {
            name: float(value)
            for name, value in zip(_VIT_METRIC_NAMES, result)
            if isinstance(value, (int, float))
        }
    if isinstance(result, dict):
        return {
            str(key): float(value)
            for key, value in result.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    return {}
