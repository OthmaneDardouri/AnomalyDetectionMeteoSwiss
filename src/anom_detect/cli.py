"""Command-line entry point for training and testing every model.

Run via the ``anom-detect`` console script, ``python -m anom_detect.cli``, or
the ``train_test.py`` shim at the repository root.
"""
import argparse
from pathlib import Path
from typing import Union

from anom_detect.base_model.base_autoencoder import BaseAEManager
from anom_detect.deep_feature_ad.deep_feature_ad_manager import DeepFeatureADManager
from anom_detect.logging_utils import get_logger
from anom_detect.patchcore.patchcore_class import PatchCoreManager
from anom_detect.tracking import track_run
from anom_detect.trafo_model.trafo_autoencoder import TransAEManager
from anom_detect.vit_model.ViT import ViTManager

logger = get_logger(__name__)

MANAGERS = {
    "base": BaseAEManager,
    "trafo": TransAEManager,
    "vit": ViTManager,
    "patchcore": PatchCoreManager,
    "deep_feature_ad": DeepFeatureADManager,
}


def _default_config_path() -> Path:
    """``config.yaml`` at the repository root."""
    return Path(__file__).parent.parent.parent / "config.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--product_class", default="hazelnut", help="class name or 'all'")
    parser.add_argument("--model_name", choices=tuple(MANAGERS), default="vit")
    parser.add_argument("--train_path", default="runs/train")
    parser.add_argument("--test_path", default="runs/test")
    parser.add_argument(
        "--mode",
        choices=("train", "test", "all"),
        default="train",
        help="'all' runs training followed by evaluation",
    )
    parser.add_argument(
        "--num_seg_samples",
        default="5",
        help="How many test images to save as segmentation comparison PNGs, "
        "or 'all' for every one.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log params, metrics and small output files to MLflow. Needs "
        "`pip install mlflow`. Equivalent to ANOM_DETECT_MLFLOW=1.",
    )
    return parser.parse_args()


def parse_num_seg_samples(value: str) -> Union[int, None]:
    """An int, or ``None`` for 'all' (no limit)."""
    if value.strip().lower() == "all":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("--num_seg_samples must be non-negative or 'all'.")
    return parsed


def build_manager(args: argparse.Namespace):
    """Construct the manager for ``args.model_name``."""
    try:
        manager_cls = MANAGERS[args.model_name]
    except KeyError:
        raise ValueError(f"Model name '{args.model_name}' not defined.") from None
    return manager_cls(
        args.product_class,
        args.config,
        args.train_path,
        args.test_path,
        num_seg_samples=parse_num_seg_samples(args.num_seg_samples),
    )


def run(args: argparse.Namespace) -> None:
    """Run the requested ``args.mode``; ``all`` means train then test.

    Each step gets its own namespace: ``save_model`` dumps ``vars(args)`` into
    ``args.yaml``, so a shared one would record the wrong mode.
    """
    if not Path(args.config).is_file():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    modes = ("train", "test") if args.mode == "all" else (args.mode,)
    for mode in modes:
        _run_step(argparse.Namespace(**{**vars(args), "mode": mode}))


def _run_step(args: argparse.Namespace) -> None:
    """Run a single train or test step.

    ``train()``/``test()`` are called identically for every manager; only the
    threshold step below genuinely differs per model. ``run`` is None unless
    --mlflow was given, and every method on it swallows tracking errors.
    """
    manager = build_manager(args)

    with track_run(args, getattr(args, "mlflow", False)) as run:
        if args.mode != "train":
            result = manager.test()
            if run is not None:
                run.log_result(result)
                run.log_output_files(args.test_path)
            return

        manager.train()
        if args.model_name == "vit":
            # Calibrate on training data. Using test masks here would leak labels.
            threshold = manager.training_threshold()
            manager.save_model(args, threshold)
        elif args.model_name == "deep_feature_ad":
            threshold = manager.compute_threshold()
        else:
            mean_error, std_error, threshold = manager.compute_thresh()
            manager.save_model(args, mean_error, std_error, threshold)

        if run is not None:
            if threshold is not None:
                run.log_metrics({"threshold": threshold})
            run.log_output_files(args.train_path)


def main() -> None:
    run(parse_arguments())


if __name__ == "__main__":
    main()
