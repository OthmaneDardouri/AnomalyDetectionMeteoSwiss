"""Turn a deep-feature-AD training run into a small, committable serving bundle.

A training checkpoint is ~115 MB, nearly all of it the frozen ResNet50 that
the detector reloads from torchvision anyway. Dropping those keys leaves the
~12 MB the run actually produced, so a ready-to-serve model fits in the repo.

    python scripts/export_serving_model.py \
        --train-path runs/train/deep_feature_ad_wood \
        --product-class wood --output models/dfr_wood

It writes the layout ``anom_detect.serve`` expects, so a bundle and a real run
are interchangeable there.
"""
import argparse
import shutil
from pathlib import Path

import torch
import yaml

BACKBONE_MARKER = "feature_extractor.backbone."
# The backbone's frozen ImageNet *parameters* are pure duplication (~100 MB).
# Its BatchNorm *buffers* are not -- the detector trains in train() mode, so
# those running stats adapt to the class. At 0.2 MB the bundle keeps them and
# stays numerically identical to the full checkpoint.
BACKBONE_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-path", required=True, help="The --train_path of a deep_feature_ad run"
    )
    parser.add_argument("--product-class", required=True, help="e.g. wood")
    parser.add_argument("--output", required=True, help="Directory to write the bundle to")
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Which calibrated threshold becomes the operating point, as its "
        "sigma multiplier (e.g. 1.0 for mean+1*std). `serve` reads the first "
        "entry, so this reorders the file. Defaults to training's own order.",
    )
    return parser.parse_args()


def _select_operating_threshold(thresholds_path: Path, sigma: float) -> None:
    """Move the ``sigma``-multiplier threshold to the front of the file.

    ``serve.load_threshold`` reads the first entry, so reordering here avoids
    a second way to pick an operating point at serving time.
    """
    document = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    block = document["thresholds"]
    multipliers = [float(m) for m in block["sigma_multiplier"]]
    if sigma not in multipliers:
        raise ValueError(
            f"--sigma {sigma} was not calibrated; this run has {multipliers}."
        )
    index = multipliers.index(sigma)
    order = [index] + [i for i in range(len(multipliers)) if i != index]
    block["sigma_multiplier"] = [block["sigma_multiplier"][i] for i in order]
    block["thresholds"] = [block["thresholds"][i] for i in order]
    thresholds_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"Operating threshold: mean+{sigma}*std = {block['thresholds'][0]:.4f}")


def export(train_path: Path, product_class: str, output: Path, sigma=None) -> Path:
    """Write the trimmed bundle and return the weights file it produced."""
    weights_path = train_path / "checkpoints" / f"{product_class}_dfad_weights.pth"
    thresholds_path = train_path / f"{product_class}_thresholds.yaml"
    if not weights_path.is_file():
        raise FileNotFoundError(f"No trained weights at {weights_path}")
    if not thresholds_path.is_file():
        raise FileNotFoundError(
            f"No thresholds at {thresholds_path}; run --mode train to completion first."
        )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    trimmed = {
        key: value
        for key, value in state_dict.items()
        if BACKBONE_MARKER not in key or key.endswith(BACKBONE_BUFFERS)
    }
    if not trimmed:
        raise ValueError(f"{weights_path} contained only backbone weights; nothing to export.")

    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    out_weights = output / "checkpoints" / weights_path.name
    torch.save(trimmed, out_weights)
    shutil.copy2(thresholds_path, output / thresholds_path.name)
    if sigma is not None:
        _select_operating_threshold(output / thresholds_path.name, sigma)

    # Records the architecture the weights belong to, so the bundle documents itself.
    model_config = weights_path.parent / f"{product_class}_dfad_config.yaml"
    if model_config.is_file():
        shutil.copy2(model_config, output / "checkpoints" / model_config.name)

    print(
        f"{out_weights} ({out_weights.stat().st_size / 1e6:.1f} MB, "
        f"down from {weights_path.stat().st_size / 1e6:.1f} MB): "
        f"{len(state_dict) - len(trimmed)} frozen backbone tensors dropped, "
        f"{len(trimmed)} kept."
    )
    return out_weights


def main() -> None:
    args = parse_arguments()
    export(Path(args.train_path), args.product_class, Path(args.output), args.sigma)


if __name__ == "__main__":
    main()
