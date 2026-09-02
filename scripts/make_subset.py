"""Carve a small, committable subset out of the full ~5 GB MVTec AD dataset.

Copies downscaled images per class into a self-contained MVTec-AD layout the
pipelines train on unmodified. Selection is deterministic (evenly spaced
indices over the sorted file list), so re-running reproduces the same subset.

    python scripts/make_subset.py --source mvtec_anomaly_detection
    python scripts/make_subset.py --classes screw wood tile --size 512
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image

DEFAULT_CLASSES = ("screw", "wood")
# MVTec AD is CC BY-NC-SA 4.0: redistributing a subset is allowed, but the
# attribution and license terms have to travel with it.
ATTRIBUTION = """\
This directory contains a small downscaled subset of the MVTec Anomaly
Detection (MVTec AD) dataset, redistributed for testing and CI only.

  Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
  "MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly
  Detection." CVPR 2019.
  https://www.mvtec.com/company/research/datasets/mvtec-ad

Licensed by MVTec Software GmbH under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/

The images here have been resized and sampled; they are NOT suitable for
reporting benchmark numbers. Use the full dataset for that.
"""


def evenly_spaced(items: list, count: int) -> list:
    """Pick ``count`` items spread across ``items``.

    Spreading beats the first N: MVTec filenames are sequential, so
    consecutive shots of the same part are near-duplicates.
    """
    if count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def copy_image(src: Path, dest: Path, size: int, is_mask: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if size <= 0:
        shutil.copy2(src, dest)
        return
    with Image.open(src) as image:
        # Nearest-neighbour on masks: anything that interpolates invents
        # fractional labels along the defect boundary.
        resample = Image.NEAREST if is_mask else Image.LANCZOS
        image.resize((size, size), resample).save(dest)


def build_class(
    source_class: Path,
    dest_class: Path,
    size: int,
    n_train: int,
    n_test_good: int,
    n_per_defect: int,
) -> dict:
    counts = {"train": 0, "test": 0, "masks": 0}

    train_good = sorted((source_class / "train" / "good").glob("*.png"))
    if not train_good:
        raise FileNotFoundError(f"No training images under {source_class / 'train' / 'good'}")
    for path in evenly_spaced(train_good, n_train):
        copy_image(path, dest_class / "train" / "good" / path.name, size, is_mask=False)
        counts["train"] += 1

    for defect_dir in sorted(d for d in (source_class / "test").iterdir() if d.is_dir()):
        defect = defect_dir.name
        wanted = n_test_good if defect == "good" else n_per_defect
        for path in evenly_spaced(sorted(defect_dir.glob("*.png")), wanted):
            copy_image(path, dest_class / "test" / defect / path.name, size, is_mask=False)
            counts["test"] += 1
            if defect == "good":
                continue
            # Every non-good test image must bring its mask along: the test
            # split loader raises FileNotFoundError on a missing one.
            mask = source_class / "ground_truth" / defect / f"{path.stem}_mask.png"
            if not mask.is_file():
                raise FileNotFoundError(f"Ground-truth mask missing for {path}: {mask}")
            copy_image(
                mask,
                dest_class / "ground_truth" / defect / mask.name,
                size,
                is_mask=True,
            )
            counts["masks"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "mvtec_anomaly_detection",
        help="Directory containing the full per-class MVTec AD folders",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root / "data" / "mvtec_subset",
        help="Where to write the subset (deleted and rebuilt if it exists)",
    )
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Square edge length to resize to; 0 copies the originals untouched",
    )
    parser.add_argument("--train-images", type=int, default=20)
    parser.add_argument("--test-good", type=int, default=4)
    parser.add_argument("--test-per-defect", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Source dataset not found: {args.source}")

    missing = [name for name in args.classes if not (args.source / name).is_dir()]
    if missing:
        raise SystemExit(f"Classes not found under {args.source}: {', '.join(missing)}")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True)
    (args.dest / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")

    for name in args.classes:
        counts = build_class(
            args.source / name,
            args.dest / name,
            size=args.size,
            n_train=args.train_images,
            n_test_good=args.test_good,
            n_per_defect=args.test_per_defect,
        )
        print(
            f"{name}: {counts['train']} train, {counts['test']} test, "
            f"{counts['masks']} masks"
        )

    total = sum(p.stat().st_size for p in args.dest.rglob("*") if p.is_file())
    print(f"Wrote {args.dest} ({total / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
