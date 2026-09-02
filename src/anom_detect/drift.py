"""Score-distribution drift detection.

Unsupervised: production has no labels, so "has the model degraded?" cannot
be answered. What *can* be: has the **anomaly rate** -- the fraction of live
scores above the trained threshold -- moved from the rate training implied?

That one signal is deliberately all this reports, and it's *calibrated*: the
threshold and the rate share a reference, so its bias cancels on both sides.

A verdict needs training scores on disk. DFR persists them in
``<class>_thresholds.yaml`` and PatchCore in ``memory_bank.pth``. The other
managers save only mean/std/threshold, so they get the live rate and no
verdict -- a real gap, not a filled-in guess.
"""
import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from anom_detect.logging_utils import get_logger

logger = get_logger(__name__)

# A live anomaly rate this far from the reference rate is worth flagging.
DEFAULT_RATE_DELTA = 0.20
# Below this many live scores, the comparison is noise; report "insufficient
# data" instead of a verdict nobody should act on.
DEFAULT_MIN_SAMPLES = 10


@dataclass
class Reference:
    """The training-time baseline a live batch is compared against."""

    threshold: float
    scores: Optional[list[float]] = None
    source: str = ""

    @property
    def anomaly_rate(self) -> Optional[float]:
        """Fraction of training scores above the threshold, if scores are known."""
        if not self.scores:
            return None
        return float(np.mean(np.asarray(self.scores) > self.threshold))


@dataclass
class DriftReport:
    """The verdict, plus every number it was derived from."""

    drifted: bool
    reasons: list[str]
    n_reference: int
    n_live: int
    threshold: float
    live_anomaly_rate: float
    reference_anomaly_rate: Optional[float] = None
    anomaly_rate_delta: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        verdict = "DRIFT DETECTED" if self.drifted else "no drift"
        lines = [
            f"  Verdict                 {verdict}",
            f"  Live samples            {self.n_live}",
            f"  Reference samples       {self.n_reference}",
            f"  Threshold               {self.threshold:.4f}",
            f"  Live anomaly rate       {self.live_anomaly_rate:.1%}",
        ]
        if self.reference_anomaly_rate is not None:
            lines.append(f"  Reference anomaly rate  {self.reference_anomaly_rate:.1%}")
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        for note in self.notes:
            lines.append(f"  . {note}")
        return "\n".join(lines)


def compute_drift(
    reference: Reference,
    live_scores: Sequence[float],
    rate_delta_threshold: float = DEFAULT_RATE_DELTA,
    min_live_samples: int = DEFAULT_MIN_SAMPLES,
) -> DriftReport:
    """Compare ``live_scores`` against the training-time ``reference``.

    Drift when the live anomaly rate moves more than ``rate_delta_threshold``
    from the reference rate. With no reference rate, the live rate is still
    reported but no verdict is possible.
    """
    live = np.asarray(list(live_scores), dtype=float)

    if live.size < min_live_samples:
        return DriftReport(
            drifted=False,
            reasons=[
                f"insufficient data: {live.size} live scores, "
                f"{min_live_samples} needed for a verdict"
            ],
            n_reference=len(reference.scores) if reference.scores else 0,
            n_live=int(live.size),
            threshold=reference.threshold,
            live_anomaly_rate=float(np.mean(live > reference.threshold)) if live.size else 0.0,
            reference_anomaly_rate=reference.anomaly_rate,
        )

    live_rate = float(np.mean(live > reference.threshold))
    reference_rate = reference.anomaly_rate
    reasons: list[str] = []
    notes: list[str] = []

    rate_delta = None
    if reference_rate is not None:
        rate_delta = live_rate - reference_rate
        if abs(rate_delta) > rate_delta_threshold:
            reasons.append(
                f"anomaly rate moved {rate_delta:+.1%} "
                f"({reference_rate:.1%} -> {live_rate:.1%})"
            )
    else:
        notes.append("reference has no known anomaly rate; rate comparison skipped")

    return DriftReport(
        drifted=bool(reasons),
        reasons=reasons or ["live scores are consistent with training"],
        notes=notes,
        n_reference=len(reference.scores) if reference.scores else 0,
        n_live=int(live.size),
        threshold=reference.threshold,
        live_anomaly_rate=live_rate,
        reference_anomaly_rate=reference_rate,
        anomaly_rate_delta=rate_delta,
    )


def _reference_from_dfr_thresholds(path: Path) -> Reference:
    """Read a DFR ``<class>_thresholds.yaml``.

    ``thresholds`` holds one float or one per sigma multiplier; index 0 is the
    operating point, matching what ``serve`` and ``DeepFeatureADManager.test`` use.
    """
    info = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("thresholds", {})
    thresholds = info.get("thresholds")
    if isinstance(thresholds, list):
        if not thresholds:
            raise ValueError(f"{path} contains an empty threshold list.")
        thresholds = thresholds[0]
    if thresholds is None:
        raise ValueError(f"{path} has no 'thresholds' value.")

    scores = info.get("train_scores")
    scores = [float(s) for s in scores] if scores else None
    return Reference(threshold=float(thresholds), scores=scores, source=str(path))


def load_reference(train_output_dir: Path, product_class: Optional[str] = None) -> Reference:
    """Build a :class:`Reference` from what a training run left on disk.

    Three shapes, in order of preference: ``<class>_thresholds.yaml`` (DFR,
    needs ``product_class``), ``memory_bank.pth`` (PatchCore, full score
    array), then ``training_statistics.yaml`` (no scores, so no rate).
    """
    train_output_dir = Path(train_output_dir)
    memory_bank = train_output_dir / "memory_bank.pth"
    stats_file = train_output_dir / "training_statistics.yaml"

    if product_class:
        for name in (
            f"{product_class}_thresholds.yaml",
            f"{product_class}_foundational_thresholds.yaml",
        ):
            if (train_output_dir / name).is_file():
                return _reference_from_dfr_thresholds(train_output_dir / name)

    if memory_bank.is_file():
        # weights_only=True: this only ever needs the tensors and floats it
        # wrote itself, and a checkpoint is untrusted input.
        checkpoint = torch.load(memory_bank, map_location="cpu", weights_only=True)
        scores = checkpoint.get("train_scores")
        if scores:
            return Reference(
                threshold=float(checkpoint["threshold"]),
                scores=[float(s) for s in scores],
                source=str(memory_bank),
            )

    if stats_file.is_file():
        stats = yaml.safe_load(stats_file.read_text(encoding="utf-8")) or {}
        if "threshold" not in stats:
            raise ValueError(f"{stats_file} is missing key: 'threshold'")
        return Reference(threshold=float(stats["threshold"]), source=str(stats_file))

    hint = "" if product_class else " (pass --product-class for a deep-feature run)"
    raise FileNotFoundError(
        f"No training reference in {train_output_dir}{hint}. Expected "
        "<class>_thresholds.yaml, memory_bank.pth, or training_statistics.yaml "
        "from a completed `--mode train` run."
    )


def load_scores(path: Path) -> list[float]:
    """Read live scores from a JSON file: either a bare list or ``{"scores": [...]}."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores") if isinstance(payload, dict) else payload
    if not isinstance(scores, list):
        raise ValueError(f"{path} must hold a JSON list of scores or a 'scores' key.")
    return [float(s) for s in scores]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a batch of anomaly scores against a training run's baseline."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Training output directory, e.g. runs/train/patchcore_wood/train_patchcore_wood",
    )
    parser.add_argument(
        "--live",
        required=True,
        type=Path,
        help="JSON file of live scores, e.g. runs/test/patchcore_wood/scores.json",
    )
    parser.add_argument(
        "--product-class",
        default=None,
        help="Required for deep-feature runs, whose baseline file is named "
        "<class>_thresholds.yaml",
    )
    parser.add_argument("--rate-delta", type=float, default=DEFAULT_RATE_DELTA)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON instead of a table"
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit 1 when drift is detected, for use as a CI or cron gate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    reference = load_reference(args.reference, args.product_class)
    report = compute_drift(
        reference,
        load_scores(args.live),
        rate_delta_threshold=args.rate_delta,
        min_live_samples=args.min_samples,
    )

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"\n  Reference               {reference.source}")
        print(report.summary())
        print()

    return 1 if (report.drifted and args.fail_on_drift) else 0


if __name__ == "__main__":
    raise SystemExit(main())
