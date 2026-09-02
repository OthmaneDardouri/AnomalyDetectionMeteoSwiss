"""Tests for score-distribution drift detection.

The unit tests below pin the decision boundaries with synthetic score arrays.
The integration test at the bottom is the one that matters: it trains
PatchCore on ``screw`` and then scores ``wood`` images through it, which is a
guaranteed distribution shift, and asserts the detector actually fires.
"""
import json

import numpy as np
import pytest

from anom_detect.dataset_preprocessor import MVTecAD2
from anom_detect.drift import (
    Reference,
    compute_drift,
    load_reference,
    load_scores,
)
from anom_detect.patchcore.patchcore_class import PatchCoreManager


def _reference(scores, threshold):
    return Reference(threshold=threshold, scores=list(scores))


def test_same_distribution_is_not_drift():
    rng = np.random.default_rng(0)
    reference = _reference(rng.normal(10, 1, 300), threshold=13.0)
    live = rng.normal(10, 1, 100)

    report = compute_drift(reference, live)

    assert report.drifted is False
    assert report.n_live == 100


def test_shifted_distribution_is_drift():
    rng = np.random.default_rng(0)
    reference = _reference(rng.normal(10, 1, 300), threshold=13.0)
    live = rng.normal(14, 1, 100)  # whole distribution moved past the threshold

    report = compute_drift(reference, live)

    assert report.drifted is True
    assert report.live_anomaly_rate > 0.5
    assert any("anomaly rate" in reason for reason in report.reasons)


def test_shift_below_threshold_is_not_drift():
    """A distribution shift that never crosses the threshold isn't flagged.

    This is the training-fit-reference case: held-out scores can sit
    slightly higher than training scores without anything having drifted.
    """
    rng = np.random.default_rng(0)
    reference = _reference(rng.normal(10, 1, 300), threshold=20.0)
    live = rng.normal(11, 1, 100)  # shifted, but still far below the threshold

    report = compute_drift(reference, live)

    assert report.drifted is False


def test_anomaly_rate_jump_is_drift():
    reference = _reference([1.0] * 100, threshold=1.5)
    live = [2.0] * 50  # every live image trips the threshold

    report = compute_drift(reference, live)

    assert report.drifted is True
    assert report.reference_anomaly_rate == 0.0
    assert report.live_anomaly_rate == 1.0
    assert report.anomaly_rate_delta == pytest.approx(1.0)
    assert any("anomaly rate" in reason for reason in report.reasons)


def test_too_few_live_scores_returns_no_verdict():
    """Below the sample floor the answer is 'don't know', never 'no drift'."""
    rng = np.random.default_rng(0)
    reference = _reference(rng.normal(10, 1, 300), threshold=13.0)

    report = compute_drift(reference, [99.0, 99.0, 99.0])

    assert report.drifted is False
    assert "insufficient data" in report.reasons[0]


def test_summary_only_reference_has_no_verdict():
    """Managers that save just mean/std/threshold get no reference rate to compare."""
    reference = Reference(threshold=13.0)
    assert reference.anomaly_rate is None

    report = compute_drift(reference, [20.0] * 50)

    assert report.drifted is False
    assert report.reference_anomaly_rate is None
    assert report.live_anomaly_rate == 1.0
    assert any("rate comparison skipped" in note for note in report.notes)


def test_report_round_trips_through_json():
    reference = _reference([1.0, 2.0, 3.0] * 40, threshold=2.5)
    report = compute_drift(reference, [5.0] * 40)

    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["drifted"] is True
    assert payload["n_live"] == 40
    assert isinstance(report.summary(), str)


def test_load_scores_accepts_both_json_shapes(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([1.0, 2.0]), encoding="utf-8")
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"scores": [1.0, 2.0]}), encoding="utf-8")

    assert load_scores(bare) == [1.0, 2.0] == load_scores(wrapped)


def test_load_reference_reports_a_missing_run_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No training reference"):
        load_reference(tmp_path / "never_trained")


@pytest.mark.slow
def test_screw_model_scoring_wood_images_is_detected_as_drift(tmp_path, subset_config_path):
    """The end-to-end check: a real model, real images, a real shift.

    A PatchCore memory bank built from screw images has never seen wood, so
    wood scores land far from the screw training distribution. If the detector
    misses this, it will miss anything.
    """
    train_path = tmp_path / "train"
    manager = PatchCoreManager("screw", str(subset_config_path), str(train_path), str(train_path))
    manager.train()
    _mean_error, _std_error, threshold = manager.compute_thresh()

    # In-distribution control: screw test images through the screw model.
    screw_scores = manager._score_dataset(manager.test_dataset, desc="screw")

    wood_dataset = MVTecAD2(
        "wood", "test", transform=manager.transform,
        config_path=str(subset_config_path),
    )
    wood_scores = manager._score_dataset(wood_dataset, desc="wood")

    reference = Reference(threshold=threshold, scores=manager.train_scores)

    wood_report = compute_drift(reference, wood_scores)
    assert wood_report.drifted is True, wood_report.summary()
    assert wood_report.live_anomaly_rate > 0.5

    # And the control must look markedly less shifted: screw test images come
    # from the same class the memory bank was built from. This is the half
    # that catches a detector which simply always says "drift".
    screw_report = compute_drift(reference, screw_scores)
    assert screw_report.live_anomaly_rate < wood_report.live_anomaly_rate
