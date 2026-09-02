"""Tests for the FastAPI serving layer.

Run against a real DFR model trained on the committed ``wood`` subset, so
``/predict`` exercises the deployed path: image bytes in, score out. Marked
``slow`` because training needs the ImageNet ResNet50 weights.

The model is trained once per module; each test builds its own client over
that checkpoint, keeping the in-memory drift window isolated per test.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="serving extras not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

import train_test  # noqa: E402
from anom_detect.serve import DeepFeatureScorer, create_app  # noqa: E402

PRODUCT_CLASS = "wood"
SUBSET_ROOT = Path(__file__).resolve().parent.parent / "data" / "mvtec_subset"
# The subset ships 8 good test images per class. Only those work as an
# in-distribution control; train/good images *are* the reference distribution.
MIN_DRIFT_SAMPLES = 8


def test_missing_checkpoint_names_the_training_command(tmp_path, subset_config_path):
    """Startup must fail loudly, with the command that fixes it."""
    with pytest.raises(FileNotFoundError, match="--model_name deep_feature_ad"):
        DeepFeatureScorer(str(subset_config_path), PRODUCT_CLASS, str(tmp_path))


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, subset_config_path):
    """Train deep_feature_ad on wood once via the CLI; returns the train path."""
    train_path = tmp_path_factory.mktemp("serve_train")
    argv = sys.argv
    sys.argv = [
        "train_test.py",
        "--config", str(subset_config_path),
        "--product_class", PRODUCT_CLASS,
        "--model_name", "deep_feature_ad",
        "--train_path", str(train_path),
        "--test_path", str(train_path),
        "--mode", "train",
    ]
    try:
        train_test.main()
    finally:
        sys.argv = argv

    assert (train_path / "checkpoints" / f"{PRODUCT_CLASS}_dfad_weights.pth").is_file()
    return train_path


@pytest.fixture
def client(trained_run, subset_config_path):
    """A client with its own empty drift window, over the shared checkpoint."""
    app = create_app(
        str(subset_config_path),
        PRODUCT_CLASS,
        str(trained_run),
        min_drift_samples=MIN_DRIFT_SAMPLES,
    )
    with TestClient(app) as test_client:
        yield test_client


def _good_images(limit):
    paths = sorted((SUBSET_ROOT / PRODUCT_CLASS / "test" / "good").glob("*.png"))[:limit]
    assert len(paths) == limit, f"subset has too few good {PRODUCT_CLASS} images"
    return paths


def _post_image(client, path):
    return client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/png")})


@pytest.mark.slow
def test_health_reports_the_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "deep_feature_ad"
    assert body["product_class"] == PRODUCT_CLASS
    assert body["threshold"] > 0
    # Training persists its score distribution, so a reference must exist.
    assert body["reference_samples"] > 0
    assert body["scored_since_start"] == 0


@pytest.mark.slow
def test_predict_scores_an_uploaded_image(client):
    image_path = _good_images(1)[0]

    response = _post_image(client, image_path)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == image_path.name
    assert body["score"] > 0
    assert body["threshold"] > 0
    assert isinstance(body["is_anomalous"], bool)
    assert client.get("/health").json()["scored_since_start"] == 1


@pytest.mark.slow
def test_defective_image_scores_above_a_good_one(client):
    """Checks the wiring (transform, weights, top-k scoring), not accuracy.

    A serving layer applying the wrong normalisation passes every other test
    here and fails this one.
    """
    good = sorted((SUBSET_ROOT / PRODUCT_CLASS / "test" / "good").glob("*.png"))
    defect = sorted((SUBSET_ROOT / PRODUCT_CLASS / "test" / "hole").glob("*.png"))

    good_scores = [_post_image(client, p).json()["score"] for p in good]
    defect_scores = [_post_image(client, p).json()["score"] for p in defect]

    assert max(defect_scores) > min(good_scores)


@pytest.mark.slow
def test_predict_rejects_a_non_image_upload(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"this is not a png", "text/plain")}
    )

    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"].lower()


@pytest.mark.slow
def test_predict_rejects_an_empty_upload(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})

    assert response.status_code == 400


@pytest.mark.slow
def test_drift_withholds_a_verdict_until_enough_images(client):
    """A fresh window must answer 'insufficient data', never 'no drift'."""
    body = client.get("/drift").json()

    assert body["drifted"] is False
    assert body["n_live"] == 0
    assert "insufficient data" in body["reasons"][0]


@pytest.mark.slow
def test_drift_reports_a_verdict_once_the_window_fills(client):
    for path in _good_images(MIN_DRIFT_SAMPLES):
        assert _post_image(client, path).status_code == 200

    body = client.get("/drift").json()

    assert body["n_live"] == MIN_DRIFT_SAMPLES
    assert body["reference_anomaly_rate"] is not None
    assert body["n_reference"] > 0
    assert isinstance(body["drifted"], bool)
    assert body["reasons"]
