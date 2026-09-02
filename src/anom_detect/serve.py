"""FastAPI service around a trained deep-feature-AD (DFR) model.

``GET /health`` reports the loaded model, ``POST /predict`` scores an uploaded
image, and ``GET /drift`` compares the scores served so far against training.
DFR is served because it scores in one forward pass and needs only its weights
and threshold file at inference time.

The drift window lives in memory; a restart clears it, which is the right
trade for a few hundred floats.

    anom-detect-serve                                            # models/dfr_hazelnut
    anom-detect-serve --product-class wood --train-path runs/train/dfr_wood
"""
import argparse
import io
from collections import deque
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError

from anom_detect.deep_feature_ad.deep_feature_ad_manager import DeepFeatureADManager
from anom_detect.drift import DEFAULT_MIN_SAMPLES, Reference, compute_drift, load_reference
from anom_detect.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_WINDOW_SIZE = 500
# Reject oversized uploads before decoding; PIL allocates for decompression bombs.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
# Must match the trainer's top-k, or live scores are incomparable to the threshold.
ANOMALY_SCORE_TOP_K = 10


class DeepFeatureScorer:
    """Loads a trained DFR checkpoint and scores single images with it."""

    def __init__(self, config_path: str, product_class: str, train_path: str) -> None:
        self.product_class = product_class
        self.train_path = Path(train_path)

        weights_path = self.train_path / "checkpoints" / f"{product_class}_dfad_weights.pth"
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"No trained weights at {weights_path}. Train first with:\n"
                f"  python train_test.py --model_name deep_feature_ad "
                f"--product_class {product_class} --mode train --train_path {train_path}"
            )

        # Reuse the manager so live scoring gets the exact transform training
        # used. Its dataset access is lazy, so nothing is read from disk here.
        self._manager = DeepFeatureADManager(
            product_class, config_path, str(train_path), str(train_path)
        )
        self.device = self._manager.device
        self.transform = self._manager.transform
        self.detector = self._manager.detector
        _load_detector_weights(self.detector, weights_path, self.device)
        self.detector.eval()
        self.weights_path = weights_path
        self.reference = load_reference(self.train_path, product_class)
        self.threshold = self.reference.threshold

    def score(self, image: Image.Image) -> tuple:
        """Return ``(score, is_anomalous)`` for one PIL image."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features, reconstructed = self.detector(tensor)
            error_map = self.detector.compute_reconstruction_error(features, reconstructed)
            score = float(
                self.detector.compute_anomaly_score(error_map, k=ANOMALY_SCORE_TOP_K)[0]
            )
        return score, bool(score > self.threshold)


def _load_detector_weights(detector, weights_path: Path, device) -> None:
    """Load a DFR checkpoint, accepting bundles that omit the frozen backbone.

    ``scripts/export_serving_model.py`` strips the ~100 MB of frozen ResNet50
    weights the detector reloads anyway, so missing keys are tolerated only
    when every one of them belongs to that backbone.
    """
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    missing, unexpected = detector.load_state_dict(state_dict, strict=False)
    unexpected_keys = list(unexpected)
    stray = [key for key in missing if "feature_extractor.backbone." not in key]
    if unexpected_keys or stray:
        raise RuntimeError(
            f"Checkpoint {weights_path} does not match the configured model "
            f"(missing: {stray[:5]}, unexpected: {unexpected_keys[:5]}). The "
            "--config must use the same DeepFeatureAE settings the model was "
            "trained with."
        )


def create_app(
    config_path: str,
    product_class: str,
    train_path: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_drift_samples: int = DEFAULT_MIN_SAMPLES,
):
    """Build the FastAPI app with the model already loaded.

    A factory rather than a module-level ``app`` so tests can point it at the
    committed subset, and so a missing checkpoint fails at startup.
    """
    from fastapi import FastAPI, File, HTTPException, UploadFile

    scorer = DeepFeatureScorer(config_path, product_class, train_path)
    reference: Optional[Reference] = scorer.reference
    # Rolling window of served scores; a deque drops the oldest when full.
    window: deque = deque(maxlen=window_size)

    app = FastAPI(
        title="ANOM-DETECT",
        description="Anomaly scoring and score-drift monitoring for MVTec AD.",
        version="0.1.0",
    )
    app.state.scorer = scorer
    app.state.window = window

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": "deep_feature_ad",
            "product_class": product_class,
            "threshold": scorer.threshold,
            "reference_samples": len(reference.scores or []),
            "scored_since_start": len(window),
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty upload.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        try:
            image = Image.open(io.BytesIO(payload))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Not a readable image: {exc}") from exc

        score, is_anomalous = scorer.score(image)
        window.append(score)
        return {
            "filename": file.filename,
            "score": score,
            "threshold": scorer.threshold,
            "is_anomalous": is_anomalous,
        }

    @app.get("/drift")
    def drift() -> dict:
        report = compute_drift(reference, list(window), min_live_samples=min_drift_samples)
        return report.as_dict()

    return app


def parse_arguments() -> argparse.Namespace:
    repo_root = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser(description="Serve a trained deep-feature-AD model over HTTP.")
    # Defaults serve the committed models/dfr_hazelnut bundle: no dataset, no training.
    parser.add_argument("--config", default=str(repo_root / "config.yaml"))
    parser.add_argument("--product-class", default="hazelnut")
    parser.add_argument(
        "--train-path",
        default=str(repo_root / "models" / "dfr_hazelnut"),
        help="A completed deep_feature_ad --train_path, or a serving bundle like "
        "the committed models/dfr_hazelnut (the default)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    return parser.parse_args()


def main() -> None:
    # Parse first, so --help works without the serving stack installed.
    args = parse_arguments()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit(
            "Serving needs FastAPI and uvicorn, which aren't installed.\n"
            "  pip install -r requirements.txt"
        ) from exc

    app = create_app(
        args.config, args.product_class, args.train_path, window_size=args.window_size
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
