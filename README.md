# ANOM-DETECT: Vision Anomaly Detection

[![CI](https://github.com/OthmaneDardouri/MVTEC2AnomalyDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/OthmaneDardouri/MVTEC2AnomalyDetection/actions/workflows/ci.yml)

Five unsupervised anomaly-detection models for industrial defect detection on
MVTec AD, behind one CLI: PatchCore, a ViT autoencoder, a deep-feature
reconstruction autoencoder (DFR), a CNN+Transformer autoencoder, and a plain
CNN autoencoder baseline. Every model trains on normal images only.

## Results

Image-level AUC-ROC, measured on this codebase:

| Model | Hazelnut | Leather | Wood | Carpet | Bottle |
| --- | --- | --- | --- | --- | --- |
| Base autoencoder | 0.66 | 0.41 | 0.59 | 0.46 | 0.63 |
| ViT autoencoder | 0.87 | 0.75 | 0.57 | 0.52 | 0.71 |
| **Deep feature reconstruction (DFR)** | **1.00** | **1.00** | **0.99** | **0.97** | **0.95** |

PatchCore and the Transformer autoencoder aren't in this sweep; PatchCore
scored 0.95 AUC-ROC on `toothbrush` in a separate run
([docs/sample_results/](docs/sample_results/)). These are single runs with no
seed averaging, so treat small gaps as noise.

**DFR wins because it reconstructs frozen ResNet50 features instead of raw
pixels.** The backbone already knows what edges and textures are, so the
autoencoder only has to learn what "normal" looks like in that space. It's the
model this repo serves. For reference, the original PatchCore paper (Roth et
al., CVPR 2022, [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)) reports
up to 99.6% mean image-level AUROC across all 15 MVTec AD classes — a
different codebase, hyperparameters and the full dataset, so not directly
comparable.

## Quick start (no download needed)

The repo ships a 5 MB real-image subset (`screw` + `wood`, in
[data/mvtec_subset/](data/mvtec_subset/)), so this runs immediately:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

python train_test.py --config config.subset.yaml --model_name patchcore \
  --product_class wood --mode all \
  --train_path runs/train/subset_wood --test_path runs/test/subset_wood
```

That trains, tests, and prints an AUC-ROC score in about a minute on CPU.
**These numbers aren't meaningful.** 20 training images is far too few; this
is here to prove the pipeline runs. Use the full dataset for real results.

Verify the install without any dataset at all:
```bash
pip install pytest ruff httpx && python -m pytest && python -m ruff check .
```
(`httpx` backs FastAPI's `TestClient`; without it the serving tests skip
rather than fail, which is easy to miss.)

## The full dataset

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('thtuan/mvtecad-mvtec-anomaly-detection'))"
```
Point `config.yaml` at the printed path (the folder that directly contains
`bottle/`, `cable/`, `hazelnut/`, ...):
```yaml
DATASET_PATH: "/path/to/mvtec_anomaly_detection"
```
It's git-ignored, so never commit it. Only the small subset above is tracked.
(Registration-gated official download: the
[MVTec AD dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad).)

## Training and testing

One pattern for every model:
```bash
python train_test.py --model_name <name> --product_class <class> --mode <mode> \
  --train_path runs/train/<run> --test_path runs/test/<run>
```
- `--model_name`: `base`, `vit`, `deep_feature_ad`, `trafo`, or `patchcore`
- `--mode`: `train`, `test`, or `all` (both, one process)
- Keep `--train_path`/`--test_path` names matched, so results stay traceable
  to the model that produced them.

```bash
python train_test.py --model_name patchcore --product_class toothbrush --mode all \
  --train_path runs/train/patchcore_toothbrush --test_path runs/test/patchcore_toothbrush
```
`toothbrush` is the smallest class (60 images), so it's the fastest way to
confirm real data works end to end. Any name in `DATASET_OBJECTS`
(config.yaml) works.

**Multi-class ("foundational") training** trains one shared DFR model across
several classes, saving one threshold per class:
```bash
python train_test.py --model_name deep_feature_ad --product_class foundational \
  --mode train --train_path runs/train/deep_feature_foundational
```

Per-epoch losses go to TensorBoard (`tensorboard --logdir runs/`).

## Experiment tracking (MLflow, optional)

**Example runs are committed under `examples/mlflow_runs/`** — 33 real runs on
the full MVTec dataset, with their params, metrics and output files (1.2 MB).
Browse them with no training and no dataset:
```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///examples/mlflow_runs/mlflow.db   # from the repo root
```

Three models on `wood`, and the DFR model that `serve` ships:

| Run | ROC AUC (image) | Notes |
| --- | --- | --- |
| `patchcore-wood` | 0.9842 | best on wood; no gradient training at all |
| `base-wood` | 0.9614 | plain pixel autoencoder baseline |
| `deep_feature_ad-wood` | 0.9281 | same architecture the service runs |
| `deep_feature_ad-hazelnut` | 1.0000 | the model in `models/dfr_hazelnut` |
| `patchcore-` bottle / hazelnut / leather | 1.0000 / 1.0000 / 0.9966 | one config, six classes |
| `patchcore-` carpet / toothbrush / zipper | 0.9707 / 0.9528 / 0.9512 | |

And a PatchCore hyperparameter sweep on `wood`, which is the part worth
opening in the UI, because it separates two things that are easy to conflate:

| memory bank ratio | threshold multiplier | ROC AUC | accuracy at threshold | wall clock |
| --- | --- | --- | --- | --- |
| 0.01 | 2.0 | 0.9789 | 0.9114 | 114 s |
| 0.05 | 2.0 | 0.9833 | 0.8987 | 148 s |
| 0.10 | 2.0 | 0.9842 | 0.8734 | 150 s |
| 0.25 | 2.0 | 0.9860 | 0.7848 | 257 s |
| 0.50 | 2.0 | 0.9860 | 0.7595 | 418 s |
| 0.10 | 1.0 | 0.9842 | 0.7848 | 138 s |
| 0.10 | 3.0 | 0.9842 | 0.8987 | 137 s |

Keeping more of the patch memory bank improves **ranking** (AUC climbs to
0.986 and saturates) while costing 3.7x the wall clock — and it makes accuracy
at the calibrated threshold *worse*, because the threshold moves with it. The
multiplier rows show the other half: AUC is identical across all three,
because the multiplier only moves the operating point. Ranking quality and
calibration are separate problems, and only one of them is what production
sees.

Add `--mlflow` to any training or testing command to record your own:
```bash
python train_test.py --model_name patchcore --product_class toothbrush --mode all \
  --train_path runs/train/pc_tb --test_path runs/test/pc_tb --mlflow
mlflow ui   # reads the ./mlflow.db this writes; open http://127.0.0.1:5000
```
It logs the model/class/mode plus that model's hyperparameters from
`config.yaml` as params, the threshold (train) and ROC-AUC (test) as metrics,
and the small `.json`/`.txt`/`.yaml` files from the run directory as
artifacts. Weights and segmentation PNGs stay on disk.

Tracking is opt-in and never fatal: without the flag nothing changes, and if
`mlflow` is missing or the server is unreachable the run logs a warning and
carries on. `ANOM_DETECT_MLFLOW=1` is equivalent to the flag,
`MLFLOW_TRACKING_URI` points at a server instead of the local `./mlflow.db`,
and `ANOM_DETECT_MLFLOW_EXPERIMENT` renames the experiment.

## Serving

**Serve the committed model — no dataset, no training:**
```bash
pip install -r requirements.txt
python serve.py                 # loads models/dfr_hazelnut; leave running

curl -F "file=@examples/sample_images/hazelnut/print_000.png" http://127.0.0.1:8000/predict
# -> {"score":110.89,"threshold":45.04,"is_anomalous":true}
curl -F "file=@examples/sample_images/hazelnut/good_000.png" http://127.0.0.1:8000/predict
# -> {"score":33.33,"threshold":45.04,"is_anomalous":false}
```
`models/dfr_hazelnut/` is a real DFR run on the full `hazelnut` class (1.0000
ROC AUC), exported by `scripts/export_serving_model.py`: it drops the ~100 MB
of frozen ImageNet weights torchvision reloads anyway and keeps the 12 MB the
run actually produced, which is what makes a servable model small enough to
live in the repo. `examples/sample_images/hazelnut/` holds five test images to
POST at it: two normal, three defective, and this model gets all five right.

**Which threshold serves is a decision, not a default.** Training calibrates
several (mean + 1/3/5 x std) and saves them all. This bundle was exported with
`--sigma 1.0`, because mean+3*std — what training happens to list first — sits
above two of the three defects even though the model ranks every defect above
every normal image at 1.00 AUC. Perfect ranking, wrong operating point. The
flag reorders the thresholds in the bundle rather than adding a second way to
choose one at serving time, so `serve` keeps reading the first entry.

Export one of your own the same way:
```bash
python train_test.py --model_name deep_feature_ad --product_class wood --mode train \
  --train_path runs/train/dfr_wood
python scripts/export_serving_model.py --train-path runs/train/dfr_wood \
  --product-class wood --output models/dfr_wood --sigma 1.0
python serve.py --product-class wood --train-path models/dfr_wood
```
`serve.py` also accepts a training directory directly (`--train-path
runs/train/dfr_wood`), so exporting is only needed to keep a model small.

`python serve.py` and `python drift.py` need only `pip install -r
requirements.txt` — they put `src` on the path themselves, so they work on a
fresh clone with no `pip install -e .`. After `pip install -e .` the
`anom-detect-serve` / `anom-detect-drift` commands do the same thing from any
directory. `--config`/`--product-class` default to
`config.yaml`/`hazelnut`, matching the committed bundle; for any other class,
train it first and pass `--product-class <class> --train-path <that run>`.

DFR is served because it's the model worth serving (see Results) and it scores
an image in one forward pass from just a weights file.

**Endpoints:**
| | |
| --- | --- |
| `GET /health` | which model, which threshold, how many scored |
| `POST /predict` | upload an image → `{score, threshold, is_anomalous}` |
| `GET /drift` | has the recently-served score distribution moved? |

## Drift monitoring

**How it works.** There are no labels in production, so "did the model get
worse?" is unanswerable. What is answerable: has the **anomaly rate** — the
share of served images landing above the threshold — moved from what training
implied? That's the one signal this reports, and it's *calibrated*: the
threshold and the rate share the same training-time reference, so whatever
bias that reference has cancels out on both sides of the comparison.

Only DFR and PatchCore persist their training score distribution, so only they
get a verdict; the other three report the live rate and say so. That's a real
gap, not a filled-in guess.

**Offline, on a finished test run** (every `--mode test` writes `scores.json`):
```bash
python drift.py --reference runs/train/dfr_wood --product-class wood \
  --live runs/test/dfr_wood/segmentation_wood/scores.json
# add --fail-on-drift for a CI/cron gate, --json for machine-readable output
```
`curl http://127.0.0.1:8000/drift` answers the same question for the live
service, over a rolling in-memory window of recently served scores.

**In Docker:**
```bash
docker compose run --rm anom-detect --config config.subset.yaml \
  --model_name deep_feature_ad --product_class wood --mode train \
  --train_path runs/train/dfr_wood
docker compose up serve
curl -F "file=@data/mvtec_subset/wood/test/hole/000.png" http://localhost:8000/predict
```

## Docker

```bash
docker build -t anom-detect .   # CPU; prefetches ResNet50 weights (~2.5 GB)
docker run --rm --shm-size=2g \
  -v /path/to/mvtec_anomaly_detection:/data:ro -v "$PWD/runs:/app/runs" \
  anom-detect --model_name patchcore --product_class toothbrush --mode all \
              --train_path runs/train/patchcore_toothbrush --test_path runs/test/patchcore_toothbrush
```
Or `docker compose run --rm anom-detect ...` with `DATASET_PATH=/path/... `
set, which handles the volumes and `--shm-size` for you.
`docker compose up tensorboard` serves the same `runs/` at `localhost:6006`.
For an NVIDIA GPU: `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124`
and `--gpus all`.

## Testing & CI

```bash
python -m pytest              # everything, ~1 min on CPU
python -m pytest -m "not slow"  # skip the pretrained-backbone tests
python -m ruff check .
```
Most tests build a synthetic dataset on disk, so no download is needed.
`test_subset_dataset.py`, `test_drift.py` and `test_serve.py` run against the
real committed subset, including training PatchCore on `screw` and confirming
it correctly flags `wood` images as drifted.

| CI job | Runs |
| --- | --- |
| `lint` | ruff |
| `test` | fast suite on Python 3.9 + 3.12 |
| `test-slow` | full suite + a real CLI run + a drift check; PR-only with the `run-slow-tests` label |
| `docker` | builds the image, runs the baked-in subset end to end |

## Project structure

```
config.yaml / config.subset.yaml   # full-dataset / bundled-subset configs
train_test.py / serve.py / drift.py  # CLI shims -> anom_detect.*
data/mvtec_subset/                 # ~5 MB real screw+wood images (tracked)
models/dfr_hazelnut/               # servable 12 MB DFR model (tracked)
examples/mlflow_runs/              # MLflow store for the 33 example runs
examples/sample_images/hazelnut/   # images to POST at the running service
scripts/make_subset.py             # rebuilds the subset from a full checkout
scripts/export_serving_model.py    # training run -> small serving bundle
src/anom_detect/
  cli.py                           # train/test dispatch
  tracking.py                      # optional MLflow logging (--mlflow)
  serve.py / drift.py              # HTTP service + drift detection
  dataset_preprocessor.py          # MVTec dataset loader
  base_model/ vit_model/ deep_feature_ad/ patchcore/ trafo_model/
tests/                             # synthetic + real-subset tests
```

## Models, briefly

| Model | Approach |
| --- | --- |
| PatchCore | Memory bank of normal ResNet50 patch features; 1-NN distance. Simplified vs. the paper: random subsample instead of greedy coreset. |
| ViT autoencoder | ViT encoder + CNN decoder + Mixture Density Network; MSE + SSIM + Gaussian likelihood loss. |
| Deep feature autoencoder (DFR) | Reconstructs multi-layer ResNet50 features; top-k reconstruction error. |
| Transformer autoencoder | Reconstructs frozen-ResNet50 patch features via a custom encoder-decoder. |
| Base autoencoder | 3-layer conv encoder/decoder, pixel-space MSE. The baseline the feature-space models improve on. |

Thresholds are calibrated on held-out normal data at three sigma multipliers
(1σ/3σ/5σ = aggressive/standard/conservative), never on test masks.

## License

Code: see [LICENSE](LICENSE). The bundled dataset subset is MVTec AD under
CC BY-NC-SA 4.0; see
[data/mvtec_subset/ATTRIBUTION.txt](data/mvtec_subset/ATTRIBUTION.txt).
