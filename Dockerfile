# CPU-only by default. For an NVIDIA GPU, build with
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# and run with `--gpus all` (host needs the NVIDIA Container Toolkit). The
# CUDA wheels bundle their own runtime libraries, so the base image below
# works for both.
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Bakes the ImageNet ResNet50 weights into the image so PatchCore and
# deep_feature_ad don't need internet on first run. Pass 0 to skip and save
# ~100 MB (they'll download into the /cache/torch volume instead).
ARG PREFETCH_WEIGHTS=1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/cache/torch \
    MPLBACKEND=Agg

# torch's OpenMP runtime; everything else in requirements.txt ships self-contained wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision come from the PyTorch index (that index only carries
# those, hence the separate step); the rest resolves from PyPI and finds
# torch>=2.1 already satisfied.
COPY requirements.txt ./
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision \
    && python -m pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE config.yaml config.subset.yaml train_test.py ./
COPY src ./src
COPY tests ./tests
# ~5 MB of real screw/wood images so the image can run end-to-end with
# `--config config.subset.yaml` and no dataset mount.
COPY data/mvtec_subset ./data/mvtec_subset
# The 12 MB serving bundle, so `docker compose up serve` answers requests
# without training anything first (scripts/export_serving_model.py made it).
COPY models ./models
COPY examples/sample_images ./examples/sample_images
RUN python -m pip install --no-deps -e .

# The dataset is bind-mounted at /data rather than copied in (it's ~5 GB and
# separately licensed). Point the shipped config at it; mounting your own
# config.yaml over /app/config.yaml also works as long as it uses /data.
RUN sed -i 's|^DATASET_PATH:.*|DATASET_PATH: "/data"|' config.yaml

RUN if [ "$PREFETCH_WEIGHTS" = "1" ]; then \
        python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"; \
    fi

VOLUME ["/data", "/app/runs", "/cache/torch"]

# anom-detect-serve listens here (see the `serve` service in docker-compose.yml).
EXPOSE 8000

ENTRYPOINT ["anom-detect"]
CMD ["--help"]
