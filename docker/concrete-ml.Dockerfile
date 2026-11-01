# Concrete-ML for the baseline. It pins concrete-python 2.10, so it cannot share
# the main image; our circuit is measured here too, on the same version.
#
#   docker build -f docker/concrete-ml.Dockerfile -t cipherforest-ml .
#   docker run --rm --memory 3g -v <repo>/experiments:/work/experiments cipherforest-ml \
#       python experiments/concrete_ml_bench.py adult xgb-3-6
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first, or pip pulls ~2 GB of CUDA libs
RUN pip install --no-cache-dir "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "concrete-ml==1.9.0"

WORKDIR /work
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
