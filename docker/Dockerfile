# Linux image for the TFHE parts; Concrete has no Windows wheels.
#
#   docker build -f docker/Dockerfile -t cipherforest .
#   docker run --rm cipherforest
#
# keygen and evaluation keys peak at a few GB. On a small Docker Desktop VM run the
# fhe test files one container at a time, e.g.
#   docker run --rm --memory 3g cipherforest pytest -q tests/test_deploy.py
FROM python:3.11-slim

# Concrete links each compiled circuit with ld
RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/*

# concrete-python pulls torch; the default wheel brings ~2 GB of CUDA libs, so take the CPU build
RUN pip install --no-cache-dir "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "concrete-python==2.11.0" "numpy==1.26.4" "pytest==9.1.1" "hypothesis==6.170.12"

WORKDIR /work
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir --no-deps -e .

CMD ["pytest", "-q"]
