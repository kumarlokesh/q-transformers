# Minimal image to run Python benchmarks with CPU PyTorch
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch and project deps
# Torch CPU wheels are hosted on PyTorch index
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install -r /workspace/requirements.txt

WORKDIR /workspace

COPY python/ /workspace/python/
COPY test_basic_functionality.py /workspace/
COPY README.md /workspace/README.md
COPY README.md /workspace/python/README.md

# Install project in editable mode (Python packages only)
RUN python -m pip install -e /workspace/python

# Default command: run basic functionality test
CMD ["python", "/workspace/test_basic_functionality.py"]
