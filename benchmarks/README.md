# Benchmarks

Performance and accuracy benchmarks for Q-Transformers.

## Structure

- `accuracy/` - Downstream task accuracy benchmarks
- `latency/` - Speed and memory usage benchmarks  
- `results/` - Benchmark results and analysis
- `scripts/` - Benchmark execution scripts

## Scripts

- `run_quick_benchmarks.py` — quick benchmarks comparing exact, prototype, stratified/adaptive, and classical baselines
- `run_sampling_benchmarks.py` — comprehensive sampling + multihead + MPS simulation evaluation
- `run_acceleration_benchmarks.py` — GPU/CUDA acceleration, error mitigation, and transformer blocks
- `run_full_evaluation_suite.py` — end-to-end comprehensive evaluation (GLUE/SuperGLUE, supremacy, training, deployment)

## Usage

### Docker-based (Recommended)

```bash
# Quick benchmarks
make bench-quick

# Sampling benchmarks  
make bench-sampling

# Full evaluation suite
make bench-full

# Custom parameters
make shell
# Inside container:
python benchmarks/run_quick_benchmarks.py --batch 2 --seq 64 --dim 64 --samples 32 --device cpu
```

### Local Development

```bash
# Quick benchmarks
python benchmarks/run_quick_benchmarks.py --batch 2 --seq 64 --dim 64 --samples 32 --device cpu

# Sampling benchmarks
python benchmarks/run_sampling_benchmarks.py --batch 2 --seq 64 --dim 128 --samples 32 --scaling

# Acceleration benchmarks
python benchmarks/run_acceleration_benchmarks.py --batch_size 4 --seq_len 64 --embed_dim 256 --num_samples 32

# Full evaluation suite
python benchmarks/run_full_evaluation_suite.py
```

## Rust-backed Backends

The optional Rust extension (`qtransformers_core`) provides high-performance backends:

- **`rust-classical`** — classical top-k attention accelerated in Rust
- **`rust-quantum`** — sampling-based approximation in Rust

### Docker Setup (Recommended)

```bash
make build  # Rust extension built automatically
make test   # Includes Rust backend tests
```

### Local Setup

```bash
# Prerequisites: Rust toolchain + maturin
# See https://rustup.rs for Rust installation

# Build and install the PyO3 extension
maturin develop -m rust-core/Cargo.toml
pip install -e python/
```

Then in Python:

```python
from qtransformers import quantum_attention
import torch
Q = torch.randn(2, 64, 64)
K = torch.randn(2, 64, 64)
V = torch.randn(2, 64, 64)
out = quantum_attention(Q, K, V, top_k=32, backend="rust-classical")
