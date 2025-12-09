# Q-Transformers: Quantum-Enhanced NLP

> **v0.1.0** - Library implementing quantum-inspired attention mechanisms for transformer models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/kumarlokesh/q-transformers)

## Table of contents

- [Q-Transformers: Quantum-Enhanced NLP](#q-transformers-quantum-enhanced-nlp)
  - [Table of contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Benchmarks](#benchmarks)
  - [Developer Quick Start](#developer-quick-start)
    - [Basic Usage](#basic-usage)
  - [Documentation](#documentation)
    - [Benchmarking and Evaluation](#benchmarking-and-evaluation)
  - [Technical Overview](#technical-overview)
    - [Core Components](#core-components)
  - [Technical Approach](#technical-approach)
  - [License](#license)
  - [Additional documentation](#additional-documentation)

## Key Features

- **Drop-in PyTorch compatibility** - Use with existing transformer code
- **Multi-GPU distributed training** - Quantum-aware gradient synchronization
- **Docker/Kubernetes deployment** - Containerized deployment
- **Real quantum hardware support** - IBM Quantum integration via Qiskit
- **Comprehensive benchmarking** - GLUE/SuperGLUE validation with 19 NLP tasks

## Benchmarks

Benchmarks and evaluation scripts are provided under the `benchmarks/` directory. Results depend on hardware, backend configuration, and random seeds; reproduce experiments using the provided scripts rather than relying on summarized claims in the README.

## Developer Quick Start

The repository includes a `Makefile` with common developer tasks. Use the `Makefile` targets from the project root to build, run checks, and execute tests. The targets orchestrate the toolchain (Python, Rust) and ensure consistent environments across machines.

Common tasks:

- Build development image and prepare environment:

```bash
make build
```

- Open an interactive shell with the repository mounted:

```bash
make shell
```

- Run Python unit tests:

```bash
make test-python
```

- Run Rust tests:

```bash
make test-rust
```

- Run the full test suite (Python + Rust + integration):

```bash
make test
```

If you need or prefer a local Python virtual environment, a local install is supported but may require system toolchains for some dependencies:

```bash
git clone https://github.com/kumarlokesh/q-transformers
cd q-transformers
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e python[dev]
```

### Basic Usage

```python
import torch
from qtransformers import QuantumMultiheadAttention

# Drop-in replacement for nn.MultiheadAttention
attn = QuantumMultiheadAttention(
    embed_dim=512,
    num_heads=8,
    quantum_backend="stratified",  # Best performing backend
    num_samples=32
)

# Use exactly like PyTorch MultiheadAttention
query = torch.randn(10, 32, 512)  # seq_len, batch, embed_dim
key = torch.randn(10, 32, 512)
value = torch.randn(10, 32, 512)

output, attn_weights = attn(query, key, value)
```

## Documentation

- **[Core Architecture](docs/core-architecture.md)** - Technical overview and benchmarks
- **[Advanced Features](docs/advanced-features.md)** - Quantum hardware integration
- **[Mathematical Foundations](docs/mathematical-foundations.md)** - Theory and algorithms

### Benchmarking and Evaluation

```python
from qtransformers import GLUEBenchmarkSuite, QuantumSupremacyVerifier

# Run comprehensive NLP benchmarks
benchmark_suite = GLUEBenchmarkSuite()
results = benchmark_suite.run_full_evaluation(
    quantum_model=quantum_model,
    classical_model=classical_model
)

# Verify quantum supremacy
verifier = QuantumSupremacyVerifier()
supremacy_results = verifier.verify_quantum_advantage(
    quantum_results=results["quantum"],
    classical_results=results["classical"]
)
```

## Technical Overview

### Core Components

- **Quantum-Inspired Attention**: Sampling-based attention approximation with O(n·S) complexity where S << n
- **Multi-GPU Training**: Distributed training support with quantum-aware gradient synchronization
- **Benchmark Suite**: GLUE/SuperGLUE evaluation framework for reproducible experiments

## Technical Approach

- **Sampling-Based Approximation**: Reduces full O(n²) attention to sparse sampling, trading exactness for efficiency
- **Variance Reduction**: Stratified sampling and control variates improve approximation quality (42% error reduction in internal benchmarks)
- **Memory Efficiency**: Sparse attention patterns reduce memory footprint

> **Note**: Performance claims are based on internal benchmarks. Results may vary by hardware, configuration, and task. See `benchmarks/` for reproduction scripts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Additional documentation

- [Mathematical Foundations](docs/mathematical-foundations.md)
- [Core Architecture](docs/core-architecture.md)
- [Advanced Features](docs/advanced-features.md)
- [Benchmark Results](benchmarks/README.md)

---
