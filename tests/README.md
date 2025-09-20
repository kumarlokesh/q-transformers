# Tests

Comprehensive test suite for Q-Transformers with Docker-first workflow.

## Structure

- **`python/`** - Python unit tests for quantum attention, multihead attention, and simulation
- **`rust/`** - Rust unit tests for high-performance attention kernels  
- **`integration/`** - End-to-end integration tests for complete workflows
- **`conftest.py`** - Shared pytest fixtures and configuration

## Test Categories

- **Unit Tests**: Individual function/class testing with mocked dependencies
- **Integration Tests**: End-to-end attention layer and transformer testing
- **Rust Backend Tests**: High-performance kernel validation (marked with `@pytest.mark.rust`)
- **Benchmark Tests**: Performance regression and accuracy validation

## Running Tests

### Docker-based (Recommended)

```bash
# All tests (Python + Rust + Integration)
make test

# Python tests only
make test-python

# Rust tests only
make test-rust

# Integration tests only  
make test-integration

# With coverage report
make test-coverage
```

### Local Development

```bash
# Prerequisites: build Rust extension first
maturin develop -m rust-core/Cargo.toml

# Python tests
python -m pytest tests/python/ -v

# Rust tests
cd rust-core && cargo test

# Integration tests
python -m pytest tests/integration/ -v
```

## Test Features

- ✅ **Automatic Rust extension detection** - Tests skip gracefully if extension not built
- ✅ **Shared fixtures** - Consistent test tensors and configurations
- ✅ **Performance validation** - Benchmark regression detection
