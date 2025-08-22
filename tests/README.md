# Tests

Unit and integration tests for Q-Transformers.

## Structure

- `python/` - Python package tests
- `rust/` - Rust crate tests
- `integration/` - Cross-language integration tests

## Test Categories

- **Unit Tests**: Individual function/class testing
- **Integration Tests**: End-to-end attention layer testing
- **Benchmark Tests**: Performance regression testing
- **Property Tests**: Rust proptest-based testing

## Running Tests

```bash
# Python tests
cd python && python -m pytest tests/

# Rust tests  
cd rust-core && cargo test

# Integration tests
python -m pytest tests/integration/
```
