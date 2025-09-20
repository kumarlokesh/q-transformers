"""
Pytest configuration and shared fixtures for Q-Transformers tests.
"""

import sys
import pytest
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

# Try importing the optional Rust extension
try:
    from qtransformers_core import classical_attention_rs, quantum_attention_rs  # type: ignore
    HAS_RUST_CORE = True
except Exception:
    HAS_RUST_CORE = False


@pytest.fixture
def sample_tensors():
    """Fixture providing standard test tensors."""
    B, N, D = 2, 8, 16
    Q = torch.randn(B, N, D)
    K = torch.randn(B, N, D)
    V = torch.randn(B, N, D)
    return Q, K, V


@pytest.fixture
def small_tensors():
    """Fixture providing small test tensors for quick tests."""
    B, N, D = 1, 4, 8
    Q = torch.randn(B, N, D)
    K = torch.randn(B, N, D)  
    V = torch.randn(B, N, D)
    return Q, K, V


@pytest.fixture
def device():
    """Fixture providing the test device (CPU for CI)."""
    return torch.device("cpu")


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "rust: mark test as requiring Rust extension"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip Rust tests if extension not available."""
    for item in items:
        if "rust" in item.keywords and not HAS_RUST_CORE:
            skip_rust = pytest.mark.skip(
                reason="qtransformers_core not built; run `make install` first"
            )
            item.add_marker(skip_rust)
