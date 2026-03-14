#!/usr/bin/env python3
"""
Basic functionality tests for Q-Transformers.
"""

import torch
import pytest


def test_basic_imports():
    """Test that core modules import successfully."""
    from qtransformers.attention import QuantumAttentionLayer
    from qsim.quantum_simulator import QuantumAttentionSimulator

    assert QuantumAttentionLayer is not None
    assert QuantumAttentionSimulator is not None


def test_quantum_attention_forward():
    """Test basic quantum attention forward pass."""
    from qtransformers.attention import QuantumAttentionLayer

    attention = QuantumAttentionLayer(embed_dim=64, num_heads=8)

    seq_len, batch_size = 10, 2
    x = torch.randn(seq_len, batch_size, 64)

    output, attn_weights = attention(x, x, x)

    assert output.shape == (seq_len, batch_size, 64)
    assert torch.isfinite(output).all()


def test_quantum_simulator_forward():
    """Test basic quantum simulator forward pass."""
    from qsim.quantum_simulator import QuantumAttentionSimulator

    simulator = QuantumAttentionSimulator(device="cpu")

    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 8)
    value = torch.randn(2, 4, 8)

    result, attn_weights = simulator.simulate_attention(query, key, value, num_samples=16)

    assert result.shape == query.shape
    assert torch.isfinite(result).all()


def test_package_version():
    """Test that version info is accessible."""
    import qtransformers

    version = getattr(qtransformers, "__version__", None)
    assert version is not None
    assert version.startswith("0.")
