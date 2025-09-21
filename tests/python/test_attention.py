"""
Tests for quantum-inspired attention mechanisms.
"""

import pytest
import torch

from qtransformers import QuantumAttentionLayer, quantum_attention


class TestQuantumAttentionLayer:
    """Test suite for QuantumAttentionLayer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = QuantumAttentionLayer(embed_dim=512, num_heads=8)
        assert layer.embed_dim == 512
        assert layer.num_heads == 8
        assert layer.head_dim == 64

    def test_forward_pass(self):
        """Test forward pass with dummy inputs."""
        layer = QuantumAttentionLayer(embed_dim=256, num_heads=4)

        # Create dummy inputs
        seq_len, batch_size = 10, 2
        query = torch.randn(seq_len, batch_size, 256)
        key = torch.randn(seq_len, batch_size, 256)
        value = torch.randn(seq_len, batch_size, 256)

        # Forward pass
        output, attn_weights = layer(query, key, value, need_weights=True)

        assert output.shape == (seq_len, batch_size, 256)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_batch_first(self):
        """Test batch_first option."""
        layer = QuantumAttentionLayer(embed_dim=128, num_heads=2, batch_first=True)

        # Create dummy inputs (batch_first format)
        batch_size, seq_len = 3, 8
        query = torch.randn(batch_size, seq_len, 128)
        key = torch.randn(batch_size, seq_len, 128)
        value = torch.randn(batch_size, seq_len, 128)

        # Forward pass
        output, _ = layer(query, key, value, need_weights=False)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

    def test_different_kdim_vdim(self):
        """Test with different key and value dimensions."""
        layer = QuantumAttentionLayer(embed_dim=256, num_heads=4, kdim=128, vdim=192)

        seq_len, batch_size = 6, 2
        query = torch.randn(seq_len, batch_size, 256)
        key = torch.randn(seq_len, batch_size, 128)  # Different kdim
        value = torch.randn(seq_len, batch_size, 192)  # Different vdim

        output, _ = layer(query, key, value)
        assert output.shape == (seq_len, batch_size, 256)


class TestQuantumAttentionFunction:
    """Test suite for quantum_attention function."""

    def test_classical_backend(self):
        """Test classical backend."""
        batch_size, seq_len, d_model = 2, 8, 64
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        output = quantum_attention(Q, K, V, top_k=4, backend="classical")
        assert output.shape == V.shape

    def test_quantum_sim_backend(self):
        """Test quantum simulation backend."""
        batch_size, seq_len, d_model = 1, 4, 32
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        output = quantum_attention(Q, K, V, top_k=2, backend="quantum-sim")
        assert output.shape == V.shape

    def test_invalid_backend(self):
        """Test invalid backend raises error."""
        Q = torch.randn(1, 4, 32)
        K = torch.randn(1, 4, 32)
        V = torch.randn(1, 4, 32)

        with pytest.raises(ValueError, match="Unknown backend"):
            quantum_attention(Q, K, V, backend="invalid")


@pytest.mark.parametrize(
    "embed_dim,num_heads",
    [
        (128, 4),
        (256, 8),
        (512, 16),
    ],
)
def test_attention_layer_dimensions(embed_dim, num_heads):
    """Test various embed_dim and num_heads combinations."""
    layer = QuantumAttentionLayer(embed_dim=embed_dim, num_heads=num_heads)

    seq_len, batch_size = 5, 2
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)

    output, _ = layer(query, key, value)
    assert output.shape == (seq_len, batch_size, embed_dim)


def test_attention_mask():
    """Test attention with mask (placeholder - to be implemented)."""
    # This test will be expanded when attention masking is implemented
    layer = QuantumAttentionLayer(embed_dim=64, num_heads=2)

    seq_len, batch_size = 4, 1
    query = torch.randn(seq_len, batch_size, 64)
    key = torch.randn(seq_len, batch_size, 64)
    value = torch.randn(seq_len, batch_size, 64)

    # Create a simple attention mask
    attn_mask = torch.zeros(seq_len, seq_len)
    attn_mask[2:, :2] = float("-inf")  # Mask future tokens

    output, _ = layer(query, key, value, attn_mask=attn_mask)
    assert output.shape == (seq_len, batch_size, 64)
