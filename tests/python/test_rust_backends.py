import numpy as np
import pytest
import torch

from qtransformers import quantum_attention


def _classical_topk_python(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, k: int
) -> torch.Tensor:
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    top_scores, top_indices = torch.topk(scores, k=min(k, scores.shape[-1]), dim=-1)
    top_weights = torch.softmax(top_scores, dim=-1)

    # Gather top values using advanced indexing
    batch_size, seq_len, _ = V.shape

    # Expand indices for gathering
    batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, seq_len, k)

    # Gather values: (batch_size, seq_len, k, d_model)
    top_values = V[batch_indices, top_indices]

    # Apply attention weights and sum
    out = torch.sum(top_weights.unsqueeze(-1) * top_values, dim=-2)
    return out


@pytest.mark.rust
def test_rust_classical_shapes_and_parity():
    B, N, D = 1, 8, 16
    Q = torch.randn(B, N, D)
    K = torch.randn(B, N, D)
    V = torch.randn(B, N, D)
    k = 4

    # Rust backend via Python wrapper
    out_rust = quantum_attention(Q, K, V, top_k=k, backend="rust-classical")
    assert out_rust.shape == V.shape

    # Compare to Python impl on CPU
    out_py = _classical_topk_python(Q, K, V, k)
    # Allow small numerical tolerance
    assert torch.allclose(out_rust, out_py, atol=1e-5, rtol=1e-5)


@pytest.mark.rust
def test_rust_quantum_sampling_shapes():
    B, N, D = 2, 10, 32
    Q = torch.randn(B, N, D)
    K = torch.randn(B, N, D)
    V = torch.randn(B, N, D)

    out = quantum_attention(Q, K, V, top_k=8, backend="rust-quantum")
    assert out.shape == V.shape
    assert torch.isfinite(out).all()


@pytest.mark.rust
def test_function_level_rs_bindings():
    """Directly call PyO3 functions for a single matrix."""
    from qtransformers_core import classical_attention_rs, quantum_attention_rs

    N, D = 6, 8
    Q = np.random.randn(N, D).astype(np.float32)
    K = np.random.randn(N, D).astype(np.float32)
    V = np.random.randn(N, D).astype(np.float32)

    out_classical = classical_attention_rs(Q, K, V, 3)
    out_quantum = quantum_attention_rs(Q, K, V, 4)

    assert out_classical.shape == (N, D)
    assert out_quantum.shape == (N, D)
