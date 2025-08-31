"""
Quantum-inspired attention mechanisms for Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QuantumAttentionLayer(nn.Module):
    """
    Quantum-inspired multi-head attention layer.
    
    This is a placeholder implementation that will be developed in Phase 2.
    Currently provides the same interface as nn.MultiheadAttention for compatibility.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of quantum-inspired attention.
        
        Args:
            query: Query tensor of shape (L, N, E) or (N, L, E) if batch_first
            key: Key tensor of shape (S, N, E) or (N, S, E) if batch_first  
            value: Value tensor of shape (S, N, E) or (N, S, E) if batch_first
            key_padding_mask: Mask for padding tokens
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights across heads
            
        Returns:
            Attention output and optional attention weights
        """
        # Placeholder implementation - classical attention for now
        # Will be replaced with quantum-inspired implementation in Phase 2
        
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        L, N, E = query.shape
        S = key.shape[0]
        
        # Project to Q, K, V
        q = self.q_proj(query)  # (L, N, E)
        k = self.k_proj(key)    # (S, N, E)  
        v = self.v_proj(value)  # (S, N, E)
        
        # Reshape for multi-head attention
        q = q.view(L, N, self.num_heads, self.head_dim).transpose(1, 2)  # (L, H, N, D)
        k = k.view(S, N, self.num_heads, self.head_dim).transpose(1, 2)  # (S, H, N, D)
        v = v.view(S, N, self.num_heads, self.head_dim).transpose(1, 2)  # (S, H, N, D)
        
        # Compute attention (classical for now)
        attn_output, attn_weights = self._classical_attention(q, k, v, attn_mask, key_padding_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(L, N, E)
        attn_output = self.out_proj(attn_output)
        
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
            
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # Average over heads
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def _classical_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classical scaled dot-product attention (placeholder)."""
        L, H, N, D = q.shape
        S = k.shape[0]

        # Reorder to (N, H, L, D) to compute per-batch scores
        q_nhld = q.permute(2, 1, 0, 3)  # (N, H, L, D)
        k_nhsd = k.permute(2, 1, 0, 3)  # (N, H, S, D)
        v_nhsd = v.permute(2, 1, 0, 3)  # (N, H, S, D)

        # Compute attention scores: (N, H, L, S)
        scores = torch.matmul(q_nhld, k_nhsd.transpose(-2, -1)) / (D ** 0.5)

        # Apply masks with correct broadcasting
        if attn_mask is not None:
            # attn_mask expected shape (L, S)
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            # key_padding_mask expected shape (N, S)
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Softmax over keys
        attn_weights = F.softmax(scores, dim=-1)  # (N, H, L, S)
        attn_weights = self.dropout_layer(attn_weights)

        # Weighted sum: (N, H, L, S) @ (N, H, S, D) -> (N, H, L, D)
        attn_out_nhld = torch.matmul(attn_weights, v_nhsd)

        # Back to (L, H, N, D)
        attn_output = attn_out_nhld.permute(2, 1, 0, 3)

        # Return attn_weights as (N, H, L, S) so caller can average over heads
        return attn_output, attn_weights


def quantum_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    top_k: int = 32,
    backend: str = "classical"
) -> torch.Tensor:
    """
    Functional interface for quantum-inspired attention.
    
    Args:
        Q: Query tensor of shape (..., seq_len, d_model)
        K: Key tensor of shape (..., seq_len, d_model)
        V: Value tensor of shape (..., seq_len, d_model)
        top_k: Number of top attention weights to consider
        backend: Attention backend ("classical", "quantum-sim", "quantum-hw")
        
    Returns:
        Attention output tensor of same shape as V
    """
    # Placeholder implementation
    # Will integrate with Rust backend in Phase 2
    
    if backend == "classical":
        # Classical efficient attention approximation
        return _classical_efficient_attention(Q, K, V, top_k)
    elif backend == "quantum-sim":
        # Quantum simulation (to be implemented in Phase 1)
        return _quantum_sim_attention(Q, K, V, top_k)
    elif backend == "quantum-hw":
        # Real quantum hardware (to be implemented in Phase 4)
        return _quantum_hw_attention(Q, K, V, top_k)
    elif backend == "phase0-proto":
        # Phase 0 prototype approximation via sampling
        return quantum_inspired_attention_prototype(Q, K, V, num_samples=max(1, top_k))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _classical_efficient_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, top_k: int) -> torch.Tensor:
    """Classical efficient attention approximation."""
    # Simple top-k approximation for now
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    
    # Get top-k attention weights
    top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1)
    
    # Apply softmax to top-k scores
    top_weights = F.softmax(top_scores, dim=-1)
    
    # Gather corresponding values and compute weighted sum
    batch_size, seq_len, d_model = V.shape
    top_values = torch.gather(
        V.unsqueeze(-3).expand(-1, seq_len, -1, -1),
        dim=-2,
        index=top_indices.unsqueeze(-1).expand(-1, -1, -1, d_model)
    )
    
    output = torch.sum(top_weights.unsqueeze(-1) * top_values, dim=-2)
    return output


def _quantum_sim_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, top_k: int) -> torch.Tensor:
    """Quantum simulation attention (placeholder - to be implemented in Phase 1)."""
    # For now, fall back to classical implementation
    return _classical_efficient_attention(Q, K, V, top_k)


def _quantum_hw_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, top_k: int) -> torch.Tensor:
    """Quantum hardware attention (placeholder - to be implemented in Phase 4)."""
    # For now, fall back to classical implementation  
    return _classical_efficient_attention(Q, K, V, top_k)


def quantum_inspired_attention_prototype(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_samples: int = 32,
) -> torch.Tensor:
    """
    Phase 0 prototype: quantum-inspired approximation of softmax attention via
    amplitude sampling. Implements the algorithm described in docs/phase0-mathematical-foundations.md.

    Args:
        Q: (..., seq_len_q, d_model)
        K: (..., seq_len_k, d_model)
        V: (..., seq_len_k, d_model)
        num_samples: number of samples per query row

    Returns:
        Output tensor with shape (..., seq_len_q, d_model)
    """
    assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "Prototype expects 3D tensors (B, N, D)"
    B, Nq, D = Q.shape
    Nk = K.shape[-2]
    device = Q.device

    # 1) logits = Q K^T / sqrt(D)
    logits = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # (B, Nq, Nk)

    # 2) amplitudes = exp(logits/2), row-normalize
    amplitudes = torch.exp(logits / 2)
    amplitudes = amplitudes / (amplitudes.norm(dim=-1, keepdim=True) + 1e-12)

    # 3) Sampling from amplitudes^2 per row
    probs = amplitudes.square()  # (B, Nq, Nk)
    probs_2d = probs.reshape(B * Nq, Nk)
    # Guard against NaNs and rows of zeros
    probs_2d = torch.nan_to_num(probs_2d, nan=0.0)
    row_sums = probs_2d.sum(dim=-1, keepdim=True)
    safe_probs_2d = torch.where(row_sums > 0, probs_2d / row_sums, torch.full_like(probs_2d, 1.0 / Nk))

    samples = torch.multinomial(safe_probs_2d, num_samples=num_samples, replacement=True)  # (B*Nq, S)

    # 4) Reconstruct empirical distribution
    recon = torch.zeros_like(safe_probs_2d, device=device)
    recon.scatter_add_(1, samples, torch.ones_like(samples, dtype=recon.dtype))
    recon = recon / float(num_samples)
    recon = recon.view(B, Nq, Nk)

    # 5) Apply probabilities to values: (B, Nq, Nk) @ (B, Nk, D) -> (B, Nq, D)
    out = torch.matmul(recon, V)
    return out
