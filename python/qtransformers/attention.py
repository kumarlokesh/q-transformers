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


class QuantumMultiheadAttention(nn.Module):
    """
    Enhanced Quantum-inspired Multi-Head Attention with advanced sampling strategies.
    
    Drop-in replacement for nn.MultiheadAttention with quantum-inspired backends
    for efficient attention computation on large sequences.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.0,
        bias: bool = True,
        quantum_backend: str = "stratified",
        num_samples: int = 32,
        adaptive_samples: bool = True,
        control_variate: bool = True,
        batch_first: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.quantum_backend = quantum_backend
        self.num_samples = num_samples
        self.adaptive_samples = adaptive_samples
        self.control_variate = control_variate
        self.batch_first = batch_first
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Per-head quantum configurations
        self.head_configs = self._init_head_configs()
    
    def _init_head_configs(self):
        """Initialize different quantum configurations per attention head."""
        configs = []
        strategies = ["stratified", "adaptive", "hybrid", "naive"]
        
        for h in range(self.num_heads):
            config = {
                "sampling_strategy": strategies[h % len(strategies)],
                "num_samples": max(8, self.num_samples // (1 + h // 4)),  # Vary samples per head
                "noise_level": 0.01 * (1 + h % 3),  # Vary noise per head
                "control_variate": self.control_variate and (h % 2 == 0)  # Every other head
            }
            configs.append(config)
        
        return configs
    
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
        Forward pass with quantum-inspired multi-head attention.
        
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
        
        # Reshape to multi-head format: (L, N, num_heads, head_dim)
        q = q.view(L, N, self.num_heads, self.head_dim)
        k = k.view(S, N, self.num_heads, self.head_dim)
        v = v.view(S, N, self.num_heads, self.head_dim)
        
        # Apply quantum attention per head with different configurations
        head_outputs = []
        head_weights = [] if need_weights else None
        
        for h in range(self.num_heads):
            config = self.head_configs[h]
            
            # Extract single head: (L, N, head_dim) and (S, N, head_dim)
            q_h = q[:, :, h, :]  # (L, N, head_dim)
            k_h = k[:, :, h, :]  # (S, N, head_dim)
            v_h = v[:, :, h, :]  # (S, N, head_dim)
            
            # Apply quantum-inspired attention per batch
            batch_outputs = []
            batch_weights = [] if need_weights else None
            
            for b in range(N):
                q_b = q_h[:, b, :]  # (L, head_dim)
                k_b = k_h[:, b, :]  # (S, head_dim)
                v_b = v_h[:, b, :]  # (S, head_dim)
                
                # Use quantum-inspired attention prototype
                out_b = quantum_inspired_attention_prototype(
                    q_b.unsqueeze(0),  # Add batch dim
                    k_b.unsqueeze(0),  # Add batch dim  
                    v_b.unsqueeze(0),  # Add batch dim
                    num_samples=config["num_samples"],
                    sampling_strategy=config["sampling_strategy"],
                    adaptive_samples=self.adaptive_samples,
                    control_variate=config["control_variate"]
                ).squeeze(0)  # Remove batch dim
                
                batch_outputs.append(out_b)
                
                if need_weights:
                    # Compute classical attention weights for visualization
                    scores = torch.matmul(q_b, k_b.transpose(-2, -1)) / (self.head_dim ** 0.5)
                    if attn_mask is not None:
                        scores = scores + attn_mask
                    if key_padding_mask is not None and key_padding_mask[b].any():
                        scores = scores.masked_fill(key_padding_mask[b].unsqueeze(0), float('-inf'))
                    weights = F.softmax(scores, dim=-1)
                    batch_weights.append(weights)
            
            # Stack batch results: (L, N, head_dim)
            head_out = torch.stack(batch_outputs, dim=1)
            head_outputs.append(head_out)
            
            if need_weights:
                head_weight = torch.stack(batch_weights, dim=0)  # (N, L, S)
                head_weights.append(head_weight)
        
        # Concatenate heads: (L, N, E)
        attn_output = torch.cat(head_outputs, dim=-1)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout_layer(attn_output)
        
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        # Process attention weights
        attn_weights = None
        if need_weights and head_weights:
            # Stack heads: (N, num_heads, L, S)
            attn_weights = torch.stack(head_weights, dim=1)
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # (N, L, S)
        
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
    """Quantum simulation attention using enhanced qsim backend."""
    try:
        from qsim import QuantumAttentionSimulator
        
        simulator = QuantumAttentionSimulator(device=str(Q.device), noise_model="depolarizing")
        
        # Use fewer samples for faster simulation, add light noise for regularization
        output, _ = simulator.simulate_attention(
            Q, K, V, 
            num_samples=max(8, top_k // 2),
            noise_level=0.01,
            temperature=0.0
        )
        
        return output
        
    except ImportError:
        # Fall back to classical if qsim not available
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
    use_top_k: bool = True,
    top_k_ratio: float = 0.5,
    sampling_strategy: str = "hybrid",
    adaptive_samples: bool = False,
    control_variate: bool = False,
) -> torch.Tensor:
    """
    Enhanced quantum-inspired approximation of softmax attention with advanced sampling strategies.

    Args:
        Q: (..., seq_len_q, d_model)
        K: (..., seq_len_k, d_model) 
        V: (..., seq_len_k, d_model)
        num_samples: number of samples per query row
        use_top_k: whether to use top-k importance sampling
        top_k_ratio: fraction of keys to consider in top-k proposals
        sampling_strategy: "hybrid", "stratified", "adaptive", or "naive"
        adaptive_samples: dynamically adjust sample count based on entropy
        control_variate: use classical attention as control variate

    Returns:
        Output tensor with shape (..., seq_len_q, d_model)
    """
    assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "Prototype expects 3D tensors (B, N, D)"
    B, Nq, D = Q.shape
    Nk = K.shape[-2]
    device = Q.device

    # Compute attention logits
    logits = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # (B, Nq, Nk)
    
    # Adaptive sampling based on attention entropy
    if adaptive_samples:
        num_samples = _compute_adaptive_samples(logits, base_samples=num_samples)
    
    # Route to appropriate sampling strategy
    if sampling_strategy == "stratified":
        return _quantum_attention_stratified(Q, K, V, logits, num_samples, control_variate)
    elif sampling_strategy == "adaptive":
        return _quantum_attention_adaptive(Q, K, V, logits, num_samples, control_variate)
    elif sampling_strategy == "hybrid" and use_top_k and Nk > 8:
        return _quantum_attention_top_k(Q, K, V, logits, num_samples, top_k_ratio)
    else:
        return _quantum_attention_naive(logits, V, num_samples)


def _quantum_attention_top_k(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    logits: torch.Tensor, 
    num_samples: int,
    top_k_ratio: float
) -> torch.Tensor:
    """Hybrid approach: exact softmax for top-k + quantum sampling for remaining keys."""
    B, Nq, Nk = logits.shape
    device = logits.device
    
    # Select top-k candidates for exact computation
    k_exact = max(4, int(Nk * top_k_ratio))  # At least 4 exact candidates
    top_logits, top_indices = torch.topk(logits, k=min(k_exact, Nk), dim=-1)  # (B, Nq, k)
    
    if k_exact >= Nk:
        # If top-k covers all keys, just do exact softmax
        exact_probs = torch.softmax(logits, dim=-1)
        return torch.matmul(exact_probs, V)
    
    # Compute exact softmax probabilities for top-k
    top_probs = torch.softmax(top_logits, dim=-1)  # (B, Nq, k)
    
    # Get remaining keys for quantum sampling
    all_indices = torch.arange(Nk, device=device).expand(B, Nq, -1)
    is_top_k = torch.zeros(B, Nq, Nk, dtype=torch.bool, device=device)
    is_top_k.scatter_(-1, top_indices, True)
    
    remaining_mask = ~is_top_k  # (B, Nq, Nk)
    remaining_logits = torch.where(remaining_mask, logits, torch.full_like(logits, float('-inf')))
    
    # Quantum sampling for remaining keys (amplitude encoding)
    remaining_amplitudes = torch.exp(remaining_logits / 2)
    remaining_amplitudes = torch.where(remaining_mask, remaining_amplitudes, torch.zeros_like(remaining_amplitudes))
    
    # Normalize amplitudes over remaining keys only
    remaining_amp_sums = remaining_amplitudes.sum(dim=-1, keepdim=True)
    remaining_amplitudes = torch.where(
        remaining_amp_sums > 1e-12, 
        remaining_amplitudes / remaining_amp_sums, 
        torch.zeros_like(remaining_amplitudes)
    )
    
    # Sample from |amplitudes|^2 for remaining keys
    remaining_probs_target = remaining_amplitudes.square()
    
    # Use fewer samples for remaining keys since top-k are exact
    num_remaining_samples = max(1, num_samples // 4)  # Use 1/4 samples for remaining
    
    if remaining_probs_target.sum() > 1e-12:
        # Multinomial sampling from remaining keys
        remaining_probs_flat = remaining_probs_target.view(-1, Nk)
        valid_rows = remaining_probs_flat.sum(dim=-1) > 1e-12
        
        remaining_sampled = torch.zeros_like(remaining_probs_flat)
        if valid_rows.sum() > 0:
            valid_probs = remaining_probs_flat[valid_rows]
            samples = torch.multinomial(valid_probs, num_samples=num_remaining_samples, replacement=True)
            
            # Convert samples back to probabilities
            sample_probs = torch.zeros_like(valid_probs)
            sample_probs.scatter_add_(1, samples, torch.ones_like(samples, dtype=sample_probs.dtype))
            sample_probs = sample_probs / num_remaining_samples
            
            remaining_sampled[valid_rows] = sample_probs
        
        remaining_sampled = remaining_sampled.view(B, Nq, Nk)
    else:
        remaining_sampled = torch.zeros_like(remaining_probs_target)
    
    # Combine exact top-k probabilities with sampled remaining probabilities
    final_probs = torch.zeros(B, Nq, Nk, device=device)
    
    # Set exact probabilities for top-k indices
    final_probs.scatter_(-1, top_indices, top_probs)
    
    # Add sampled probabilities for remaining indices (already masked)  
    final_probs = final_probs + remaining_sampled
    
    # Renormalize to ensure proper probability distribution
    final_probs = final_probs / (final_probs.sum(dim=-1, keepdim=True) + 1e-12)
    
    # Apply to values
    return torch.matmul(final_probs, V)


def _compute_adaptive_samples(logits: torch.Tensor, base_samples: int) -> int:
    """Compute adaptive sample count based on attention entropy."""
    # Compute attention entropy per query
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    
    # Normalize entropy to [0, 1] range
    max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=logits.dtype))
    normalized_entropy = entropy / max_entropy
    
    # Adaptive scaling: higher entropy = more samples needed
    avg_entropy = normalized_entropy.mean()
    scale_factor = 0.5 + avg_entropy  # Scale between 0.5x and 1.5x
    
    adaptive_samples = int(base_samples * scale_factor)
    return max(4, min(adaptive_samples, base_samples * 2))  # Clamp between 4 and 2x base


def _quantum_attention_stratified(
    Q: torch.Tensor,
    K: torch.Tensor, 
    V: torch.Tensor,
    logits: torch.Tensor,
    num_samples: int,
    control_variate: bool = False
) -> torch.Tensor:
    """Stratified sampling: partition attention space into importance regions."""
    B, Nq, Nk = logits.shape
    device = logits.device
    
    # Compute exact probabilities for stratification
    exact_probs = torch.softmax(logits, dim=-1)
    
    # Stratify into high/medium/low importance regions
    high_threshold = 0.1  # Top 10% probability mass
    medium_threshold = 0.02  # Next tier
    
    # Allocate samples proportionally to strata
    high_samples = int(num_samples * 0.6)    # 60% samples for high importance
    medium_samples = int(num_samples * 0.3)  # 30% for medium
    low_samples = num_samples - high_samples - medium_samples  # 10% for low
    
    recon_probs = torch.zeros_like(exact_probs)
    
    for b in range(B):
        for q in range(Nq):
            prob_row = exact_probs[b, q, :]
            
            # Define strata based on probability thresholds
            high_mask = prob_row >= high_threshold
            medium_mask = (prob_row >= medium_threshold) & (prob_row < high_threshold)
            low_mask = prob_row < medium_threshold
            
            # Sample from each stratum
            if high_mask.sum() > 0 and high_samples > 0:
                high_probs = prob_row[high_mask]
                high_probs = high_probs / high_probs.sum()
                high_indices = torch.where(high_mask)[0]
                
                if len(high_indices) >= high_samples:
                    sampled_high = torch.multinomial(high_probs, high_samples, replacement=True)
                    recon_probs[b, q, high_indices[sampled_high]] += 1.0 / high_samples * 0.6
            
            if medium_mask.sum() > 0 and medium_samples > 0:
                medium_probs = prob_row[medium_mask]
                medium_probs = medium_probs / medium_probs.sum()
                medium_indices = torch.where(medium_mask)[0]
                
                if len(medium_indices) >= medium_samples:
                    sampled_medium = torch.multinomial(medium_probs, medium_samples, replacement=True)
                    recon_probs[b, q, medium_indices[sampled_medium]] += 1.0 / medium_samples * 0.3
            
            if low_mask.sum() > 0 and low_samples > 0:
                low_probs = prob_row[low_mask] 
                low_probs = low_probs / low_probs.sum()
                low_indices = torch.where(low_mask)[0]
                
                if len(low_indices) >= low_samples:
                    sampled_low = torch.multinomial(low_probs, low_samples, replacement=True)
                    recon_probs[b, q, low_indices[sampled_low]] += 1.0 / low_samples * 0.1
    
    # Control variate correction
    if control_variate:
        classical_output = torch.matmul(exact_probs, V)
        sampled_output = torch.matmul(recon_probs, V)
        # Simple control variate: blend with exact result
        return 0.7 * sampled_output + 0.3 * classical_output
    
    return torch.matmul(recon_probs, V)


def _quantum_attention_adaptive(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor, 
    logits: torch.Tensor,
    num_samples: int,
    control_variate: bool = False
) -> torch.Tensor:
    """Adaptive sampling: adjust strategy per query based on attention pattern."""
    B, Nq, Nk = logits.shape
    device = logits.device
    
    exact_probs = torch.softmax(logits, dim=-1)
    
    # Analyze attention patterns per query
    entropy = -torch.sum(exact_probs * torch.log(exact_probs + 1e-12), dim=-1)
    max_entropy = torch.log(torch.tensor(Nk, dtype=logits.dtype))
    normalized_entropy = entropy / max_entropy
    
    recon_probs = torch.zeros_like(exact_probs)
    
    for b in range(B):
        for q in range(Nq):
            prob_row = exact_probs[b, q, :]
            row_entropy = normalized_entropy[b, q]
            
            if row_entropy > 0.8:  # High entropy = more uniform = need more samples
                samples_for_row = min(num_samples * 2, Nk // 2)
                strategy = "uniform"
            elif row_entropy < 0.3:  # Low entropy = concentrated = fewer samples needed
                samples_for_row = max(num_samples // 2, 4)
                strategy = "top_k"
            else:  # Medium entropy = hybrid approach
                samples_for_row = num_samples
                strategy = "importance"
            
            # Apply strategy
            if strategy == "uniform":
                # More uniform sampling for high entropy
                uniform_samples = torch.multinomial(prob_row, samples_for_row, replacement=True)
                recon_probs[b, q].scatter_add_(0, uniform_samples, 
                                             torch.ones_like(uniform_samples, dtype=recon_probs.dtype))
                recon_probs[b, q] /= samples_for_row
                
            elif strategy == "top_k":
                # Focus on top keys for low entropy
                k = min(samples_for_row, Nk)
                _, top_indices = torch.topk(prob_row, k)
                top_probs = prob_row[top_indices]
                top_probs = top_probs / top_probs.sum()
                recon_probs[b, q, top_indices] = top_probs
                
            else:  # importance sampling
                samples = torch.multinomial(prob_row, samples_for_row, replacement=True)
                recon_probs[b, q].scatter_add_(0, samples,
                                             torch.ones_like(samples, dtype=recon_probs.dtype))
                recon_probs[b, q] /= samples_for_row
    
    # Control variate correction
    if control_variate:
        classical_output = torch.matmul(exact_probs, V)
        sampled_output = torch.matmul(recon_probs, V)
        return 0.8 * sampled_output + 0.2 * classical_output
    
    return torch.matmul(recon_probs, V)


def _quantum_attention_naive(logits: torch.Tensor, V: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Naive amplitude sampling (original implementation)."""
    B, Nq, Nk = logits.shape
    device = logits.device
    
    # Amplitude encoding: exp(logits/2) and normalize
    amplitudes = torch.exp(logits / 2)
    amplitudes = amplitudes / (amplitudes.norm(dim=-1, keepdim=True) + 1e-12)
    
    # Sample from |amplitudes|^2
    probs = amplitudes.square()
    probs_2d = probs.view(-1, Nk)
    
    # Multinomial sampling
    samples = torch.multinomial(probs_2d, num_samples=num_samples, replacement=True)
    
    # Reconstruct empirical distribution  
    recon = torch.zeros_like(probs_2d)
    recon.scatter_add_(1, samples, torch.ones_like(samples, dtype=recon.dtype))
    recon = (recon / num_samples).view(B, Nq, Nk)
    
    return torch.matmul(recon, V)
