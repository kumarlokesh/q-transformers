# Mathematical Foundations

> **v0.1.0** - Mathematical framework for quantum-enhanced attention

This document provides the mathematical foundation for Q-Transformers' quantum-enhanced attention mechanisms.

## Quantum-Enhanced Attention

Q-Transformers achieves improvements on reasoning tasks through quantum-inspired sampling of attention patterns.

**Complexity**: Quantum sampling approximates softmax attention with O(n·S) complexity where S << n, using advanced sampling strategies for better approximation quality.

## Quantum-Inspired Attention Algorithm

### Core Concept

Traditional attention computes:

```text
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Our quantum-inspired approach approximates this through:

1. **Amplitude Encoding**: Encode Q, K vectors into quantum amplitude states
2. **Probabilistic Sampling**: Use quantum measurement to approximate softmax probabilities
3. **Efficient Computation**: Reduce complexity from O(n²) to O(n log n) or better

### Mathematical Framework

#### 1. Amplitude Encoding

For query vector q_i and key vectors K = [k_1, k_2, ..., k_n]:

```text
|ψ_q⟩ = Σ_j α_j |j⟩  where α_j = q_i · k_j / ||q_i · K||
```

#### 2. Quantum Kernel Estimation

The attention weight a_ij can be approximated through quantum measurement:

```text
a_ij ≈ |⟨i|ψ_q⟩|² = |α_i|²
```

#### 3. Complexity Analysis

- **Classical softmax**: O(n²d) for n×n attention matrix
- **Quantum-inspired**: O(nd log n) using probabilistic sampling
- **Memory**: O(n) instead of O(n²)

### Notation and Assumptions

- **Notation**:
  - n: sequence length, d: embedding dimension, h: number of heads
  - Q, K, V ∈ R^{n×d}; q_i, k_j, v_j denote rows
  - τ = √d is the softmax temperature denominator
  - A = softmax(QK^T/τ) ∈ R^{n×n}
- **Assumptions**:
  - Inputs are L2-normalized per head: ||q_i||_2 ≤ C_q, ||k_j||_2 ≤ C_k
  - Logit range bounded: |⟨q_i, k_j⟩|/τ ≤ B (prevents numerical overflow)
  - Sampling budget S per query scales sublinearly with n (e.g., S = O(log n))

### Approximation and Error Bounds (sketch)

We approximate row i of A via measurement-driven sampling of indices j with probabilities p_{ij} ∝ exp(⟨q_i,k_j⟩/τ).

Let Â_i be the empirical distribution from S samples and Ĥ_i = Â_i V be the approximated output row. Under standard multinomial concentration (e.g., Bretagnolle–Huber–Carol inequality) and bounded values ||v_j||_2 ≤ C_v,

```text
P( ||Â_i − A_i||_1 ≥ ε ) ≤ 2 exp(− S ε^2 / 2)
```

then

```text
||Ĥ_i − (A_i V)||_2 ≤ ||Â_i − A_i||_1 · max_j ||v_j||_2 ≤ C_v · ε
```

Choosing S = O((1/ε^2)·log(1/δ)) yields ||Ĥ_i − (A_i V)||_2 ≤ C_v ε with probability ≥ 1−δ.

### Sampling Complexity and Runtime

- Per-row sampling cost with alias tables or top-k candidate sets: O(S log n) (or O(S) with precomputed alias method)
- Building proposals (optional top-k shortlist via ANN): O(n log n) per layer or amortized via caching
- Overall per layer per head (ignoring projection costs): O(n S + build) with S ≪ n

## Experimental Design

### Toy Problem Setup

**Dataset**: Synthetic attention matrices (32×32, 64×64, 128×128)
**Baselines**:

- Vanilla softmax attention
- Linformer (linear attention)
- Performer (random feature attention)

**Metrics**:

- Approximation error: ||A_quantum - A_exact||_F
- Computational time: latency comparison
- Memory usage: peak memory consumption

### Implementation Plan

```python
def quantum_inspired_attention_prototype(Q, K, V, num_samples=32):
    """
    Prototype implementation for mathematical validation
    """
    # 1. Compute attention logits
    logits = Q @ K.T / math.sqrt(Q.shape[-1])
    
    # 2. Quantum-inspired sampling
    probabilities = approximate_softmax_quantum(logits, num_samples)
    
    # 3. Apply to values
    output = probabilities @ V
    return output

def approximate_softmax_quantum(logits, num_samples):
    """
    Quantum-inspired approximation of softmax using amplitude encoding
    """
    # Convert logits to amplitudes
    amplitudes = torch.exp(logits / 2)  # sqrt of probabilities
    amplitudes = amplitudes / torch.norm(amplitudes, dim=-1, keepdim=True)
    
    # Simulate quantum measurement (sampling)
    samples = torch.multinomial(amplitudes**2, num_samples, replacement=True)
    
    # Reconstruct probability distribution
    probs = torch.zeros_like(logits)
    probs.scatter_add_(-1, samples, torch.ones_like(samples, dtype=torch.float))
    probs = probs / num_samples
    
    return probs
```

## Experimental Protocol Checklist

- **[data_setup]** Synthetic data generation seeds logged and fixed (e.g., 1337, 2024, 4096).
- **[preprocessing]** Input normalization verified per head: `||q_i||_2`, `||k_j||_2` bounded; temperature `τ=√d` documented.
- **[baselines]** Implement and lock baselines: exact softmax, Linformer, Performer (versions, commit hashes recorded).
- **[hyperparams]** Grid specified and pre-registered:
  - Samples S ∈ {8, 16, 32, 64, 128}
  - Sequence length n ∈ {32, 64, 128, 256}
  - Heads h ∈ {1, 4, 8}; d_head ∈ {32, 64}
  - Noise params for `qsim` (depolarizing p ∈ {0, 1e-3, 1e-2})
- **[metrics]**
  - Approx error: `||A_approx − A_exact||_F / ||A_exact||_F`
  - Output error: `||H_approx − H_exact|| / ||H_exact||`
  - Latency (ms), peak memory (MB), sample count used
- **[evaluation]** K independent runs (e.g., K=5) for each setting; report mean±std.
- **[logging]** Structured logs (JSON/CSV). Capture env: Python/Rust versions, CUDA, torch, commit SHA.
- **[plots]**
  - Error vs S (log scale S)
  - Latency vs sequence length n
  - Error vs noise parameter p
- **[ablations]** With/without top-k proposals; alias vs naive sampling; normalization on/off.
- **[failure_modes]** Record divergence cases, probability mass collapse, instability with large logits; mitigation notes.
- **[reproducibility]** Scripts/notebooks stored under `examples/` with README; random seeds, configs under `benchmarks/configs/` (planned).
- **[compute]** Record hardware, batch sizes, wall-clock; cap runtime per experiment.

## Expected Outcomes

### Success Criteria

- **Approximation Error**: < 5% Frobenius norm difference
- **Speed Improvement**: ≥ 2× faster than vanilla attention
- **Memory Reduction**: ≥ 50% memory usage reduction

### Risk Mitigation

- **Poor Approximation**: Fall back to classical efficient attention methods
- **No Speed Gain**: Focus on interpretability benefits
- **Implementation Complexity**: Start with simplified quantum-inspired algorithms

## References

- [Quantum Machine Learning Survey](https://arxiv.org/abs/2011.00027)
- [Efficient Attention Mechanisms](https://arxiv.org/abs/2009.06732)
- [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)

---
