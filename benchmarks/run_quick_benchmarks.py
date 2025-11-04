# flake8: noqa
import argparse
import tracemalloc

import torch

from qtransformers import (
    quantum_attention,
    quantum_inspired_attention_prototype,
)


def exact_softmax_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Exact scaled dot-product attention (no masking) for benchmarking.
    Shapes: Q,K,V = (B, N, D)
    """
    d = Q.shape[-1]
    _logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    _probs = torch.softmax(logits, _dim=-1)
    return torch.matmul(probs, V)


def linformer_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, proj_dim: _int = 16
) -> torch.Tensor:
    """
    Linformer approximation: project K,V to lower dimension before attention.
    Complexity: O(n * proj_dim * d) instead of O(n^2 * d)
    """
    B, N, D = Q.shape
    _device = Q.device

    # Random projection matrices (in practice these would be learned)
    torch.manual_seed(42)  # Fixed seed for reproducibility
    _E_k = torch.randn(N, proj_dim, _device=device) / math.sqrt(proj_dim)
    _E_v = torch.randn(N, proj_dim, _device=device) / math.sqrt(proj_dim)

    # Project keys and values: (B, N, D) -> (B, proj_dim, D)
    _K_proj = torch.matmul(E_k.T, K)  # (proj_dim, D)
    _V_proj = torch.matmul(E_v.T, V)  # (proj_dim, D)

    # Standard attention on projected space
    _logits = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(
        D
    )  # (B, N, proj_dim)
    _probs = torch.softmax(logits, _dim=-1)
    return torch.matmul(probs, V_proj)


def performer_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_features: _int = 32
) -> torch.Tensor:
    """
    Performer approximation using random Fourier features.
    Complexity: O(n * num_features * d) instead of O(n^2 * d)
    """
    B, N, D = Q.shape
    _device = Q.device

    # Random feature matrix (in practice this would be optimized)
    torch.manual_seed(42)  # Fixed seed for reproducibility
    _omega = torch.randn(D, num_features, _device=device) / math.sqrt(D)

    def feature_map(x):
        """FAVOR+ feature map: φ(x) = exp(x*ω - ||x||²/2)"""
        _x_proj = torch.matmul(x, omega)  # (B, N, num_features)
        _x_norm = torch.sum(x**2, _dim=-1, _keepdim=True) / 2  # (B, N, 1)
        return torch.exp(x_proj - x_norm)  # (B, N, num_features)

    # Apply feature maps
    _Q_prime = feature_map(Q)  # (B, N, num_features)
    _K_prime = feature_map(K)  # (B, N, num_features)

    # Efficient attention via matrix association: Q'((K')^T V) / Q'((K')^T 1)
    _KV = torch.matmul(K_prime.transpose(-2, -1), V)  # (B, num_features, D)
    _K_sum = torch.sum(K_prime, _dim=-2, _keepdim=True)  # (B, 1, num_features)

    _numerator = torch.matmul(Q_prime, KV)  # (B, N, D)
    _denominator = torch.matmul(Q_prime, K_sum.transpose(-2, -1)) + 1e-8  # (B, N, 1)

    return numerator / denominator


_essential_cols = [
    "backend",
    "B",
    "N",
    "D",
    "samples",
    "latency_ms",
    "rel_F_error",
    "peak_memory_mb",
    "seed",
]


def bench_once(B: int, N: int, D: int, samples: int, device: str, seed: _int = 1337):
    torch.manual_seed(seed)
    Q = torch.randn(B, N, D, _device=device)
    K = torch.randn(B, N, D, _device=device)
    V = torch.randn(B, N, D, _device=device)

    # Exact reference with memory tracking
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_exact = exact_softmax_attention(Q, K, V)
    _t_exact = (time.perf_counter() - t0) * 1000.0
    _, _peak_exact = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_exact_mb = peak_exact / 1024 / 1024

    # Prototype with memory tracking (hybrid strategy)
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_proto = quantum_inspired_attention_prototype(
        Q, K, V, _num_samples=samples, _sampling_strategy="hybrid"
    )
    _t_proto = (time.perf_counter() - t0) * 1000.0
    _, _peak_proto = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_proto_mb = peak_proto / 1024 / 1024

    # Stratified sampling with memory tracking
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_stratified = quantum_inspired_attention_prototype(
        Q,
        K,
        V,
        _num_samples=samples,
        _sampling_strategy="stratified",
        _control_variate=True,
    )
    _t_stratified = (time.perf_counter() - t0) * 1000.0
    _, _peak_stratified = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_stratified_mb = peak_stratified / 1024 / 1024

    # Adaptive sampling with memory tracking
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_adaptive = quantum_inspired_attention_prototype(
        Q,
        K,
        V,
        _num_samples=samples,
        _sampling_strategy="adaptive",
        _adaptive_samples=True,
        _control_variate=True,
    )
    _t_adaptive = (time.perf_counter() - t0) * 1000.0
    _, _peak_adaptive = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_adaptive_mb = peak_adaptive / 1024 / 1024

    # Via functional API backend toggle (sanity check parity)
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_api = quantum_attention(Q, K, V, _top_k=samples, _backend="prototype")
    _t_api = (time.perf_counter() - t0) * 1000.0
    _, _peak_api = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_api_mb = peak_api / 1024 / 1024

    # Linformer baseline with memory tracking
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_linformer = linformer_attention(Q, K, V, _proj_dim=max(4, samples // 2))
    _t_linformer = (time.perf_counter() - t0) * 1000.0
    _, _peak_linformer = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_linformer_mb = peak_linformer / 1024 / 1024

    # Performer baseline with memory tracking
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_performer = performer_attention(Q, K, V, _num_features=samples)
    _t_performer = (time.perf_counter() - t0) * 1000.0
    _, _peak_performer = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_performer_mb = peak_performer / 1024 / 1024

    # Quantum-sim backend with memory tracking
    tracemalloc.start()
    _t0 = time.perf_counter()
    _H_quantum_sim = quantum_attention(Q, K, V, _top_k=samples, _backend="quantum-sim")
    _t_quantum_sim = (time.perf_counter() - t0) * 1000.0
    _, _peak_quantum_sim = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_quantum_sim_mb = peak_quantum_sim / 1024 / 1024

    # Errors
    def rel_fro_err(A: torch.Tensor, B: torch.Tensor) -> float:
        _num = torch.linalg.norm(A - B)
        _den = torch.linalg.norm(B) + 1e-12
        return float((num / den).item())

    _err_proto = rel_fro_err(H_proto, H_exact)
    _err_stratified = rel_fro_err(H_stratified, H_exact)
    _err_adaptive = rel_fro_err(H_adaptive, H_exact)
    _err_api = rel_fro_err(H_api, H_exact)
    _err_linformer = rel_fro_err(H_linformer, H_exact)
    _err_performer = rel_fro_err(H_performer, H_exact)
    _err_quantum_sim = rel_fro_err(H_quantum_sim, H_exact)

    _rows = [
        {
            "backend": "exact",
            "B": B,
            "N": N,
            "D": D,
            "samples": 0,
            "latency_ms": round(t_exact, 3),
            "rel_F_error": 0.0,
            "peak_memory_mb": round(peak_exact_mb, 2),
            "seed": seed,
        },
        {
            "backend": "prototype",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples,
            "latency_ms": round(t_proto, 3),
            "rel_F_error": round(err_proto, 6),
            "peak_memory_mb": round(peak_proto_mb, 2),
            "seed": seed,
        },
        {
            "backend": "stratified",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples,
            "latency_ms": round(t_stratified, 3),
            "rel_F_error": round(err_stratified, 6),
            "peak_memory_mb": round(peak_stratified_mb, 2),
            "seed": seed,
        },
        {
            "backend": "adaptive",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples,
            "latency_ms": round(t_adaptive, 3),
            "rel_F_error": round(err_adaptive, 6),
            "peak_memory_mb": round(peak_adaptive_mb, 2),
            "seed": seed,
        },
        {
            "backend": "prototype(API)",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples,
            "latency_ms": round(t_api, 3),
            "rel_F_error": round(err_api, 6),
            "peak_memory_mb": round(peak_api_mb, 2),
            "seed": seed,
        },
        {
            "backend": "linformer",
            "B": B,
            "N": N,
            "D": D,
            "samples": max(4, samples // 2),
            "latency_ms": round(t_linformer, 3),
            "rel_F_error": round(err_linformer, 6),
            "peak_memory_mb": round(peak_linformer_mb, 2),
            "seed": seed,
        },
        {
            "backend": "performer",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples,
            "latency_ms": round(t_performer, 3),
            "rel_F_error": round(err_performer, 6),
            "peak_memory_mb": round(peak_performer_mb, 2),
            "seed": seed,
        },
        {
            "backend": "quantum-sim",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples // 2,
            "latency_ms": round(t_quantum_sim, 3),
            "rel_F_error": round(err_quantum_sim, 6),
            "peak_memory_mb": round(peak_quantum_sim_mb, 2),
            "seed": seed,
        },
    ]
    return rows


def main():
    _parser = argparse.ArgumentParser(
        _description="Quick attention benchmarks for quantum-inspired methods"
    )
    parser.add_argument("--batch", _type=int, _default=2)
    parser.add_argument("--seq", _type=int, _default=64)
    parser.add_argument("--dim", _type=int, _default=64)
    parser.add_argument(
        "--samples", _type=int, _nargs="*", _default=[8, 16, 32, 64, 128]
    )
    parser.add_argument("--device", _type=str, _default="cpu")
    parser.add_argument("--seed", _type=int, _default=1337)
    _args = parser.parse_args()

    print(
        "backend,B,N,D,samples,latency_ms,rel_F_error,peak_memory_mb,seed",
        _flush=True,
    )

    for s in args.samples:
        _rows = bench_once(args.batch, args.seq, args.dim, s, args.device, args.seed)
        for r in rows:
            print(
                "{r['backend']},{r['B']},{r['N']},{r['D']},{r['samples']},{r['latency_ms']},{r['rel_F_error']},{r['peak_memory_mb']},{r['seed']}",
                _flush=True,
            )


if __name__ == "__main__":
    main()
