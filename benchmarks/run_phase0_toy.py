import argparse
import math
import time
from typing import List

import torch

from qtransformers import (
    quantum_attention,
    quantum_inspired_attention_prototype,
)


def exact_softmax_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Exact scaled dot-product attention (no masking) for benchmarking.
    Shapes: Q,K,V = (B, N, D)
    """
    d = Q.shape[-1]
    logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, V)


essential_cols = ["backend", "B", "N", "D", "samples", "latency_ms", "rel_F_error", "seed"]


def bench_once(B: int, N: int, D: int, samples: int, device: str, seed: int = 1337):
    torch.manual_seed(seed)
    Q = torch.randn(B, N, D, device=device)
    K = torch.randn(B, N, D, device=device)
    V = torch.randn(B, N, D, device=device)

    # Exact reference
    t0 = time.perf_counter()
    H_exact = exact_softmax_attention(Q, K, V)
    t_exact = (time.perf_counter() - t0) * 1000.0

    # Prototype
    t0 = time.perf_counter()
    H_proto = quantum_inspired_attention_prototype(Q, K, V, num_samples=samples)
    t_proto = (time.perf_counter() - t0) * 1000.0

    # Via functional API backend toggle (sanity check parity)
    t0 = time.perf_counter()
    H_api = quantum_attention(Q, K, V, top_k=samples, backend="phase0-proto")
    t_api = (time.perf_counter() - t0) * 1000.0

    # Errors
    def rel_fro_err(A: torch.Tensor, B: torch.Tensor) -> float:
        num = torch.linalg.norm(A - B)
        den = torch.linalg.norm(B) + 1e-12
        return float((num / den).item())

    err_proto = rel_fro_err(H_proto, H_exact)
    err_api = rel_fro_err(H_api, H_exact)

    rows = [
        {
            "backend": "exact",
            "B": B,
            "N": N,
            "D": D,
            "samples": 0,
            "latency_ms": round(t_exact, 3),
            "rel_F_error": 0.0,
            "seed": seed,
        },
        {
            "backend": "phase0-proto",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples,
            "latency_ms": round(t_proto, 3),
            "rel_F_error": round(err_proto, 6),
            "seed": seed,
        },
        {
            "backend": "phase0-proto(API)",
            "B": B,
            "N": N,
            "D": D,
            "samples": samples,
            "latency_ms": round(t_api, 3),
            "rel_F_error": round(err_api, 6),
            "seed": seed,
        },
    ]
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 0 toy benchmark for quantum-inspired attention prototype")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--samples", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    print(
        "backend,B,N,D,samples,latency_ms,rel_F_error,seed",
        flush=True,
    )

    for s in args.samples:
        rows = bench_once(args.batch, args.seq, args.dim, s, args.device, args.seed)
        for r in rows:
            print(
                f"{r['backend']},{r['B']},{r['N']},{r['D']},{r['samples']},{r['latency_ms']},{r['rel_F_error']},{r['seed']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
