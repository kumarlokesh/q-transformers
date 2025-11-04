#!/usr/bin/env python3
"""
Comprehensive Sampling Benchmark for Q-Transformers

Tests the following components:
- Advanced sampling strategies (stratified, adaptive, control variates)
- QuantumMultiheadAttention module
- MPS quantum simulation
- Comprehensive memory profiling
- Scaling analysis
- Approximation quality metrics
"""

# flake8: noqa

import argparse
import sys

import torch

sys.path.append("/workspace/python")
from qsim.quantum_simulator import (
    MatrixProductStateSimulator,
    QuantumAttentionSimulator,
)
from qtransformers import (
    QuantumMultiheadAttention,
    quantum_attention,
    quantum_inspired_attention_prototype,
)
from qtransformers.memory_profiler import (
    AdvancedMemoryProfiler,
    compare_attention_memory_usage,
)


def exact_softmax_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """Reference exact attention implementation."""
    _d_k = Q.shape[-1]
    _scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
    _attn_weights = F.softmax(scores, _dim=-1)
    return torch.matmul(attn_weights, V)


def linformer_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, k: _int = 16
) -> torch.Tensor:
    """Linformer efficient attention baseline."""
    batch_size, seq_len, _d_model = Q.shape

    if seq_len <= k:
        return exact_softmax_attention(Q, K, V)

    # Random projection matrices
    E = torch.randn(seq_len, k, _device=Q.device) / (k**0.5)
    F = torch.randn(seq_len, k, _device=Q.device) / (k**0.5)

    # Project keys and values
    _K_proj = torch.matmul(K.transpose(-2, -1), E).transpose(-2, -1)  # (B, k, d_model)
    _V_proj = torch.matmul(V.transpose(-2, -1), F).transpose(-2, -1)  # (B, k, d_model)

    # Compute attention with projected keys/values
    _scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / (d_model**0.5)
    _attn_weights = F.softmax(scores, _dim=-1)
    return torch.matmul(attn_weights, V_proj)


def performer_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, m: _int = 32
) -> torch.Tensor:
    """Performer (FAVOR+) efficient attention baseline."""
    batch_size, seq_len, _d_model = Q.shape

    # Random feature mapping
    _omega = torch.randn(m, d_model, _device=Q.device) / (d_model**0.5)

    def phi(x):
        # Positive random features
        return torch.exp(
            torch.matmul(x, omega.T) - torch.norm(x, _dim=-1, _keepdim=True) ** 2 / 2
        )

    _Q_prime = phi(Q)  # (B, seq_len, m)
    _K_prime = phi(K)  # (B, seq_len, m)

    # Linear attention computation
    _KV = torch.matmul(K_prime.transpose(-2, -1), V)  # (B, m, d_model)
    Z = torch.sum(K_prime, _dim=1, _keepdim=True)  # (B, 1, m)

    _numerator = torch.matmul(Q_prime, KV)  # (B, seq_len, d_model)
    _denominator = torch.matmul(Q_prime, Z.transpose(-2, -1))  # (B, seq_len, 1)

    return numerator / (denominator + 1e-8)


class SamplingBenchmarkSuite:
    """Comprehensive benchmark suite for sampling and attention components."""

    def __init__(self):
        self.profiler = AdvancedMemoryProfiler()
        self.quantum_sim = QuantumAttentionSimulator()
        self.mps_sim = MatrixProductStateSimulator(max_bond_dim=32)

    def benchmark_sampling_strategies(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_samples: _int = 32
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark all quantum sampling strategies."""

        _strategies = [
            ("naive", {"sampling_strategy": "naive"}),
            ("hybrid", {"sampling_strategy": "hybrid"}),
            (
                "stratified",
                {"sampling_strategy": "stratified", "control_variate": True},
            ),
            (
                "adaptive",
                {
                    "sampling_strategy": "adaptive",
                    "adaptive_samples": True,
                    "control_variate": True,
                },
            ),
        ]

        _exact_output = exact_softmax_attention(Q, K, V)
        _results = {}

        for strategy_name, kwargs in strategies:
            self.profiler.reset()
            self.profiler.set_baseline()

            with self.profiler.profile_block(strategy_name):
                _start_time = time.perf_counter()

                _output = quantum_inspired_attention_prototype(
                    Q, K, V, _num_samples=num_samples, **kwargs
                )

                _latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Calculate approximation error
            _error = torch.norm(output - exact_output) / (
                torch.norm(exact_output) + 1e-12
            )

            _memory_report = self.profiler.generate_report()

            results[strategy_name] = {
                "latency_ms": latency_ms,
                "relative_error": float(error.item()),
                "peak_memory_mb": memory_report["peak_usage"]["peak_cpu_mb"],
                "memory_trend": memory_report["trends"]["cpu_trend_mb"],
            }

        return results

    def benchmark_multihead_attention(
        self,
        batch_size: _int = 2,
        seq_len: _int = 64,
        embed_dim: _int = 512,
        num_heads: _int = 8,
    ) -> Dict[str, Any]:
        """Benchmark QuantumMultiheadAttention vs standard attention."""

        # Generate test inputs
        _query = torch.randn(seq_len, batch_size, embed_dim)
        _key = torch.randn(seq_len, batch_size, embed_dim)
        _value = torch.randn(seq_len, batch_size, embed_dim)

        # Standard PyTorch attention
        _standard_attn = torch.nn.MultiheadAttention(
            _embed_dim=embed_dim, _num_heads=num_heads, _batch_first=False
        )

        # Our quantum attention
        _quantum_attn = QuantumMultiheadAttention(
            _embed_dim=embed_dim,
            _num_heads=num_heads,
            _quantum_backend="stratified",
            _num_samples=32,
            _batch_first=False,
        )

        _results = {}

        # Benchmark standard attention
        self.profiler.reset()
        self.profiler.set_baseline()

        with self.profiler.profile_block("standard_attention"):
            _start_time = time.perf_counter()
            standard_output, _standard_weights = standard_attn(query, key, value)
            _standard_latency = (time.perf_counter() - start_time) * 1000.0

        _standard_report = self.profiler.generate_report()

        # Benchmark quantum attention
        self.profiler.reset()
        self.profiler.set_baseline()

        with self.profiler.profile_block("quantum_attention"):
            _start_time = time.perf_counter()
            quantum_output, _quantum_weights = quantum_attn(query, key, value)
            _quantum_latency = (time.perf_counter() - start_time) * 1000.0

        _quantum_report = self.profiler.generate_report()

        # Calculate approximation error
        _error = torch.norm(quantum_output - standard_output) / (
            torch.norm(standard_output) + 1e-12
        )

        _results = {
            "standard_attention": {
                "latency_ms": standard_latency,
                "peak_memory_mb": standard_report["peak_usage"]["peak_cpu_mb"],
                "output_shape": list(standard_output.shape),
            },
            "quantum_attention": {
                "latency_ms": quantum_latency,
                "peak_memory_mb": quantum_report["peak_usage"]["peak_cpu_mb"],
                "output_shape": list(quantum_output.shape),
                "relative_error": float(error.item()),
            },
            "speedup": (
                standard_latency / quantum_latency
                if quantum_latency > 0
                else float("in")
            ),
            "memory_ratio": (
                standard_report["peak_usage"]["peak_cpu_mb"]
                / quantum_report["peak_usage"]["peak_cpu_mb"]
                if quantum_report["peak_usage"]["peak_cpu_mb"] > 0
                else float("in")
            ),
        }

        return results

    def benchmark_mps_simulation(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> Dict[str, Any]:
        """Benchmark Matrix Product State simulation."""

        # Standard quantum simulation
        self.profiler.reset()
        self.profiler.set_baseline()

        with self.profiler.profile_block("standard_quantum_sim"):
            _start_time = time.perf_counter()
            standard_output, _ = self.quantum_sim.simulate_attention(
                Q, K, V, _num_samples=32
            )
            _standard_latency = (time.perf_counter() - start_time) * 1000.0

        _standard_report = self.profiler.generate_report()

        # MPS simulation
        self.profiler.reset()
        self.profiler.set_baseline()

        with self.profiler.profile_block("mps_quantum_sim"):
            _start_time = time.perf_counter()

            _mps_tensors = self.mps_sim.encode_attention_mps(Q, K, V)
            _mps_output = self.mps_sim.mps_attention_forward(
                mps_tensors, V, _num_samples=32
            )

            _mps_latency = (time.perf_counter() - start_time) * 1000.0

        _mps_report = self.profiler.generate_report()

        # MPS efficiency metrics
        _mps_metrics = self.mps_sim.compute_mps_metrics(mps_tensors)

        # Calculate approximation error
        _error = torch.norm(mps_output - standard_output) / (
            torch.norm(standard_output) + 1e-12
        )

        return {
            "standard_simulation": {
                "latency_ms": standard_latency,
                "peak_memory_mb": standard_report["peak_usage"]["peak_cpu_mb"],
            },
            "mps_simulation": {
                "latency_ms": mps_latency,
                "peak_memory_mb": mps_report["peak_usage"]["peak_cpu_mb"],
                "relative_error": float(error.item()),
                "compression_ratio": mps_metrics["compression_ratio"],
                "memory_saving": mps_metrics["memory_saving"],
                "num_mps_tensors": mps_metrics["num_tensors"],
                "max_bond_dim": mps_metrics["max_bond_dim"],
            },
        }

    def benchmark_scaling_analysis(
        self, sequence_lengths: List[int] = [16, 32, 64, 128, 256], d_model: _int = 64
    ) -> Dict[str, List[float]]:
        """Analyze scaling behavior with sequence length."""

        _attention_functions = {
            "exact": exact_softmax_attention,
            "quantum_stratified": lambda Q, K, V: quantum_inspired_attention_prototype(
                Q,
                K,
                V,
                _num_samples=32,
                _sampling_strategy="stratified",
                _control_variate=True,
            ),
            "linformer": lambda Q, K, V: linformer_attention(Q, K, V, k=16),
            "performer": lambda Q, K, V: performer_attention(Q, K, V, m=32),
        }

        _results = {
            name: {"latencies": [], "errors": [], "memories": []}
            for name in attention_functions.keys()
        }
        results["sequence_lengths"] = sequence_lengths

        for seq_len in sequence_lengths:
            print("Testing sequence length: {seq_len}")

            Q = torch.randn(1, seq_len, d_model)
            K = torch.randn(1, seq_len, d_model)
            V = torch.randn(1, seq_len, d_model)

            # Exact reference
            _exact_output = exact_softmax_attention(Q, K, V)

            for name, attention_fn in attention_functions.items():
                self.profiler.reset()
                self.profiler.set_baseline()

                try:
                    with self.profiler.profile_block("{name}_{seq_len}"):
                        _start_time = time.perf_counter()
                        _output = attention_fn(Q, K, V)
                        _latency = (time.perf_counter() - start_time) * 1000.0

                    _report = self.profiler.generate_report()

                    # Calculate error
                    if _name == "exact":
                        _error = 0.0
                    else:
                        _error = torch.norm(output - exact_output) / (
                            torch.norm(exact_output) + 1e-12
                        )
                        _error = float(error.item())

                    results[name]["latencies"].append(latency)
                    results[name]["errors"].append(error)
                    results[name]["memories"].append(
                        report["peak_usage"]["peak_cpu_mb"]
                    )

                except Exception as _e:
                    print("Failed {name} at seq_len {seq_len}: {e}")
                    results[name]["latencies"].append(float("in"))
                    results[name]["errors"].append(float("in"))
                    results[name]["memories"].append(float("in"))

        return results

    def run_comprehensive_benchmark(self, args) -> Dict[str, Any]:
        """Run the full comprehensive benchmark suite."""

        print("🚀 Starting Comprehensive Sampling Benchmark")
        print("=" * 50)

        _results = {
            "config": {
                "batch_size": args.batch,
                "seq_len": args.seq,
                "d_model": args.dim,
                "num_samples": args.samples,
                "device": "cpu",
            },
            "sampling_suite": {},
        }

        # Generate test data
        Q = torch.randn(args.batch, args.seq, args.dim)
        K = torch.randn(args.batch, args.seq, args.dim)
        V = torch.randn(args.batch, args.seq, args.dim)

        # 1. Sampling Strategies Benchmark
        print("\n📊 Benchmarking Advanced Sampling Strategies...")
        _sampling_results = self.benchmark_sampling_strategies(Q, K, V, args.samples)
        results["sampling_suite"]["sampling_strategies"] = sampling_results

        print("Sampling Strategy Results:")
        for strategy, metrics in sampling_results.items():
            print(
                "  {strategy:12} | Error: {metrics['relative_error']:.3f} | "
                "Latency: {metrics['latency_ms']:.1f}ms | "
                "Memory: {metrics['peak_memory_mb']:.1f}MB"
            )

        # 2. Multi-head Attention Benchmark
        print("\n🧠 Benchmarking QuantumMultiheadAttention...")
        _multihead_results = self.benchmark_multihead_attention(
            _batch_size=args.batch, _seq_len=args.seq, _embed_dim=args.dim, _num_heads=8
        )
        results["sampling_suite"]["multihead_attention"] = multihead_results

        print("Multi-head Attention Comparison:")
        print(
            "  Standard    | Latency: {multihead_results['standard_attention']['latency_ms']:.1f}ms | "
            "Memory: {multihead_results['standard_attention']['peak_memory_mb']:.1f}MB"
        )
        print(
            "  Quantum     | Latency: {multihead_results['quantum_attention']['latency_ms']:.1f}ms | "
            "Memory: {multihead_results['quantum_attention']['peak_memory_mb']:.1f}MB | "
            "Error: {multihead_results['quantum_attention']['relative_error']:.3f}"
        )

        # 3. MPS Quantum Simulation Benchmark
        print("\n⚛️  Benchmarking MPS Quantum Simulation...")
        _mps_results = self.benchmark_mps_simulation(Q, K, V)
        results["sampling_suite"]["mps_simulation"] = mps_results

        print("MPS Simulation Results:")
        print(
            "  Standard Sim | Latency: {mps_results['standard_simulation']['latency_ms']:.1f}ms | "
            "Memory: {mps_results['standard_simulation']['peak_memory_mb']:.1f}MB"
        )
        print(
            "  MPS Sim      | Latency: {mps_results['mps_simulation']['latency_ms']:.1f}ms | "
            "Memory: {mps_results['mps_simulation']['peak_memory_mb']:.1f}MB | "
            "Compression: {mps_results['mps_simulation']['compression_ratio']:.1f}x"
        )

        # 4. Scaling Analysis
        if args.scaling:
            print("\n📈 Running Scaling Analysis...")
            _scaling_results = self.benchmark_scaling_analysis()
            results["sampling_suite"]["scaling_analysis"] = scaling_results

            print("Scaling Analysis Complete - see results for details")

        # Summary
        print("\n" + "=" * 50)
        print("🎯 Sampling Benchmark Summary")
        print("=" * 50)

        _best_sampling = min(
            sampling_results.items(), _key=lambda x: x[1]["relative_error"]
        )
        print(
            "Best Sampling Strategy: {best_sampling[0]} (Error: {best_sampling[1]['relative_error']:.3f})"
        )

        _quantum_speedup = multihead_results.get("speedup", 1.0)
        print("Quantum Multi-head Speedup: {quantum_speedup:.2f}x")

        _mps_compression = mps_results["mps_simulation"]["compression_ratio"]
        print("MPS Memory Compression: {mps_compression:.1f}x")

        return results


def main():
    _parser = argparse.ArgumentParser(description="Comprehensive Sampling Benchmark")
    parser.add_argument("--batch", _type=int, _default=2, _help="Batch size")
    parser.add_argument("--seq", _type=int, _default=64, _help="Sequence length")
    parser.add_argument("--dim", _type=int, _default=128, _help="Model dimension")
    parser.add_argument(
        "--samples", _type=int, _default=32, _help="Number of quantum samples"
    )
    parser.add_argument("--scaling", _action="store_true", _help="Run scaling analysis")
    parser.add_argument("--output", _type=str, _help="Output JSON file for results")

    _args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run benchmark
    _benchmark = SamplingBenchmarkSuite()
    _results = benchmark.run_comprehensive_benchmark(args)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, _indent=2)
        print("\nResults saved to {args.output}")

    print("\n✅ Sampling Benchmark Complete!")


if __name__ == "__main__":
    main()
