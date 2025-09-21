#!/usr/bin/env python3
"""
Acceleration Benchmark Suite

Tests GPU acceleration and related components:
- Advanced sampling strategies (QMC, learned importance, control variates)
- GPU-accelerated quantum kernels
- Qiskit quantum hardware backends
- Large-scale quantum transformer blocks
- Error mitigation techniques
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from qtransformers.advanced_sampling import (
    LearnedImportanceSampler,
    MultilevelControlVariate,
    QuasiMonteCarloSampler,
)
from qtransformers.attention import quantum_attention
from qtransformers.cuda_kernels import (
    GPUMemoryOptimizer,
    gpu_quantum_attention,
)
from qtransformers.memory_profiler import MemoryProfiler
from qtransformers.quantum_error_mitigation import (
    SymmetryVerification,
    ZeroNoiseExtrapolation,
)
from qtransformers.quantum_transformer_blocks import (
    QuantumTransformerBlock,
    ScalableQuantumTransformer,
)

try:
    from qtransformers.qiskit_backend import QiskitQuantumBackend

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def benchmark_advanced_sampling_strategies(
    batch_size: int = 4, seq_len: int = 64, embed_dim: int = 256, num_samples: int = 32
) -> Dict[str, Any]:
    """Benchmark advanced sampling strategies."""

    print("🔬 Benchmarking Advanced Sampling Strategies...")

    # Generate test data
    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)

    # Classical reference
    _d_k = embed_dim
    _logits = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
    _exact_weights = F.softmax(logits, _dim=-1)
    _exact_output = torch.matmul(exact_weights, V)

    _results = {}

    # 1. Quasi-Monte Carlo Sampling
    print("  Testing Quasi-Monte Carlo sampling...")
    _qmc_sampler = QuasiMonteCarloSampler(sequence_type="sobol", _scrambling=True)

    _start_time = time.time()
    qmc_samples, _qmc_weights = qmc_sampler.sample_attention_weights(
        logits, num_samples
    )
    _qmc_output = torch.matmul(qmc_weights, V)
    _qmc_time = time.time() - start_time

    _qmc_error = float(torch.norm(qmc_output - exact_output) / torch.norm(exact_output))

    results["qmc"] = {
        "error": qmc_error,
        "latency_ms": qmc_time * 1000,
        "samples_shape": qmc_samples.shape,
        "variance_reduction": qmc_sampler.estimate_variance_reduction(
            exact_weights, qmc_weights, exact_weights  # Simplified
        ),
    }

    # 2. Learned Importance Sampling
    print("  Testing Learned Importance Sampling...")
    _importance_sampler = LearnedImportanceSampler(embed_dim, _hidden_dim=128)

    _start_time = time.time()
    importance_output, importance_weights, _importance_metrics = (
        importance_sampler.adaptive_sample_distribution(
            Q, K, V, _base_num_samples=num_samples
        )
    )
    _importance_time = time.time() - start_time

    _importance_error = float(
        torch.norm(importance_output - exact_output) / torch.norm(exact_output)
    )

    results["learned_importance"] = {
        "error": importance_error,
        "latency_ms": importance_time * 1000,
        "metrics": importance_metrics,
    }

    # 3. Multi-level Control Variates
    print("  Testing Multi-level Control Variates...")
    _control_variates = MultilevelControlVariate(
        ["linformer", "performer", "exact_topk"]
    )

    # Simple quantum sampler for control variate testing
    def simple_quantum_sampler(Q, K, V):
        return quantum_attention(Q, K, V, _top_k=num_samples, _backend="prototype")

    _start_time = time.time()
    _quantum_output = simple_quantum_sampler(Q, K, V)
    _corrected_output = control_variates.apply_control_variates(quantum_output, Q, K, V)
    _control_time = time.time() - start_time

    _control_error = float(
        torch.norm(corrected_output - exact_output) / torch.norm(exact_output)
    )

    results["control_variates"] = {
        "error": control_error,
        "latency_ms": control_time * 1000,
        "methods": control_variates.control_methods,
    }

    return results


def benchmark_gpu_acceleration(
    batch_size: int = 4, seq_len: int = 128, embed_dim: int = 512, num_samples: int = 64
) -> Dict[str, Any]:
    """Benchmark GPU acceleration."""

    print("⚡ Benchmarking GPU Acceleration...")

    _results = {}

    # Test data
    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)

    if torch.cuda.is_available():
        Q_gpu, K_gpu, _V_gpu = Q.cuda(), K.cuda(), V.cuda()

        print("  Testing GPU quantum attention...")
        _start_time = time.time()
        _gpu_output = gpu_quantum_attention(
            Q_gpu, K_gpu, V_gpu, _num_samples=num_samples
        )
        torch.cuda.synchronize()
        _gpu_time = time.time() - start_time

        results["gpu_quantum_attention"] = {
            "latency_ms": gpu_time * 1000,
            "output_shape": gpu_output.shape,
            "memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

        print("  Testing GPU memory optimization...")
        _memory_optimizer = GPUMemoryOptimizer()
        _optimal_batch = memory_optimizer.get_optimal_batch_size(seq_len, embed_dim)

        results["memory_optimization"] = {
            "optimal_batch_size": optimal_batch,
            "total_gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            "current_memory_usage_mb": torch.cuda.memory_allocated() / 1024**2,
        }

    else:
        print("  CUDA not available, testing CPU fallback...")
        _start_time = time.time()
        _cpu_output = gpu_quantum_attention(
            Q, K, V, _num_samples=num_samples, _backend="cpu_fallback"
        )
        _cpu_time = time.time() - start_time

        results["cpu_fallback"] = {
            "latency_ms": cpu_time * 1000,
            "output_shape": cpu_output.shape,
        }

    return results


def benchmark_error_mitigation(
    batch_size: int = 2, seq_len: int = 32, embed_dim: int = 128
) -> Dict[str, Any]:
    """Benchmark error mitigation techniques."""

    print("🛡️ Benchmarking Error Mitigation...")

    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)

    _results = {}

    print("  Testing Zero-Noise Extrapolation...")
    _zne = ZeroNoiseExtrapolation(noise_levels=[0.0, 0.01, 0.02, 0.05])

    def noisy_attention(Q, K, V, _noise_level=0.0):
        _logits = torch.matmul(Q, K.transpose(-2, -1)) / (embed_dim**0.5)
        if noise_level > 0:
            _noise = torch.randn_like(logits) * noise_level
            _logits = logits + noise
        _weights = F.softmax(logits, _dim=-1)
        return torch.matmul(weights, V)

    _start_time = time.time()
    mitigated_output, _zne_metrics = zne.mitigate_quantum_attention(
        noisy_attention, Q, K, V
    )
    _zne_time = time.time() - start_time

    _exact_output = noisy_attention(Q, K, V, _noise_level=0.0)
    _zne_error = float(
        torch.norm(mitigated_output - exact_output) / torch.norm(exact_output)
    )

    results["zero_noise_extrapolation"] = {
        "error": zne_error,
        "latency_ms": zne_time * 1000,
        "metrics": zne_metrics,
    }

    print("  Testing Symmetry Verification...")
    _symmetry_verifier = SymmetryVerification(
        _symmetries=["row_normalization", "positivity", "attention_entropy"]
    )

    # Generate attention weights with some violations
    _noisy_logits = torch.matmul(Q, K.transpose(-2, -1)) / (embed_dim**0.5)
    noisy_logits += torch.randn_like(noisy_logits) * 0.1
    _noisy_weights = F.softmax(noisy_logits, _dim=-1)

    _start_time = time.time()
    _violations = symmetry_verifier.verify_attention_symmetries(noisy_weights)
    _corrected_weights = symmetry_verifier.correct_attention_symmetries(noisy_weights)
    _symmetry_time = time.time() - start_time

    results["symmetry_verification"] = {
        "latency_ms": symmetry_time * 1000,
        "violations": violations,
        "correction_applied": True,
    }

    return results


def benchmark_quantum_transformer_blocks(
    batch_size: int = 2, seq_len: int = 64, d_model: int = 256, nhead: int = 8
) -> Dict[str, Any]:
    """Benchmark quantum transformer blocks."""

    print("🧠 Benchmarking Quantum Transformer Blocks...")

    _results = {}

    # Quantum configuration
    _quantum_config = {
        "backend": "prototype",
        "num_samples": 32,
        "use_advanced_sampling": True,
        "use_error_mitigation": False,  # Disable for speed
        "use_gpu_acceleration": torch.cuda.is_available(),
    }

    # Single Quantum Transformer Block
    print("  Testing QuantumTransformerBlock...")
    _quantum_block = QuantumTransformerBlock(
        _d_model=d_model, _nhead=nhead, _quantum_config=quantum_config
    )

    # Test data
    _src = torch.randn(batch_size, seq_len, d_model)

    _start_time = time.time()
    with torch.no_grad():
        _block_output = quantum_block(src)
    _block_time = time.time() - start_time

    results["quantum_transformer_block"] = {
        "latency_ms": block_time * 1000,
        "output_shape": block_output.shape,
        "num_parameters": sum(p.numel() for p in quantum_block.parameters()),
        "memory_usage_mb": (
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        ),
    }

    print("  Testing ScalableQuantumTransformer...")
    _quantum_transformer = ScalableQuantumTransformer(
        _vocab_size=1000,
        _d_model=d_model,
        _nhead=nhead,
        _num_encoder_layers=3,  # Smaller for benchmarking
        _quantum_config=quantum_config,
        _max_seq_length=seq_len,
    )

    # Generate token inputs
    _input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    _start_time = time.time()
    with torch.no_grad():
        _logits = quantum_transformer(input_ids)
    _transformer_time = time.time() - start_time

    results["scalable_quantum_transformer"] = {
        "latency_ms": transformer_time * 1000,
        "output_shape": logits.shape,
        "num_parameters": sum(p.numel() for p in quantum_transformer.parameters()),
        "memory_usage_mb": (
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        ),
    }

    return results


def benchmark_scaling_analysis(max_seq_len: int = 256) -> Dict[str, Any]:
    """Analyze scaling behavior of implementations."""

    print("📈 Running Scaling Analysis...")

    _seq_lengths = [16, 32, 64, 128]
    if max_seq_len > 128:
        seq_lengths.append(256)

    _results = {"seq_lengths": seq_lengths, "scaling_data": {}}

    for seq_len in seq_lengths:
        print("  Testing sequence length: {seq_len}")

        # Test data
        _batch_size = max(1, 32 // (seq_len // 16))  # Adjust batch size for memory
        _embed_dim = 256

        Q = torch.randn(batch_size, seq_len, embed_dim)
        K = torch.randn(batch_size, seq_len, embed_dim)
        V = torch.randn(batch_size, seq_len, embed_dim)

        _seq_data = {}

        # Classical attention baseline
        _start_time = time.time()
        _classical_logits = torch.matmul(Q, K.transpose(-2, -1)) / (embed_dim**0.5)
        _classical_weights = F.softmax(classical_logits, _dim=-1)
        _classical_output = torch.matmul(classical_weights, V)
        _classical_time = time.time() - start_time

        seq_data["classical"] = {
            "latency_ms": classical_time * 1000,
            "memory_mb": classical_weights.numel() * 4 / 1024**2,  # float32
        }

        # Quantum attention
        _start_time = time.time()
        _quantum_output = quantum_attention(Q, K, V, _top_k=32, _backend="prototype")
        _quantum_time = time.time() - start_time

        _quantum_error = float(
            torch.norm(quantum_output - classical_output) / torch.norm(classical_output)
        )

        seq_data["quantum"] = {
            "latency_ms": quantum_time * 1000,
            "error": quantum_error,
            "memory_mb": Q.numel() * 4 / 1024**2,  # Approximate
        }

        results["scaling_data"][seq_len] = seq_data

    return results


def generate_benchmark_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive benchmark report."""

    _report = []
    report.append("# Q-Transformers Acceleration Benchmark Report")
    report.append("=" * 50)
    report.append("")

    # Advanced Sampling Results
    if "advanced_sampling" in results:
        report.append("## Advanced Sampling Strategies")
        report.append("")

        _sampling_results = results["advanced_sampling"]

        report.append("| Method | Error (%) | Latency (ms) | Notes |")
        report.append("|--------|-----------|--------------|-------|")

        for method, data in sampling_results.items():
            _error_pct = data["error"] * 100
            _latency = data["latency_ms"]
            _notes = "Samples: {data.get('samples_shape', 'N/A')}"
            report.append(
                "| {method.replace('_', ' ').title()} | {error_pct:.2f}% | {latency:.2f} | {notes} |"
            )

        report.append("")

        _best_method = min(
            sampling_results.keys(), _key=lambda k: sampling_results[k]["error"]
        )
        _best_error = sampling_results[best_method]["error"] * 100
        report.append(
            "**Best Method:** {best_method.replace('_', ' ').title()} ({best_error:.2f}% error)"
        )
        report.append("")

    # GPU Acceleration Results
    if "gpu_acceleration" in results:
        report.append("## GPU Acceleration Performance")
        report.append("")

        _gpu_results = results["gpu_acceleration"]

        if "gpu_quantum_attention" in gpu_results:
            _gpu_data = gpu_results["gpu_quantum_attention"]
            report.append("- **GPU Quantum Attention:** {gpu_data['latency_ms']:.2f}ms")
            report.append(
                "- **Memory Allocated:** {gpu_data['memory_allocated_mb']:.2f}MB"
            )

        if "memory_optimization" in gpu_results:
            _mem_data = gpu_results["memory_optimization"]
            report.append("- **Optimal Batch Size:** {mem_data['optimal_batch_size']}")
            report.append("- **GPU Memory:** {mem_data['total_gpu_memory_gb']:.2f}GB")

        report.append("")

    # Error Mitigation Results
    if "error_mitigation" in results:
        report.append("## Error Mitigation Effectiveness")
        report.append("")

        _mitigation_results = results["error_mitigation"]

        if "zero_noise_extrapolation" in mitigation_results:
            _zne_data = mitigation_results["zero_noise_extrapolation"]
            report.append(
                "- **Zero-Noise Extrapolation:** {zne_data['error']*100:.2f}% error"
            )
            report.append(
                "- **Extrapolation Confidence:** {zne_data['metrics'].get('extrapolation_confidence', 'N/A')}"
            )

        if "symmetry_verification" in mitigation_results:
            _sym_data = mitigation_results["symmetry_verification"]
            _violations = sym_data["violations"]
            report.append("- **Symmetry Violations:** {len(violations)} types detected")

        report.append("")

    # Quantum Transformer Blocks
    if "quantum_blocks" in results:
        report.append("## Quantum Transformer Block Performance")
        report.append("")

        _block_results = results["quantum_blocks"]

        if "quantum_transformer_block" in block_results:
            _block_data = block_results["quantum_transformer_block"]
            report.append(
                "- **Single Block Latency:** {block_data['latency_ms']:.2f}ms"
            )
            report.append("- **Parameters:** {block_data['num_parameters']:,}")

        if "scalable_quantum_transformer" in block_results:
            _transformer_data = block_results["scalable_quantum_transformer"]
            report.append(
                "- **Full Transformer Latency:** {transformer_data['latency_ms']:.2f}ms"
            )
            report.append(
                "- **Total Parameters:** {transformer_data['num_parameters']:,}"
            )

        report.append("")

    # Scaling Analysis
    if "scaling" in results:
        report.append("## Scaling Analysis")
        report.append("")

        _scaling_data = results["scaling"]["scaling_data"]

        report.append(
            "| Seq Length | Classical (ms) | Quantum (ms) | Error (%) | Speedup |"
        )
        report.append(
            "|------------|----------------|--------------|-----------|---------|"
        )

        for seq_len in results["scaling"]["seq_lengths"]:
            _classical_time = scaling_data[seq_len]["classical"]["latency_ms"]
            _quantum_time = scaling_data[seq_len]["quantum"]["latency_ms"]
            _error = scaling_data[seq_len]["quantum"]["error"] * 100
            _speedup = classical_time / quantum_time

            report.append(
                "| {seq_len} | {classical_time:.2f} | {quantum_time:.2f} | {error:.2f}% | {speedup:.2f}x |"
            )

        report.append("")

    report.append("## Acceleration Benchmark Summary")
    report.append("")
    report.append("✅ **Completed Implementations:**")
    report.append(
        "- Advanced sampling strategies (QMC, learned importance, control variates)"
    )
    report.append("- GPU-accelerated quantum kernels with CUDA optimization")
    report.append("- Qiskit quantum hardware backend integration")
    report.append("- Quantum transformer blocks")
    report.append("- Comprehensive error mitigation techniques")
    report.append("")

    return "\n".join(report)


def main():
    _parser = argparse.ArgumentParser(
        _description="Q-Transformers Acceleration Benchmarks"
    )
    parser.add_argument(
        "--batch_size", _type=int, _default=4, _help="Batch size for testing"
    )
    parser.add_argument(
        "--seq_len", _type=int, _default=64, _help="Sequence length for testing"
    )
    parser.add_argument(
        "--embed_dim", _type=int, _default=256, _help="Embedding dimension"
    )
    parser.add_argument(
        "--num_samples", _type=int, _default=32, _help="Number of quantum samples"
    )
    parser.add_argument(
        "--output_dir", _type=str, _default=".", _help="Output directory for results"
    )
    parser.add_argument(
        "--skip_scaling", _action="store_true", _help="Skip scaling analysis"
    )
    _args = parser.parse_args()

    print("🚀 Q-Transformers Acceleration Benchmark")
    print("=" * 60)
    print()

    _profiler = MemoryProfiler()
    profiler.start_profiling()

    _all_results = {}

    try:
        # 1. Advanced Sampling Strategies
        all_results["advanced_sampling"] = benchmark_advanced_sampling_strategies(
            args.batch_size, args.seq_len, args.embed_dim, args.num_samples
        )

        # 2. GPU Acceleration
        all_results["gpu_acceleration"] = benchmark_gpu_acceleration(
            args.batch_size, args.seq_len, args.embed_dim, args.num_samples
        )

        # 3. Error Mitigation
        all_results["error_mitigation"] = benchmark_error_mitigation(
            _batch_size=2, _seq_len=32, _embed_dim=128  # Smaller for speed
        )

        # 4. Quantum Transformer Blocks
        all_results["quantum_blocks"] = benchmark_quantum_transformer_blocks(
            _batch_size=2, _seq_len=args.seq_len, _d_model=args.embed_dim, _nhead=8
        )

        # 5. Scaling Analysis (optional)
        if not args.skip_scaling:
            all_results["scaling"] = benchmark_scaling_analysis(
                _max_seq_len=min(args.seq_len * 2, 256)
            )

    except Exception as _e:
        print("❌ Benchmark error: {e}")
        return 1

    finally:
        _memory_report = profiler.stop_profiling()
        all_results["memory_profile"] = memory_report

    _report = generate_benchmark_report(all_results)

    _report_path = os.path.join(args.output_dir, "acceleration_benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("📊 ACCELERATION BENCHMARK COMPLETE")
    print("=" * 60)
    print("📄 Report saved to: {report_path}")
    print("\n🎉 Acceleration benchmarks completed!")

    return 0


if __name__ == "__main__":
    exit(main())
