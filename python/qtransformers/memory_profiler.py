"""
Comprehensive memory profiling utilities for Q-Transformers.

Tracks CPU memory, GPU memory, and provides detailed analysis 
of memory usage patterns during quantum attention computations.
"""

import torch
import tracemalloc
import psutil
import gc
import time
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    torch_allocated_mb: float
    torch_cached_mb: float
    python_tracemalloc_mb: float
    process_rss_mb: float
    process_vms_mb: float


class AdvancedMemoryProfiler:
    """
    Advanced memory profiler with CPU and GPU tracking.
    
    Provides detailed analysis of memory usage patterns,
    peak memory tracking, and memory leak detection.
    """
    
    def __init__(self, enable_tracemalloc: bool = True):
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        
        if self.enable_tracemalloc:
            tracemalloc.start()
            
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a comprehensive memory snapshot."""
        timestamp = time.time()
        
        # CPU memory from psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_memory_mb = memory_info.rss / 1024 / 1024
        
        # GPU memory tracking
        gpu_memory_mb = 0.0
        torch_allocated_mb = 0.0
        torch_cached_mb = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            torch_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            torch_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
        
        # Python tracemalloc
        python_tracemalloc_mb = 0.0
        if self.enable_tracemalloc:
            current, peak = tracemalloc.get_traced_memory()
            python_tracemalloc_mb = current / 1024 / 1024
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            cpu_memory_mb=cpu_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            torch_allocated_mb=torch_allocated_mb,
            torch_cached_mb=torch_cached_mb,
            python_tracemalloc_mb=python_tracemalloc_mb,
            process_rss_mb=memory_info.rss / 1024 / 1024,
            process_vms_mb=memory_info.vms / 1024 / 1024
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def set_baseline(self) -> MemorySnapshot:
        """Set baseline memory usage for comparison."""
        self.baseline_snapshot = self.take_snapshot("baseline")
        return self.baseline_snapshot
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager for profiling a code block."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_snapshot = self.take_snapshot(f"{block_name}_start")
        
        try:
            yield self
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            end_snapshot = self.take_snapshot(f"{block_name}_end")
            
            # Calculate memory delta
            cpu_delta = end_snapshot.cpu_memory_mb - start_snapshot.cpu_memory_mb
            gpu_delta = end_snapshot.gpu_memory_mb - start_snapshot.gpu_memory_mb
            
            print(f"Memory usage for {block_name}:")
            print(f"  CPU delta: {cpu_delta:+.2f} MB")
            print(f"  GPU delta: {gpu_delta:+.2f} MB")
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage across all snapshots."""
        if not self.snapshots:
            return {}
        
        return {
            "peak_cpu_mb": max(s.cpu_memory_mb for s in self.snapshots),
            "peak_gpu_mb": max(s.gpu_memory_mb for s in self.snapshots),
            "peak_torch_allocated_mb": max(s.torch_allocated_mb for s in self.snapshots),
            "peak_torch_cached_mb": max(s.torch_cached_mb for s in self.snapshots),
            "peak_python_tracemalloc_mb": max(s.python_tracemalloc_mb for s in self.snapshots),
        }
    
    def analyze_memory_trends(self) -> Dict[str, Any]:
        """Analyze memory usage trends and potential leaks."""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots for trend analysis"}
        
        # Calculate trends
        cpu_trend = (self.snapshots[-1].cpu_memory_mb - self.snapshots[0].cpu_memory_mb)
        gpu_trend = (self.snapshots[-1].gpu_memory_mb - self.snapshots[0].gpu_memory_mb)
        
        # Memory growth rate per snapshot
        cpu_growth_rate = cpu_trend / len(self.snapshots)
        gpu_growth_rate = gpu_trend / len(self.snapshots)
        
        # Detect potential memory leaks (consistent upward trend)
        cpu_increases = sum(1 for i in range(1, len(self.snapshots)) 
                           if self.snapshots[i].cpu_memory_mb > self.snapshots[i-1].cpu_memory_mb)
        gpu_increases = sum(1 for i in range(1, len(self.snapshots)) 
                           if self.snapshots[i].gpu_memory_mb > self.snapshots[i-1].gpu_memory_mb)
        
        total_intervals = len(self.snapshots) - 1
        cpu_leak_probability = cpu_increases / total_intervals if total_intervals > 0 else 0
        gpu_leak_probability = gpu_increases / total_intervals if total_intervals > 0 else 0
        
        return {
            "total_snapshots": len(self.snapshots),
            "cpu_trend_mb": cpu_trend,
            "gpu_trend_mb": gpu_trend,
            "cpu_growth_rate_mb_per_snapshot": cpu_growth_rate,
            "gpu_growth_rate_mb_per_snapshot": gpu_growth_rate,
            "cpu_leak_probability": cpu_leak_probability,
            "gpu_leak_probability": gpu_leak_probability,
            "potential_cpu_leak": cpu_leak_probability > 0.7 and cpu_trend > 10,
            "potential_gpu_leak": gpu_leak_probability > 0.7 and gpu_trend > 10,
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        peak_memory = self.get_peak_memory()
        trends = self.analyze_memory_trends()
        
        # Calculate efficiency metrics
        baseline_memory = self.baseline_snapshot.cpu_memory_mb if self.baseline_snapshot else 0
        current_memory = self.snapshots[-1].cpu_memory_mb
        memory_overhead = current_memory - baseline_memory
        
        return {
            "summary": {
                "total_snapshots": len(self.snapshots),
                "baseline_memory_mb": baseline_memory,
                "current_memory_mb": current_memory,
                "memory_overhead_mb": memory_overhead,
                "efficiency_ratio": baseline_memory / current_memory if current_memory > 0 else 0
            },
            "peak_usage": peak_memory,
            "trends": trends,
            "latest_snapshot": {
                "cpu_memory_mb": self.snapshots[-1].cpu_memory_mb,
                "gpu_memory_mb": self.snapshots[-1].gpu_memory_mb,
                "torch_allocated_mb": self.snapshots[-1].torch_allocated_mb,
                "torch_cached_mb": self.snapshots[-1].torch_cached_mb,
            }
        }
    
    def reset(self):
        """Reset profiler state."""
        self.snapshots.clear()
        self.baseline_snapshot = None
        
        if self.enable_tracemalloc:
            tracemalloc.stop()
            tracemalloc.start()


@contextmanager
def memory_profile_attention(attention_fn, *args, profiler_name: str = "attention", **kwargs):
    """
    Context manager for profiling attention function memory usage.
    
    Args:
        attention_fn: Attention function to profile
        *args, **kwargs: Arguments to pass to attention function
        profiler_name: Name for the profiler
        
    Yields:
        Tuple of (attention_output, memory_report)
    """
    profiler = AdvancedMemoryProfiler()
    profiler.set_baseline()
    
    with profiler.profile_block(profiler_name):
        output = attention_fn(*args, **kwargs)
    
    report = profiler.generate_report()
    yield output, report


def compare_attention_memory_usage(
    attention_functions: Dict[str, callable],
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Compare memory usage across different attention implementations.
    
    Args:
        attention_functions: Dict of {name: function} pairs
        Q, K, V: Attention tensors
        **kwargs: Additional arguments for attention functions
        
    Returns:
        Dict of memory reports per attention function
    """
    results = {}
    
    for name, attention_fn in attention_functions.items():
        profiler = AdvancedMemoryProfiler()
        profiler.set_baseline()
        
        # Run attention with memory profiling
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with profiler.profile_block(name):
            try:
                output = attention_fn(Q, K, V, **kwargs)
                success = True
            except Exception as e:
                success = False
                output = None
        
        report = profiler.generate_report()
        report["success"] = success
        report["function_name"] = name
        
        if success and output is not None:
            report["output_shape"] = list(output.shape)
            report["output_dtype"] = str(output.dtype)
        
        results[name] = report
    
    return results


def benchmark_memory_efficiency(
    attention_fn: callable,
    sequence_lengths: List[int],
    batch_size: int = 1,
    d_model: int = 64,
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Benchmark memory usage scaling with sequence length.
    
    Args:
        attention_fn: Attention function to benchmark
        sequence_lengths: List of sequence lengths to test
        batch_size: Batch size
        d_model: Model dimension
        device: Device to run on
        
    Returns:
        Dict with memory usage data per sequence length
    """
    cpu_memory_usage = []
    gpu_memory_usage = []
    torch_allocated = []
    
    for seq_len in sequence_lengths:
        # Generate test tensors
        Q = torch.randn(batch_size, seq_len, d_model, device=device)
        K = torch.randn(batch_size, seq_len, d_model, device=device)
        V = torch.randn(batch_size, seq_len, d_model, device=device)
        
        profiler = AdvancedMemoryProfiler()
        profiler.set_baseline()
        
        with profiler.profile_block(f"seq_len_{seq_len}"):
            try:
                output = attention_fn(Q, K, V)
            except Exception:
                # Skip if attention fails for this sequence length
                cpu_memory_usage.append(float('inf'))
                gpu_memory_usage.append(float('inf'))
                torch_allocated.append(float('inf'))
                continue
        
        report = profiler.generate_report()
        cpu_memory_usage.append(report["latest_snapshot"]["cpu_memory_mb"])
        gpu_memory_usage.append(report["latest_snapshot"]["gpu_memory_mb"])
        torch_allocated.append(report["latest_snapshot"]["torch_allocated_mb"])
        
        # Cleanup
        del Q, K, V, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {
        "sequence_lengths": sequence_lengths,
        "cpu_memory_mb": cpu_memory_usage,
        "gpu_memory_mb": gpu_memory_usage,
        "torch_allocated_mb": torch_allocated,
    }
