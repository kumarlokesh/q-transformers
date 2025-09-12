"""
Q-Transformers: Quantum-Inspired Transformer Attention Mechanisms

This package implements quantum-inspired attention mechanisms for Transformers,
providing both theoretical foundations and practical implementations for 
improved attention approximation through quantum sampling techniques.

Phase 2 Features:
- Advanced sampling strategies (QMC, learned importance, control variates)  
- GPU-accelerated quantum kernels with CUDA optimization
- Qiskit quantum hardware backend integration
- Production-ready quantum transformer blocks
- Comprehensive error mitigation techniques
"""

# Core attention mechanisms
from .attention import (
    QuantumAttentionLayer,
    quantum_attention,
    QuantumMultiheadAttention
)

# Phase 2: Advanced sampling strategies
from .advanced_sampling import (
    QuasiMonteCarloSampler,
    LearnedImportanceSampler, 
    MultilevelControlVariate
)

# Phase 2: Error mitigation techniques
from .quantum_error_mitigation import (
    ZeroNoiseExtrapolation,
    SymmetryVerification,
    ProbabilisticErrorCancellation,
    VirtualDistillation
)

# Phase 2: GPU acceleration
from .cuda_kernels import (
    gpu_quantum_attention,
    get_cuda_kernels,
    GPUMemoryOptimizer
)

# Phase 2: Quantum transformer blocks
from .quantum_transformer_blocks import (
    QuantumTransformerBlock,
    QuantumTransformerEncoder,
    ScalableQuantumTransformer,
    create_quantum_gpt,
    create_quantum_bert
)

# Phase 2: Qiskit backend (optional import)
try:
    from .qiskit_backend import (
        QiskitQuantumBackend,
        QuantumAttentionCircuit,
        HybridQuantumClassical,
        create_qiskit_backend
    )
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Memory profiling utilities
from .memory_profiler import MemoryProfiler

# Visualization tools
from .visualization import (
    plot_attention_heatmap,
    analyze_attention_patterns,
    QuantumAttentionAnalyzer
)

__version__ = "2.0.0"
__phase__ = "Phase 2: Production-Ready Quantum Transformers"

# Core exports
__all__ = [
    # Core attention
    "QuantumAttentionLayer", 
    "quantum_attention",
    "QuantumMultiheadAttention",
    
    # Advanced sampling
    "QuasiMonteCarloSampler",
    "LearnedImportanceSampler",
    "MultilevelControlVariate",
    
    # Error mitigation
    "ZeroNoiseExtrapolation", 
    "SymmetryVerification",
    "ProbabilisticErrorCancellation",
    "VirtualDistillation",
    
    # GPU acceleration
    "gpu_quantum_attention",
    "get_cuda_kernels", 
    "GPUMemoryOptimizer",
    
    # Transformer blocks
    "QuantumTransformerBlock",
    "QuantumTransformerEncoder", 
    "ScalableQuantumTransformer",
    "create_quantum_gpt",
    "create_quantum_bert",
    
    # Memory and visualization
    "MemoryProfiler",
    "plot_attention_heatmap",
    "analyze_attention_patterns", 
    "QuantumAttentionAnalyzer",
    
    # Constants
    "QISKIT_AVAILABLE"
]

# Conditional Qiskit exports
if QISKIT_AVAILABLE:
    __all__.extend([
        "QiskitQuantumBackend",
        "QuantumAttentionCircuit", 
        "HybridQuantumClassical",
        "create_qiskit_backend"
    ])