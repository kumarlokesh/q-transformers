"""
Quantum Transformers: Advanced quantum-inspired attention mechanisms for neural networks.

This package provides implementations of quantum-inspired attention mechanisms,
including quantum sampling strategies, advanced sampling techniques, GPU acceleration,
quantum hardware integration, large-scale training infrastructure, and production deployment tools.

Key Components:
- QuantumAttentionLayer: Core quantum attention implementation
- QuantumMultiheadAttention: Multi-head quantum attention
- Advanced sampling strategies (Quasi-Monte Carlo, learned importance sampling)
- Quantum error mitigation techniques
- CUDA acceleration for quantum operations
- Quantum transformer blocks for large-scale models
- Qiskit backend for quantum hardware integration
- Real-world NLP benchmarking and evaluation
- Quantum supremacy verification protocols
- Large-scale distributed training infrastructure
- Production deployment tools and APIs
- Memory profiling and visualization tools
"""

__version__ = "0.1.0"

# Core quantum attention
from .attention import (
    QuantumAttentionLayer,
    QuantumMultiheadAttention,
    quantum_attention
)

from qsim.quantum_simulator import QuantumAttentionSimulator

from .advanced_sampling import (
    QuasiMonteCarloSampler,
    LearnedImportanceSampler,
    MultilevelControlVariate
)

from .quantum_error_mitigation import (
    ZeroNoiseExtrapolation,
    SymmetryVerification,
    ProbabilisticErrorCancellation,
    VirtualDistillation
)

from .cuda_kernels import (
    gpu_quantum_attention,
    GPUMemoryOptimizer
)

try:
    from .qiskit_backend import (
        QiskitQuantumBackend,
        QuantumAttentionCircuit,
        QuantumErrorCorrection,
        HybridQuantumClassical
    )
except Exception:
    # Optional dependency; backend available when qiskit is installed
    pass

from .quantum_transformer_blocks import (
    QuantumTransformerBlock,
    ScalableQuantumTransformer,
    create_quantum_gpt,
    create_quantum_bert
)

try:
    from .nlp_benchmarks import (
        GLUEBenchmarkSuite,
        SuperGLUEBenchmarkSuite,
        QuantumAdvantageAnalyzer,
        ComplexReasoningBenchmark
    )
except ImportError:
    pass

try:
    from .quantum_supremacy import (
        QuantumSupremacyTester,
        StatisticalVerificationFramework,
        ComplexityAnalyzer,
        PatternAnalyzer
    )
except ImportError:
    pass

try:
    from .large_scale_training import (
        QuantumTrainer,
        DistributedQuantumTraining,
        QuantumDataCollator,
        MixedPrecisionQuantumTraining
    )
except ImportError:
    pass

try:
    from .production_deployment import (
        QuantumNLPServer,
        ModelQuantizer,
        AutoScaler,
        HealthMonitor
    )
except ImportError:
    pass

try:
    from .distributed_quantum import (
        DistributedQuantumAttention,
        MultiGPUQuantumTransformer,
        QuantumGradientSynchronizer,
        setup_distributed_quantum_training,
        create_distributed_quantum_transformer
    )
except ImportError:
    pass

try:
    from .deployment import (
        QuantumModelServer,
        DeploymentConfig,
        ModelVersionManager,
        create_app,
        run_server
    )
except ImportError:
    pass

try:
    from .memory_profiler import MemoryProfiler
except ImportError:
    pass

try:
    from .visualization import (
        plot_attention_heatmap,
        analyze_quantum_attention,
        save_attention_analysis
    )
except ImportError:
    pass

__all__ = [
    # Core attention
    "QuantumAttentionLayer",
    "QuantumMultiheadAttention", 
    "quantum_attention",
    
    # Quantum simulation
    "QuantumSimulator",
    
    # Advanced sampling
    "QuasiMonteCarloSampler",
    "LearnedImportanceSampler", 
    "MultiLevelControlVariates",
    
    # Error mitigation
    "ZeroNoiseExtrapolation",
    "SymmetryVerification",
    "ProbabilisticErrorCancellation",
    "VirtualDistillation",
    
    # GPU acceleration
    "gpu_quantum_attention",
    "gpu_mps_contraction", 
    "GPUMemoryOptimizer",
    
    # Quantum hardware
    "QiskitQuantumBackend",
    "QuantumCircuitBuilder",
    "QuantumHardwareManager",
    
    # Transformer blocks
    "QuantumTransformerBlock",
    "ScalableQuantumTransformer",
    "create_quantum_gpt",
    "create_quantum_bert",
    
    # NLP benchmarking
    "GLUEBenchmarkSuite",
    "SuperGLUEBenchmarkSuite", 
    "QuantumAdvantageAnalyzer",
    "NLPEvaluationFramework",
    
    # Quantum supremacy
    "QuantumSupremacyVerifier",
    "ComplexityAnalyzer",
    "QuantumClassicalComparator",
    "SupremacyBenchmarkSuite",
    
    # Training infrastructure
    "QuantumTrainer",
    "TrainingConfig",
    "QuantumDataCollator",
    "create_quantum_trainer",
    "launch_distributed_training",
    
    # Distributed training
    "DistributedQuantumAttention",
    "MultiGPUQuantumTransformer",
    "QuantumGradientSynchronizer", 
    "setup_distributed_quantum_training",
    "create_distributed_quantum_transformer",
    
    # Production deployment
    "QuantumModelServer",
    "DeploymentConfig",
    "ModelVersionManager",
    "create_app",
    "run_server",
    
    # Utilities
    "MemoryProfiler",
    "plot_attention_heatmap",
    "analyze_quantum_attention", 
    "save_attention_analysis"
]