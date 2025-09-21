"""
Quantum Transformers: Advanced quantum-inspired attention mechanisms for neural networks.

This package provides implementations of quantum-inspired attention mechanisms,
including quantum sampling strategies, advanced sampling techniques, GPU acceleration,
quantum hardware integration,
    large-scale training infrastructure,
    and production deployment tools.

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


from .advanced_sampling import (
    LearnedImportanceSampler,
    MultilevelControlVariate,
    QuasiMonteCarloSampler,
)

# Core quantum attention
from .attention import (
    QuantumAttentionLayer,
    QuantumMultiheadAttention,
    quantum_attention,
)
from .quantum_error_mitigation import (
    ProbabilisticErrorCancellation,
    SymmetryVerification,
    VirtualDistillation,
    ZeroNoiseExtrapolation,
)

try:
    from .qiskit_backend import (
        HybridQuantumClassical,
        QiskitQuantumBackend,
        QuantumAttentionCircuit,
        QuantumErrorCorrection,
    )
except Exception:
    # Optional dependency; backend available when qiskit is installed
    pass

from .quantum_transformer_blocks import (
    QuantumTransformerBlock,
    ScalableQuantumTransformer,
    create_quantum_bert,
    create_quantum_gpt,
)

try:
    from .nlp_benchmarks import (
        ComplexReasoningBenchmark,
        GLUEBenchmarkSuite,
        QuantumAdvantageAnalyzer,
        SuperGLUEBenchmarkSuite,
    )
except ImportError:
    pass

try:
    from .quantum_supremacy import (
        ComplexityAnalyzer,
        PatternAnalyzer,
        QuantumSupremacyTester,
        StatisticalVerificationFramework,
    )
except ImportError:
    pass

try:
    from .training_infrastructure import (
        DistributedQuantumTraining,
        MixedPrecisionQuantumTraining,
        QuantumDataCollator,
        QuantumTrainer,
    )
except ImportError:
    pass

try:
    from .distributed_quantum import (
        DistributedQuantumAttention,
        MultiGPUQuantumTransformer,
        QuantumGradientSynchronizer,
        create_distributed_quantum_transformer,
        setup_distributed_quantum_training,
    )
except ImportError:
    pass

try:
    from .deployment import (
        DeploymentConfig,
        ModelVersionManager,
        QuantumModelServer,
        create_app,
        run_server,
    )
except ImportError:
    pass

try:
    pass  # Empty try block for now
except ImportError:
    pass

try:
    from .visualization import (
        analyze_quantum_attention,
        plot_attention_heatmap,
        save_attention_analysis,
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
    "save_attention_analysis",
]  # Test comment
