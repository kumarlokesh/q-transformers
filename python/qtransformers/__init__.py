"""
Q-Transformers: Quantum-inspired attention mechanisms for Transformers.

A hybrid Rust + Python library for efficient, interpretable attention using 
quantum-inspired algorithms.
"""

__version__ = "0.1.0"
__author__ = "Q-Transformers Team"

from .attention import (
    QuantumAttentionLayer,
    quantum_attention,
    quantum_inspired_attention_prototype,
)

# Try to import Rust backend (will be available after Phase 2)
try:
    from qtransformers_core import quantum_attention_rs, classical_attention_rs
    HAS_RUST_BACKEND = True
except ImportError:
    HAS_RUST_BACKEND = False

__all__ = [
    "QuantumAttentionLayer", 
    "quantum_attention",
    "quantum_inspired_attention_prototype",
    "HAS_RUST_BACKEND",
]

if HAS_RUST_BACKEND:
    __all__.extend(["quantum_attention_rs", "classical_attention_rs"])