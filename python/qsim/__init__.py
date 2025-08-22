"""
QSim: Quantum simulation layer for Q-Transformers.

Classical simulator for quantum-inspired attention mechanisms.
"""

__version__ = "0.1.0"

from .quantum_simulator import (
    QuantumAttentionSimulator,
    amplitude_encode,
    quantum_measure,
)

__all__ = [
    "QuantumAttentionSimulator",
    "amplitude_encode", 
    "quantum_measure",
]