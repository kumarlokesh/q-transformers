"""
Qiskit Backend for Quantum Hardware Integration

Implements real quantum hardware backends for Q-Transformers:
- IBM Quantum hardware integration
- Quantum circuit construction for attention
- Hardware-aware noise models
- Quantum error correction protocols
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
import math

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import Aer, execute, IBMQ
    from qiskit.providers import Backend
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import Statevector, DensityMatrix
    from qiskit.algorithms.optimizers import SPSA
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not available. Install with: pip install qiskit qiskit-aer")


class QuantumAttentionCircuit:
    """Quantum circuit implementation of attention mechanism."""
    
    def __init__(
        self,
        num_qubits: int,
        depth: int = 3,
        entanglement: str = "linear"
    ):
        """
        Initialize quantum attention circuit.
        
        Args:
            num_qubits: Number of qubits for attention representation
            depth: Circuit depth (number of layers)
            entanglement: Entanglement pattern ("linear", "full", "circular")
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum hardware backend")
        
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        
        # Create parameterized quantum circuit for attention
        self.qreg = QuantumRegister(num_qubits, 'attention')
        self.creg = ClassicalRegister(num_qubits, 'measurement')
        self.circuit = QuantumCircuit(self.qreg, self.creg)
        
        # Parameters for quantum attention
        self.theta_params = []
        self._build_attention_circuit()
    
    def _build_attention_circuit(self):
        """Build parameterized quantum attention circuit."""
        
        for layer in range(self.depth):
            # Rotation layer (encoding query/key information)
            for i in range(self.num_qubits):
                theta = Parameter(f'theta_{layer}_{i}')
                self.theta_params.append(theta)
                self.circuit.ry(theta, i)
            
            # Entanglement layer (attention interactions)
            if self.entanglement == "linear":
                for i in range(self.num_qubits - 1):
                    self.circuit.cx(i, i + 1)
            elif self.entanglement == "circular":
                for i in range(self.num_qubits - 1):
                    self.circuit.cx(i, i + 1)
                if self.num_qubits > 2:
                    self.circuit.cx(self.num_qubits - 1, 0)
            elif self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        self.circuit.cx(i, j)
        
        # Final measurement layer
        self.circuit.measure_all()
    
    def encode_attention_data(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> List[float]:
        """
        Encode query-key attention data into circuit parameters.
        
        Args:
            query: Query vector (embed_dim,)
            key: Key vector (embed_dim,)
            
        Returns:
            Parameter values for quantum circuit
        """
        # Compute attention logits
        attention_logit = torch.dot(query, key) / math.sqrt(query.shape[0])
        
        # Map to quantum circuit parameters
        param_values = []
        
        # Use attention logit and vector components to set rotation angles
        for layer in range(self.depth):
            for i in range(self.num_qubits):
                if i < query.shape[0] and i < key.shape[0]:
                    # Use query-key product for this qubit
                    angle = float(query[i] * key[i] + attention_logit * 0.1)
                else:
                    # Pad with scaled attention logit
                    angle = float(attention_logit * (i + 1) / self.num_qubits)
                
                # Normalize angle to [0, 2π]
                angle = math.atan2(math.sin(angle), math.cos(angle)) + math.pi
                param_values.append(angle)
        
        return param_values
    
    def get_parameterized_circuit(self) -> QuantumCircuit:
        """Get the parameterized quantum circuit."""
        return self.circuit


class QiskitQuantumBackend:
    """Qiskit-based quantum backend for attention computation."""
    
    def __init__(
        self,
        backend_name: str = "aer_simulator",
        num_qubits: int = 8,
        shots: int = 1024,
        noise_model: Optional[NoiseModel] = None
    ):
        """
        Initialize Qiskit quantum backend.
        
        Args:
            backend_name: Qiskit backend ("aer_simulator", "ibmq_qasm_simulator", etc.)
            num_qubits: Number of qubits for quantum attention
            shots: Number of measurement shots
            noise_model: Optional noise model for realistic simulation
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum hardware backend")
        
        self.backend_name = backend_name
        self.num_qubits = num_qubits
        self.shots = shots
        self.noise_model = noise_model
        
        # Initialize backend
        self.backend = self._get_backend()
        
        # Create quantum attention circuit
        self.attention_circuit = QuantumAttentionCircuit(
            num_qubits=num_qubits,
            depth=3,
            entanglement="linear"
        )
        
    def _get_backend(self) -> Backend:
        """Get Qiskit backend instance."""
        
        if self.backend_name == "aer_simulator":
            backend = AerSimulator()
        elif self.backend_name.startswith("ibmq"):
            # Try to load IBM Quantum backend
            try:
                IBMQ.load_account()
                provider = IBMQ.get_provider()
                backend = provider.get_backend(self.backend_name)
            except Exception as e:
                print(f"Failed to load IBM backend {self.backend_name}: {e}")
                print("Falling back to Aer simulator")
                backend = AerSimulator()
        else:
            # Default to Aer simulator
            backend = AerSimulator()
        
        return backend
    
    def create_hardware_noise_model(
        self,
        gate_error_rate: float = 0.001,
        measurement_error_rate: float = 0.02,
        thermal_relaxation: bool = True
    ) -> NoiseModel:
        """
        Create realistic hardware noise model.
        
        Args:
            gate_error_rate: Single/two-qubit gate error rate
            measurement_error_rate: Measurement error rate
            thermal_relaxation: Include thermal relaxation effects
            
        Returns:
            Qiskit noise model
        """
        noise_model = NoiseModel()
        
        # Gate errors
        single_qubit_error = depolarizing_error(gate_error_rate, 1)
        two_qubit_error = depolarizing_error(gate_error_rate * 2, 2)
        
        # Add errors to all gates
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['ry', 'rz', 'sx'])
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
        
        # Measurement errors
        if measurement_error_rate > 0:
            from qiskit.providers.aer.noise import ReadoutError
            readout_error = ReadoutError([
                [1 - measurement_error_rate, measurement_error_rate],
                [measurement_error_rate, 1 - measurement_error_rate]
            ])
            noise_model.add_all_qubit_readout_error(readout_error)
        
        # Thermal relaxation (simplified)
        if thermal_relaxation:
            t1 = 50e-6  # T1 relaxation time (50 μs)
            t2 = 70e-6  # T2 dephasing time (70 μs)
            gate_time = 50e-9  # Gate time (50 ns)
            
            from qiskit.providers.aer.noise import thermal_relaxation_error
            thermal_error = thermal_relaxation_error(t1, t2, gate_time)
            noise_model.add_all_qubit_quantum_error(thermal_error, ['ry', 'rz', 'sx'])
        
        return noise_model
    
    def quantum_attention_measurement(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        num_measurements: int = None
    ) -> torch.Tensor:
        """
        Perform quantum attention measurement.
        
        Args:
            query: Query vector (embed_dim,)
            key: Key vector (embed_dim,)
            num_measurements: Number of quantum measurements (default: self.shots)
            
        Returns:
            Quantum measurement probabilities
        """
        if num_measurements is None:
            num_measurements = self.shots
        
        # Encode query-key data into circuit parameters
        param_values = self.attention_circuit.encode_attention_data(query, key)
        
        # Get parameterized circuit
        qc = self.attention_circuit.get_parameterized_circuit()
        
        # Bind parameters
        bound_circuit = qc.bind_parameters(dict(zip(
            self.attention_circuit.theta_params, param_values
        )))
        
        # Execute on quantum backend
        job = execute(
            bound_circuit,
            backend=self.backend,
            shots=num_measurements,
            noise_model=self.noise_model
        )
        
        result = job.result()
        counts = result.get_counts()
        
        # Convert measurement counts to probabilities
        probs = torch.zeros(2**self.num_qubits)
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Convert bitstring to index
            index = int(bitstring, 2)
            probs[index] = count / total_shots
        
        return probs
    
    def batched_quantum_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        max_sequence_length: int = 16
    ) -> torch.Tensor:
        """
        Compute attention for batched inputs using quantum hardware.
        
        Args:
            Q: Query tensor (batch_size, seq_len, embed_dim)
            K: Key tensor (batch_size, seq_len, embed_dim)  
            V: Value tensor (batch_size, seq_len, embed_dim)
            max_sequence_length: Maximum sequence length for quantum processing
            
        Returns:
            Quantum attention output
        """
        batch_size, seq_len_q, embed_dim = Q.shape
        seq_len_k = K.shape[1]
        
        # Limit sequence length for quantum hardware
        actual_seq_len = min(seq_len_q, max_sequence_length)
        
        # Initialize output
        output = torch.zeros(batch_size, actual_seq_len, embed_dim)
        
        for b in range(batch_size):
            for i in range(actual_seq_len):
                query_vec = Q[b, i, :]
                
                # Compute quantum attention weights for this query
                attention_weights = torch.zeros(min(seq_len_k, max_sequence_length))
                
                for j in range(min(seq_len_k, max_sequence_length)):
                    key_vec = K[b, j, :]
                    
                    # Quantum measurement for this query-key pair
                    probs = self.quantum_attention_measurement(query_vec, key_vec)
                    
                    # Extract attention weight (use first few measurement outcomes)
                    attention_weight = probs[:min(4, len(probs))].sum()  # Sum first few outcomes
                    attention_weights[j] = attention_weight
                
                # Normalize attention weights
                attention_weights = F.softmax(attention_weights, dim=0)
                
                # Apply attention to values
                weighted_values = torch.zeros(embed_dim)
                for j in range(len(attention_weights)):
                    weighted_values += attention_weights[j] * V[b, j, :]
                
                output[b, i, :] = weighted_values
        
        # Pad output if needed
        if seq_len_q > actual_seq_len:
            padding = torch.zeros(batch_size, seq_len_q - actual_seq_len, embed_dim)
            output = torch.cat([output, padding], dim=1)
        
        return output


class QuantumErrorCorrection:
    """Quantum error correction for attention circuits."""
    
    def __init__(self, code_type: str = "repetition"):
        """
        Initialize quantum error correction.
        
        Args:
            code_type: Type of error correction code ("repetition", "surface")
        """
        self.code_type = code_type
    
    def encode_logical_qubit(
        self,
        circuit: QuantumCircuit,
        data_qubit: int,
        ancilla_qubits: List[int]
    ) -> QuantumCircuit:
        """
        Encode a logical qubit with error correction.
        
        Args:
            circuit: Quantum circuit to modify
            data_qubit: Data qubit index
            ancilla_qubits: Ancilla qubit indices for error correction
            
        Returns:
            Modified circuit with error correction encoding
        """
        if self.code_type == "repetition":
            # Simple repetition code (3-qubit)
            if len(ancilla_qubits) >= 2:
                circuit.cx(data_qubit, ancilla_qubits[0])
                circuit.cx(data_qubit, ancilla_qubits[1])
        
        return circuit
    
    def decode_logical_qubit(
        self,
        circuit: QuantumCircuit,
        data_qubit: int,
        ancilla_qubits: List[int],
        syndrome_bits: List[int]
    ) -> QuantumCircuit:
        """
        Decode and error-correct a logical qubit.
        
        Args:
            circuit: Quantum circuit to modify
            data_qubit: Data qubit index
            ancilla_qubits: Ancilla qubit indices
            syndrome_bits: Classical bits for syndrome measurement
            
        Returns:
            Modified circuit with error correction decoding
        """
        if self.code_type == "repetition" and len(ancilla_qubits) >= 2:
            # Syndrome measurement
            circuit.cx(data_qubit, ancilla_qubits[0])
            circuit.cx(data_qubit, ancilla_qubits[1])
            circuit.measure(ancilla_qubits[0], syndrome_bits[0])
            circuit.measure(ancilla_qubits[1], syndrome_bits[1])
            
            # Error correction (simplified - would need classical control)
            # In practice, this requires post-processing of measurement results
        
        return circuit


class HybridQuantumClassical:
    """Hybrid quantum-classical attention computation."""
    
    def __init__(
        self,
        quantum_backend: QiskitQuantumBackend,
        classical_fallback: bool = True
    ):
        """
        Initialize hybrid quantum-classical backend.
        
        Args:
            quantum_backend: Quantum backend for quantum computations
            classical_fallback: Use classical computation when quantum fails
        """
        self.quantum_backend = quantum_backend
        self.classical_fallback = classical_fallback
        
    def hybrid_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        quantum_ratio: float = 0.5,
        sequence_threshold: int = 32
    ) -> torch.Tensor:
        """
        Compute attention using hybrid quantum-classical approach.
        
        Args:
            Q, K, V: Attention tensors
            quantum_ratio: Fraction of computation to perform quantumly
            sequence_threshold: Use classical for sequences longer than this
            
        Returns:
            Hybrid attention output
        """
        batch_size, seq_len, embed_dim = Q.shape
        
        if seq_len > sequence_threshold:
            # Use classical attention for long sequences
            if self.classical_fallback:
                return self._classical_attention(Q, K, V)
            else:
                # Truncate for quantum processing
                Q_trunc = Q[:, :sequence_threshold, :]
                K_trunc = K[:, :sequence_threshold, :]
                V_trunc = V[:, :sequence_threshold, :]
                return self.quantum_backend.batched_quantum_attention(Q_trunc, K_trunc, V_trunc)
        
        # Determine which attention heads to compute quantumly
        num_quantum_queries = int(seq_len * quantum_ratio)
        
        output = torch.zeros_like(Q)
        
        # Quantum computation for first part
        if num_quantum_queries > 0:
            Q_quantum = Q[:, :num_quantum_queries, :]
            quantum_output = self.quantum_backend.batched_quantum_attention(Q_quantum, K, V)
            output[:, :num_quantum_queries, :] = quantum_output[:, :num_quantum_queries, :]
        
        # Classical computation for remaining part
        if num_quantum_queries < seq_len:
            Q_classical = Q[:, num_quantum_queries:, :]
            classical_output = self._classical_attention(Q_classical, K, V)
            output[:, num_quantum_queries:, :] = classical_output
        
        return output
    
    def _classical_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """Classical scaled dot-product attention."""
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)
    
    def benchmark_quantum_advantage(
        self,
        test_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        quantum_ratios: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    ) -> Dict[str, Any]:
        """
        Benchmark quantum advantage across different quantum ratios.
        
        Args:
            test_inputs: List of (Q, K, V) test cases
            quantum_ratios: Ratios of quantum vs classical computation
            
        Returns:
            Benchmarking results
        """
        results = {
            "quantum_ratios": quantum_ratios,
            "execution_times": [],
            "approximation_errors": [],
            "quantum_circuit_depths": []
        }
        
        for ratio in quantum_ratios:
            ratio_times = []
            ratio_errors = []
            
            for Q, K, V in test_inputs:
                # Classical reference
                classical_output = self._classical_attention(Q, K, V)
                
                # Hybrid quantum-classical
                import time
                start_time = time.time()
                hybrid_output = self.hybrid_attention(Q, K, V, quantum_ratio=ratio)
                execution_time = time.time() - start_time
                
                # Compute approximation error
                error = float(torch.norm(hybrid_output - classical_output) / torch.norm(classical_output))
                
                ratio_times.append(execution_time)
                ratio_errors.append(error)
            
            results["execution_times"].append(ratio_times)
            results["approximation_errors"].append(ratio_errors)
            results["quantum_circuit_depths"].append(
                self.quantum_backend.attention_circuit.depth
            )
        
        return results


def create_qiskit_backend(
    backend_name: str = "aer_simulator",
    enable_noise: bool = True,
    **kwargs
) -> QiskitQuantumBackend:
    """
    Factory function to create Qiskit quantum backend.
    
    Args:
        backend_name: Qiskit backend name
        enable_noise: Whether to include realistic noise
        **kwargs: Additional backend parameters
        
    Returns:
        Configured Qiskit quantum backend
    """
    backend = QiskitQuantumBackend(backend_name=backend_name, **kwargs)
    
    if enable_noise and backend_name == "aer_simulator":
        # Add realistic noise model
        noise_model = backend.create_hardware_noise_model()
        backend.noise_model = noise_model
    
    return backend
