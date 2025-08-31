"""
Tests for quantum simulation layer (qsim).
"""

import pytest
import torch
import numpy as np
from qsim import QuantumAttentionSimulator, amplitude_encode, quantum_measure


class TestQuantumAttentionSimulator:
    """Test suite for QuantumAttentionSimulator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.simulator = QuantumAttentionSimulator(device="cpu")
        
    def test_initialization(self):
        """Test simulator initialization."""
        sim = QuantumAttentionSimulator(device="cpu")
        assert sim.device == "cpu"
        
    def test_simulate_attention_shapes(self):
        """Test that simulate_attention returns correct shapes."""
        batch_size, seq_len, d_model = 2, 4, 8
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = self.simulator.simulate_attention(Q, K, V, num_samples=16)
        
        assert output.shape == V.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
        
    def test_amplitude_encoding(self):
        """Test amplitude encoding produces normalized amplitudes."""
        Q = torch.randn(1, 3, 4)
        K = torch.randn(1, 3, 4)
        
        amplitudes = self.simulator._encode_amplitudes(Q, K)
        
        # Check normalization (amplitudes should have unit norm)
        norms = torch.norm(amplitudes, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0.0, atol=1e-6)
        
    def test_quantum_measurement_probabilistic(self):
        """Test quantum measurement produces valid probabilities."""
        amplitudes = torch.tensor([[0.6, 0.8, 0.0]])  # Simple amplitude state
        
        probs = self.simulator._quantum_measure(amplitudes, num_samples=100, noise_level=0.0)
        
        # Probabilities should sum to 1
        prob_sums = torch.sum(probs, dim=-1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), rtol=0.0, atol=1e-6)
        
        # Should be non-negative
        assert torch.all(probs >= 0)
        
    def test_noise_effect(self):
        """Test that noise affects the measurement."""
        amplitudes = torch.ones(1, 4) / 2  # Uniform amplitudes
        
        # Measure without noise
        probs_clean = self.simulator._quantum_measure(amplitudes, num_samples=1000, noise_level=0.0)
        
        # Measure with noise  
        probs_noisy = self.simulator._quantum_measure(amplitudes, num_samples=1000, noise_level=0.1)
        
        # Results should be different (with high probability)
        assert not torch.allclose(probs_clean, probs_noisy, atol=0.05)


class TestAmplitudeEncode:
    """Test suite for amplitude_encode function."""
    
    def test_single_query_encoding(self):
        """Test encoding single query with multiple keys."""
        query = torch.randn(4)  # d_model = 4
        keys = torch.randn(3, 4)  # 3 keys, d_model = 4
        
        amplitudes = amplitude_encode(query, keys)
        
        assert amplitudes.shape == (3,)  # Should have 3 amplitudes
        
        # Check normalization
        norm = torch.norm(amplitudes)
        torch.testing.assert_close(norm, torch.tensor(1.0), rtol=0.0, atol=1e-6)
        
    def test_encoding_properties(self):
        """Test mathematical properties of amplitude encoding."""
        query = torch.tensor([1.0, 0.0])
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        
        amplitudes = amplitude_encode(query, keys)
        
        # Higher inner product should lead to higher amplitude
        assert amplitudes[0] > amplitudes[1]  # [1,0]·[1,0] > [1,0]·[0,1]


class TestQuantumMeasure:
    """Test suite for quantum_measure function."""
    
    def test_exact_measurement(self):
        """Test measurement without sampling (num_samples=0)."""
        amplitudes = torch.tensor([0.6, 0.8])
        
        probs = quantum_measure(amplitudes, num_samples=0)
        expected_probs = amplitudes ** 2
        
        torch.testing.assert_close(probs, expected_probs, rtol=0.0, atol=1e-6)
        
    def test_sampled_measurement(self):
        """Test measurement with sampling."""
        amplitudes = torch.tensor([0.0, 1.0, 0.0])  # Pure state |1⟩
        
        probs = quantum_measure(amplitudes, num_samples=100)
        
        # Should concentrate probability on state |1⟩
        assert probs[1] > 0.8  # Most probability should be on index 1
        assert torch.sum(probs) <= 1.0 + 1e-6  # Probabilities sum to ~1
        
    def test_measurement_with_noise(self):
        """Test measurement with noise."""
        amplitudes = torch.tensor([1.0, 0.0])  # Pure state |0⟩
        
        probs_clean = quantum_measure(amplitudes, num_samples=0, noise_level=0.0)
        probs_noisy = quantum_measure(amplitudes, num_samples=0, noise_level=0.2)
        
        # Noise should spread probability
        assert probs_noisy[1] > probs_clean[1]


@pytest.mark.parametrize("seq_len,d_model", [
    (4, 8),
    (8, 16),
    (16, 32),
])
def test_simulator_scaling(seq_len, d_model):
    """Test simulator with different sequence lengths and model dimensions."""
    simulator = QuantumAttentionSimulator()
    
    Q = torch.randn(1, seq_len, d_model)
    K = torch.randn(1, seq_len, d_model)
    V = torch.randn(1, seq_len, d_model)
    
    output, attn_weights = simulator.simulate_attention(Q, K, V, num_samples=32)
    
    assert output.shape == (1, seq_len, d_model)
    assert attn_weights.shape == (1, seq_len, seq_len)


def test_consistency_with_classical_attention():
    """Test that quantum simulation approximates classical attention."""
    # This is a placeholder test - will be expanded in Phase 1
    simulator = QuantumAttentionSimulator()
    
    # Small test case
    Q = torch.randn(1, 3, 4)
    K = torch.randn(1, 3, 4)
    V = torch.randn(1, 3, 4)
    
    # Quantum simulation (with many samples should approximate classical)
    quantum_output, _ = simulator.simulate_attention(Q, K, V, num_samples=1000)
    
    # Classical attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    classical_weights = torch.softmax(scores, dim=-1)
    classical_output = torch.matmul(classical_weights, V)
    
    # They should be approximately equal (within sampling error)
    # This test will be refined as we develop better quantum simulation
    assert quantum_output.shape == classical_output.shape
