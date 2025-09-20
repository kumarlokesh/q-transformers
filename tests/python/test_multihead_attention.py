"""
Tests for QuantumMultiheadAttention module.
"""

import torch
import numpy as np
from qtransformers import QuantumMultiheadAttention


class TestQuantumMultiheadAttention:
    
    def test_initialization(self):
        """Test QuantumMultiheadAttention initialization."""
        embed_dim = 64
        num_heads = 8
        
        attn = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            quantum_backend="stratified",
            num_samples=32
        )
        
        assert attn.embed_dim == embed_dim
        assert attn.num_heads == num_heads
        assert attn.head_dim == embed_dim // num_heads
        assert len(attn.head_configs) == num_heads
        
    def test_forward_pass_basic(self):
        """Test basic forward pass with quantum attention."""
        batch_size, seq_len, embed_dim = 2, 16, 64
        num_heads = 8
        
        attn = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_samples=16,
            batch_first=False
        )
        
        # Input tensors (L, N, E) format
        query = torch.randn(seq_len, batch_size, embed_dim)
        key = torch.randn(seq_len, batch_size, embed_dim)  
        value = torch.randn(seq_len, batch_size, embed_dim)
        
        output, attn_weights = attn(query, key, value)
        
        assert output.shape == (seq_len, batch_size, embed_dim)
        
        if attn_weights is not None:
            assert attn_weights.shape == (batch_size, seq_len, seq_len)

        assert torch.isfinite(output).all()
        
    def test_batch_first_format(self):
        """Test batch_first=True format."""
        batch_size, seq_len, embed_dim = 2, 16, 64
        num_heads = 4
        
        attn = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_samples=16,
            batch_first=True
        )
        
        # Input tensors (N, L, E) format
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        output, attn_weights = attn(query, key, value)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        
    def test_different_quantum_backends(self):
        """Test different quantum sampling strategies."""
        batch_size, seq_len, embed_dim = 1, 8, 32
        num_heads = 4
        
        backends = ["stratified", "adaptive", "hybrid", "naive"]
        
        for backend in backends:
            attn = QuantumMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                quantum_backend=backend,
                num_samples=8
            )
            
            query = torch.randn(seq_len, batch_size, embed_dim)
            key = torch.randn(seq_len, batch_size, embed_dim)
            value = torch.randn(seq_len, batch_size, embed_dim)
            
            output, _ = attn(query, key, value, need_weights=False)
            
            assert output.shape == (seq_len, batch_size, embed_dim)
            assert torch.isfinite(output).all()
            
    def test_attention_weights_computation(self):
        """Test attention weights are properly computed and normalized."""
        batch_size, seq_len, embed_dim = 1, 4, 16
        num_heads = 2
        
        attn = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_samples=4
        )
        
        query = torch.randn(seq_len, batch_size, embed_dim)
        key = torch.randn(seq_len, batch_size, embed_dim)
        value = torch.randn(seq_len, batch_size, embed_dim)
        
        output, attn_weights = attn(query, key, value, need_weights=True, average_attn_weights=True)
        
        if attn_weights is not None:
            # Check weights are approximately normalized per row
            row_sums = torch.sum(attn_weights, dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)

            assert (attn_weights >= 0).all()
            
    def test_per_head_configurations(self):
        """Test that different heads use different configurations."""
        embed_dim = 32
        num_heads = 4
        
        attn = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_samples=16
        )
        
        configs = attn.head_configs
        
        # Check we have configs for all heads
        assert len(configs) == num_heads
        
        # Check configs have required keys
        for config in configs:
            assert "sampling_strategy" in config
            assert "num_samples" in config
            assert "control_variate" in config
            
        # Check some variation in configs (not all identical)
        strategies = [config["sampling_strategy"] for config in configs]
        assert len(set(strategies)) > 1  # Should have different strategies
        
    def test_dropout_application(self):
        """Test dropout is applied correctly."""
        batch_size, seq_len, embed_dim = 1, 8, 32
        num_heads = 2
        
        # Test with dropout
        attn_with_dropout = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.5,  # High dropout for testing
            num_samples=8
        )
        
        attn_with_dropout.train()  # Enable dropout
        
        query = torch.randn(seq_len, batch_size, embed_dim)
        key = torch.randn(seq_len, batch_size, embed_dim)
        value = torch.randn(seq_len, batch_size, embed_dim)
        
        output1, _ = attn_with_dropout(query, key, value)
        output2, _ = attn_with_dropout(query, key, value)
        
        # Outputs should be different due to dropout randomness
        assert not torch.allclose(output1, output2)
        
        # Test without dropout (eval mode)
        attn_with_dropout.eval()
        output3, _ = attn_with_dropout(query, key, value)
        output4, _ = attn_with_dropout(query, key, value)
        
        # Note: Even in eval mode, quantum sampling introduces randomness
        # So we just check outputs are finite and reasonable
        assert torch.isfinite(output3).all()
        assert torch.isfinite(output4).all()
        
    def test_gradient_flow(self):
        """Test gradients flow properly through quantum attention."""
        batch_size, seq_len, embed_dim = 1, 4, 16
        num_heads = 2
        
        attn = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_samples=4
        )
        
        query = torch.randn(seq_len, batch_size, embed_dim, requires_grad=True)
        key = torch.randn(seq_len, batch_size, embed_dim, requires_grad=True)
        value = torch.randn(seq_len, batch_size, embed_dim, requires_grad=True)
        
        output, _ = attn(query, key, value)
        loss = output.sum()
        
        loss.backward()
        
        # Check gradients exist and are finite
        assert query.grad is not None
        assert key.grad is not None  
        assert value.grad is not None
        
        assert torch.isfinite(query.grad).all()
        assert torch.isfinite(key.grad).all()
        assert torch.isfinite(value.grad).all()
        
    def test_compatibility_with_nn_multiheadattention(self):
        """Test interface compatibility with nn.MultiheadAttention."""
        batch_size, seq_len, embed_dim = 2, 8, 32
        num_heads = 4
        
        # Our quantum attention
        quantum_attn = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False
        )
        
        # Standard PyTorch attention
        standard_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False
        )
        
        query = torch.randn(seq_len, batch_size, embed_dim)
        key = torch.randn(seq_len, batch_size, embed_dim)
        value = torch.randn(seq_len, batch_size, embed_dim)
        
        # Both should accept same inputs and return compatible outputs
        quantum_output, quantum_weights = quantum_attn(query, key, value)
        standard_output, standard_weights = standard_attn(query, key, value)
        
        # Check output shapes match
        assert quantum_output.shape == standard_output.shape
        if quantum_weights is not None and standard_weights is not None:
            assert quantum_weights.shape == standard_weights.shape


def test_integration_with_transformer():
    """Integration test: Use QuantumMultiheadAttention in a simple transformer layer."""
    
    class SimpleTransformerBlock(torch.nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.attn = QuantumMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_samples=16
            )
            self.norm = torch.nn.LayerNorm(embed_dim)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim * 4, embed_dim)
            )
            
        def forward(self, x):
            # Self-attention with residual connection
            attn_out, _ = self.attn(x, x, x)
            x = self.norm(x + attn_out)
            
            # MLP with residual connection  
            mlp_out = self.mlp(x)
            x = self.norm(x + mlp_out)
            
            return x
    
    # Test the transformer block
    batch_size, seq_len, embed_dim = 2, 16, 64
    num_heads = 8
    
    transformer_block = SimpleTransformerBlock(embed_dim, num_heads)
    
    x = torch.randn(seq_len, batch_size, embed_dim)
    output = transformer_block(x)
    
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    
    # Test gradient flow through the whole block
    loss = output.sum()
    loss.backward()
    
    # Check some parameters have gradients
    for param in transformer_block.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


if __name__ == "__main__":
    # Run basic tests
    test_suite = TestQuantumMultiheadAttention()
    
    print("Testing QuantumMultiheadAttention...")
    
    test_suite.test_initialization()
    print("✓ Initialization test passed")
    
    test_suite.test_forward_pass_basic()
    print("✓ Basic forward pass test passed")
    
    test_suite.test_batch_first_format()
    print("✓ Batch first format test passed")
    
    test_suite.test_different_quantum_backends()
    print("✓ Different quantum backends test passed")
    
    test_suite.test_attention_weights_computation()
    print("✓ Attention weights computation test passed")
    
    test_suite.test_per_head_configurations()
    print("✓ Per-head configurations test passed")
    
    test_suite.test_dropout_application()
    print("✓ Dropout application test passed")
    
    test_suite.test_gradient_flow()
    print("✓ Gradient flow test passed")
    
    test_suite.test_compatibility_with_nn_multiheadattention()
    print("✓ Compatibility with nn.MultiheadAttention test passed")
    
    test_integration_with_transformer()
    print("✓ Integration with transformer test passed")
    
    print("\nAll tests passed! QuantumMultiheadAttention is working correctly.")
