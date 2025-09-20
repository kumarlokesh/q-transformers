#!/usr/bin/env python3
"""
Basic functionality test for Q-Transformers

Tests the most fundamental components to verify they actually work.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python"))

def test_basic_imports():
    """Test if basic imports work."""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        # Test core attention import
        from qtransformers.attention import QuantumAttentionLayer
        print("✅ QuantumAttentionLayer imported")
    except ImportError as e:
        print(f"❌ QuantumAttentionLayer import failed: {e}")
        return False
    
    try:
        # Test quantum simulator import
        from qsim.quantum_simulator import QuantumAttentionSimulator
        print("✅ QuantumSimulator imported")
    except ImportError as e:
        print(f"❌ QuantumSimulator import failed: {e}")
        return False
    
    return True

def test_quantum_attention_basic():
    """Test basic quantum attention functionality."""
    print("\nTesting basic quantum attention...")
    
    try:
        import torch
        from qtransformers.attention import QuantumAttentionLayer

        attention = QuantumAttentionLayer(
            embed_dim=64,
            num_heads=8
        )
        
        # Test forward pass
        batch_size, seq_len, embed_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # QuantumAttentionLayer expects query, key, value tensors
        output, attn_weights = attention(x, x, x)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output finite: {torch.isfinite(output).all()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum attention test failed: {e}")
        return False

def test_quantum_simulator():
    """Test basic quantum simulator."""
    print("\nTesting quantum simulator...")
    
    try:
        from qsim.quantum_simulator import QuantumAttentionSimulator
        import torch
        
        simulator = QuantumAttentionSimulator(device="cpu")
        
        # Test basic state preparation - need 3D tensors (batch_size, seq_len, d_model)
        query = torch.randn(2, 4, 8)  # Small tensors: batch=2, seq_len=4, d_model=8
        key = torch.randn(2, 4, 8)
        value = torch.randn(2, 4, 8)
        
        result, attn_weights = simulator.simulate_attention(query, key, value, num_samples=16)
        
        print(f"✅ Quantum simulation successful")
        print(f"   Query shape: {query.shape}")
        print(f"   Result shape: {result.shape}")
        print(f"   Result finite: {torch.isfinite(result).all()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum simulator test failed: {e}")
        return False

def main():
    """Run basic functionality tests."""
    print("🧪 Q-Transformers Basic Functionality Test")
    print("=" * 50)
    
    imports_ok = test_basic_imports()
    
    if not imports_ok:
        print("\n❌ Basic imports failed. Cannot proceed with functionality tests.")
        return False
    
    attention_ok = test_quantum_attention_basic()
    simulator_ok = test_quantum_simulator()
    
    print(f"\n📊 Test Summary:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   Quantum Attention: {'✅' if attention_ok else '❌'}")
    print(f"   Quantum Simulator: {'✅' if simulator_ok else '❌'}")
    
    if imports_ok and attention_ok and simulator_ok:
        print(f"\n🎉 Basic functionality verified!")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Code needs debugging.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
