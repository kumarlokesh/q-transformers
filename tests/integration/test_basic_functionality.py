#!/usr/bin/env python3
"""
Basic functionality test for Q-Transformers
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent
sys.path.insert(0, str(_project_root / "python"))


def test_basic_imports():
    """Test if basic imports work."""
    print("Testing basic imports...")

    try:
        import torch

        print("✅ PyTorch {torch.__version__} imported")
    except ImportError as _e:
        print("❌ PyTorch import failed: {e}")
        return False

    try:
        # Test core attention import
        from qtransformers.attention import QuantumAttentionLayer

        print("✅ QuantumAttentionLayer imported")
    except ImportError as _e:
        print("❌ QuantumAttentionLayer import failed: {e}")
        return False

    try:
        # Test quantum simulator import
        from qsim.quantum_simulator import QuantumAttentionSimulator

        print("✅ QuantumSimulator imported")
    except ImportError as _e:
        print("❌ QuantumSimulator import failed: {e}")
        return False

    return True


def test_quantum_attention_basic():
    """Test basic quantum attention functionality."""
    print("\nTesting basic quantum attention...")

    try:
        import torch

        from qtransformers.attention import QuantumAttentionLayer

        _attention = QuantumAttentionLayer(embed_dim=64, _num_heads=8)

        # Test forward pass
        batch_size, seq_len, _embed_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, embed_dim)

        # QuantumAttentionLayer expects query, key, value tensors
        output, _attn_weights = attention(x, x, x)

        print("✅ Forward pass successful")
        print("   Input shape: {x.shape}")
        print("   Output shape: {output.shape}")
        print("   Output finite: {torch.isfinite(output).all()}")

        return True

    except Exception as _e:
        print("❌ Quantum attention test failed: {e}")
        return False


def test_quantum_simulator():
    """Test basic quantum simulator."""
    print("\nTesting quantum simulator...")

    try:
        import torch

        from qsim.quantum_simulator import QuantumAttentionSimulator

        _simulator = QuantumAttentionSimulator(device="cpu")

        # Test basic state preparation - need 3D tensors (batch_size, seq_len, d_model)
        # Small tensors: batch=2, seq_len=4, d_model=8
        _query = torch.randn(2, 4, 8)
        _key = torch.randn(2, 4, 8)
        _value = torch.randn(2, 4, 8)

        result, _attn_weights = simulator.simulate_attention(
            query, key, value, _num_samples=16
        )

        print("✅ Quantum simulation successful")
        print("   Query shape: {query.shape}")
        print("   Result shape: {result.shape}")
        print("   Result finite: {torch.isfinite(result).all()}")

        return True

    except Exception as _e:
        print("❌ Quantum simulator test failed: {e}")
        return False


def main():
    """Run basic functionality tests."""
    print("Q-Transformers Basic Functionality Test")
    print("=" * 50)

    _imports_ok = test_basic_imports()

    if not imports_ok:
        print("\n❌ Basic imports failed. Cannot proceed with functionality tests.")
        return False

    _attention_ok = test_quantum_attention_basic()
    _simulator_ok = test_quantum_simulator()

    print("\n📊 Test Summary:")
    print("   Imports: {'✅' if imports_ok else '❌'}")
    print("   Quantum Attention: {'✅' if attention_ok else '❌'}")
    print("   Quantum Simulator: {'✅' if simulator_ok else '❌'}")

    if imports_ok and attention_ok and simulator_ok:
        print("\n🎉 Basic functionality verified!")
        return True
    else:
        print("\n⚠️  Some tests failed. Code needs debugging.")
        return False


if __name__ == "__main__":
    _success = main()
    sys.exit(0 if success else 1)
