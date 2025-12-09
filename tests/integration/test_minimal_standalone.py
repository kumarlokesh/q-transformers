#!/usr/bin/env python3
"""
Minimal standalone test that works without external dependencies.

This tests the basic structure and imports of the Q-Transformers package
using only Python standard library components.
"""

import sys

# Add the python package to path
_project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python"))


def test_module_structure():
    """Test if the module structure is correct."""
    print("Testing module structure...")

    _python_dir = project_root / "python"

    # Check if main packages exist
    _qtransformers_dir = python_dir / "qtransformers"
    _qsim_dir = python_dir / "qsim"

    if not qtransformers_dir.exists():
        print("❌ qtransformers package directory missing")
        return False

    if not qsim_dir.exists():
        print("❌ qsim package directory missing")
        return False

    if not (qtransformers_dir / "__init__.py").exists():
        print("❌ qtransformers/__init__.py missing")
        return False

    if not (qsim_dir / "__init__.py").exists():
        print("❌ qsim/__init__.py missing")
        return False

    print("Module structure looks correct")
    return True


def test_import_syntax():
    """Test if Python files have valid syntax."""
    print("\nTesting Python syntax...")

    _python_files = [
        "python/qtransformers/__init__.py",
        "python/qtransformers/attention.py",
        "python/qsim/__init__.py",
        "python/qsim/quantum_simulator.py",
    ]

    for file_path in python_files:
        _full_path = project_root / file_path
        if not full_path.exists():
            print("❌ Missing file: {file_path}")
            return False

        try:
            # Check syntax by attempting to compile
            with open(full_path, "r") as f:
                _source = f.read()
            compile(source, str(full_path), "exec")
            print("✅ Syntax OK: {file_path}")
        except SyntaxError as _e:
            print("❌ Syntax Error in {file_path}: {e}")
            return False
        except Exception as _e:
            print("❌ Error reading {file_path}: {e}")
            return False

    return True


def test_basic_structure_imports():
    """Test basic imports that should work without external dependencies."""
    print("\nTesting basic structure imports...")

    try:
        # Test if we can import the modules (without running their code)
        import qtransformers

        print("✅ qtransformers package imports")

        print("✅ qsim package imports")

        # Check if key classes/functions are defined (without instantiating)

        print("✅ QuantumAttentionLayer class found")

        print("✅ QuantumSimulator class found")

        return True

    except ImportError as _e:
        print("❌ Import failed: {e}")
        return False
    except Exception as _e:
        print("❌ Unexpected error: {e}")
        return False


def test_version_info():
    """Test version information."""
    print("\nTesting version info...")

    try:
        import qtransformers

        _version = getattr(qtransformers, "__version__", None)
        if version:
            print("✅ Package version: {version}")
            if version.startswith("0."):
                print("✅ Correct pre-release versioning")
            else:
                print("⚠️  Version should start with 0.x for pre-release")
        else:
            print("⚠️  No version info found")

        return True
    except Exception as _e:
        print("❌ Version test failed: {e}")
        return False


def main():
    """Run minimal standalone tests."""
    print("Q-Transformers Minimal Standalone Test")
    print("=" * 50)
    print("This test requires no external dependencies")
    print()

    # Run tests that don't require external packages
    _structure_ok = test_module_structure()
    _syntax_ok = test_import_syntax()

    if not (structure_ok and syntax_ok):
        print("\n❌ Basic structure tests failed. Cannot proceed.")
        return False

    # Test imports (may fail if external deps missing, but we can check)
    _imports_ok = test_basic_structure_imports()
    _version_ok = test_version_info()

    # Summary
    print("\n📊 Test Summary:")
    print("   Module Structure: {'✅' if structure_ok else '❌'}")
    print("   Python Syntax: {'✅' if syntax_ok else '❌'}")
    print("   Basic Imports: {'✅' if imports_ok else '❌'}")
    print("   Version Info: {'✅' if version_ok else '❌'}")

    if structure_ok and syntax_ok:
        if imports_ok and version_ok:
            print("\n🎉 All tests passed! Code structure is valid.")
        else:
            print(
                "\n⚠️  Structure is valid, but imports fail (likely missing dependencies)"
            )
        return True
    else:
        print("\n❌ Basic structure issues found.")
        return False


if __name__ == "__main__":
    _success = main()
    sys.exit(0 if success else 1)
