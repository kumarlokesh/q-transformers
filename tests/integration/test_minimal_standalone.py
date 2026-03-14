#!/usr/bin/env python3
"""
Minimal standalone tests for Q-Transformers package structure.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_module_structure():
    """Test that required package directories and init files exist."""
    python_dir = PROJECT_ROOT / "python"
    qtransformers_dir = python_dir / "qtransformers"
    qsim_dir = python_dir / "qsim"

    assert qtransformers_dir.exists(), "qtransformers package directory missing"
    assert qsim_dir.exists(), "qsim package directory missing"
    assert (qtransformers_dir / "__init__.py").exists(), "qtransformers/__init__.py missing"
    assert (qsim_dir / "__init__.py").exists(), "qsim/__init__.py missing"


def test_key_source_files_exist():
    """Test that key source files are present."""
    python_dir = PROJECT_ROOT / "python"
    expected_files = [
        "qtransformers/__init__.py",
        "qtransformers/attention.py",
        "qtransformers/advanced_sampling.py",
        "qsim/__init__.py",
        "qsim/quantum_simulator.py",
    ]
    for rel_path in expected_files:
        full_path = python_dir / rel_path
        assert full_path.exists(), f"Missing file: {rel_path}"


def test_import_syntax():
    """Test that key Python files have valid syntax."""
    python_dir = PROJECT_ROOT / "python"
    files_to_check = [
        "qtransformers/__init__.py",
        "qtransformers/attention.py",
        "qsim/__init__.py",
        "qsim/quantum_simulator.py",
    ]
    for rel_path in files_to_check:
        full_path = python_dir / rel_path
        source = full_path.read_text()
        # compile() raises SyntaxError on bad syntax
        compile(source, str(full_path), "exec")


def test_version_attribute():
    """Test that the package exposes a __version__ attribute."""
    import qtransformers

    assert hasattr(qtransformers, "__version__")
    assert isinstance(qtransformers.__version__, str)
