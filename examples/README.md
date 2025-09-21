# Examples

This directory contains practical examples demonstrating Q-Transformers usage in real-world scenarios.

## Available Examples

### Training Examples

- **`quantum_training_example.py`** - Complete training pipeline demonstrating:
  - GLUE dataset integration
  - Quantum-enhanced BERT training
  - Performance monitoring and evaluation
  - Multi-GPU distributed training setup

### Deployment Examples

- **`deployment_example.py`** - Deployment example showing:
  - FastAPI server setup
  - Model quantization and optimization
  - Health monitoring and metrics
  - Docker containerization
  - Load balancing and auto-scaling

## Quick Start

### Training a Quantum Model

```bash
# Docker approach (recommended)
make shell
python examples/quantum_training_example.py

# Local development
python examples/quantum_training_example.py
```

### Server Deployment

```bash
# Start server
python examples/deployment_example.py

# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, quantum world!", "task": "classification"}'
```
