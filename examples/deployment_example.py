#!/usr/bin/env python3
"""
Example: Deploying Quantum Transformers

This script demonstrates how to deploy trained quantum transformers
using the production-ready deployment infrastructure.
"""

import asyncio

import requests

from qtransformers.deployment import (
    DeploymentConfig,
    QuantumModelServer,
    create_app,
    run_server,
)


def create_demo_deployment():
    """Create a demo deployment configuration."""

    print("🚀 Quantum Transformer Production Deployment Example")
    print("=" * 60)

    # Deployment configuration
    _config = DeploymentConfig(
        _model_name="quantum-bert-demo",
        _model_version="3.0.0",
        _model_path="./demo_checkpoints/best",  # Path to trained model
        _host="0.0.0.0",
        _port=8000,
        _workers=1,
        # Performance settings
        _max_batch_size=16,
        _max_sequence_length=512,
        _enable_batching=True,
        _batch_timeout_ms=50,
        # Optimization settings
        _enable_quantization=True,
        _quantization_bits=8,
        # Monitoring
        _enable_metrics=True,
        _log_level="INFO",
        # Security (disabled for demo)
        _enable_cors=True,
        _api_key_required=False,
    )

    print("Deployment configuration:")
    print("  Model: {config.model_name} v{config.model_version}")
    print("  Host: {config.host}:{config.port}")
    print("  Max batch size: {config.max_batch_size}")
    print("  Quantization: {config.enable_quantization}")
    print("  Metrics: {config.enable_metrics}")

    return config


def run_deployment_server(config: DeploymentConfig):
    """Run the deployment server."""

    print("\n🌐 Starting quantum model server...")
    print("Server will be available at: http://{config.host}:{config.port}")
    print("API documentation: http://{config.host}:{config.port}/docs")
    print("Health check: http://{config.host}:{config.port}/health")
    print("Metrics: http://{config.host}:{config.port}/metrics")

    # Start server (this would run indefinitely in production)
    try:
        run_server(config)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as _e:
        print("\n❌ Server error: {e}")


async def test_deployment_api():
    """Test the deployed quantum transformer API."""

    _base_url = "http://localhost:8000"

    print("\n🧪 Testing deployment API at {base_url}")

    print("\n1. Health Check")
    try:
        _response = requests.get("{base_url}/health", _timeout=5)
        if response.status_code == 200:
            _health_data = response.json()
            print("✅ Health check passed")
            print("   Status: {health_data['status']}")
            print("   Model loaded: {health_data['model_loaded']}")
            print("   GPU available: {health_data['gpu_available']}")
            print("   Uptime: {health_data['uptime']:.2f}s")
        else:
            print("❌ Health check failed: {response.status_code}")
    except Exception as _e:
        print("❌ Health check error: {e}")
        return False

    print("\n2. Single Text Prediction")
    try:
        _test_input = {
            "text": "This is a test sentence for quantum attention processing.",
            "max_length": 128,
            "temperature": 1.0,
        }

        _response = requests.post("{base_url}/predict", _json=test_input, _timeout=30)

        if response.status_code == 200:
            _result = response.json()
            print("✅ Single prediction successful")
            print("   Processing time: {result['processing_time']:.3f}s")
            print("   Quantum metrics: {result['quantum_metrics']}")
            print("   Model version: {result['model_version']}")
        else:
            print("❌ Single prediction failed: {response.status_code}")
            print("   Error: {response.text}")
    except Exception as _e:
        print("❌ Single prediction error: {e}")

    print("\n3. Batch Text Prediction")
    try:
        _batch_input = {
            "texts": [
                "Quantum computing enables new possibilities for AI.",
                "Natural language processing benefits from quantum attention.",
                "This is a demonstration of quantum transformer capabilities.",
            ],
            "max_length": 128,
            "batch_size": 3,
        }

        _response = requests.post(
            "{base_url}/predict/batch", _json=batch_input, _timeout=30
        )

        if response.status_code == 200:
            _results = response.json()
            print("✅ Batch prediction successful")
            print("   Processed {len(results)} texts")
            for i, result in enumerate(results):
                print("   Text {i+1}: {len(result['embeddings'])} features")
        else:
            print("❌ Batch prediction failed: {response.status_code}")
            print("   Error: {response.text}")
    except Exception as _e:
        print("❌ Batch prediction error: {e}")

    print("\n4. Model Information")
    try:
        _response = requests.get("{base_url}/model/info", _timeout=5)
        if response.status_code == 200:
            _model_info = response.json()
            print("✅ Model info retrieved")
            print("   Model name: {model_info['model_name']}")
            print("   Version: {model_info['model_version']}")
            print("   Device: {model_info['device']}")
            print("   Quantization: {model_info['quantization_enabled']}")
        else:
            print("❌ Model info failed: {response.status_code}")
    except Exception as _e:
        print("❌ Model info error: {e}")

    print("\n5. Metrics Collection")
    try:
        _response = requests.get("{base_url}/metrics", _timeout=5)
        if response.status_code == 200:
            _metrics_data = response.text
            print("✅ Metrics retrieved")
            print("   Metrics format: Prometheus")
            print("   Data size: {len(metrics_data)} bytes")
        else:
            print("❌ Metrics failed: {response.status_code}")
    except Exception as _e:
        print("❌ Metrics error: {e}")

    print("\n🎉 API testing completed!")
    return True


def create_docker_deployment():
    """Create Docker deployment configuration."""

    print("\n🐳 Docker Deployment Configuration")

    _dockerfile_content = """
# Quantum Transformers Production Deployment
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ curl git && \\
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install Q-Transformers
RUN pip install -e python/

# Create non-root user
RUN useradd -m -u 1001 qtransformer && \\
    chown -R qtransformer:qtransformer /app
USER qtransformer

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "-m", "qtransformers.deployment", \\
     "--model-path", "/app/models", \\
     "--host", "0.0.0.0", \\
     "--port", "8000"]
"""

    # Create requirements.txt
    _requirements_content = """
torch>=2.0.0
transformers>=4.20.0
datasets>=2.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
numpy>=1.21.0
scipy>=1.7.0
qiskit>=0.40.0
prometheus-client>=0.15.0
psutil>=5.9.0
"""

    # Create docker-compose.yml
    _docker_compose_content = """
version: '3.8'

services:
  quantum-transformers:
    build: .
    ports:
      - "8000:8000"
    environment:
      - _MODEL_PATH =/app/models
      - _LOG_LEVEL =INFO
      - _ENABLE_QUANTIZATION =true
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - _GF_SECURITY_ADMIN_PASSWORD =quantum123
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
"""

    print("Generated Docker deployment files:")
    print("  - Dockerfile")
    print("  - requirements.txt")
    print("  - docker-compose.yml")
    print("\nTo deploy:")
    print("  1. docker-compose up --build")
    print("  2. Access API: http://localhost:8000")
    print("  3. Access Grafana: http://localhost:3000 (admin/quantum123)")
    print("  4. Access Prometheus: http://localhost:9090")

    return dockerfile_content, requirements_content, docker_compose_content


def main():
    """Main deployment example."""

    # Create deployment config
    _config = create_demo_deployment()

    print("\nChoose deployment mode:")
    print("1. Start production server")
    print("2. Test API client")
    print("3. Generate Docker configuration")
    print("4. Show deployment guide")

    _choice = input("\nEnter choice (1-4): ").strip()

    if _choice == "1":
        print("\n🚀 Starting production server...")
        print("Note: This will start the server. Use Ctrl+C to stop.")
        input("Press Enter to continue or Ctrl+C to cancel...")
        run_deployment_server(config)

    elif _choice == "2":
        print("\n🧪 Testing API (make sure server is running separately)")
        asyncio.run(test_deployment_api())

    elif _choice == "3":
        dockerfile, requirements, _docker_compose = create_docker_deployment()

        # Write files
        Path("Dockerfile").write_text(dockerfile)
        Path("requirements.txt").write_text(requirements)
        Path("docker-compose.yml").write_text(docker_compose)

        print("\n✅ Docker files created successfully!")

    elif _choice == "4":
        print("\n📋 Quantum Transformer Deployment Guide")
        print("=" * 50)
        print(
            """
1. DEVELOPMENT DEPLOYMENT:
   python examples/deployment_example.py
   
2. DOCKER DEPLOYMENT:
   docker-compose up --build
   
3. KUBERNETES DEPLOYMENT:
   kubectl apply -f k8s/quantum-transformers.yaml
   
4. CLOUD DEPLOYMENT:
   - AWS: Use ECS/EKS with GPU instances
   - GCP: Use GKE with TPU/GPU nodes  
   - Azure: Use AKS with GPU node pools
   
5. MONITORING:
   - Health: /health endpoint
   - Metrics: /metrics (Prometheus format)
   - Logs: Structured JSON logging
   
6. SCALING:
   - Horizontal: Multiple replicas
   - Vertical: GPU memory and compute
   - Load balancing: nginx/istio
   
7. SECURITY:
   - API keys: Set _api_key_required =True
   - HTTPS: Use reverse proxy with TLS
   - Network: VPC/security groups
        """
        )

    else:
        print("❌ Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
