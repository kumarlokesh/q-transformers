#!/usr/bin/env python3
"""
Example: Deploying Quantum Transformers in Production

This script demonstrates how to deploy trained quantum transformers
using the production-ready deployment infrastructure.
"""

import asyncio
import requests
import json
from pathlib import Path

from qtransformers.deployment import (
    DeploymentConfig,
    QuantumModelServer,
    create_app,
    run_server
)


def create_demo_deployment():
    """Create a demo deployment configuration."""
    
    print("🚀 Quantum Transformer Production Deployment Example")
    print("=" * 60)
    
    # Deployment configuration
    config = DeploymentConfig(
        model_name="quantum-bert-demo",
        model_version="3.0.0",
        model_path="./demo_checkpoints/best",  # Path to trained model
        host="0.0.0.0",
        port=8000,
        workers=1,
        
        # Performance settings
        max_batch_size=16,
        max_sequence_length=512,
        enable_batching=True,
        batch_timeout_ms=50,
        
        # Optimization settings
        enable_quantization=True,
        quantization_bits=8,
        
        # Monitoring
        enable_metrics=True,
        log_level="INFO",
        
        # Security (disabled for demo)
        enable_cors=True,
        api_key_required=False
    )
    
    print(f"Deployment configuration:")
    print(f"  Model: {config.model_name} v{config.model_version}")
    print(f"  Host: {config.host}:{config.port}")
    print(f"  Max batch size: {config.max_batch_size}")
    print(f"  Quantization: {config.enable_quantization}")
    print(f"  Metrics: {config.enable_metrics}")
    
    return config


def run_deployment_server(config: DeploymentConfig):
    """Run the deployment server."""
    
    print(f"\n🌐 Starting quantum model server...")
    print(f"Server will be available at: http://{config.host}:{config.port}")
    print(f"API documentation: http://{config.host}:{config.port}/docs")
    print(f"Health check: http://{config.host}:{config.port}/health")
    print(f"Metrics: http://{config.host}:{config.port}/metrics")
    
    # Start server (this would run indefinitely in production)
    try:
        run_server(config)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")


async def test_deployment_api():
    """Test the deployed quantum transformer API."""
    
    base_url = "http://localhost:8000"
    
    print(f"\n🧪 Testing deployment API at {base_url}")
    
    # Test health check
    print("\n1. Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   GPU available: {health_data['gpu_available']}")
            print(f"   Uptime: {health_data['uptime']:.2f}s")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test single prediction
    print("\n2. Single Text Prediction")
    try:
        test_input = {
            "text": "This is a test sentence for quantum attention processing.",
            "max_length": 128,
            "temperature": 1.0
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_input,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single prediction successful")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Quantum metrics: {result['quantum_metrics']}")
            print(f"   Model version: {result['model_version']}")
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
    
    # Test batch prediction
    print("\n3. Batch Text Prediction")
    try:
        batch_input = {
            "texts": [
                "Quantum computing enables new possibilities for AI.",
                "Natural language processing benefits from quantum attention.",
                "This is a demonstration of quantum transformer capabilities."
            ],
            "max_length": 128,
            "batch_size": 3
        }
        
        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch_input,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Batch prediction successful")
            print(f"   Processed {len(results)} texts")
            for i, result in enumerate(results):
                print(f"   Text {i+1}: {len(result['embeddings'])} features")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
    
    # Test model info
    print("\n4. Model Information")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model name: {model_info['model_name']}")
            print(f"   Version: {model_info['model_version']}")
            print(f"   Device: {model_info['device']}")
            print(f"   Quantization: {model_info['quantization_enabled']}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    # Test metrics endpoint
    print("\n5. Metrics Collection")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 200:
            metrics_data = response.text
            print(f"✅ Metrics retrieved")
            print(f"   Metrics format: Prometheus")
            print(f"   Data size: {len(metrics_data)} bytes")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")
    
    print(f"\n🎉 API testing completed!")
    return True


def create_docker_deployment():
    """Create Docker deployment configuration."""
    
    print(f"\n🐳 Docker Deployment Configuration")
    
    # Create Dockerfile
    dockerfile_content = """
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
    requirements_content = """
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
    docker_compose_content = """
version: '3.8'

services:
  quantum-transformers:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
      - ENABLE_QUANTIZATION=true
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
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
      - GF_SECURITY_ADMIN_PASSWORD=quantum123
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
    config = create_demo_deployment()
    
    print(f"\nChoose deployment mode:")
    print(f"1. Start production server")
    print(f"2. Test API client")
    print(f"3. Generate Docker configuration")
    print(f"4. Show deployment guide")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print(f"\n🚀 Starting production server...")
        print(f"Note: This will start the server. Use Ctrl+C to stop.")
        input("Press Enter to continue or Ctrl+C to cancel...")
        run_deployment_server(config)
        
    elif choice == "2":
        print(f"\n🧪 Testing API (make sure server is running separately)")
        asyncio.run(test_deployment_api())
        
    elif choice == "3":
        dockerfile, requirements, docker_compose = create_docker_deployment()
        
        # Write files
        Path("Dockerfile").write_text(dockerfile)
        Path("requirements.txt").write_text(requirements)  
        Path("docker-compose.yml").write_text(docker_compose)
        
        print(f"\n✅ Docker files created successfully!")
        
    elif choice == "4":
        print(f"\n📋 Quantum Transformer Deployment Guide")
        print(f"=" * 50)
        print(f"""
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
   - API keys: Set api_key_required=True
   - HTTPS: Use reverse proxy with TLS
   - Network: VPC/security groups
        """)
    
    else:
        print(f"❌ Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
