# Variables
DOCKER_TAG := qtransformers-dev:latest
DOCKER_RUN := docker run --rm -v "$(PWD)":/workspace -w /workspace
DOCKER_RUN_IT := docker run --rm -it -v "$(PWD)":/workspace -w /workspace

# Help target (default)
.PHONY: help
help:
	@echo "Q-Transformers Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup & Build:"
	@echo "  make build          Build Docker development image"
	@echo "  make build-force    Force rebuild Docker image (no cache)"
	@echo "  make clean          Clean Docker images and artifacts"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests (Python + Rust + Integration)"
	@echo "  make test-python    Run Python unit tests only"
	@echo "  make test-rust      Run Rust unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  make install        Install packages in development mode"
	@echo "  make format         Format code (Python: black, isort; Rust: rustfmt)"
	@echo "  make lint           Run linters (Python: flake8, mypy; Rust: clippy)"
	@echo "  make check          Run format + lint + test (CI pipeline)"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench-quick    Run quick benchmarks"
	@echo "  make bench-sampling Run sampling benchmarks"
	@echo "  make bench-full     Run full evaluation suite"
	@echo ""
	@echo "Interactive:"
	@echo "  make shell          Open interactive shell in container"
	@echo "  make jupyter        Start Jupyter notebook server"

# Build targets
.PHONY: build
build:
	docker build -f Dockerfile.dev -t $(DOCKER_TAG) .

.PHONY: docker-build
docker-build: build
	@echo "Docker image built as $(DOCKER_TAG)"

.PHONY: build-force
build-force:
	docker build --no-cache -f Dockerfile.dev -t $(DOCKER_TAG) .

.PHONY: clean
clean:
	docker image prune -f
	docker system prune -f
	rm -rf .pytest_cache __pycache__ .coverage htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Installation (packages are built into Docker image)
.PHONY: install
install:
	@echo "Packages are built into Docker image"
	@echo "Run 'make build' to build the image with all dependencies"

# Testing targets (use existing image if available)
.PHONY: test
test: 
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "\
		echo 'Running Python tests...' && \
		python -m pytest tests/python/ -v && \
		echo 'Running Rust tests...' && \
		cd rust-core && cargo test --quiet && cd .. && \
		echo 'Running integration tests...' && \
		python -m pytest tests/integration/ -v"

.PHONY: test-python
test-python:
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) python -m pytest tests/python/ -v

.PHONY: test-rust
test-rust:
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "cd rust-core && cargo test"

.PHONY: test-integration
test-integration:
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) python -m pytest tests/integration/ -v

.PHONY: test-coverage
test-coverage:
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "\
		coverage run --source=python/qtransformers -m pytest tests/python/ && \
		coverage report -m && \
		coverage html"

# Code quality (use existing image if available)
.PHONY: format
format: 
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "\
		black python/qtransformers/ tests/ benchmarks/ examples/ && \
		isort python/qtransformers/ tests/ benchmarks/ examples/ && \
		cd rust-core && cargo fmt"

.PHONY: lint
lint:
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "\
		flake8 python/qtransformers/ tests/ benchmarks/ examples/ --max-line-length=88 --extend-ignore=E203,W503 && \
		mypy python/qtransformers/ --ignore-missing-imports && \
		cd rust-core && cargo clippy -- -D warnings"


.PHONY: flake8
flake8:
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "flake8 python/qtransformers/ tests/ benchmarks/ examples/ --max-line-length=88 --extend-ignore=E203,W503"

.PHONY: flake8-ci
flake8-ci:
	@if ! docker image inspect $(DOCKER_TAG) >/dev/null 2>&1; then \
		echo "Building Docker image..."; \
		$(MAKE) build; \
	fi
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "flake8 python/qtransformers/ tests/ benchmarks/ examples/ --max-line-length=88 --extend-ignore=E203,W503 --show-source --statistics"

.PHONY: check
check: format lint test

# Benchmarks
.PHONY: bench-quick
bench-quick: build
	$(DOCKER_RUN) $(DOCKER_TAG) python benchmarks/run_quick_benchmarks.py --batch 2 --seq 32 --dim 64 --samples 16

.PHONY: bench-sampling
bench-sampling: build
	$(DOCKER_RUN) $(DOCKER_TAG) python benchmarks/run_sampling_benchmarks.py --batch 2 --seq 64 --dim 128 --samples 32

.PHONY: bench-full
bench-full: build
	$(DOCKER_RUN) $(DOCKER_TAG) python benchmarks/run_full_evaluation_suite.py

# Interactive development
.PHONY: shell
shell: build
	$(DOCKER_RUN_IT) $(DOCKER_TAG) bash

.PHONY: docker-shell
docker-shell: shell
	@echo "Opened shell in $(DOCKER_TAG)"

.PHONY: jupyter
jupyter: build
	$(DOCKER_RUN_IT) -p 8888:8888 $(DOCKER_TAG) jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Debugging
.PHONY: debug-env
debug-env: build
	$(DOCKER_RUN) $(DOCKER_TAG) bash -c "\
		echo 'Python version:' && python --version && \
		echo 'Rust version:' && rustc --version && \
		echo 'Cargo version:' && cargo --version && \
		echo 'Maturin version:' && maturin --version"

# CI/CD helpers
.PHONY: ci-test
ci-test: build test

.PHONY: ci-full
ci-full: build check bench-quick
