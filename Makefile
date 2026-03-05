.PHONY: help install run run-v2 up down up-v2 down-v2 test test-v2 clean build build-v2

# Default target
help:
	@echo "╔═══════════════════════════════════════════════════════╗"
	@echo "║        SIMPLE OCR API - Makefile Commands            ║"
	@echo "╚═══════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 SIMPLE VERSION (Recommended):"
	@echo "  make simple       - Run simple OCR API locally"
	@echo "  make simple-build - Build simple Docker image"
	@echo "  make simple-up    - Start simple Docker container"
	@echo "  make simple-down  - Stop simple container"
	@echo "  make simple-test  - Test simple API"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make install      - Install dependencies"
	@echo ""
	@echo "🚀 Run Locally (Legacy v1):"
	@echo "  make run          - Start Legacy API (port 8000)"
	@echo "  make test         - Test Legacy pipeline"
	@echo ""
	@echo "🔮 Run Locally (Modern v2):"
	@echo "  make run-v2       - Start Modern API (port 8000)"
	@echo "  make test-v2      - Test Modern pipeline"
	@echo ""
	@echo "🐳 Docker (Legacy v1):"
	@echo "  make build        - Build Docker image"
	@echo "  make up           - Start containers (Legacy)"
	@echo "  make down         - Stop containers"
	@echo "  make logs         - Show container logs"
	@echo "  make test-docker  - Build, start and test"
	@echo ""
	@echo "🐳 Docker (Modern v2):"
	@echo "  make build-v2     - Build Docker image (Modern)"
	@echo "  make up-v2        - Start containers (Modern)"
	@echo "  make down-v2      - Stop containers"
	@echo "  make logs-v2      - Show container logs"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test-api     - Test Legacy API endpoints"
	@echo "  make test-api-v2  - Test Modern API endpoints"
	@echo "  make test-all     - Run all tests"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean        - Remove cache files"
	@echo "  make clean-all    - Full cleanup (cache + models)"
	@echo "  make clean-docker - Remove Docker images"
	@echo ""

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

# Legacy API (v1)
run:
	@echo "🚀 Starting Legacy API (v1) on port 8000..."
	@echo "📍 Endpoint: POST http://localhost:8000/analyze-flyer"
	python api.py

# Modern API (v2)
run-v2:
	@echo "🔮 Starting Modern API (v2) on port 8000..."
	@echo "📍 Endpoint: POST http://localhost:8000/flyer/extract"
	python src/serving/api_v2.py

# Docker - Legacy (v1)
build:
	@echo "🔨 Building Legacy Docker image..."
	docker-compose build
	@echo "✅ Build complete!"

up:
	@echo "🐳 Starting Legacy containers..."
	docker-compose up -d
	@echo "✅ Legacy API running at http://localhost:8000"
	@echo "📖 Docs: http://localhost:8000/docs"

down:
	@echo "🛑 Stopping Legacy containers..."
	docker-compose down
	@echo "✅ Containers stopped"

logs:
	@echo "📋 Showing Legacy container logs..."
	docker-compose logs -f

test-docker:
	@echo "🧪 Testing Docker build and deployment..."
	./test_docker.sh

# Docker - Modern (v2)
build-v2:
	@echo "🔨 Building Modern Docker image..."
	docker-compose -f docker-compose-v2.yml build
	@echo "✅ Build complete!"

up-v2:
	@echo "🐳 Starting Modern containers..."
	docker-compose -f docker-compose-v2.yml up -d
	@echo "✅ Modern API running at http://localhost:8000"
	@echo "📖 Docs: http://localhost:8000/docs"

down-v2:
	@echo "🛑 Stopping Modern containers..."
	docker-compose -f docker-compose-v2.yml down
	@echo "✅ Containers stopped"

logs-v2:
	@echo "📋 Showing Modern container logs..."
	docker-compose -f docker-compose-v2.yml logs -f

# Testing
test:
	@echo "🧪 Testing Legacy pipeline..."
	python tests/test_pipeline.py examples/image_example.png

test-v2:
	@echo "🧪 Testing Modern pipeline..."
	python tests/test_modern_pipeline.py examples/image_example.png

test-api:
	@echo "🧪 Testing Legacy API..."
	python tests/test_api.py

test-api-v2:
	@echo "🧪 Testing Modern API..."
	python tests/test_api_v2.py

test-all:
	@echo "🧪 Running all tests..."
	@make test
	@make test-v2
	@make test-api
	@make test-api-v2

# Cleanup
clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cache cleaned"

clean-all: clean
	@echo "🧹 Full cleanup (including models)..."
	rm -rf models/*.pth models/*.pt 2>/dev/null || true
	@echo "✅ Full cleanup complete"

clean-docker:
	@echo "🧹 Cleaning Docker images and containers..."
	docker-compose down -v 2>/dev/null || true
	docker-compose -f docker-compose-v2.yml down -v 2>/dev/null || true
	docker rmi flyer-understanding-api:latest 2>/dev/null || true
	docker rmi flyer-extraction-modern:latest 2>/dev/null || true
	docker rmi flyer-extraction-legacy:latest 2>/dev/null || true
	@echo "✅ Docker cleanup complete"

# Development helpers
dev-legacy:
	@echo "🔧 Starting Legacy API in development mode..."
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

dev-modern:
	@echo "🔧 Starting Modern API in development mode..."
	cd src/serving && uvicorn api_v2:app --reload --host 0.0.0.0 --port 8000

# Health check
health:
	@echo "🏥 Checking API health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ API not running"

# Quick demo
demo:
	@echo "🎬 Running quick demo..."
	@echo "1️⃣  Starting Legacy API..."
	@make run &
	@sleep 5
	@echo "2️⃣  Testing with example image..."
	@curl -X POST http://localhost:8000/analyze-flyer -F "file=@examples/image_example.png" | python -m json.tool

# ============================================
# SIMPLE VERSION (Recommended for basic use)
# ============================================

simple:
	@echo "🚀 Starting Simple OCR API..."
	python api_simple.py

simple-build:
	@echo "🔨 Building Simple Docker image..."
	docker-compose -f docker-compose-simple.yml build
	@echo "✅ Build complete!"

simple-up:
	@echo "🐳 Starting Simple OCR API container..."
	docker-compose -f docker-compose-simple.yml up -d
	@echo "✅ API running at http://localhost:8000"
	@echo "📖 Docs: http://localhost:8000/docs"
	@echo ""
	@echo "Test with:"
	@echo "  curl -X POST http://localhost:8000/extract -F 'file=@image.png'"

simple-down:
	@echo "🛑 Stopping Simple OCR API..."
	docker-compose -f docker-compose-simple.yml down
	@echo "✅ Container stopped"

simple-logs:
	@echo "📋 Showing Simple API logs..."
	docker-compose -f docker-compose-simple.yml logs -f

simple-test:
	@echo "🧪 Testing Simple OCR API..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ API not running"
