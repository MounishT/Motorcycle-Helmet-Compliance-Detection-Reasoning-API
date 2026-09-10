.PHONY: help install train evaluate test run docker clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the model"
	@echo "  make evaluate   - Evaluate the model"
	@echo "  make test       - Run tests"
	@echo "  make run        - Start the API server"
	@echo "  make docker     - Build and run Docker container"
	@echo "  make clean      - Clean temporary files"

# Install dependencies
install:
	pip install -r requirements.txt

# Train the model
train:
	python src/train.py --config configs/data.yaml --epochs 30

# Evaluate the model
evaluate:
	python src/evaluate.py --weights weights/best.pt --data configs/data.yaml --split test

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ --cov=src --cov-report=html

# Start the API server
run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start in production mode
run-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Build and run Docker container
docker:
	docker build -t helmet-detection-api .
	docker run -d --name helmet-api -p 8000:8000 helmet-detection-api

# Stop Docker container
docker-stop:
	docker stop helmet-api
	docker rm helmet-api

# Run Docker Compose
docker-compose:
	docker-compose up -d

# Stop Docker Compose
docker-compose-down:
	docker-compose down

# Create sample test images
create-samples:
	python tests/create_sample_images.py

# Download model weights
download-weights:
	python scripts/download_weights.py

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf evaluation_results
	rm -rf runs

# Lint code
lint:
	black src/ api/ tests/
	flake8 src/ api/ tests/

# Format code
format:
	black src/ api/ tests/
	isort src/ api/ tests/

# Generate requirements.txt from current environment
freeze:
	pip freeze > requirements.txt
