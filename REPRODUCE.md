# Reproduction Guide

This document provides step-by-step instructions to reproduce the results of this submission.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended) or CPU
- 10GB+ storage space
- Git

## Environment Setup

### Option 1: Using venv (Recommended)

```bash
# Clone repository
git clone https://github.com/MounishT/Motorcycle-Helmet-Compliance-Detection-Reasoning-API.git
cd Motorcycle-Helmet-Compliance-Detection-Reasoning-API

# Create virtual environment
python -m venv venv

# Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Clone repository
git clone https://github.com/MounishT/Motorcycle-Helmet-Compliance-Detection-Reasoning-API.git
cd Motorcycle-Helmet-Compliance-Detection-Reasoning-API

# Create conda environment
conda create -n helmet-detection python=3.10
conda activate helmet-detection

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Option 1: Use Provided Dataset

If dataset is included in the repository:

```bash
# Dataset should already be in data/processed/
ls data/processed/
# Should show: train/ val/ test/
```

### Option 2: Prepare Your Own Dataset

```bash
# Prepare dataset from YOLO format
python src/prepare_dataset.py \
  --source /path/to/raw/dataset \
  --output data/processed \
  --generate-config
```

### Verify Dataset Structure

```bash
# Check directory structure
find data/processed -type d | head -20

# Expected structure:
# data/processed/
# ├── train/
# │   ├── images/
# │   └── labels/
# ├── val/
# │   ├── images/
# │   └── labels/
# └── test/
#     ├── images/
#     └── labels/
```

## Model Training

### Train from Scratch

```bash
# Set random seed (optional, done automatically)
export PYTHONHASHSEED=42

# Train RT-DETR model
python src/train.py \
  --config configs/data.yaml \
  --epochs 30 \
  --batch-size 8 \
  --img-size 640 \
  --device cuda
```

### Training Output

- Model checkpoints: `runs/helmet_detection/weights/`
- Training logs: `training.log`
- TensorBoard logs: `runs/`

### Copy Best Weights

```bash
# Copy best checkpoint to weights directory
cp runs/helmet_detection/weights/best.pt weights/best.pt
```

## Model Evaluation

### Run Evaluation

```bash
# Evaluate on test set
python src/evaluate.py \
  --weights weights/best.pt \
  --data configs/data.yaml \
  --split test \
  --device cuda
```

### Evaluation Outputs

- Metrics JSON: `evaluation_results/evaluation_metrics.json`
- Confusion matrix: `evaluation_results/confusion_matrix.png`
- Per-class metrics: `evaluation_results/per_class_metrics.png`

### Expected Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.82 |
| mAP@0.5:0.95 | 0.58 |
| Precision | 0.79 |
| Recall | 0.84 |

## API Testing

### Start API Server

```bash
# Development mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Detection
curl -X POST "http://localhost:8000/detect" \
  -F "file=@tests/sample.jpg"

# Question answering
curl -X POST "http://localhost:8000/ask" \
  -F "file=@tests/sample.jpg" \
  -F "question=How many helmets are detected?"
```

### Python Client Test

```python
import requests

# Test detection
with open('tests/sample.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    print(response.json())

# Test question answering
with open('tests/sample.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ask',
        files={'file': f},
        data={'question': 'Is anyone not wearing a helmet?'}
    )
    print(response.json())
```

## Running Tests

### Unit Tests

```bash
# Run inference tests
pytest tests/test_inference.py -v

# Run reasoning tests
pytest tests/test_reasoning.py -v
```

### Integration Tests

```bash
# Run API tests
pytest tests/test_api.py -v
```

### All Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Docker Deployment

### Build Image

```bash
docker build -t helmet-detection-api .
```

### Run Container

```bash
docker run -d \
  --name helmet-api \
  -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  helmet-detection-api
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Hardware Specifications

### Training Hardware

- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 32GB DDR4
- **Storage**: 512GB NVMe SSD
- **CUDA**: 11.8
- **cuDNN**: 8.6

### Training Time

- **GPU Training**: ~2.5 hours (30 epochs)
- **CPU Training**: ~24 hours (not recommended)

### Inference Performance

- **GPU**: ~120ms per image
- **CPU**: ~800ms per image

## Random Seeds

All random operations use seed 42 for reproducibility:

```python
import torch
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `--batch-size 4`
   - Use smaller image size: `--img-size 480`

2. **Module not found errors**
   - Ensure you're in the project root directory
   - Activate virtual environment

3. **Port already in use**
   - Use different port: `--port 8001`
   - Kill existing process: `lsof -ti:8000 | xargs kill -9`

4. **Model weights not found**
   - Ensure `weights/best.pt` exists
   - Or set environment variable: `export WEIGHTS_PATH=/path/to/weights`

## Contact

For questions about reproduction, contact:
- Email: mounishrt@gmail.com
- GitHub: https://github.com/MounishT
