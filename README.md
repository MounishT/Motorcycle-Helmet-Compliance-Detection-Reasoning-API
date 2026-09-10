# Motorcycle Helmet Compliance Detection & Reasoning API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RT-DETR based object detection system for motorcycle helmet compliance with a minimal reasoning API.

## 🎯 Overview

This project implements an end-to-end object detection system for monitoring motorcycle helmet compliance. It features:

- **Object Detection**: RT-DETR model fine-tuned on helmet detection dataset
- **Reasoning Layer**: Natural language Q&A about detected objects
- **REST API**: FastAPI endpoints for detection and reasoning
- **Production Ready**: Docker containerization, logging, error handling

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Failure Cases](#failure-cases)
- [License](#license)

## ✨ Features

### Detection Capabilities
- **Classes**: `helmet`, `no-helmet`, `motorcycle`
- **Model**: RT-DETR-L (Real-Time Detection Transformer)
- **Input**: Images (JPEG, PNG, BMP, WebP)
- **Output**: Bounding boxes with class labels and confidence scores

### Reasoning Layer
- Intent routing (image-related vs. unrelated questions)
- Structured reasoning over detection results
- Confidence guardrails (returns "insufficient information" when confidence is low)
- Support for counting, existence, and comparison questions

### API Features
- RESTful endpoints with OpenAPI documentation
- File upload validation
- Rate limiting ready
- CORS support
- Health checks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐        ┌─────────────────────────┐    │
│  │  /detect    │        │         /ask             │    │
│  │  Endpoint   │        │       Endpoint           │    │
│  └──────┬──────┘        └───────────┬─────────────┘    │
│         │                           │                   │
│         ▼                           ▼                   │
│  ┌─────────────┐        ┌─────────────────────────┐    │
│  │  Inference  │        │    Reasoning Layer       │    │
│  │  Module     │        │    - Intent Router       │    │
│  │  (RT-DETR)  │        │    - Structured Reasoner │    │
│  └──────┬──────┘        └───────────┬─────────────┘    │
│         │                           │                   │
│         └───────────┬───────────────┘                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Detection Results                  │   │
│  │  [class, bbox, confidence, inference_time]      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MounishT/Motorcycle-Helmet-Compliance-Detection-Reasoning-API.git
   cd Motorcycle-Helmet-Compliance-Detection-Reasoning-API
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights** (required for inference)
   ```bash
   # Option A: Download via script (recommended)
   python scripts/download_weights.py
   
   # Option B: Manual download
   # 1. Go to: https://github.com/MounishT/Motorcycle-Helmet-Compliance-Detection-Reasoning-API/releases
   # 2. Download 'best.pt' from the latest release
   # 3. Place it in the weights/ directory
   ```
   
   > **Note**: The `weights/best.pt` file is ~170MB. It contains the RT-DETR-L model fine-tuned on the helmet detection dataset. Without this file, the API will start but `/detect` and `/ask` endpoints will return errors.

5. **Run the API**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Access API documentation**
   Open browser: http://localhost:8000/docs

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-09-10T10:30:00"
}
```

#### 2. Detect Objects
```http
POST /detect
Content-Type: multipart/form-data
```

**Parameters**:
- `file` (required): Image file
- `confidence` (optional): Confidence threshold (0-1)

**Request Example** (cURL):
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg" \
  -F "confidence=0.25"
```

**Response**:
```json
{
  "detections": [
    {
      "class": "helmet",
      "class_id": 0,
      "confidence": 0.93,
      "bbox": [120.5, 85.2, 245.8, 210.3]
    },
    {
      "class": "no-helmet",
      "class_id": 1,
      "confidence": 0.87,
      "bbox": [350.0, 120.5, 480.2, 265.8]
    }
  ],
  "inference_ms": 125.4,
  "image_shape": [640, 640]
}
```

#### 3. Ask Question
```http
POST /ask
Content-Type: multipart/form-data
```

**Parameters**:
- `file` (required): Image file
- `question` (required): Natural language question
- `confidence` (optional): Confidence threshold (0-1)

**Request Example** (cURL):
```bash
curl -X POST "http://localhost:8000/ask" \
  -F "file=@test_image.jpg" \
  -F "question=How many riders are not wearing helmets?"
```

**Response**:
```json
{
  "answer": "1 rider not wearing a helmet",
  "status": "ok",
  "support": {
    "detections": [
      {
        "class": "no-helmet",
        "confidence": 0.87,
        "bbox": [350.0, 120.5, 480.2, 265.8]
      }
    ],
    "class_counts": {
      "helmet": 1,
      "no-helmet": 1,
      "motorcycle": 1
    },
    "has_violations": true
  },
  "confidence_score": 0.87
}
```

**Question Types**:
- Counting: "How many helmets?", "Number of motorcycles"
- Existence: "Is anyone wearing a helmet?", "Are there violations?"
- Common: "What is the most common object?"
- Yes/No: "Is there a helmet present?"

**Status Codes**:
- `ok`: Question answered successfully
- `insufficient_info`: Low detection confidence
- `question_not_image_related`: Question unrelated to image

### Python Client Example

```python
import requests

# Detect objects
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    detections = response.json()
    print(f"Found {len(detections['detections'])} objects")

# Ask question
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ask',
        files={'file': f},
        data={'question': 'How many riders are not wearing helmets?'}
    )
    result = response.json()
    print(f"Answer: {result['answer']}")
    print(f"Status: {result['status']}")
```

## 🎓 Training

### Dataset Preparation

1. **Download dataset** (or use your own):
   ```bash
   python src/prepare_dataset.py --source /path/to/raw/data --output data/processed --generate-config
   ```

2. **Verify dataset structure**:
   ```
   data/processed/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

### Train Model

```bash
python src/train.py \
  --config configs/data.yaml \
  --epochs 30 \
  --batch-size 8 \
  --img-size 640
```

**Training Options**:
- `--config`: Path to data configuration YAML
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Training batch size (default: 8)
- `--img-size`: Input image size (default: 640)
- `--device`: Training device (auto-detect if not specified)

**Training Logs**:
- TensorBoard logs: `runs/`
- Training metrics: `training.log`
- Model checkpoints: `runs/helmet_detection/weights/`

## 📊 Evaluation

### Run Evaluation

```bash
python src/evaluate.py \
  --weights weights/best.pt \
  --data configs/data.yaml \
  --split test
```

**Evaluation Outputs**:
- `evaluation_results/evaluation_metrics.json`: Full metrics
- `evaluation_results/confusion_matrix.png`: Confusion matrix
- `evaluation_results/per_class_metrics.png`: Per-class visualization

### Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| mAP@0.5 | Mean Average Precision at IoU 0.5 | > 0.7 |
| mAP@0.5:0.95 | mAP at multiple IoU thresholds | > 0.5 |
| Precision | True positives / (True + False positives) | > 0.7 |
| Recall | True positives / (True + False negatives) | > 0.7 |
| F1-Score | Harmonic mean of Precision and Recall | > 0.7 |

## 🐳 Deployment

### Docker

1. **Build image**:
   ```bash
   docker build -t helmet-detection-api .
   ```

2. **Run container**:
   ```bash
   docker run -d \
     --name helmet-api \
     -p 8000:8000 \
     -v $(pwd)/weights:/app/weights \
     helmet-detection-api
   ```

### Docker Compose

```bash
docker-compose up -d
```

This starts:
- API server (port 8000)
- Redis (port 6379) for rate limiting
- Prometheus (port 9090) for metrics (optional)

### Production Deployment

```bash
# With Gunicorn (multi-worker)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With Nginx reverse proxy (see configs/nginx.conf)
```

## 📁 Project Structure

```
helmet-detection-api/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI application
├── configs/
│   └── data.yaml               # Dataset configuration
├── data/
│   ├── processed/              # Processed dataset
│   └── raw/                    # Raw dataset
├── notebooks/
│   └── eda_analysis.ipynb      # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── evaluate.py             # Evaluation script
│   ├── inference.py            # Inference module
│   ├── prepare_dataset.py      # Dataset preparation
│   ├── reasoning.py            # Reasoning layer
│   └── train.py                # Training script
├── tests/
│   ├── test_api.py             # API integration tests
│   └── test_inference.py       # Unit tests
├── weights/
│   └── best.pt                 # Trained model weights
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── memo.md                     # 2-page submission memo
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## 🔄 Reproducibility

### Environment Setup

```bash
# Create environment
conda create -n helmet-detection python=3.10
conda activate helmet-detection

# Or use venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Training

```bash
# Set seed (done automatically in train.py)
export PYTHONHASHSEED=42

# Train
python src/train.py --config configs/data.yaml --epochs 30
```

### Reproduce Evaluation

```bash
python src/evaluate.py --weights weights/best.pt --data configs/data.yaml --split test
```

### Hardware Specifications

- **GPU**: NVIDIA GTX 1080 / RTX 3060 / equivalent
- **RAM**: 16GB minimum
- **Storage**: 10GB for dataset + weights

### Training Time

- **GPU**: ~2.5 hours (30 epochs)
- **CPU**: ~24 hours (not recommended)

## ⚠️ Failure Cases

### Case 1: Motion Blur (`images/new42.jpg`)
- **Cause**: Fast-moving riders create horizontal motion blur
- **Effect**: Helmet detection missed (confidence 0.12, below threshold)
- **Root Cause**: Blurred edges break RT-DETR's attention token aggregation
- **Mitigation**: Temporal smoothing, motion deblurring preprocessing, blur augmentation

### Case 2: Small/Distant Objects (`images/new108.jpg`)
- **Cause**: Rider ~80px tall in deep background
- **Effect**: Background rider's helmet not detected
- **Root Cause**: Object below effective FPN receptive field at 5× downsampling
- **Mitigation**: Multi-scale inference at 1280px, copy-paste augmentation for small objects

### Case 3: Occlusion (`images/new63.jpg`)
- **Cause**: Rider partially hidden behind a parked truck
- **Effect**: Bounding box extends into truck region (IoU 0.42 with GT)
- **Root Cause**: Self-attention merges rider and truck features across occlusion boundary
- **Mitigation**: Synthetic occlusion augmentation, Soft-NMS, occlusion-aware training

### Case 4: Low-Light (`images/new113.jpg`)
- **Cause**: Nighttime scene with single streetlight
- **Effect**: Low confidence (0.31), pillion rider's no-helmet missed
- **Root Cause**: 92% of training data is daytime — domain gap for nighttime inputs
- **Mitigation**: CLAHE preprocessing, mixed day/night training data, brightness augmentation

### Case 5: Class Confusion (`images/new31.jpg`)
- **Cause**: Rider wearing dark baseball cap (viewed from behind)
- **Effect**: Cap misclassified as helmet (confidence 0.78, false positive)
- **Root Cause**: Similar circular silhouette and dark color histogram
- **Mitigation**: Add `cap` negative class, hard negative mining, higher-resolution input

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHTS_PATH` | `weights/best.pt` | Path to model weights |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detection confidence threshold |
| `REASONING_CONFIDENCE_THRESHOLD` | `0.5` | Reasoning confidence threshold |

### Data Configuration (configs/data.yaml)

```yaml
path: data/processed
train: train
val: val
test: test
nc: 3
names:
  0: helmet
  1: no-helmet
  2: motorcycle
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/test_inference.py -v

# Run API tests
pytest tests/test_api.py -v

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

- **Author**: [T MOUNISH]
- **Email**: [mounishrt@gmail.com]
- **GitHub**: [github.com/MounishT](https://github.com/MounishT)

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for RT-DETR implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- Dataset contributors from Kaggle and academic repositories

---

**Note**: This is a submission for the RAP AI/ML Task Assignment. For questions or issues, please contact the project maintainer.
