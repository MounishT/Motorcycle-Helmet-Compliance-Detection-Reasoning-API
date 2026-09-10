# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-13

### Added

- **Core Detection**
  - RT-DETR model fine-tuned on helmet detection dataset
  - Support for 3 classes: helmet, no-helmet, motorcycle
  - Confidence thresholding and NMS

- **API Endpoints**
  - `/detect` endpoint for object detection
  - `/ask` endpoint for natural language Q&A
  - `/health` endpoint for health checks
  - OpenAPI documentation at `/docs`

- **Reasoning Layer**
  - Intent routing for image-related questions
  - Structured reasoning over detection results
  - Confidence guardrails with "insufficient information" responses
  - Support for counting, existence, and comparison questions

- **Training Pipeline**
  - RT-DETR training script with logging
  - TensorBoard integration
  - Checkpoint saving and early stopping
  - Reproducible training with seed=42

- **Evaluation**
  - mAP@0.5 and mAP@0.5:0.95 metrics
  - Per-class precision, recall, F1
  - Confusion matrix generation
  - Visualization plots

- **Dataset Handling**
  - YOLO format dataset preparation
  - COCO format conversion
  - Train/val/test splitting with statistics

- **Deployment**
  - Dockerfile for containerization
  - docker-compose.yml with Redis and Prometheus
  - Health checks and logging

- **Testing**
  - Unit tests for inference and reasoning
  - Integration tests for API endpoints
  - Pytest configuration

- **Documentation**
  - Comprehensive README
  - API usage examples
  - Reproduction guide
  - 2-page submission memo

### Technical Decisions

1. **Model Choice**: RT-DETR-L selected for balance of accuracy and speed
2. **Framework**: Ultralytics implementation for ease of use and maintenance
3. **API Framework**: FastAPI for automatic OpenAPI docs and async support
4. **Reasoning Approach**: Hand-written logic without frameworks per constraints
5. **Dataset Split**: 70/15/15 train/val/test with stratification

### Known Limitations

- Motion blur affects detection on fast-moving riders
- Small objects (< 32x32 pixels) may be missed
- Low-light conditions reduce confidence
- Class confusion between helmets and similar headwear

### Future Improvements

- Add video stream processing
- Implement temporal smoothing
- Add more diverse training data
- Optimize for edge deployment
- Add multi-language support

## [0.1.0] - 2026-09-10

### Added

- Initial project structure
- Basic RT-DETR training script
- Preliminary dataset preparation

### Notes

- This was the initial prototype for the RAP AI/ML Task Assignment
- All core features implemented and tested
- Ready for submission

---

## Decision Log

### 2026-09-10: Dataset Selection
- **Decision**: Use public helmet detection datasets from Kaggle
- **Rationale**: Reproducible, well-documented, diverse scenarios
- **Alternative considered**: Custom web scraping (rejected due to time constraints)

### 2026-09-10: Model Architecture
- **Decision**: RT-DETR-L with ResNet-50 backbone
- **Rationale**: Good balance of accuracy and inference speed
- **Alternative considered**: RT-DETR-X (too slow for real-time), RT-DETR-S (lower accuracy)

### 2026-09-11: Reasoning Layer Design
- **Decision**: Hand-written intent routing without frameworks
- **Rationale**: Meets constraint of no LangChain/AutoGen
- **Alternative considered**: Rule-based system (rejected for flexibility)

### 2026-09-11: Confidence Threshold
- **Decision**: 0.5 threshold for reasoning confidence
- **Rationale**: Balances false positives and coverage
- **Alternative considered**: 0.3 (too many false positives), 0.7 (too restrictive)
