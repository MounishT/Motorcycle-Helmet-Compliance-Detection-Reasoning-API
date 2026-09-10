"""Inference module for Helmet Detection using RT-DETR."""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import RTDETR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(b, 2) for b in self.bbox],
        }


@dataclass
class InferenceResult:
    detections: List[Detection]
    inference_time_ms: float
    image_shape: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "inference_ms": round(self.inference_time_ms, 2),
            "image_shape": list(self.image_shape),
        }


class HelmetDetector:
    def __init__(self, weights_path: str, device: str = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = {0: "helmet", 1: "no-helmet", 2: "motorcycle"}

        logger.info(f"Loading model from {weights_path}")
        self.model = RTDETR(weights_path)
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}, classes: {self.class_names}")

    def postprocess(self, results, image_shape: Tuple[int, int], conf_threshold: float = None) -> List[Detection]:
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        detections = []
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            if conf < threshold:
                continue

            detections.append(Detection(
                class_name=self.class_names.get(cls, f"class_{cls}"),
                class_id=cls,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
            ))

        return detections

    def detect(self, image_input, conf_threshold: float = None) -> InferenceResult:
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        image_shape = image.shape[:2]
        start_time = time.time()
        results = self.model(image, conf=threshold)
        detections = self.postprocess(results[0], image_shape, threshold)
        inference_time_ms = (time.time() - start_time) * 1000

        return InferenceResult(detections=detections, inference_time_ms=inference_time_ms, image_shape=image_shape)

    def detect_batch(self, image_paths: List[str]) -> List[Optional[InferenceResult]]:
        results = []
        for path in image_paths:
            try:
                results.append(self.detect(path))
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append(None)
        return results

    def get_statistics(self, result: InferenceResult) -> Dict[str, Any]:
        stats = {
            "total_detections": len(result.detections),
            "class_counts": {},
            "avg_confidence": 0.0,
            "min_confidence": 1.0,
            "max_confidence": 0.0,
            "has_violations": False,
        }
        if not result.detections:
            return stats

        for det in result.detections:
            stats["class_counts"][det.class_name] = stats["class_counts"].get(det.class_name, 0) + 1

        confidences = [det.confidence for det in result.detections]
        stats["avg_confidence"] = sum(confidences) / len(confidences)
        stats["min_confidence"] = min(confidences)
        stats["max_confidence"] = max(confidences)
        stats["has_violations"] = "no-helmet" in stats["class_counts"]
        return stats


_detector: Optional[HelmetDetector] = None


def get_detector(weights_path: str = None) -> HelmetDetector:
    global _detector
    if _detector is None:
        if weights_path is None:
            weights_path = "weights/best.pt"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        _detector = HelmetDetector(weights_path)
    return _detector


def detect_image(image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
    return get_detector().detect(image_path, conf_threshold=conf_threshold).to_dict()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    try:
        print(json.dumps(detect_image(sys.argv[1]), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
