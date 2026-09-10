"""
Inference Helper for Helmet Detection
Loads model and provides structured detection outputs
"""

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
    """Single detection result."""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in absolute pixels
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'class': self.class_name,
            'class_id': self.class_id,
            'confidence': round(self.confidence, 4),
            'bbox': [round(b, 2) for b in self.bbox]
        }


@dataclass
class InferenceResult:
    """Complete inference result for an image."""
    detections: List[Detection]
    inference_time_ms: float
    image_shape: Tuple[int, int]  # (height, width)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detections': [d.to_dict() for d in self.detections],
            'inference_ms': round(self.inference_time_ms, 2),
            'image_shape': list(self.image_shape)
        }


class HelmetDetector:
    """RT-DETR Helmet Detection Inference Engine."""
    
    def __init__(self, weights_path: str, device: str = None, 
                 conf_threshold: float = 0.25):
        """
        Initialize the helmet detector.
        
        Args:
            weights_path: Path to trained model weights
            device: Inference device (auto-detect if None)
            conf_threshold: Minimum confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logger.info(f"Loading model from {weights_path}")
        self.model = RTDETR(weights_path)
        self.model.to(self.device)
        
        # Class mapping
        self.class_names = {0: 'helmet', 1: 'no-helmet', 2: 'motorcycle'}
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Classes: {self.class_names}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: BGR image as numpy array
        
        Returns:
            Preprocessed image
        """
        # Resize to model input size
        img_resized = cv2.resize(image, (640, 640))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        return img_normalized
    
    def postprocess(self, results, image_shape: Tuple[int, int], conf_threshold: float = None) -> List[Detection]:
        """
        Postprocess model output to structured detections.
        
        Args:
            results: Ultralytics Results object
            image_shape: Original image shape (height, width)
            conf_threshold: Optional confidence threshold override (uses instance default if None)
        
        Returns:
            List of Detection objects
        """
        # Use provided threshold or instance default
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        detections = []
        
        # Extract boxes, classes, and confidences
        boxes = results.boxes
        
        if boxes is None or len(boxes) == 0:
            return detections
        
        # Process each detection
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box.tolist()
            
            # Get class and confidence
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            
            # Apply confidence threshold
            if conf < threshold:
                continue
            
            # Get class name
            class_name = self.class_names.get(cls, f'class_{cls}')
            
            # Create detection object
            detection = Detection(
                class_name=class_name,
                class_id=cls,
                confidence=conf,
                bbox=[x1, y1, x2, y2]
            )
            detections.append(detection)
        
        return detections
    
    def detect(self, image_input, conf_threshold: float = None) -> InferenceResult:
        """
        Run detection on an image.
        
        Args:
            image_input: File path, numpy array, or PIL Image
            conf_threshold: Optional confidence threshold override (uses instance default if None)
        
        Returns:
            InferenceResult with detections and timing
        """
        # Use provided threshold or instance default
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # Load image if needed
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
        
        # Get original image shape
        image_shape = image.shape[:2]  # (height, width)
        
        # Measure inference time
        start_time = time.time()
        
        # Run inference
        results = self.model(image, conf=threshold)
        
        # Postprocess results
        detections = self.postprocess(results[0], image_shape, threshold)
        
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_time_ms,
            image_shape=image_shape
        )
    
    def detect_batch(self, image_paths: List[str]) -> List[InferenceResult]:
        """
        Run detection on multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of InferenceResult objects
        """
        results = []
        for path in image_paths:
            try:
                result = self.detect(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append(None)
        return results
    
    def get_statistics(self, result: InferenceResult) -> Dict[str, Any]:
        """
        Compute detection statistics for an inference result.
        
        Args:
            result: InferenceResult object
        
        Returns:
            Dictionary with detection statistics
        """
        stats = {
            'total_detections': len(result.detections),
            'class_counts': {},
            'avg_confidence': 0.0,
            'min_confidence': 1.0,
            'max_confidence': 0.0,
            'has_violations': False
        }
        
        if not result.detections:
            return stats
        
        # Count classes
        for det in result.detections:
            if det.class_name not in stats['class_counts']:
                stats['class_counts'][det.class_name] = 0
            stats['class_counts'][det.class_name] += 1
        
        # Confidence statistics
        confidences = [det.confidence for det in result.detections]
        stats['avg_confidence'] = sum(confidences) / len(confidences)
        stats['min_confidence'] = min(confidences)
        stats['max_confidence'] = max(confidences)
        
        # Check for violations (no-helmet detections)
        stats['has_violations'] = 'no-helmet' in stats['class_counts']
        
        return stats


# Global detector instance (lazy loaded)
_detector: Optional[HelmetDetector] = None


def get_detector(weights_path: str = None) -> HelmetDetector:
    """
    Get or initialize the global detector instance.
    
    Args:
        weights_path: Path to model weights (uses default if None)
    
    Returns:
        HelmetDetector instance
    """
    global _detector
    
    if _detector is None:
        if weights_path is None:
            weights_path = 'weights/best.pt'
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        _detector = HelmetDetector(weights_path)
    
    return _detector


def detect_image(image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
    """
    Convenience function to detect objects in an image.
    
    Args:
        image_path: Path to image file
        conf_threshold: Minimum confidence threshold
    
    Returns:
        Dictionary with detection results
    """
    detector = get_detector()
    result = detector.detect(image_path, conf_threshold=conf_threshold)
    return result.to_dict()


if __name__ == '__main__':
    # Test inference
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        result = detect_image(image_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
