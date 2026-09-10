"""Unit tests for inference output structure and reasoning logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.inference import Detection, InferenceResult
from src.reasoning import IntentRouter, StructuredReasoner, ReasoningEngine


class TestInferenceOutput:
    def test_detection_dataclass(self):
        det = Detection(class_name="helmet", class_id=0, confidence=0.92, bbox=[100.0, 100.0, 200.0, 200.0])
        assert det.class_name == "helmet"
        assert det.class_id == 0
        assert det.confidence == 0.92
        d = det.to_dict()
        assert d["class"] == "helmet"
        assert d["confidence"] == 0.92

    def test_inference_result_structure(self):
        dets = [Detection("helmet", 0, 0.92, [100, 100, 200, 200]), Detection("no-helmet", 1, 0.87, [300, 150, 400, 250])]
        result = InferenceResult(detections=dets, inference_time_ms=150.5, image_shape=(640, 640))
        assert len(result.detections) == 2
        d = result.to_dict()
        assert len(d["detections"]) == 2
        assert d["inference_ms"] == 150.5

    def test_detection_bbox_format(self):
        det = Detection("helmet", 0, 0.95, [10.5, 20.3, 30.7, 40.1])
        bbox = det.to_dict()["bbox"]
        assert isinstance(bbox, list) and len(bbox) == 4
        assert all(isinstance(b, float) for b in bbox)


class TestIntentRouter:
    def test_image_related(self):
        router = IntentRouter()
        assert router.is_image_related("How many helmets are there?")
        assert router.is_image_related("Is anyone not wearing a helmet?")
        assert router.is_image_related("What objects are detected?")
        assert router.is_image_related("Count the motorcycles")
        assert not router.is_image_related("What is the weather?")
        assert not router.is_image_related("Tell me a joke")
        assert not router.is_image_related("What is 2+2?")

    def test_question_type_classification(self):
        router = IntentRouter()
        assert router.get_question_type("How many helmets?") == "counting"
        assert router.get_question_type("Number of riders") == "counting"
        assert router.get_question_type("Is there a helmet?") == "yes_no"
        assert router.get_question_type("Are there motorcycles?") == "yes_no"
        assert router.get_question_type("Is anyone wearing a helmet?") == "existence"
        assert router.get_question_type("What is the most common object?") == "common"

    def test_target_class_extraction(self):
        router = IntentRouter()
        assert router.extract_target_class("How many helmets?") == "helmet"
        assert router.extract_target_class("Count motorcycles") == "motorcycle"
        assert router.extract_target_class("Any no-helmet?") == "no-helmet"
        assert router.extract_target_class("Is anyone wearing headgear?") == "helmet"
        assert router.extract_target_class("What is the time?") is None


class TestStructuredReasoner:
    def test_analyze_detections(self):
        reasoner = StructuredReasoner(confidence_threshold=0.5)
        dets = [
            {"class": "helmet", "confidence": 0.92, "bbox": [100, 100, 200, 200]},
            {"class": "no-helmet", "confidence": 0.87, "bbox": [300, 150, 400, 250]},
            {"class": "motorcycle", "confidence": 0.95, "bbox": [50, 50, 450, 350]},
        ]
        stats = reasoner.analyze_detections(dets)
        assert stats["total"] == 3
        assert stats["class_counts"]["helmet"] == 1
        assert stats["has_violations"]
        assert stats["reliable"]

    def test_empty_detections(self):
        stats = StructuredReasoner(0.5).analyze_detections([])
        assert stats["total"] == 0
        assert not stats["reliable"]

    def test_low_confidence(self):
        dets = [{"class": "helmet", "confidence": 0.3}, {"class": "no-helmet", "confidence": 0.2}]
        stats = StructuredReasoner(0.5).analyze_detections(dets)
        assert stats["total"] == 2
        assert not stats["reliable"]

    def test_count_by_class(self):
        r = StructuredReasoner()
        dets = [{"class": "helmet"}, {"class": "helmet"}, {"class": "no-helmet"}]
        assert r.count_by_class(dets, "helmet") == 2
        assert r.count_by_class(dets, "no-helmet") == 1
        assert r.count_by_class(dets, "motorcycle") == 0

    def test_get_most_common(self):
        r = StructuredReasoner()
        dets = [{"class": "helmet"}] * 3 + [{"class": "no-helmet"}]
        assert r.get_most_common(dets) == "helmet"
        assert r.get_most_common([]) is None


class TestReasoningEngine:
    def test_counting(self):
        engine = ReasoningEngine(0.5)
        dets = [{"class": "helmet", "confidence": 0.92}, {"class": "no-helmet", "confidence": 0.87}]
        result = engine.reason("How many helmets?", dets)
        assert result.status == "ok"
        assert "1 helmet" in result.answer

    def test_yes_no(self):
        engine = ReasoningEngine(0.5)
        dets = [{"class": "helmet", "confidence": 0.92}]
        result = engine.reason("Is anyone wearing a helmet?", dets)
        assert result.status == "ok"
        assert "Yes" in result.answer or "helmet" in result.answer

    def test_unrelated(self):
        engine = ReasoningEngine(0.5)
        result = engine.reason("What is the weather?", [{"class": "helmet", "confidence": 0.92}])
        assert result.status == "question_not_image_related"

    def test_insufficient_info(self):
        engine = ReasoningEngine(0.8)
        result = engine.reason("How many helmets?", [{"class": "helmet", "confidence": 0.3}])
        assert result.status == "insufficient_info"
        assert "Insufficient" in result.answer

    def test_violations(self):
        engine = ReasoningEngine(0.5)
        dets = [
            {"class": "helmet", "confidence": 0.92},
            {"class": "no-helmet", "confidence": 0.87},
            {"class": "no-helmet", "confidence": 0.82},
        ]
        result = engine.reason("How many riders not wearing helmets?", dets)
        assert result.status == "ok"
        assert "2" in result.answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
