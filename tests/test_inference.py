"""
Unit Tests for Helmet Detection System
Tests inference output structure and reasoning logic
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInferenceOutput:
    """Test inference output structure and format."""
    
    def test_detection_dataclass(self):
        """Test Detection dataclass structure."""
        from src.inference import Detection
        
        det = Detection(
            class_name='helmet',
            class_id=0,
            confidence=0.92,
            bbox=[100.0, 100.0, 200.0, 200.0]
        )
        
        assert det.class_name == 'helmet'
        assert det.class_id == 0
        assert det.confidence == 0.92
        assert det.bbox == [100.0, 100.0, 200.0, 200.0]
        
        # Test to_dict method
        d = det.to_dict()
        assert 'class' in d
        assert 'class_id' in d
        assert 'confidence' in d
        assert 'bbox' in d
        assert d['confidence'] == 0.92
    
    def test_inference_result_structure(self):
        """Test InferenceResult dataclass structure."""
        from src.inference import Detection, InferenceResult
        
        detections = [
            Detection('helmet', 0, 0.92, [100, 100, 200, 200]),
            Detection('no-helmet', 1, 0.87, [300, 150, 400, 250])
        ]
        
        result = InferenceResult(
            detections=detections,
            inference_time_ms=150.5,
            image_shape=(640, 640)
        )
        
        assert len(result.detections) == 2
        assert result.inference_time_ms == 150.5
        assert result.image_shape == (640, 640)
        
        # Test to_dict
        d = result.to_dict()
        assert 'detections' in d
        assert 'inference_ms' in d
        assert 'image_shape' in d
        assert len(d['detections']) == 2
    
    def test_detection_bbox_format(self):
        """Test bounding box format is consistent."""
        from src.inference import Detection
        
        det = Detection('helmet', 0, 0.95, [10.5, 20.3, 30.7, 40.1])
        d = det.to_dict()
        
        # Check bbox is list of 4 floats
        assert isinstance(d['bbox'], list)
        assert len(d['bbox']) == 4
        assert all(isinstance(b, float) for b in d['bbox'])


class TestReasoningLogic:
    """Test reasoning layer logic and intent routing."""
    
    def test_intent_router_image_related(self):
        """Test intent router identifies image-related questions."""
        from src.reasoning import IntentRouter
        
        router = IntentRouter()
        
        # Should be image-related
        assert router.is_image_related("How many helmets are there?") == True
        assert router.is_image_related("Is anyone not wearing a helmet?") == True
        assert router.is_image_related("What objects are detected?") == True
        assert router.is_image_related("Count the motorcycles") == True
        
        # Should NOT be image-related
        assert router.is_image_related("What is the weather?") == False
        assert router.is_image_related("Tell me a joke") == False
        assert router.is_image_related("What is 2+2?") == False
    
    def test_question_type_classification(self):
        """Test question type classification."""
        from src.reasoning import IntentRouter
        
        router = IntentRouter()
        
        assert router.get_question_type("How many helmets?") == 'counting'
        assert router.get_question_type("Number of riders") == 'counting'
        assert router.get_question_type("Is there a helmet?") == 'yes_no'
        assert router.get_question_type("Are there motorcycles?") == 'yes_no'
        assert router.get_question_type("Is anyone wearing a helmet?") == 'existence'
        assert router.get_question_type("What is the most common object?") == 'common'
    
    def test_target_class_extraction(self):
        """Test target class extraction from questions."""
        from src.reasoning import IntentRouter
        
        router = IntentRouter()
        
        assert router.extract_target_class("How many helmets?") == 'helmet'
        assert router.extract_target_class("Count motorcycles") == 'motorcycle'
        assert router.extract_target_class("Any no-helmet?") == 'no-helmet'
        assert router.extract_target_class("Is anyone wearing headgear?") == 'helmet'
        assert router.extract_target_class("What is the time?") is None
    
    def test_structured_reasoner_analyze(self):
        """Test structured reasoner analysis."""
        from src.reasoning import StructuredReasoner
        
        reasoner = StructuredReasoner(confidence_threshold=0.5)
        
        detections = [
            {'class': 'helmet', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]},
            {'class': 'no-helmet', 'confidence': 0.87, 'bbox': [300, 150, 400, 250]},
            {'class': 'motorcycle', 'confidence': 0.95, 'bbox': [50, 50, 450, 350]}
        ]
        
        stats = reasoner.analyze_detections(detections)
        
        assert stats['total'] == 3
        assert stats['class_counts']['helmet'] == 1
        assert stats['class_counts']['no-helmet'] == 1
        assert stats['class_counts']['motorcycle'] == 1
        assert stats['avg_confidence'] > 0
        assert stats['has_violations'] == True
        assert stats['reliable'] == True
    
    def test_structured_reasoner_empty_detections(self):
        """Test structured reasoner with empty detections."""
        from src.reasoning import StructuredReasoner
        
        reasoner = StructuredReasoner(confidence_threshold=0.5)
        
        stats = reasoner.analyze_detections([])
        
        assert stats['total'] == 0
        assert stats['class_counts'] == {}
        assert stats['reliable'] == False
    
    def test_structured_reasoner_low_confidence(self):
        """Test structured reasoner with low confidence detections."""
        from src.reasoning import StructuredReasoner
        
        reasoner = StructuredReasoner(confidence_threshold=0.5)
        
        detections = [
            {'class': 'helmet', 'confidence': 0.3, 'bbox': [100, 100, 200, 200]},
            {'class': 'no-helmet', 'confidence': 0.2, 'bbox': [300, 150, 400, 250]}
        ]
        
        stats = reasoner.analyze_detections(detections)
        
        assert stats['total'] == 2
        assert stats['reliable'] == False  # Below threshold
    
    def test_count_by_class(self):
        """Test counting by class."""
        from src.reasoning import StructuredReasoner
        
        reasoner = StructuredReasoner()
        
        detections = [
            {'class': 'helmet', 'confidence': 0.9},
            {'class': 'helmet', 'confidence': 0.85},
            {'class': 'no-helmet', 'confidence': 0.7}
        ]
        
        assert reasoner.count_by_class(detections, 'helmet') == 2
        assert reasoner.count_by_class(detections, 'no-helmet') == 1
        assert reasoner.count_by_class(detections, 'motorcycle') == 0
    
    def test_get_most_common(self):
        """Test getting most common class."""
        from src.reasoning import StructuredReasoner
        
        reasoner = StructuredReasoner()
        
        detections = [
            {'class': 'helmet', 'confidence': 0.9},
            {'class': 'helmet', 'confidence': 0.85},
            {'class': 'helmet', 'confidence': 0.8},
            {'class': 'no-helmet', 'confidence': 0.7}
        ]
        
        assert reasoner.get_most_common(detections) == 'helmet'
        
        # Empty detections
        assert reasoner.get_most_common([]) is None


class TestReasoningIntegration:
    """Integration tests for the reasoning engine."""
    
    def test_reasoning_engine_counting(self):
        """Test reasoning engine with counting question."""
        from src.reasoning import ReasoningEngine
        
        engine = ReasoningEngine(confidence_threshold=0.5)
        
        detections = [
            {'class': 'helmet', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]},
            {'class': 'no-helmet', 'confidence': 0.87, 'bbox': [300, 150, 400, 250]}
        ]
        
        result = engine.reason("How many helmets?", detections)
        
        assert result.status == 'ok'
        assert '1 helmet' in result.answer
    
    def test_reasoning_engine_yes_no(self):
        """Test reasoning engine with yes/no question."""
        from src.reasoning import ReasoningEngine
        
        engine = ReasoningEngine(confidence_threshold=0.5)
        
        detections = [
            {'class': 'helmet', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]}
        ]
        
        result = engine.reason("Is anyone wearing a helmet?", detections)
        
        assert result.status == 'ok'
        assert 'Yes' in result.answer or 'helmet' in result.answer
    
    def test_reasoning_engine_unrelated(self):
        """Test reasoning engine with unrelated question."""
        from src.reasoning import ReasoningEngine
        
        engine = ReasoningEngine(confidence_threshold=0.5)
        
        detections = [
            {'class': 'helmet', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]}
        ]
        
        result = engine.reason("What is the weather?", detections)
        
        assert result.status == 'question_not_image_related'
    
    def test_reasoning_engine_insufficient_info(self):
        """Test reasoning engine with low confidence detections."""
        from src.reasoning import ReasoningEngine
        
        engine = ReasoningEngine(confidence_threshold=0.8)
        
        detections = [
            {'class': 'helmet', 'confidence': 0.3, 'bbox': [100, 100, 200, 200]}
        ]
        
        result = engine.reason("How many helmets?", detections)
        
        assert result.status == 'insufficient_info'
        assert 'Insufficient information' in result.answer
    
    def test_reasoning_engine_violations(self):
        """Test reasoning engine with violation detection."""
        from src.reasoning import ReasoningEngine
        
        engine = ReasoningEngine(confidence_threshold=0.5)
        
        detections = [
            {'class': 'helmet', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]},
            {'class': 'no-helmet', 'confidence': 0.87, 'bbox': [300, 150, 400, 250]},
            {'class': 'no-helmet', 'confidence': 0.82, 'bbox': [500, 200, 600, 300]}
        ]
        
        result = engine.reason("How many riders not wearing helmets?", detections)
        
        assert result.status == 'ok'
        assert '2' in result.answer or 'no-helmet' in result.answer.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
