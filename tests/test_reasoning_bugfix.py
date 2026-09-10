"""Regression tests for reasoning layer keyword matching bugs.

Covers: hyphenated forms, plurals, articles, verb tenses, and
correct class priority (no-helmet checked before helmet).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning import IntentRouter, ReasoningEngine


class TestExtractTargetClass:
    """Verify extract_target_class handles hyphens, plurals, articles."""

    def setup_method(self):
        self.router = IntentRouter()

    def test_hyphenated_no_helmet(self):
        assert self.router.extract_target_class("Any no-helmet?") == 'no-helmet'

    def test_plural_helmets(self):
        assert self.router.extract_target_class("How many helmets?") == 'helmet'

    def test_not_wearing_helmets_plural(self):
        assert self.router.extract_target_class("How many riders not wearing helmets?") == 'no-helmet'

    def test_not_wearing_a_helmet(self):
        assert self.router.extract_target_class("Is anyone not wearing a helmet?") == 'no-helmet'

    def test_without_a_helmet(self):
        assert self.router.extract_target_class("Riders without a helmet") == 'no-helmet'

    def test_motorcycle_still_works(self):
        assert self.router.extract_target_class("Count motorcycles") == 'motorcycle'

    def test_plain_helmet(self):
        assert self.router.extract_target_class("How many helmets?") == 'helmet'


class TestImageRelated:
    """Verify is_image_related matches verb forms."""

    def setup_method(self):
        self.router = IntentRouter()

    def test_detected(self):
        assert self.router.is_image_related("What objects are detected?") is True

    def test_detecting(self):
        assert self.router.is_image_related("What are you detecting?") is True

    def test_identified(self):
        assert self.router.is_image_related("What was identified?") is True


class TestEndToEnd:
    """Full reasoning engine integration test for violation questions."""

    def test_violation_counting(self):
        engine = ReasoningEngine(confidence_threshold=0.5)
        detections = [
            {'class': 'helmet', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]},
            {'class': 'no-helmet', 'confidence': 0.87, 'bbox': [300, 150, 400, 250]},
            {'class': 'no-helmet', 'confidence': 0.82, 'bbox': [500, 200, 600, 300]},
        ]
        result = engine.reason("How many riders not wearing helmets?", detections)
        assert result.status == 'ok'
        assert '2' in result.answer

    def test_violation_existence(self):
        engine = ReasoningEngine(confidence_threshold=0.5)
        detections = [
            {'class': 'no-helmet', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]},
        ]
        result = engine.reason("Is anyone not wearing a helmet?", detections)
        assert result.status == 'ok'
        assert 'not wearing' in result.answer.lower() or 'no-helmet' in result.answer.lower()


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
