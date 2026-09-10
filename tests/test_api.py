"""
Integration Tests for Helmet Detection API
Tests end-to-end API functionality
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a sample image for testing."""
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new('RGB', (640, 640), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.read()


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns valid response."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'timestamp' in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'endpoints' in data
        assert '/detect' in data['endpoints']
        assert '/ask' in data['endpoints']


class TestDetectEndpoint:
    """Test /detect endpoint."""
    
    def test_detect_missing_file(self, client):
        """Test detect endpoint without file returns error."""
        response = client.post("/detect")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_detect_invalid_file_type(self, client):
        """Test detect endpoint with invalid file type."""
        response = client.post(
            "/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()['detail']
    
    def test_detect_valid_image_format(self, client, sample_image_bytes):
        """Test detect endpoint accepts valid image format."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        # May fail if model not loaded, but should not be 400
        assert response.status_code != 400


class TestAskEndpoint:
    """Test /ask endpoint."""
    
    def test_ask_missing_file(self, client):
        """Test ask endpoint without file returns error."""
        response = client.post(
            "/ask",
            data={"question": "How many helmets?"}
        )
        
        assert response.status_code == 422
    
    def test_ask_missing_question(self, client, sample_image_bytes):
        """Test ask endpoint without question returns error."""
        response = client.post(
            "/ask",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 422
    
    def test_ask_invalid_file_type(self, client):
        """Test ask endpoint with invalid file type."""
        response = client.post(
            "/ask",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"question": "How many helmets?"}
        )
        
        assert response.status_code == 400


class TestReasoningLogic:
    """Test reasoning logic integration."""
    
    def test_unrelated_question(self):
        """Test that unrelated questions are handled correctly."""
        from src.reasoning import answer_question
        
        detections = [
            {'class': 'helmet', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]}
        ]
        
        result = answer_question("What is the capital of France?", detections)
        
        assert result['status'] == 'question_not_image_related'
    
    def test_counting_question(self):
        """Test counting question logic."""
        from src.reasoning import answer_question
        
        detections = [
            {'class': 'helmet', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]},
            {'class': 'helmet', 'confidence': 0.85, 'bbox': [300, 150, 400, 250]},
            {'class': 'no-helmet', 'confidence': 0.8, 'bbox': [500, 200, 600, 300]}
        ]
        
        result = answer_question("How many helmets?", detections)
        
        assert result['status'] == 'ok'
        assert '2' in result['answer']
    
    def test_violation_detection(self):
        """Test violation detection question."""
        from src.reasoning import answer_question
        
        detections = [
            {'class': 'no-helmet', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]}
        ]
        
        result = answer_question("Is anyone not wearing a helmet?", detections)
        
        assert result['status'] == 'ok'
        assert 'not wearing' in result['answer'].lower() or 'no-helmet' in result['answer'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
