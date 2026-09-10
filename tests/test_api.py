"""Integration tests for Helmet Detection API endpoints."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    from PIL import Image
    import io
    img = Image.new("RGB", (640, 640), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "/detect" in data["endpoints"]
        assert "/ask" in data["endpoints"]


class TestDetectEndpoint:
    def test_missing_file(self, client):
        assert client.post("/detect").status_code == 422

    def test_invalid_file_type(self, client):
        resp = client.post("/detect", files={"file": ("test.txt", b"not an image", "text/plain")})
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_valid_image_format(self, client, sample_image_bytes):
        resp = client.post("/detect", files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")})
        assert resp.status_code != 400


class TestAskEndpoint:
    def test_missing_file(self, client):
        resp = client.post("/ask", data={"question": "How many helmets?"})
        assert resp.status_code == 422

    def test_missing_question(self, client, sample_image_bytes):
        resp = client.post("/ask", files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")})
        assert resp.status_code == 422

    def test_invalid_file_type(self, client):
        resp = client.post(
            "/ask",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"question": "How many helmets?"},
        )
        assert resp.status_code == 400


class TestReasoningIntegration:
    def test_unrelated_question(self):
        from src.reasoning import answer_question
        result = answer_question("What is the capital of France?", [{"class": "helmet", "confidence": 0.9}])
        assert result["status"] == "question_not_image_related"

    def test_counting_question(self):
        from src.reasoning import answer_question
        dets = [
            {"class": "helmet", "confidence": 0.9},
            {"class": "helmet", "confidence": 0.85},
            {"class": "no-helmet", "confidence": 0.8},
        ]
        result = answer_question("How many helmets?", dets)
        assert result["status"] == "ok"
        assert "2" in result["answer"]

    def test_violation_detection(self):
        from src.reasoning import answer_question
        dets = [{"class": "no-helmet", "confidence": 0.9, "bbox": [100, 100, 200, 200]}]
        result = answer_question("Is anyone not wearing a helmet?", dets)
        assert result["status"] == "ok"
        assert "not wearing" in result["answer"].lower() or "no-helmet" in result["answer"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
