"""FastAPI Application for Helmet Detection API."""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import cv2
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import get_detector
from src.reasoning import answer_question, get_reasoning_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
REASONING_CONFIDENCE_THRESHOLD = float(os.getenv("REASONING_CONFIDENCE_THRESHOLD", "0.5"))
MAX_FILE_SIZE = 10 * 1024 * 1024
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Helmet Detection API...")
    try:
        if os.path.exists(WEIGHTS_PATH):
            get_detector(WEIGHTS_PATH)
            get_reasoning_engine(REASONING_CONFIDENCE_THRESHOLD)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Weights file not found: {WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    yield
    logger.info("Shutting down Helmet Detection API...")


app = FastAPI(
    title="Motorcycle Helmet Compliance Detection API",
    description="RT-DETR based object detection for helmet compliance with reasoning layer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionResponse(BaseModel):
    detections: list = Field(description="List of detected objects")
    inference_ms: float = Field(description="Inference time in milliseconds")
    image_shape: list = Field(description="Image dimensions [height, width]")


class AskResponse(BaseModel):
    answer: str = Field(description="Natural language answer")
    status: str = Field(description="Response status")
    support: dict = Field(description="Supporting information")
    confidence_score: float = Field(description="Confidence score of the answer")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


def _validate_upload(file: UploadFile, contents: bytes):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Supported: {SUPPORTED_FORMATS}")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024 * 1024)}MB")


def _decode_image(contents: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@app.get("/", response_model=dict)
async def root():
    return {
        "name": "Motorcycle Helmet Compliance Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Detect objects in an image",
            "/ask": "POST - Ask questions about an image",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_loaded = os.path.exists(WEIGHTS_PATH)
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/detect", response_model=DetectionResponse)
def detect(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: Optional[float] = Form(None, description="Confidence threshold (0-1)"),
):
    start_time = time.time()
    contents = file.file.read()
    _validate_upload(file, contents)
    try:
        img = _decode_image(contents)
        detector = get_detector(WEIGHTS_PATH)
        conf = confidence if confidence is not None else CONFIDENCE_THRESHOLD
        result = detector.detect(img, conf_threshold=conf)
        total_ms = (time.time() - start_time) * 1000
        logger.info(f"Detection: {len(result.detections)} objects in {total_ms:.2f}ms")
        return DetectionResponse(
            detections=[d.to_dict() for d in result.detections],
            inference_ms=round(total_ms, 2),
            image_shape=list(result.image_shape),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(
    file: UploadFile = File(..., description="Image file to analyze"),
    question: str = Form(..., description="Natural language question about the image"),
    confidence: Optional[float] = Form(None, description="Confidence threshold (0-1)"),
):
    start_time = time.time()
    contents = file.file.read()
    _validate_upload(file, contents)
    try:
        img = _decode_image(contents)
        detector = get_detector(WEIGHTS_PATH)
        conf = confidence if confidence is not None else CONFIDENCE_THRESHOLD
        detection_result = detector.detect(img, conf_threshold=conf)
        detections = [d.to_dict() for d in detection_result.detections]
        reasoning_conf = confidence if confidence is not None else REASONING_CONFIDENCE_THRESHOLD
        reasoning_result = answer_question(question, detections, reasoning_conf)
        total_ms = (time.time() - start_time) * 1000
        logger.info(f"Ask: '{question}' -> {reasoning_result['status']} in {total_ms:.2f}ms")
        return AskResponse(
            answer=reasoning_result["answer"],
            status=reasoning_result["status"],
            support=reasoning_result["support"],
            confidence_score=reasoning_result["confidence_score"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    uvicorn.run("api.main:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Helmet Detection API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    start_server(args.host, args.port, args.reload)
