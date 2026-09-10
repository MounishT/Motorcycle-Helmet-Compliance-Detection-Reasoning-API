"""
FastAPI Application for Helmet Detection API
Exposes /detect and /ask endpoints
"""

import os
import sys
import json
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import HelmetDetector, get_detector
from src.reasoning import answer_question, get_reasoning_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
REASONING_CONFIDENCE_THRESHOLD = float(os.getenv("REASONING_CONFIDENCE_THRESHOLD", "0.5"))
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
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
    
    # Shutdown
    logger.info("Shutting down Helmet Detection API...")


# Initialize FastAPI app
app = FastAPI(
    title="Motorcycle Helmet Compliance Detection API",
    description="RT-DETR based object detection for helmet compliance with reasoning layer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionResponse(BaseModel):
    """Response model for /detect endpoint."""
    detections: list = Field(description="List of detected objects")
    inference_ms: float = Field(description="Inference time in milliseconds")
    image_shape: list = Field(description="Image dimensions [height, width]")


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""
    answer: str = Field(description="Natural language answer")
    status: str = Field(description="Response status: ok, insufficient_info, question_not_image_related")
    support: dict = Field(description="Supporting information")
    confidence_score: float = Field(description="Confidence score of the answer")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Motorcycle Helmet Compliance Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Detect objects in an image",
            "/ask": "POST - Ask questions about an image",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = os.path.exists(WEIGHTS_PATH)
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/detect", response_model=DetectionResponse)
def detect(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: Optional[float] = Form(None, description="Confidence threshold (0-1)")
):
    """
    Detect objects in an image using RT-DETR.
    
    Returns detected objects with bounding boxes, classes, and confidence scores.
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {SUPPORTED_FORMATS}"
        )
    
    # Validate file size
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    try:
        # Decode image directly from RAM (no disk I/O)
        nparr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Get detector
        detector = get_detector(WEIGHTS_PATH)
        
        # Determine confidence threshold
        conf_threshold = confidence if confidence is not None else CONFIDENCE_THRESHOLD
        
        # Run detection
        result = detector.detect(img_cv2, conf_threshold=conf_threshold)
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Detection completed: {len(result.detections)} objects in {total_time:.2f}ms")
        
        return DetectionResponse(
            detections=[d.to_dict() for d in result.detections],
            inference_ms=round(total_time, 2),
            image_shape=list(result.image_shape)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
def ask(
    file: UploadFile = File(..., description="Image file to analyze"),
    question: str = Form(..., description="Natural language question about the image"),
    confidence: Optional[float] = Form(None, description="Confidence threshold (0-1)")
):
    """
    Ask a natural language question about an image.
    
    The system will:
    1. Route the intent (image-related or not)
    2. Run detection on the image
    3. Reason over the detection results
    4. Return an answer with supporting information
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {SUPPORTED_FORMATS}"
        )
    
    # Validate file size
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    try:
        # Decode image directly from RAM (no disk I/O)
        nparr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Get detector
        detector = get_detector(WEIGHTS_PATH)
        
        # Determine confidence threshold
        conf_threshold = confidence if confidence is not None else CONFIDENCE_THRESHOLD
        
        # Run detection
        detection_result = detector.detect(img_cv2, conf_threshold=conf_threshold)
        detections = [d.to_dict() for d in detection_result.detections]
        
        # Run reasoning
        reasoning_confidence = confidence if confidence is not None else REASONING_CONFIDENCE_THRESHOLD
        reasoning_result = answer_question(question, detections, reasoning_confidence)
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Question answered: '{question}' -> {reasoning_result['status']} in {total_time:.2f}ms")
        
        return AskResponse(
            answer=reasoning_result['answer'],
            status=reasoning_result['status'],
            support=reasoning_result['support'],
            confidence_score=reasoning_result['confidence_score']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Helmet Detection API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    start_server(args.host, args.port, args.reload)
