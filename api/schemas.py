"""
Pydantic schemas for Helmet Detection API
Defines request/response models for all endpoints

NOTE: This file contains RICHER schemas for potential future endpoints.
The current /detect and /ask endpoints use simpler schemas defined in main.py.
This approach allows for backward compatibility while providing advanced
schemas for enhanced API functionality.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ComplianceStatus(str, Enum):
    """Compliance status enumeration."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_INFO = "insufficient_info"
    UNKNOWN = "unknown"


class ResponseStatus(str, Enum):
    """API response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    INSUFFICIENT_INFO = "insufficient_info"


class IntentType(str, Enum):
    """Question intent type enumeration."""
    COUNT = "count"
    COMPLIANCE_CHECK = "compliance_check"
    DETECTION_LIST = "detection_list"
    SPATIAL_QUERY = "spatial_query"
    CONFIDENCE_QUERY = "confidence_query"
    OFF_TOPIC = "off_topic"


class BoundingBoxResponse(BaseModel):
    """Bounding box response model."""
    x1: float = Field(..., description="Left x coordinate (normalized 0-1)")
    y1: float = Field(..., description="Top y coordinate (normalized 0-1)")
    x2: float = Field(..., description="Right x coordinate (normalized 0-1)")
    y2: float = Field(..., description="Bottom y coordinate (normalized 0-1)")


class DetectionResponse(BaseModel):
    """Single detection response model."""
    class_id: int = Field(..., description="Class ID (0=helmet, 1=no-helmet, 2=motorcycle)")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bbox: BoundingBoxResponse = Field(..., description="Bounding box coordinates")


class DetectionSummaryResponse(BaseModel):
    """Detection summary response model."""
    total_detections: int = Field(..., description="Total number of detections")
    helmets: int = Field(0, description="Number of helmet detections")
    no_helmets: int = Field(0, description="Number of no-helmet detections")
    motorcycles: int = Field(0, description="Number of motorcycle detections")
    compliance_rate: float = Field(0.0, ge=0, le=1, description="Compliance rate (0-1)")
    compliance_status: ComplianceStatus = Field(..., description="Compliance status")


class DetectResponse(BaseModel):
    """Detection endpoint response model."""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    detections: List[DetectionResponse] = Field(default_factory=list, description="List of detections")
    summary: Optional[DetectionSummaryResponse] = Field(None, description="Detection summary")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class AskResponse(BaseModel):
    """Question answering endpoint response model."""
    status: ResponseStatus = Field(..., description="Response status")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    intent: IntentType = Field(..., description="Detected question intent")
    confidence: float = Field(0.0, ge=0, le=1, description="Answer confidence")
    detections_used: int = Field(0, description="Number of detections used")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check endpoint response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    weights_path: Optional[str] = Field(None, description="Path to model weights")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = Field("error", description="Error status")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error detail")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
