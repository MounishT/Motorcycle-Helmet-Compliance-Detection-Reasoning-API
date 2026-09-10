"""Extended Pydantic schemas for future API endpoints.

The current /detect and /ask endpoints use simpler schemas in main.py.
These richer schemas are kept for potential future use.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_INFO = "insufficient_info"
    UNKNOWN = "unknown"


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    INSUFFICIENT_INFO = "insufficient_info"


class IntentType(str, Enum):
    COUNT = "count"
    COMPLIANCE_CHECK = "compliance_check"
    DETECTION_LIST = "detection_list"
    SPATIAL_QUERY = "spatial_query"
    CONFIDENCE_QUERY = "confidence_query"
    OFF_TOPIC = "off_topic"


class BoundingBoxResponse(BaseModel):
    x1: float = Field(..., description="Left x coordinate")
    y1: float = Field(..., description="Top y coordinate")
    x2: float = Field(..., description="Right x coordinate")
    y2: float = Field(..., description="Bottom y coordinate")


class DetectionResponse(BaseModel):
    class_id: int = Field(..., description="Class ID (0=helmet, 1=no-helmet, 2=motorcycle)")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bbox: BoundingBoxResponse = Field(..., description="Bounding box coordinates")


class DetectionSummaryResponse(BaseModel):
    total_detections: int
    helmets: int = 0
    no_helmets: int = 0
    motorcycles: int = 0
    compliance_rate: float = Field(0.0, ge=0, le=1)
    compliance_status: ComplianceStatus


class DetectResponse(BaseModel):
    status: ResponseStatus
    message: str
    detections: List[DetectionResponse] = Field(default_factory=list)
    summary: Optional[DetectionSummaryResponse] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class AskResponse(BaseModel):
    status: ResponseStatus
    question: str
    answer: str
    intent: IntentType
    confidence: float = Field(0.0, ge=0, le=1)
    detections_used: int = 0
    processing_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    weights_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
