"""Reasoning Layer for Helmet Detection — intent routing + structured reasoning."""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    answer: str
    status: str
    support: Dict[str, Any]
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "support": self.support,
            "confidence_score": round(self.confidence_score, 4),
        }


class IntentRouter:
    IMAGE_RELATED_PATTERNS = [
        r"\b(helmet|hat|headgear|head)\b",
        r"\b(rider|motorcyclist|person|people|human)\b",
        r"\b(motorcycle|bike|vehicle)\b",
        r"\b(wearing|not wearing|without)\b",
        r"\b(count|how many|number)\b",
        r"\b(detect(?:ed|ing|s)?|detection|identif(?:y|ied|ying|ies))\b",
        r"\b(safe|safety|compliance|violation)\b",
        r"\b(bounding.box|location|where)\b",
        r"\b(confident|confidence|sure)\b",
        r"\b(image|picture|photo|frame)\b",
    ]

    COUNTING_PATTERNS = [r"how many", r"number of", r"count", r"total"]
    YES_NO_PATTERNS = [r"is there", r"are there", r"does.*have", r"can you see"]
    EXISTENCE_PATTERNS = [r"is anyone", r"are.*wearing", r"is.*wearing", r"not wearing", r"without"]
    COMMON_PATTERNS = [r"most common", r"what.*most", r"which.*most", r"common object"]

    def __init__(self):
        self.image_patterns = [re.compile(p, re.IGNORECASE) for p in self.IMAGE_RELATED_PATTERNS]
        self.counting_patterns = [re.compile(p, re.IGNORECASE) for p in self.COUNTING_PATTERNS]
        self.yes_no_patterns = [re.compile(p, re.IGNORECASE) for p in self.YES_NO_PATTERNS]
        self.existence_patterns = [re.compile(p, re.IGNORECASE) for p in self.EXISTENCE_PATTERNS]
        self.common_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMMON_PATTERNS]

    def is_image_related(self, question: str) -> bool:
        return any(p.search(question) for p in self.image_patterns)

    def get_question_type(self, question: str) -> str:
        q = question.lower()
        for patterns, qtype in [
            (self.counting_patterns, "counting"),
            (self.yes_no_patterns, "yes_no"),
            (self.existence_patterns, "existence"),
            (self.common_patterns, "common"),
        ]:
            if any(p.search(q) for p in patterns):
                return qtype
        return "general"

    def extract_target_class(self, question: str) -> Optional[str]:
        q = question.lower()
        # ORDER MATTERS: specific classes before general to prevent substring false match
        class_keywords = {
            "no-helmet": [
                "no helmet", "no-helmet", "without helmet", "without a helmet",
                "not wearing helmet", "not wearing a helmet", "not wearing helmets",
                "no headgear", "bare head", "unprotected",
            ],
            "helmet": ["helmet", "helmets", "headgear", r"\bhat\b"],
            "motorcycle": ["motorcycle", "bike", "motorbike", "vehicle"],
        }
        for class_name, keywords in class_keywords.items():
            for kw in keywords:
                if kw.startswith(r"\b"):
                    if re.search(kw, q):
                        return class_name
                elif kw in q:
                    return class_name
        return None


class StructuredReasoner:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def analyze_detections(self, detections: List[Dict]) -> Dict[str, Any]:
        if not detections:
            return {
                "total": 0, "class_counts": {}, "avg_confidence": 0.0,
                "min_confidence": 0.0, "max_confidence": 0.0,
                "has_violations": False, "reliable": False,
            }

        class_counts: Dict[str, int] = {}
        for det in detections:
            cls = det.get("class", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1

        confidences = [det.get("confidence", 0) for det in detections]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)

        reliable = (
            len(detections) > 0
            and avg_conf >= self.confidence_threshold
            and min_conf >= self.confidence_threshold * 0.7
        )

        return {
            "total": len(detections),
            "class_counts": class_counts,
            "avg_confidence": avg_conf,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
            "has_violations": "no-helmet" in class_counts,
            "reliable": reliable,
        }

    def count_by_class(self, detections: List[Dict], target_class: str) -> int:
        return sum(1 for d in detections if d.get("class") == target_class)

    def get_most_common(self, detections: List[Dict]) -> Optional[str]:
        if not detections:
            return None
        counts: Dict[str, int] = {}
        for d in detections:
            cls = d.get("class", "unknown")
            counts[cls] = counts.get(cls, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def has_specific_class(self, detections: List[Dict], target_class: str) -> bool:
        return any(d.get("class") == target_class for d in detections)


class ReasoningEngine:
    def __init__(self, confidence_threshold: float = 0.5):
        self.intent_router = IntentRouter()
        self.structured_reasoner = StructuredReasoner(confidence_threshold)
        self.confidence_threshold = confidence_threshold

    def reason(self, question: str, detections: List[Dict]) -> ReasoningResult:
        if not self.intent_router.is_image_related(question):
            return ReasoningResult(
                answer="This question is not related to the image content.",
                status="question_not_image_related",
                support={"question": question},
                confidence_score=1.0,
            )

        stats = self.structured_reasoner.analyze_detections(detections)

        if not stats["reliable"] and len(detections) > 0:
            return ReasoningResult(
                answer="Insufficient information to answer confidently.",
                status="insufficient_info",
                support={
                    "detections": detections,
                    "avg_confidence": stats["avg_confidence"],
                    "min_confidence": stats["min_confidence"],
                    "threshold": self.confidence_threshold,
                    "reason": f"Average confidence {stats['avg_confidence']:.2f} < threshold {self.confidence_threshold}",
                },
                confidence_score=stats["avg_confidence"],
            )

        q_type = self.intent_router.get_question_type(question)
        target = self.intent_router.extract_target_class(question)

        handlers = {
            "counting": self._handle_counting,
            "yes_no": self._handle_yes_no,
            "existence": self._handle_existence,
            "common": self._handle_common,
        }
        handler = handlers.get(q_type, self._handle_general)
        return handler(question, detections, stats, target)

    def _handle_counting(self, question, detections, stats, target_class):
        if target_class:
            count = self.structured_reasoner.count_by_class(detections, target_class)
            answer = f"{count} {target_class} detected"
        else:
            count = stats["total"]
            answer = f"{count} objects detected in total"
        return ReasoningResult(answer=answer, status="ok",
                               support={"detections": detections, "class_counts": stats["class_counts"], "target_class": target_class},
                               confidence_score=stats["avg_confidence"])

    def _handle_yes_no(self, question, detections, stats, target_class):
        if target_class:
            present = self.structured_reasoner.has_specific_class(detections, target_class)
            answer = f"Yes, {target_class} is present" if present else f"No, {target_class} is not detected"
        else:
            answer = "Yes, objects are detected" if stats["total"] > 0 else "No objects detected"
        return ReasoningResult(answer=answer, status="ok",
                               support={"detections": detections, "class_counts": stats["class_counts"]},
                               confidence_score=stats["avg_confidence"])

    def _handle_existence(self, question, detections, stats, target_class):
        if target_class == "no-helmet":
            count = self.structured_reasoner.count_by_class(detections, "no-helmet")
            answer = f"{count} rider(s) not wearing helmet(s)" if count > 0 else "All riders are wearing helmets"
        elif target_class:
            count = self.structured_reasoner.count_by_class(detections, target_class)
            answer = f"{count} {target_class} detected" if count > 0 else f"No {target_class} detected"
        else:
            answer = f"{stats['total']} objects detected"
        return ReasoningResult(answer=answer, status="ok",
                               support={"detections": detections, "class_counts": stats["class_counts"], "has_violations": stats["has_violations"]},
                               confidence_score=stats["avg_confidence"])

    def _handle_common(self, question, detections, stats, target_class=None):
        most_common = self.structured_reasoner.get_most_common(detections)
        if most_common:
            count = stats["class_counts"].get(most_common, 0)
            answer = f"The most common object is '{most_common}' with {count} detection(s)"
        else:
            answer = "No objects detected to determine the most common"
        return ReasoningResult(answer=answer, status="ok",
                               support={"detections": detections, "class_counts": stats["class_counts"], "most_common": most_common},
                               confidence_score=stats["avg_confidence"])

    def _handle_general(self, question, detections, stats, target_class=None):
        if stats["total"] == 0:
            answer = "No objects detected in the image"
        else:
            answer = f"Detected {stats['total']} objects: {', '.join(stats['class_counts'].keys())}"
        return ReasoningResult(answer=answer, status="ok",
                               support={"detections": detections, "class_counts": stats["class_counts"], "total": stats["total"]},
                               confidence_score=stats["avg_confidence"])


_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine(confidence_threshold: float = 0.5) -> ReasoningEngine:
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine(confidence_threshold)
    return _reasoning_engine


def answer_question(question: str, detections: List[Dict], confidence_threshold: float = 0.5) -> Dict[str, Any]:
    return get_reasoning_engine(confidence_threshold).reason(question, detections).to_dict()


if __name__ == "__main__":
    test_detections = [
        {"class": "helmet", "confidence": 0.92, "bbox": [100, 100, 200, 200]},
        {"class": "no-helmet", "confidence": 0.87, "bbox": [300, 150, 400, 250]},
        {"class": "motorcycle", "confidence": 0.95, "bbox": [50, 50, 450, 350]},
    ]
    for q in ["How many riders are not wearing helmets?", "Is anyone not wearing a helmet?",
              "What is the most common object?", "What is the weather today?"]:
        r = answer_question(q, test_detections)
        print(f"\nQ: {q}\nA: {r['answer']}\nStatus: {r['status']}")
