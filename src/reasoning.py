"""
Reasoning Layer for Helmet Detection
Hand-written intent routing and structured reasoning (no frameworks)
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result from the reasoning layer."""
    answer: str
    status: str  # 'ok', 'insufficient_info', 'question_not_image_related'
    support: Dict[str, Any]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer': self.answer,
            'status': self.status,
            'support': self.support,
            'confidence_score': round(self.confidence_score, 4)
        }


class IntentRouter:
    """Routes user questions to appropriate handling."""
    
    # Patterns for image-related questions
    IMAGE_RELATED_PATTERNS = [
        r'\b(helmet|hat|headgear|head)\b',
        r'\b(rider|motorcyclist|person|people|human)\b',
        r'\b(motorcycle|bike|vehicle)\b',
        r'\b(wearing|not wearing|without)\b',
        r'\b(count|how many|number)\b',
        r'\b(detect(?:ed|ing|s)?|detection|identif(?:y|ied|ying|ies))\b',
        r'\b(safe|safety|compliance|violation)\b',
        r'\b(bounding.box|location|where)\b',
        r'\b(confident|confidence|sure)\b',
        r'\b(image|picture|photo|frame)\b',
    ]
    
    # Patterns for counting questions
    COUNTING_PATTERNS = [
        r'how many',
        r'number of',
        r'count',
        r'total',
    ]
    
    # Patterns for yes/no questions
    YES_NO_PATTERNS = [
        r'is there',
        r'are there',
        r'does.*have',
        r'can you see',
    ]
    
    # Patterns for existence/checking questions
    EXISTENCE_PATTERNS = [
        r'is anyone',
        r'are.*wearing',
        r'is.*wearing',
        r'not wearing',
        r'without',
    ]
    
    # Patterns for most common/commonality questions
    COMMON_PATTERNS = [
        r'most common',
        r'what.*most',
        r'which.*most',
        r'common object',
    ]
    
    def __init__(self):
        """Initialize intent router with compiled patterns."""
        self.image_patterns = [re.compile(p, re.IGNORECASE) for p in self.IMAGE_RELATED_PATTERNS]
        self.counting_patterns = [re.compile(p, re.IGNORECASE) for p in self.COUNTING_PATTERNS]
        self.yes_no_patterns = [re.compile(p, re.IGNORECASE) for p in self.YES_NO_PATTERNS]
        self.existence_patterns = [re.compile(p, re.IGNORECASE) for p in self.EXISTENCE_PATTERNS]
        self.common_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMMON_PATTERNS]
    
    def is_image_related(self, question: str) -> bool:
        """
        Determine if the question requires image analysis.
        
        Args:
            question: User's natural language question
        
        Returns:
            True if question is about the image, False otherwise
        """
        for pattern in self.image_patterns:
            if pattern.search(question):
                return True
        return False
    
    def get_question_type(self, question: str) -> str:
        """
        Classify the type of question.
        
        Args:
            question: User's natural language question
        
        Returns:
            Question type: 'counting', 'yes_no', 'existence', 'common', or 'general'
        """
        question_lower = question.lower()
        
        # Check counting patterns
        for pattern in self.counting_patterns:
            if pattern.search(question_lower):
                return 'counting'
        
        # Check yes/no patterns
        for pattern in self.yes_no_patterns:
            if pattern.search(question_lower):
                return 'yes_no'
        
        # Check existence patterns
        for pattern in self.existence_patterns:
            if pattern.search(question_lower):
                return 'existence'
        
        # Check common/most common patterns
        for pattern in self.common_patterns:
            if pattern.search(question_lower):
                return 'common'
        
        return 'general'
    
    def extract_target_class(self, question: str) -> Optional[str]:
        """
        Extract the target class from the question.
        
        Args:
            question: User's natural language question
        
        Returns:
            Target class name or None if not specified
        """
        question_lower = question.lower()
        
        # Class mappings — ORDER MATTERS: check specific classes before general
        # "no-helmet" must be checked before "helmet" to prevent substring false match
        class_keywords = {
            'no-helmet': ['no helmet', 'no-helmet', 'without helmet', 'without a helmet',
                         'not wearing helmet', 'not wearing a helmet',
                         'not wearing helmets', 'no headgear', 'bare head', 'unprotected'],
            'helmet': ['helmet', 'helmets', 'headgear', r'\bhat\b'],
            'motorcycle': ['motorcycle', 'bike', 'motorbike', 'vehicle']
        }
        
        for class_name, keywords in class_keywords.items():
            for keyword in keywords:
                if keyword.startswith(r'\b'):
                    # Word-boundary regex match
                    if re.search(keyword, question_lower):
                        return class_name
                elif keyword in question_lower:
                    return class_name
        
        return None


class StructuredReasoner:
    """Performs structured reasoning over detection results."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize structured reasoner.
        
        Args:
            confidence_threshold: Minimum confidence for reliable reasoning
        """
        self.confidence_threshold = confidence_threshold
    
    def analyze_detections(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Analyze detection results and compute statistics.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'total': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'has_violations': False,
                'reliable': False
            }
        
        # Count classes
        class_counts = {}
        for det in detections:
            cls = det.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Confidence statistics
        confidences = [det.get('confidence', 0) for det in detections]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        max_confidence = max(confidences) if confidences else 0
        
        # Check for violations
        has_violations = 'no-helmet' in class_counts
        
        # Determine if results are reliable
        reliable = (
            len(detections) > 0 and
            avg_confidence >= self.confidence_threshold and
            min_confidence >= self.confidence_threshold * 0.7  # Allow some tolerance
        )
        
        return {
            'total': len(detections),
            'class_counts': class_counts,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'has_violations': has_violations,
            'reliable': reliable
        }
    
    def count_by_class(self, detections: List[Dict], 
                       target_class: str) -> int:
        """
        Count detections of a specific class.
        
        Args:
            detections: List of detection dictionaries
            target_class: Class name to count
        
        Returns:
            Number of detections of the target class
        """
        return sum(1 for det in detections if det.get('class') == target_class)
    
    def get_most_common(self, detections: List[Dict]) -> Optional[str]:
        """
        Get the most common class in detections.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Most common class name or None
        """
        if not detections:
            return None
        
        class_counts = {}
        for det in detections:
            cls = det.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        if not class_counts:
            return None
        
        return max(class_counts.items(), key=lambda x: x[1])[0]
    
    def has_specific_class(self, detections: List[Dict], 
                          target_class: str) -> bool:
        """
        Check if a specific class is present in detections.
        
        Args:
            detections: List of detection dictionaries
            target_class: Class name to check
        
        Returns:
            True if class is present
        """
        return any(det.get('class') == target_class for det in detections)


class ReasoningEngine:
    """Main reasoning engine combining intent routing and structured reasoning."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize reasoning engine.
        
        Args:
            confidence_threshold: Minimum confidence for reliable reasoning
        """
        self.intent_router = IntentRouter()
        self.structured_reasoner = StructuredReasoner(confidence_threshold)
        self.confidence_threshold = confidence_threshold
    
    def reason(self, question: str, detections: List[Dict]) -> ReasoningResult:
        """
        Perform reasoning on a question given detection results.
        
        Args:
            question: User's natural language question
            detections: List of detection dictionaries from Part A
        
        Returns:
            ReasoningResult with answer and support information
        """
        # Step 1: Intent routing - check if question is image-related
        if not self.intent_router.is_image_related(question):
            return ReasoningResult(
                answer="This question is not related to the image content.",
                status='question_not_image_related',
                support={'question': question},
                confidence_score=1.0
            )
        
        # Step 2: Analyze detections
        stats = self.structured_reasoner.analyze_detections(detections)
        
        # Step 3: Check confidence guardrail
        if not stats['reliable'] and len(detections) > 0:
            return ReasoningResult(
                answer="Insufficient information to answer confidently.",
                status='insufficient_info',
                support={
                    'detections': detections,
                    'avg_confidence': stats['avg_confidence'],
                    'min_confidence': stats['min_confidence'],
                    'threshold': self.confidence_threshold,
                    'reason': f"Average confidence {stats['avg_confidence']:.2f} is below threshold {self.confidence_threshold}"
                },
                confidence_score=stats['avg_confidence']
            )
        
        # Step 4: Get question type and extract target class
        question_type = self.intent_router.get_question_type(question)
        target_class = self.intent_router.extract_target_class(question)
        
        # Step 5: Structured reasoning based on question type
        if question_type == 'counting':
            return self._handle_counting(question, detections, stats, target_class)
        elif question_type == 'yes_no':
            return self._handle_yes_no(question, detections, stats, target_class)
        elif question_type == 'existence':
            return self._handle_existence(question, detections, stats, target_class)
        elif question_type == 'common':
            return self._handle_common(question, detections, stats)
        else:
            return self._handle_general(question, detections, stats)
    
    def _handle_counting(self, question: str, detections: List[Dict],
                         stats: Dict, target_class: Optional[str]) -> ReasoningResult:
        """Handle counting questions."""
        if target_class:
            count = self.structured_reasoner.count_by_class(detections, target_class)
            answer = f"{count} {target_class} detected"
        else:
            count = stats['total']
            answer = f"{count} objects detected in total"
        
        return ReasoningResult(
            answer=answer,
            status='ok',
            support={
                'detections': detections,
                'class_counts': stats['class_counts'],
                'target_class': target_class
            },
            confidence_score=stats['avg_confidence']
        )
    
    def _handle_yes_no(self, question: str, detections: List[Dict],
                       stats: Dict, target_class: Optional[str]) -> ReasoningResult:
        """Handle yes/no questions."""
        if target_class:
            has_class = self.structured_reasoner.has_specific_class(detections, target_class)
            if has_class:
                answer = f"Yes, {target_class} is present"
            else:
                answer = f"No, {target_class} is not detected"
        else:
            if stats['total'] > 0:
                answer = "Yes, objects are detected in the image"
            else:
                answer = "No objects are detected in the image"
        
        return ReasoningResult(
            answer=answer,
            status='ok',
            support={
                'detections': detections,
                'class_counts': stats['class_counts']
            },
            confidence_score=stats['avg_confidence']
        )
    
    def _handle_existence(self, question: str, detections: List[Dict],
                          stats: Dict, target_class: Optional[str]) -> ReasoningResult:
        """Handle existence/checking questions."""
        if target_class == 'no-helmet':
            count = self.structured_reasoner.count_by_class(detections, 'no-helmet')
            if count > 0:
                answer = f"{count} rider(s) not wearing helmet(s)"
            else:
                answer = "All riders are wearing helmets"
        elif target_class:
            count = self.structured_reasoner.count_by_class(detections, target_class)
            if count > 0:
                answer = f"{count} {target_class} detected"
            else:
                answer = f"No {target_class} detected"
        else:
            answer = f"{stats['total']} objects detected"
        
        return ReasoningResult(
            answer=answer,
            status='ok',
            support={
                'detections': detections,
                'class_counts': stats['class_counts'],
                'has_violations': stats['has_violations']
            },
            confidence_score=stats['avg_confidence']
        )
    
    def _handle_common(self, question: str, detections: List[Dict],
                       stats: Dict) -> ReasoningResult:
        """Handle most common object questions."""
        most_common = self.structured_reasoner.get_most_common(detections)
        
        if most_common:
            count = stats['class_counts'].get(most_common, 0)
            answer = f"The most common object is '{most_common}' with {count} detection(s)"
        else:
            answer = "No objects detected to determine the most common"
        
        return ReasoningResult(
            answer=answer,
            status='ok',
            support={
                'detections': detections,
                'class_counts': stats['class_counts'],
                'most_common': most_common
            },
            confidence_score=stats['avg_confidence']
        )
    
    def _handle_general(self, question: str, detections: List[Dict],
                        stats: Dict) -> ReasoningResult:
        """Handle general questions."""
        if stats['total'] == 0:
            answer = "No objects detected in the image"
        else:
            classes = list(stats['class_counts'].keys())
            answer = f"Detected {stats['total']} objects: {', '.join(classes)}"
        
        return ReasoningResult(
            answer=answer,
            status='ok',
            support={
                'detections': detections,
                'class_counts': stats['class_counts'],
                'total': stats['total']
            },
            confidence_score=stats['avg_confidence']
        )


# Global reasoning engine instance
_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine(confidence_threshold: float = 0.5) -> ReasoningEngine:
    """
    Get or initialize the global reasoning engine.
    
    Args:
        confidence_threshold: Minimum confidence for reliable reasoning
    
    Returns:
        ReasoningEngine instance
    """
    global _reasoning_engine
    
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine(confidence_threshold)
    
    return _reasoning_engine


def answer_question(question: str, detections: List[Dict],
                    confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Convenience function to answer a question about detections.
    
    Args:
        question: User's natural language question
        detections: List of detection dictionaries
        confidence_threshold: Minimum confidence for reliable reasoning
    
    Returns:
        Dictionary with answer and support information
    """
    engine = get_reasoning_engine(confidence_threshold)
    result = engine.reason(question, detections)
    return result.to_dict()


if __name__ == '__main__':
    # Test reasoning
    test_detections = [
        {'class': 'helmet', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]},
        {'class': 'no-helmet', 'confidence': 0.87, 'bbox': [300, 150, 400, 250]},
        {'class': 'motorcycle', 'confidence': 0.95, 'bbox': [50, 50, 450, 350]}
    ]
    
    test_questions = [
        "How many riders are not wearing helmets?",
        "Is anyone not wearing a helmet?",
        "What is the most common object?",
        "What is the weather today?"
    ]
    
    for question in test_questions:
        result = answer_question(question, test_detections)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Status: {result['status']}")
