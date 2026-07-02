"""Core package for the obstetric emergency decision support system."""

from .patient import PatientInfo
from .risk_engine import AssessmentResult, RiskEngine, RuleBasedRiskEngine
from .recommendation import build_recommendation_summary
from .logger import setup_logger
from .validators import (
    InvalidChoiceError,
    InvalidNumericValueError,
    InvalidYesNoError,
    MissingValueError,
    ValidationError,
    validate_patient_information,
)

__all__ = [
    "PatientInfo",
    "AssessmentResult",
    "RiskEngine",
    "RuleBasedRiskEngine",
    "build_recommendation_summary",
    "setup_logger",
    "validate_patient_information",
    "ValidationError",
    "MissingValueError",
    "InvalidNumericValueError",
    "InvalidYesNoError",
    "InvalidChoiceError",
]
