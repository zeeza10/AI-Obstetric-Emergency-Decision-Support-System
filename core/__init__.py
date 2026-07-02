"""Core package for the obstetric emergency decision support system."""

from .patient import PatientInfo
from .risk_engine import AssessmentResult, ModelRiskEngine, RiskEngine, RuleBasedRiskEngine
from .recommendation import build_recommendation_summary
from .logger import setup_logger
from .validators import (
    InvalidChoiceError,
    InvalidNumericValueError,
    InvalidYesNoError,
    MissingValueError,
    ValidationError,
    parse_patient_from_form_data,
    validate_patient_information,
)

__all__ = [
    "PatientInfo",
    "AssessmentResult",
    "ModelRiskEngine",
    "RiskEngine",
    "RuleBasedRiskEngine",
    "build_recommendation_summary",
    "setup_logger",
    "parse_patient_from_form_data",
    "validate_patient_information",
    "ValidationError",
    "MissingValueError",
    "InvalidNumericValueError",
    "InvalidYesNoError",
    "InvalidChoiceError",
]
