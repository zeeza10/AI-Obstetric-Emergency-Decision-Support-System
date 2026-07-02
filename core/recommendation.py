"""Recommendation formatting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from .risk_engine import AssessmentResult


def get_clinical_guidance(risk_level: str) -> Tuple[str, str]:
    """Return a clinical explanation and recommended action for a given risk level."""
    guidance = {
        "Low Risk": (
            "The reported symptoms do not currently suggest an acute obstetric emergency.",
            "Continue routine monitoring and seek medical attention if symptoms worsen.",
        ),
        "Moderate Risk": (
            "The patient has concerning symptoms that require prompt medical evaluation and close monitoring.",
            "Arrange prompt medical assessment and monitor the patient closely.",
        ),
        "High Risk": (
            "The symptoms indicate a potentially dangerous obstetric condition that needs urgent clinical review.",
            "Seek urgent hospital assessment now and prepare for immediate medical intervention.",
        ),
        "Critical Risk": (
            "The presentation suggests a severe obstetric emergency requiring immediate intervention.",
            "Call emergency services immediately and arrange urgent hospital transfer with continuous monitoring.",
        ),
    }
    return guidance.get(risk_level, guidance["Moderate Risk"])


def build_recommendation_summary(result: AssessmentResult) -> Dict[str, str]:
    """Convert an assessment result into a user-friendly summary dictionary."""
    return {
        "Risk Level": result.risk_level,
        "Risk Score": str(result.risk_score),
        "Confidence Score": str(result.confidence_score),
        "Clinical Explanation": result.clinical_explanation,
        "Recommended Immediate Action": result.recommended_action,
    }
