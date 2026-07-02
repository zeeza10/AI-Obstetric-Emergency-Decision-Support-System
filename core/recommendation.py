"""Recommendation formatting helpers."""

from __future__ import annotations

from typing import Dict

from .risk_engine import AssessmentResult


def build_recommendation_summary(result: AssessmentResult) -> Dict[str, str]:
    """Convert an assessment result into a user-friendly summary dictionary."""
    return {
        "Risk Level": result.risk_level,
        "Risk Score": str(result.risk_score),
        "Clinical Explanation": result.clinical_explanation,
        "Recommended Immediate Action": result.recommended_action,
    }
