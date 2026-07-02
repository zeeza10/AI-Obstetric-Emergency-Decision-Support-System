"""Audit logging for clinical assessment events."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from .patient import PatientInfo
from .risk_engine import AssessmentResult


def setup_audit_logger(log_file: str = "logs/audit.log") -> logging.Logger:
    """Create a dedicated audit logger for assessment traceability."""
    from .logger import setup_logger

    return setup_logger("obstetric_audit", log_file=log_file)


def build_audit_record(
    patient: PatientInfo,
    result: AssessmentResult,
    source: str,
) -> Dict[str, Any]:
    """Build a structured audit record for an assessment event."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "patient": patient.to_summary_dict(),
        "assessment": {
            "risk_level": result.risk_level,
            "risk_score": result.risk_score,
            "confidence_score": result.confidence_score,
            "recommended_action": result.recommended_action,
        },
    }


def log_assessment_audit(
    logger: logging.Logger,
    patient: PatientInfo,
    result: AssessmentResult,
    source: str,
) -> None:
    """Write a structured audit entry for a completed assessment."""
    record = build_audit_record(patient, result, source=source)
    logger.info(json.dumps(record, sort_keys=True))
