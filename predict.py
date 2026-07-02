"""Reusable rule-based prediction module for obstetric emergency triage.

This module exposes a clean, production-oriented interface for assessing
obstetric emergency risk without collecting user input directly. It is intended
for reuse by console applications, web services, or future machine-learning
backends.
"""

from __future__ import annotations

from typing import Optional

from core import AssessmentResult, PatientInfo, RiskEngine, RuleBasedRiskEngine, validate_patient_information


def predict_risk(
    age: int,
    pregnancy_weeks: int,
    heavy_bleeding: bool,
    severe_abdominal_pain: bool,
    blood_pressure: int,
    body_temperature: float,
    fetal_movement: str,
    consciousness: str,
    engine: Optional[RiskEngine] = None,
) -> AssessmentResult:
    """Assess obstetric emergency risk from explicit patient parameters.

    Parameters
    ----------
    age: int
        Patient age in years.
    pregnancy_weeks: int
        Gestational age in weeks.
    heavy_bleeding: bool
        Whether the patient reports heavy bleeding.
    severe_abdominal_pain: bool
        Whether the patient reports severe abdominal pain.
    blood_pressure: int
        Systolic blood pressure.
    body_temperature: float
        Body temperature in degrees Celsius.
    fetal_movement: str
        One of: Normal, Reduced, Absent.
    consciousness: str
        One of: Alert, Drowsy, Unconscious.
    engine: Optional[RiskEngine]
        Optional custom engine implementation. Defaults to the built-in rule-based engine.

    Returns
    -------
    AssessmentResult
        Structured risk assessment output.
    """
    patient = PatientInfo(
        age=age,
        pregnancy_weeks=pregnancy_weeks,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        blood_pressure=blood_pressure,
        body_temperature=body_temperature,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )
    validate_patient_information(patient)
    active_engine = engine or RuleBasedRiskEngine()
    return active_engine.assess_risk(patient)
