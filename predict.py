"""Reusable rule-based prediction module for obstetric emergency triage.

This module exposes a clean, production-oriented interface for assessing
obstetric emergency risk without collecting user input directly. It is intended
for reuse by console applications, web services, or future machine-learning
backends.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

from core import (
    AssessmentResult,
    ModelRiskEngine,
    PatientInfo,
    RiskEngine,
    RuleBasedRiskEngine,
    validate_patient_information,
)

__all__ = [
    "PatientInfo",
    "AssessmentResult",
    "RiskEngine",
    "RuleBasedRiskEngine",
    "ModelRiskEngine",
    "create_patient",
    "predict_from_form_data",
    "predict_risk",
]


def create_patient(
    age: int,
    pregnancy_weeks: int,
    heavy_bleeding: bool,
    severe_abdominal_pain: bool,
    blood_pressure: int,
    body_temperature: float,
    fetal_movement: str,
    consciousness: str,
) -> PatientInfo:
    """Create a PatientInfo instance from explicit clinical inputs."""
    return PatientInfo(
        age=age,
        pregnancy_weeks=pregnancy_weeks,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        blood_pressure=blood_pressure,
        body_temperature=body_temperature,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )


def predict_from_patient(patient: PatientInfo, engine: Optional[RiskEngine] = None) -> AssessmentResult:
    """Assess risk for a preconstructed patient object."""
    validate_patient_information(patient)
    active_engine = engine or ModelRiskEngine()
    return active_engine.assess_risk(patient)


def predict_from_form_data(
    form_data: Mapping[str, object],
    engine: Optional[RiskEngine] = None,
) -> Tuple[AssessmentResult, PatientInfo]:
    """Assess risk from a request-like form mapping while keeping parsing centralized."""
    age = int(str(form_data.get("age", "0")))
    pregnancy_weeks = int(str(form_data.get("pregnancy_weeks", "0")))
    heavy_bleeding = str(form_data.get("heavy_bleeding", "false")) == "true"
    severe_abdominal_pain = str(form_data.get("severe_abdominal_pain", "false")) == "true"
    blood_pressure = int(str(form_data.get("blood_pressure", "0")))
    body_temperature = float(str(form_data.get("body_temperature", "0")))
    fetal_movement = str(form_data.get("fetal_movement", "Normal"))
    consciousness = str(form_data.get("consciousness", "Alert"))

    patient = create_patient(
        age=age,
        pregnancy_weeks=pregnancy_weeks,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        blood_pressure=blood_pressure,
        body_temperature=body_temperature,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )
    result = predict_from_patient(patient, engine=engine)
    return result, patient


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
    patient = create_patient(
        age=age,
        pregnancy_weeks=pregnancy_weeks,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        blood_pressure=blood_pressure,
        body_temperature=body_temperature,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )
    return predict_from_patient(patient, engine=engine)
