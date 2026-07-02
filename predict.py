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
    parse_patient_from_form_data,
    validate_patient_information,
)

__all__ = [
    "PatientInfo",
    "AssessmentResult",
    "RiskEngine",
    "RuleBasedRiskEngine",
    "ModelRiskEngine",
    "create_patient",
    "get_default_engine",
    "parse_patient_from_form_data",
    "predict_from_form_data",
    "predict_from_patient",
    "predict_risk",
]


def create_patient(
    age: int,
    height_cm: int,
    weight_kg: float,
    pregnancy_weeks: int,
    gravida: int,
    parity: int,
    previous_c_section: bool,
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
        height_cm=height_cm,
        weight_kg=weight_kg,
        pregnancy_weeks=pregnancy_weeks,
        gravida=gravida,
        parity=parity,
        previous_c_section=previous_c_section,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        blood_pressure=blood_pressure,
        body_temperature=body_temperature,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )


def get_default_engine() -> RiskEngine:
    """Return the configured risk engine implementation."""
    from config import get_config

    config = get_config()
    if config.engine_type.lower() == "rule":
        return RuleBasedRiskEngine()
    return ModelRiskEngine(model_path=config.model_path or None)


def predict_from_patient(patient: PatientInfo, engine: Optional[RiskEngine] = None) -> AssessmentResult:
    """Assess risk for a preconstructed patient object."""
    validate_patient_information(patient)
    active_engine = engine or get_default_engine()
    return active_engine.assess_risk(patient)


def predict_from_form_data(
    form_data: Mapping[str, object],
    engine: Optional[RiskEngine] = None,
) -> Tuple[AssessmentResult, PatientInfo]:
    """Assess risk from a request-like form mapping while keeping parsing centralized."""
    patient = parse_patient_from_form_data(form_data)
    result = predict_from_patient(patient, engine=engine)
    return result, patient


def predict_risk(
    age: int,
    height_cm: int,
    weight_kg: float,
    pregnancy_weeks: int,
    gravida: int,
    parity: int,
    previous_c_section: bool,
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
    height_cm: int
        Patient height in centimeters.
    weight_kg: float
        Patient weight in kilograms.
    pregnancy_weeks: int
        Gestational age in weeks.
    gravida: int
        Total number of pregnancies including the current one.
    parity: int
        Number of previous births at or beyond 20 weeks.
    previous_c_section: bool
        Whether the patient has had a previous cesarean section.
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
        height_cm=height_cm,
        weight_kg=weight_kg,
        pregnancy_weeks=pregnancy_weeks,
        gravida=gravida,
        parity=parity,
        previous_c_section=previous_c_section,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        blood_pressure=blood_pressure,
        body_temperature=body_temperature,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )
    return predict_from_patient(patient, engine=engine)
