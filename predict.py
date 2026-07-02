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
    heavy_vaginal_bleeding: bool,
    bleeding_severity: str,
    abdominal_pain: bool,
    pain_score: int,
    fetal_movement: str,
    loss_of_consciousness: bool,
    convulsions: bool,
    headache: bool,
    blurred_vision: bool,
    difficulty_breathing: bool,
    chest_pain: bool,
    vomiting: bool,
    systolic_bp: int,
    diastolic_bp: int,
    heart_rate: int,
    respiratory_rate: int,
    body_temperature: float,
    spo2: int,
    blood_sugar: float,
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
        heavy_vaginal_bleeding=heavy_vaginal_bleeding,
        bleeding_severity=bleeding_severity,
        abdominal_pain=abdominal_pain,
        pain_score=pain_score,
        fetal_movement=fetal_movement,
        loss_of_consciousness=loss_of_consciousness,
        convulsions=convulsions,
        headache=headache,
        blurred_vision=blurred_vision,
        difficulty_breathing=difficulty_breathing,
        chest_pain=chest_pain,
        vomiting=vomiting,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        respiratory_rate=respiratory_rate,
        body_temperature=body_temperature,
        spo2=spo2,
        blood_sugar=blood_sugar,
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
    heavy_vaginal_bleeding: bool,
    bleeding_severity: str,
    abdominal_pain: bool,
    pain_score: int,
    fetal_movement: str,
    loss_of_consciousness: bool,
    convulsions: bool,
    headache: bool,
    blurred_vision: bool,
    difficulty_breathing: bool,
    chest_pain: bool,
    vomiting: bool,
    systolic_bp: int,
    diastolic_bp: int,
    heart_rate: int,
    respiratory_rate: int,
    body_temperature: float,
    spo2: int,
    blood_sugar: float,
    engine: Optional[RiskEngine] = None,
) -> AssessmentResult:
    """Assess obstetric emergency risk from explicit patient parameters."""
    patient = create_patient(
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        pregnancy_weeks=pregnancy_weeks,
        gravida=gravida,
        parity=parity,
        previous_c_section=previous_c_section,
        heavy_vaginal_bleeding=heavy_vaginal_bleeding,
        bleeding_severity=bleeding_severity,
        abdominal_pain=abdominal_pain,
        pain_score=pain_score,
        fetal_movement=fetal_movement,
        loss_of_consciousness=loss_of_consciousness,
        convulsions=convulsions,
        headache=headache,
        blurred_vision=blurred_vision,
        difficulty_breathing=difficulty_breathing,
        chest_pain=chest_pain,
        vomiting=vomiting,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        respiratory_rate=respiratory_rate,
        body_temperature=body_temperature,
        spo2=spo2,
        blood_sugar=blood_sugar,
    )
    return predict_from_patient(patient, engine=engine)
