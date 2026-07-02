"""Reusable rule-based prediction module for obstetric emergency triage.

This module exposes a clean, production-oriented interface for assessing
obstetric emergency risk without collecting user input directly. It is intended
for reuse by console applications, web services, or future machine-learning
backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PatientInfo:
    """Represents the structured patient data required for risk assessment."""

    age: int
    pregnancy_weeks: int
    heavy_bleeding: bool
    severe_abdominal_pain: bool
    blood_pressure: int
    body_temperature: float
    fetal_movement: str
    consciousness: str


@dataclass
class AssessmentResult:
    """Represents the final assessment output returned to the caller."""

    risk_level: str
    risk_score: int
    clinical_explanation: str
    recommended_action: str


class RiskEngine(ABC):
    """Abstract interface for risk assessment engines."""

    @abstractmethod
    def assess_risk(self, patient: PatientInfo) -> AssessmentResult:
        """Return a structured assessment for the provided patient."""


class RuleBasedRiskEngine(RiskEngine):
    """Transparent rule-based engine for Version 1."""

    def assess_risk(self, patient: PatientInfo) -> AssessmentResult:
        """Evaluate a patient using explainable clinical rules."""
        score = 0
        reasons: List[str] = []

        if patient.age < 18 or patient.age > 35:
            score += 1
            reasons.append("Maternal age is outside the standard low-risk range.")

        if patient.pregnancy_weeks < 20 or patient.pregnancy_weeks > 42:
            score += 1
            reasons.append("Gestational age is outside the expected low-risk range.")

        if patient.heavy_bleeding:
            score += 3
            reasons.append("Heavy bleeding is a major warning sign.")

        if patient.severe_abdominal_pain:
            score += 2
            reasons.append("Severe abdominal pain suggests possible serious complications.")

        if patient.blood_pressure < 90 or patient.blood_pressure > 140:
            score += 2
            reasons.append("Blood pressure is outside the expected range.")

        if patient.body_temperature >= 38.0:
            score += 2
            reasons.append("Elevated temperature may indicate infection or inflammation.")

        if patient.fetal_movement == "Reduced":
            score += 2
            reasons.append("Fetal movement is reduced and needs urgent review.")
        elif patient.fetal_movement == "Absent":
            score += 4
            reasons.append("Absent fetal movement is a serious fetal concern.")

        if patient.consciousness == "Drowsy":
            score += 2
            reasons.append("The patient appears drowsy.")
        elif patient.consciousness == "Unconscious":
            score += 5
            reasons.append("The patient is unconscious.")

        if (
            patient.heavy_bleeding
            and patient.severe_abdominal_pain
            and patient.blood_pressure < 100
        ):
            score += 2
            reasons.append("Bleeding with abdominal pain and low blood pressure suggests instability.")

        risk_level, explanation, action = self._classify(score, patient)
        return AssessmentResult(
            risk_level=risk_level,
            risk_score=score,
            clinical_explanation=f"{explanation} {' '.join(reasons)}",
            recommended_action=action,
        )

    def _classify(self, score: int, patient: PatientInfo) -> tuple[str, str, str]:
        """Translate the computed score into a final risk category."""
        if score >= 9 or (patient.heavy_bleeding and patient.consciousness == "Unconscious"):
            return (
                "Critical Risk",
                "The presentation suggests a severe obstetric emergency requiring immediate intervention.",
                "Call emergency services immediately and arrange urgent hospital transfer with continuous monitoring.",
            )

        if score >= 6:
            return (
                "High Risk",
                "The symptoms indicate a potentially dangerous obstetric condition that needs urgent clinical review.",
                "Seek urgent hospital assessment now and prepare for immediate medical intervention.",
            )

        if score >= 3:
            return (
                "Moderate Risk",
                "The patient has concerning symptoms that require prompt medical evaluation and close monitoring.",
                "Arrange prompt medical assessment and monitor the patient closely.",
            )

        return (
            "Low Risk",
            "The reported symptoms do not currently suggest an acute obstetric emergency.",
            "Continue routine monitoring and seek medical attention if symptoms worsen.",
        )


def validate_patient_information(patient: PatientInfo) -> None:
    """Validate that the patient fields are clinically reasonable."""
    if patient.age <= 0:
        raise ValueError("Age must be greater than zero.")
    if patient.pregnancy_weeks <= 0 or patient.pregnancy_weeks > 45:
        raise ValueError("Pregnancy weeks must be between 1 and 45.")
    if patient.blood_pressure <= 0:
        raise ValueError("Blood pressure must be greater than zero.")
    if patient.body_temperature <= 0:
        raise ValueError("Body temperature must be greater than zero.")


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
