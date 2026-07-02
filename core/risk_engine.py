"""Risk assessment engine for the obstetric emergency support system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .patient import PatientInfo


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
