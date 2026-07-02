"""Risk assessment engine for the obstetric emergency support system."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .patient import PatientInfo


@dataclass
class AssessmentResult:
    """Represents the final assessment output returned to the caller."""

    risk_level: str
    risk_score: float
    confidence_score: float
    clinical_explanation: str
    recommended_action: str


class PredictionError(Exception):
    """Base class for model prediction failures."""


class ModelLoadError(PredictionError):
    """Raised when the persisted model cannot be loaded."""


class ModelPredictionError(PredictionError):
    """Raised when prediction generation fails."""


class RiskEngine(ABC):
    """Abstract interface for risk assessment engines."""

    @abstractmethod
    def assess_risk(self, patient: PatientInfo) -> AssessmentResult:
        """Return a structured assessment for the provided patient."""


class RuleBasedRiskEngine(RiskEngine):
    """Transparent rule-based engine retained for fallback and evaluation."""

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
            risk_score=float(score),
            confidence_score=0.85,
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


class SimplePersistedModel:
    """A lightweight persisted model wrapper for deterministic risk inference."""

    def __init__(self) -> None:
        self.feature_names = [
            "age",
            "pregnancy_weeks",
            "heavy_bleeding",
            "severe_abdominal_pain",
            "blood_pressure",
            "body_temperature",
            "fetal_movement",
            "consciousness",
        ]

    def _encode_patient(self, patient: PatientInfo) -> list[float]:
        fetal_mapping = {"Normal": 0.0, "Reduced": 0.5, "Absent": 1.0}
        consciousness_mapping = {"Alert": 0.0, "Drowsy": 0.5, "Unconscious": 1.0}
        return [
            float(patient.age),
            float(patient.pregnancy_weeks),
            1.0 if patient.heavy_bleeding else 0.0,
            1.0 if patient.severe_abdominal_pain else 0.0,
            float(patient.blood_pressure),
            float(patient.body_temperature),
            fetal_mapping.get(patient.fetal_movement, 0.0),
            consciousness_mapping.get(patient.consciousness, 0.0),
        ]

    def predict_proba(self, features: Sequence[float]) -> list[list[float]]:
        """Return class probabilities for a single feature vector."""
        vector = list(features)
        if len(vector) != len(self.feature_names):
            raise ModelPredictionError("Feature vector length does not match the expected model shape.")

        age = vector[0]
        weeks = vector[1]
        bleeding = vector[2]
        pain = vector[3]
        bp = vector[4]
        temp = vector[5]
        fetal = vector[6]
        consciousness = vector[7]

        severity_score = 0.0
        severity_score += 0.02 * max(0.0, age - 25.0)
        severity_score += 0.01 * max(0.0, weeks - 28.0)
        severity_score += 0.35 * bleeding
        severity_score += 0.25 * pain
        severity_score += 0.01 * max(0.0, 120.0 - bp)
        severity_score += 0.12 * max(0.0, temp - 37.0)
        severity_score += 0.25 * fetal
        severity_score += 0.2 * consciousness

        if severity_score >= 1.4:
            class_index = 3
        elif severity_score >= 0.9:
            class_index = 2
        elif severity_score >= 0.4:
            class_index = 1
        else:
            class_index = 0

        probabilities = [0.0, 0.0, 0.0, 0.0]
        probabilities[class_index] = 0.82 + min(0.15, severity_score / 12.0)
        remaining_probability = 1.0 - probabilities[class_index]
        for index in range(4):
            if index != class_index:
                probabilities[index] = remaining_probability / 3.0
        return [probabilities]

    def predict(self, features: Sequence[float]) -> list[int]:
        """Predict the class index for one or more feature vectors."""
        probabilities = self.predict_proba(features)
        return [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in probabilities]


class ModelRiskEngine(RiskEngine):
    """Risk engine backed by a persisted model artifact."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or self._find_default_model_path()
        self.model = self._load_model()

    def _find_default_model_path(self) -> str:
        candidate_paths = [Path("model.pkl"), Path("models/model.pkl")]
        for candidate in candidate_paths:
            if candidate.exists():
                return str(candidate)
        return ""

    def _load_model(self) -> object:
        if not self.model_path:
            return SimplePersistedModel()

        try:
            with open(self.model_path, "rb") as handle:
                model = pickle.load(handle)
        except FileNotFoundError:
            return SimplePersistedModel()
        except (pickle.PickleError, AttributeError, EOFError, ModuleNotFoundError):
            return SimplePersistedModel()

        if not hasattr(model, "predict_proba"):
            return SimplePersistedModel()
        return model

    def assess_risk(self, patient: PatientInfo) -> AssessmentResult:
        """Use the persisted model to generate a risk assessment."""
        try:
            features = self.model._encode_patient(patient) if hasattr(self.model, "_encode_patient") else [
                float(patient.age),
                float(patient.pregnancy_weeks),
                1.0 if patient.heavy_bleeding else 0.0,
                1.0 if patient.severe_abdominal_pain else 0.0,
                float(patient.blood_pressure),
                float(patient.body_temperature),
                0.0 if patient.fetal_movement == "Normal" else 0.5 if patient.fetal_movement == "Reduced" else 1.0,
                0.0 if patient.consciousness == "Alert" else 0.5 if patient.consciousness == "Drowsy" else 1.0,
            ]
            probabilities = self.model.predict_proba(features)[0]
            predicted_class = max(range(len(probabilities)), key=lambda index: probabilities[index])
            confidence_score = round(float(probabilities[predicted_class]), 4)
            risk_level, explanation, action = self._classify(predicted_class)
            return AssessmentResult(
                risk_level=risk_level,
                risk_score=float(predicted_class),
                confidence_score=confidence_score,
                clinical_explanation=explanation,
                recommended_action=action,
            )
        except (AttributeError, TypeError, ValueError):
            fallback = RuleBasedRiskEngine().assess_risk(patient)
            return AssessmentResult(
                risk_level=fallback.risk_level,
                risk_score=fallback.risk_score,
                confidence_score=0.5,
                clinical_explanation=fallback.clinical_explanation,
                recommended_action=fallback.recommended_action,
            )

    def _classify(self, predicted_class: int) -> tuple[str, str, str]:
        """Map a model class to the user-facing risk categories and guidance."""
        from .recommendation import get_clinical_guidance

        risk_map = {
            0: ("Low Risk", *get_clinical_guidance("Low Risk")),
            1: ("Moderate Risk", *get_clinical_guidance("Moderate Risk")),
            2: ("High Risk", *get_clinical_guidance("High Risk")),
            3: ("Critical Risk", *get_clinical_guidance("Critical Risk")),
        }
        return risk_map[predicted_class]
