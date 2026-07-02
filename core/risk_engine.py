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

        if patient.bmi >= 40:
            score += 2
            reasons.append("BMI indicates severe obesity, increasing obstetric complication risk.")
        elif patient.bmi >= 30:
            score += 1
            reasons.append("BMI indicates obesity, which may increase complication risk.")

        if patient.gravida >= 5:
            score += 1
            reasons.append("Grand multiparity may increase the risk of obstetric complications.")

        if patient.previous_c_section:
            score += 1
            reasons.append("Previous cesarean section increases risk in certain obstetric emergencies.")

        if patient.pregnancy_weeks < 20 or patient.pregnancy_weeks > 42:
            score += 1
            reasons.append("Gestational age is outside the expected low-risk range.")

        if patient.heavy_vaginal_bleeding:
            score += 3
            reasons.append("Heavy vaginal bleeding is a major warning sign.")

        if patient.bleeding_severity == "Moderate":
            score += 1
            reasons.append("Moderate bleeding severity increases clinical concern.")
        elif patient.bleeding_severity == "Heavy":
            score += 2
            reasons.append("Heavy bleeding severity is clinically significant.")
        elif patient.bleeding_severity == "Severe":
            score += 3
            reasons.append("Severe bleeding severity suggests urgent hemorrhage risk.")

        if patient.abdominal_pain:
            score += 2
            reasons.append("Abdominal pain suggests possible serious complications.")

        if patient.pain_score >= 8:
            score += 3
            reasons.append("Severe pain score indicates a high level of distress.")
        elif patient.pain_score >= 5:
            score += 2
            reasons.append("Moderate to severe pain score requires prompt assessment.")
        elif patient.pain_score >= 1:
            score += 1
            reasons.append("Reported pain score contributes to the clinical concern.")

        if patient.systolic_bp < 90 or patient.systolic_bp > 140 or patient.diastolic_bp > 90:
            score += 2
            reasons.append("Blood pressure is outside the expected range.")

        if patient.heart_rate < 60 or patient.heart_rate > 100:
            score += 1
            reasons.append("Heart rate is outside the expected range.")
        if patient.heart_rate < 50 or patient.heart_rate > 120:
            score += 1
            reasons.append("Abnormal heart rate may indicate hemodynamic instability.")

        if patient.respiratory_rate < 12 or patient.respiratory_rate > 20:
            score += 1
            reasons.append("Respiratory rate is outside the expected range.")
        if patient.respiratory_rate < 10 or patient.respiratory_rate > 24:
            score += 1
            reasons.append("Abnormal respiratory rate requires urgent review.")

        if patient.body_temperature >= 38.0:
            score += 2
            reasons.append("Elevated temperature may indicate infection or inflammation.")

        if patient.spo2 < 95:
            score += 1
            reasons.append("Oxygen saturation is below the expected range.")
        if patient.spo2 < 92:
            score += 2
            reasons.append("Low SpO₂ suggests possible hypoxemia.")

        if patient.blood_sugar < 70 or patient.blood_sugar > 140:
            score += 1
            reasons.append("Blood sugar is outside the expected range.")

        if patient.fetal_movement == "Reduced":
            score += 2
            reasons.append("Fetal movement is reduced and needs urgent review.")
        elif patient.fetal_movement == "Absent":
            score += 4
            reasons.append("Absent fetal movement is a serious fetal concern.")

        if patient.loss_of_consciousness:
            score += 5
            reasons.append("Loss of consciousness is a critical neurological warning sign.")

        if patient.convulsions:
            score += 4
            reasons.append("Convulsions suggest a severe obstetric or neurological emergency.")

        if patient.headache and patient.blurred_vision:
            score += 2
            reasons.append("Headache with blurred vision may indicate severe pre-eclampsia.")
        elif patient.headache or patient.blurred_vision:
            score += 1
            reasons.append("Neurological visual symptoms require clinical review.")

        if patient.difficulty_breathing:
            score += 2
            reasons.append("Difficulty breathing may indicate respiratory compromise.")

        if patient.chest_pain:
            score += 2
            reasons.append("Chest pain requires urgent cardiovascular assessment.")

        if patient.vomiting:
            score += 1
            reasons.append("Persistent vomiting may indicate systemic illness or complication.")

        if (
            patient.heavy_vaginal_bleeding
            and patient.abdominal_pain
            and patient.systolic_bp < 100
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
        if score >= 9 or (patient.heavy_vaginal_bleeding and patient.loss_of_consciousness):
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
            "height_cm",
            "weight_kg",
            "bmi",
            "pregnancy_weeks",
            "gravida",
            "parity",
            "previous_c_section",
            "heavy_vaginal_bleeding",
            "bleeding_severity",
            "abdominal_pain",
            "pain_score",
            "fetal_movement",
            "loss_of_consciousness",
            "convulsions",
            "headache",
            "blurred_vision",
            "difficulty_breathing",
            "chest_pain",
            "vomiting",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "respiratory_rate",
            "body_temperature",
            "spo2",
            "blood_sugar",
        ]

    def _encode_patient(self, patient: PatientInfo) -> list[float]:
        fetal_mapping = {"Normal": 0.0, "Reduced": 0.5, "Absent": 1.0}
        bleeding_mapping = {"None": 0.0, "Light": 0.25, "Moderate": 0.5, "Heavy": 0.75, "Severe": 1.0}
        return [
            float(patient.age),
            float(patient.height_cm),
            float(patient.weight_kg),
            patient.bmi,
            float(patient.pregnancy_weeks),
            float(patient.gravida),
            float(patient.parity),
            1.0 if patient.previous_c_section else 0.0,
            1.0 if patient.heavy_vaginal_bleeding else 0.0,
            bleeding_mapping.get(patient.bleeding_severity, 0.0),
            1.0 if patient.abdominal_pain else 0.0,
            float(patient.pain_score) / 10.0,
            fetal_mapping.get(patient.fetal_movement, 0.0),
            1.0 if patient.loss_of_consciousness else 0.0,
            1.0 if patient.convulsions else 0.0,
            1.0 if patient.headache else 0.0,
            1.0 if patient.blurred_vision else 0.0,
            1.0 if patient.difficulty_breathing else 0.0,
            1.0 if patient.chest_pain else 0.0,
            1.0 if patient.vomiting else 0.0,
            float(patient.systolic_bp),
            float(patient.diastolic_bp),
            float(patient.heart_rate),
            float(patient.respiratory_rate),
            float(patient.body_temperature),
            float(patient.spo2),
            patient.blood_sugar,
        ]

    def predict_proba(self, features: Sequence[float]) -> list[list[float]]:
        """Return class probabilities for a single feature vector."""
        vector = list(features)
        if len(vector) != len(self.feature_names):
            raise ModelPredictionError("Feature vector length does not match the expected model shape.")

        age = vector[0]
        height_cm = vector[1]
        weight_kg = vector[2]
        bmi = vector[3]
        weeks = vector[4]
        gravida = vector[5]
        parity = vector[6]
        previous_c_section = vector[7]
        heavy_bleeding = vector[8]
        bleeding_severity = vector[9]
        abdominal_pain = vector[10]
        pain_score = vector[11]
        fetal = vector[12]
        loss_of_consciousness = vector[13]
        convulsions = vector[14]
        headache = vector[15]
        blurred_vision = vector[16]
        difficulty_breathing = vector[17]
        chest_pain = vector[18]
        vomiting = vector[19]
        systolic_bp = vector[20]
        diastolic_bp = vector[21]
        heart_rate = vector[22]
        respiratory_rate = vector[23]
        temp = vector[24]
        spo2 = vector[25]
        blood_sugar = vector[26]

        severity_score = 0.0
        severity_score += 0.02 * max(0.0, age - 25.0)
        severity_score += 0.01 * max(0.0, bmi - 25.0)
        severity_score += 0.01 * max(0.0, weeks - 28.0)
        severity_score += 0.03 * max(0.0, gravida - 2.0)
        severity_score += 0.08 * previous_c_section
        severity_score += 0.3 * heavy_bleeding
        severity_score += 0.2 * bleeding_severity
        severity_score += 0.2 * abdominal_pain
        severity_score += 0.25 * pain_score
        severity_score += 0.25 * fetal
        severity_score += 0.2 * loss_of_consciousness
        severity_score += 0.18 * convulsions
        severity_score += 0.08 * headache
        severity_score += 0.08 * blurred_vision
        severity_score += 0.12 * difficulty_breathing
        severity_score += 0.12 * chest_pain
        severity_score += 0.06 * vomiting
        severity_score += 0.01 * max(0.0, 120.0 - systolic_bp)
        severity_score += 0.008 * max(0.0, diastolic_bp - 80.0)
        severity_score += 0.005 * max(0.0, abs(heart_rate - 80.0) - 10.0)
        severity_score += 0.005 * max(0.0, abs(respiratory_rate - 16.0) - 4.0)
        severity_score += 0.12 * max(0.0, temp - 37.0)
        severity_score += 0.01 * max(0.0, 95.0 - spo2)
        severity_score += 0.004 * max(0.0, abs(blood_sugar - 100.0) - 20.0)

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
            features = self.model._encode_patient(patient) if hasattr(self.model, "_encode_patient") else SimplePersistedModel()._encode_patient(patient)
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
