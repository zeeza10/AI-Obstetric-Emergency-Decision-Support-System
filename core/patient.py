"""Patient data model for the decision support system."""

from __future__ import annotations

from dataclasses import dataclass


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate body mass index from height (cm) and weight (kg)."""
    height_m = height_cm / 100.0
    if height_m <= 0:
        raise ValueError("Height must be greater than zero to calculate BMI.")
    return round(weight_kg / (height_m**2), 1)


@dataclass
class PatientInfo:
    """Represents the structured patient information used for assessment."""

    age: int
    height_cm: int
    weight_kg: float
    pregnancy_weeks: int
    gravida: int
    parity: int
    previous_c_section: bool
    heavy_vaginal_bleeding: bool
    bleeding_severity: str
    abdominal_pain: bool
    pain_score: int
    fetal_movement: str
    loss_of_consciousness: bool
    convulsions: bool
    headache: bool
    blurred_vision: bool
    difficulty_breathing: bool
    chest_pain: bool
    vomiting: bool
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    respiratory_rate: int
    body_temperature: float
    spo2: int
    blood_sugar: float
    hemoglobin: float = 12.0
    urine_protein: str = "Negative"
    platelet_count: int = 250
    urine_glucose: str = "Negative"
    hypertension: bool = False
    diabetes: bool = False
    anemia: bool = False
    heart_disease: bool = False
    multiple_pregnancy: bool = False
    previous_preeclampsia: bool = False
    previous_hemorrhage: bool = False

    @property
    def bmi(self) -> float:
        """Return the patient's calculated BMI."""
        return calculate_bmi(self.height_cm, self.weight_kg)

    def to_summary_dict(self) -> dict[str, str]:
        """Return a clinician-friendly summary of the submitted patient data."""
        return {
            "Age": f"{self.age} years",
            "Height": f"{self.height_cm} cm",
            "Weight": f"{self.weight_kg:.1f} kg",
            "BMI": f"{self.bmi:.1f}",
            "Pregnancy Weeks": str(self.pregnancy_weeks),
            "Gravida": str(self.gravida),
            "Parity": str(self.parity),
            "Previous C-Section": "Yes" if self.previous_c_section else "No",
            "Heavy Vaginal Bleeding": "Yes" if self.heavy_vaginal_bleeding else "No",
            "Bleeding Severity": self.bleeding_severity,
            "Abdominal Pain": "Yes" if self.abdominal_pain else "No",
            "Pain Score": f"{self.pain_score}/10",
            "Fetal Movement": self.fetal_movement,
            "Loss of Consciousness": "Yes" if self.loss_of_consciousness else "No",
            "Convulsions": "Yes" if self.convulsions else "No",
            "Headache": "Yes" if self.headache else "No",
            "Blurred Vision": "Yes" if self.blurred_vision else "No",
            "Difficulty Breathing": "Yes" if self.difficulty_breathing else "No",
            "Chest Pain": "Yes" if self.chest_pain else "No",
            "Vomiting": "Yes" if self.vomiting else "No",
            "Systolic BP": f"{self.systolic_bp} mmHg",
            "Diastolic BP": f"{self.diastolic_bp} mmHg",
            "Heart Rate": f"{self.heart_rate} bpm",
            "Respiratory Rate": f"{self.respiratory_rate} /min",
            "Body Temperature": f"{self.body_temperature:.1f} °C",
            "SpO₂": f"{self.spo2}%",
            "Blood Sugar": f"{self.blood_sugar:.1f} mg/dL",
            "Hemoglobin": f"{self.hemoglobin:.1f} g/dL",
            "Urine Protein": self.urine_protein,
            "Platelet Count": f"{self.platelet_count} x10^9/L",
            "Urine Glucose": self.urine_glucose,
            "Hypertension": "Yes" if self.hypertension else "No",
            "Diabetes": "Yes" if self.diabetes else "No",
            "Anemia": "Yes" if self.anemia else "No",
            "Heart Disease": "Yes" if self.heart_disease else "No",
            "Multiple Pregnancy": "Yes" if self.multiple_pregnancy else "No",
            "Previous Preeclampsia": "Yes" if self.previous_preeclampsia else "No",
            "Previous Hemorrhage": "Yes" if self.previous_hemorrhage else "No",
        }
