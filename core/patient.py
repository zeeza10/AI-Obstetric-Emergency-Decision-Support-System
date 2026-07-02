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
    heavy_bleeding: bool
    severe_abdominal_pain: bool
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    respiratory_rate: int
    body_temperature: float
    spo2: int
    blood_sugar: float
    fetal_movement: str
    consciousness: str

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
            "Heavy Bleeding": "Yes" if self.heavy_bleeding else "No",
            "Severe Abdominal Pain": "Yes" if self.severe_abdominal_pain else "No",
            "Systolic BP": f"{self.systolic_bp} mmHg",
            "Diastolic BP": f"{self.diastolic_bp} mmHg",
            "Heart Rate": f"{self.heart_rate} bpm",
            "Respiratory Rate": f"{self.respiratory_rate} /min",
            "Body Temperature": f"{self.body_temperature:.1f} °C",
            "SpO₂": f"{self.spo2}%",
            "Blood Sugar": f"{self.blood_sugar:.1f} mg/dL",
            "Fetal Movement": self.fetal_movement,
            "Consciousness": self.consciousness,
        }
