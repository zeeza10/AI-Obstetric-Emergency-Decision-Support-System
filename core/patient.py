"""Patient data model for the decision support system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatientInfo:
    """Represents the structured patient information used for assessment."""

    age: int
    pregnancy_weeks: int
    heavy_bleeding: bool
    severe_abdominal_pain: bool
    blood_pressure: int
    body_temperature: float
    fetal_movement: str
    consciousness: str

    def to_summary_dict(self) -> dict[str, str]:
        """Return a clinician-friendly summary of the submitted patient data."""
        return {
            "Age": f"{self.age} years",
            "Pregnancy Weeks": str(self.pregnancy_weeks),
            "Heavy Bleeding": "Yes" if self.heavy_bleeding else "No",
            "Severe Abdominal Pain": "Yes" if self.severe_abdominal_pain else "No",
            "Blood Pressure": f"{self.blood_pressure} mmHg",
            "Body Temperature": f"{self.body_temperature:.1f} °C",
            "Fetal Movement": self.fetal_movement,
            "Consciousness": self.consciousness,
        }
