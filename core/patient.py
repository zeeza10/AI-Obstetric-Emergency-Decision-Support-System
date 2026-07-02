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
