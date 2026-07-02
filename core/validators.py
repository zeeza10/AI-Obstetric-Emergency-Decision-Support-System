"""Validation utilities for patient input data."""

from __future__ import annotations

from .patient import PatientInfo


class ValidationError(Exception):
    """Base class for domain-specific validation errors."""


class MissingValueError(ValidationError):
    """Raised when a required value is missing."""


class InvalidNumericValueError(ValidationError):
    """Raised when a numeric input is invalid or out of range."""


class InvalidYesNoError(ValidationError):
    """Raised when a Yes/No response is not recognized."""


class InvalidChoiceError(ValidationError):
    """Raised when an input choice is invalid."""


def validate_patient_information(patient: PatientInfo) -> None:
    """Validate that the patient fields are clinically reasonable."""
    if not isinstance(patient.age, int) or isinstance(patient.age, bool):
        raise InvalidNumericValueError("Age must be a whole number.")
    if patient.age <= 0:
        raise InvalidNumericValueError("Age must be greater than zero.")

    if not isinstance(patient.pregnancy_weeks, int) or isinstance(patient.pregnancy_weeks, bool):
        raise InvalidNumericValueError("Pregnancy weeks must be a whole number.")
    if patient.pregnancy_weeks <= 0 or patient.pregnancy_weeks > 45:
        raise InvalidNumericValueError("Pregnancy weeks must be between 1 and 45.")

    if not isinstance(patient.blood_pressure, int) or isinstance(patient.blood_pressure, bool):
        raise InvalidNumericValueError("Blood pressure must be a whole number.")
    if patient.blood_pressure <= 0:
        raise InvalidNumericValueError("Blood pressure must be greater than zero.")

    if not isinstance(patient.body_temperature, (int, float)):
        raise InvalidNumericValueError("Body temperature must be a number.")
    if patient.body_temperature <= 0:
        raise InvalidNumericValueError("Body temperature must be greater than zero.")

    allowed_fetal_movements = {"Normal", "Reduced", "Absent"}
    if patient.fetal_movement not in allowed_fetal_movements:
        raise InvalidChoiceError("Fetal movement must be Normal, Reduced, or Absent.")

    allowed_consciousness_states = {"Alert", "Drowsy", "Unconscious"}
    if patient.consciousness not in allowed_consciousness_states:
        raise InvalidChoiceError("Consciousness must be Alert, Drowsy, or Unconscious.")
