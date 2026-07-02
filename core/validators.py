"""Validation utilities for patient input data."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Mapping, Optional, Set, Union

from .patient import PatientInfo, calculate_bmi

FormValue = Union[str, int, float, bool, None]

AGE_MIN = 12
AGE_MAX = 55
HEIGHT_CM_MIN = 100
HEIGHT_CM_MAX = 220
WEIGHT_KG_MIN = 30.0
WEIGHT_KG_MAX = 200.0
BMI_MIN = 10.0
BMI_MAX = 60.0
PREGNANCY_WEEKS_MIN = 1
PREGNANCY_WEEKS_MAX = 45
GRAVIDA_MIN = 1
GRAVIDA_MAX = 15
PARITY_MIN = 0
PARITY_MAX = 14
SYSTOLIC_BP_MIN = 50
SYSTOLIC_BP_MAX = 250
DIASTOLIC_BP_MIN = 30
DIASTOLIC_BP_MAX = 150
HEART_RATE_MIN = 40
HEART_RATE_MAX = 200
RESPIRATORY_RATE_MIN = 8
RESPIRATORY_RATE_MAX = 40
BODY_TEMPERATURE_MIN = 30.0
BODY_TEMPERATURE_MAX = 45.0
SPO2_MIN = 50
SPO2_MAX = 100
BLOOD_SUGAR_MIN = 30.0
BLOOD_SUGAR_MAX = 600.0

ALLOWED_FETAL_MOVEMENTS: Set[str] = {"Normal", "Reduced", "Absent"}
ALLOWED_CONSCIOUSNESS_STATES: Set[str] = {"Alert", "Drowsy", "Unconscious"}

YES_VALUES: Set[str] = {"true", "yes", "y", "1", "on"}
NO_VALUES: Set[str] = {"false", "no", "n", "0", "off"}

FIELD_LABELS = {
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "previous_c_section": "Previous C-Section",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "heart_rate": "Heart Rate",
    "respiratory_rate": "Respiratory Rate",
    "spo2": "SpO₂",
    "blood_sugar": "Blood Sugar",
}


class ValidationError(Exception):
    """Base class for domain-specific validation errors."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        self.field = field
        super().__init__(message)


class MissingValueError(ValidationError):
    """Raised when a required value is missing."""


class InvalidNumericValueError(ValidationError):
    """Raised when a numeric input is invalid or out of range."""


class InvalidYesNoError(ValidationError):
    """Raised when a Yes/No response is not recognized."""


class InvalidChoiceError(ValidationError):
    """Raised when an input choice is invalid."""


def _normalize_label(field_name: str) -> str:
    """Convert a field key into a user-facing label."""
    if field_name in FIELD_LABELS:
        return FIELD_LABELS[field_name]
    return field_name.replace("_", " ").title()


def _is_missing(value: FormValue) -> bool:
    """Return True when a submitted value should be treated as empty."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return False
    return str(value).strip() == ""


def _require_raw_value(form_data: Mapping[str, object], field_name: str) -> FormValue:
    """Return a submitted field value or raise when it is missing."""
    if field_name not in form_data:
        raise MissingValueError(f"{_normalize_label(field_name)} is required.", field=field_name)

    value = form_data[field_name]
    if _is_missing(value):
        raise MissingValueError(f"{_normalize_label(field_name)} is required.", field=field_name)
    return value


def _parse_whole_number(raw: FormValue, field_name: str) -> int:
    """Parse a required whole-number field from user input."""
    if isinstance(raw, bool):
        raise InvalidNumericValueError(
            f"{_normalize_label(field_name)} must be a whole number.",
            field=field_name,
        )

    if isinstance(raw, int):
        return raw

    if isinstance(raw, float):
        if not raw.is_integer():
            raise InvalidNumericValueError(
                f"{_normalize_label(field_name)} must be a whole number.",
                field=field_name,
            )
        return int(raw)

    text = str(raw).strip()
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as exc:
        raise InvalidNumericValueError(
            f"{_normalize_label(field_name)} must be a valid whole number.",
            field=field_name,
        ) from exc

    if decimal_value != decimal_value.to_integral_value():
        raise InvalidNumericValueError(
            f"{_normalize_label(field_name)} must be a whole number.",
            field=field_name,
        )

    return int(decimal_value)


def _parse_decimal_number(raw: FormValue, field_name: str) -> float:
    """Parse a required decimal field from user input."""
    if isinstance(raw, bool):
        raise InvalidNumericValueError(
            f"{_normalize_label(field_name)} must be a valid number.",
            field=field_name,
        )

    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    try:
        return float(Decimal(text))
    except InvalidOperation as exc:
        raise InvalidNumericValueError(
            f"{_normalize_label(field_name)} must be a valid number.",
            field=field_name,
        ) from exc


def _parse_required_int(
    raw: FormValue,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Parse and range-check a required integer field."""
    value = _parse_whole_number(raw, field_name)
    if value < minimum or value > maximum:
        raise InvalidNumericValueError(
            f"{_normalize_label(field_name)} must be between {minimum} and {maximum}.",
            field=field_name,
        )
    return value


def _parse_required_float(
    raw: FormValue,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Parse and range-check a required floating-point field."""
    value = _parse_decimal_number(raw, field_name)
    if value < minimum or value > maximum:
        raise InvalidNumericValueError(
            f"{_normalize_label(field_name)} must be between {minimum:g} and {maximum:g}.",
            field=field_name,
        )
    return round(value, 1)


def _parse_required_yes_no(raw: FormValue, field_name: str) -> bool:
    """Parse a required Yes/No field from common web and API representations."""
    if isinstance(raw, bool):
        return raw

    if isinstance(raw, int) and raw in {0, 1}:
        return bool(raw)

    if isinstance(raw, float) and raw in {0.0, 1.0}:
        return bool(int(raw))

    normalized = str(raw).strip().lower()
    if normalized in YES_VALUES:
        return True
    if normalized in NO_VALUES:
        return False

    raise InvalidYesNoError(
        f"{_normalize_label(field_name)} must be Yes or No.",
        field=field_name,
    )


def _parse_required_choice(raw: FormValue, field_name: str, allowed_values: Set[str]) -> str:
    """Parse a required choice field against an allow-list."""
    if isinstance(raw, bool):
        raise InvalidChoiceError(
            f"{_normalize_label(field_name)} must be one of: {', '.join(sorted(allowed_values))}.",
            field=field_name,
        )

    normalized = str(raw).strip()
    if not normalized:
        raise MissingValueError(f"{_normalize_label(field_name)} is required.", field=field_name)

    for allowed_value in allowed_values:
        if normalized.casefold() == allowed_value.casefold():
            return allowed_value

    allowed_text = ", ".join(sorted(allowed_values))
    raise InvalidChoiceError(
        f"{_normalize_label(field_name)} must be one of: {allowed_text}.",
        field=field_name,
    )


def _validate_obstetric_counts(gravida: int, parity: int) -> None:
    """Ensure gravida and parity are clinically consistent."""
    if parity >= gravida:
        raise InvalidNumericValueError(
            "Parity must be less than Gravida for an ongoing pregnancy.",
            field="parity",
        )


def _validate_blood_pressure(systolic_bp: int, diastolic_bp: int) -> None:
    """Ensure blood pressure values are clinically consistent."""
    if diastolic_bp >= systolic_bp:
        raise InvalidNumericValueError(
            "Diastolic BP must be lower than Systolic BP.",
            field="diastolic_bp",
        )


def parse_patient_from_form_data(form_data: Mapping[str, object]) -> PatientInfo:
    """Parse and validate patient information from a request-like mapping."""
    age = _parse_required_int(_require_raw_value(form_data, "age"), "age", AGE_MIN, AGE_MAX)
    height_cm = _parse_required_int(
        _require_raw_value(form_data, "height_cm"),
        "height_cm",
        HEIGHT_CM_MIN,
        HEIGHT_CM_MAX,
    )
    weight_kg = _parse_required_float(
        _require_raw_value(form_data, "weight_kg"),
        "weight_kg",
        WEIGHT_KG_MIN,
        WEIGHT_KG_MAX,
    )
    pregnancy_weeks = _parse_required_int(
        _require_raw_value(form_data, "pregnancy_weeks"),
        "pregnancy_weeks",
        PREGNANCY_WEEKS_MIN,
        PREGNANCY_WEEKS_MAX,
    )
    gravida = _parse_required_int(
        _require_raw_value(form_data, "gravida"),
        "gravida",
        GRAVIDA_MIN,
        GRAVIDA_MAX,
    )
    parity = _parse_required_int(
        _require_raw_value(form_data, "parity"),
        "parity",
        PARITY_MIN,
        PARITY_MAX,
    )
    _validate_obstetric_counts(gravida, parity)
    previous_c_section = _parse_required_yes_no(
        _require_raw_value(form_data, "previous_c_section"),
        "previous_c_section",
    )
    heavy_bleeding = _parse_required_yes_no(
        _require_raw_value(form_data, "heavy_bleeding"),
        "heavy_bleeding",
    )
    severe_abdominal_pain = _parse_required_yes_no(
        _require_raw_value(form_data, "severe_abdominal_pain"),
        "severe_abdominal_pain",
    )
    systolic_bp = _parse_required_int(
        _require_raw_value(form_data, "systolic_bp"),
        "systolic_bp",
        SYSTOLIC_BP_MIN,
        SYSTOLIC_BP_MAX,
    )
    diastolic_bp = _parse_required_int(
        _require_raw_value(form_data, "diastolic_bp"),
        "diastolic_bp",
        DIASTOLIC_BP_MIN,
        DIASTOLIC_BP_MAX,
    )
    _validate_blood_pressure(systolic_bp, diastolic_bp)
    heart_rate = _parse_required_int(
        _require_raw_value(form_data, "heart_rate"),
        "heart_rate",
        HEART_RATE_MIN,
        HEART_RATE_MAX,
    )
    respiratory_rate = _parse_required_int(
        _require_raw_value(form_data, "respiratory_rate"),
        "respiratory_rate",
        RESPIRATORY_RATE_MIN,
        RESPIRATORY_RATE_MAX,
    )
    body_temperature = _parse_required_float(
        _require_raw_value(form_data, "body_temperature"),
        "body_temperature",
        BODY_TEMPERATURE_MIN,
        BODY_TEMPERATURE_MAX,
    )
    spo2 = _parse_required_int(
        _require_raw_value(form_data, "spo2"),
        "spo2",
        SPO2_MIN,
        SPO2_MAX,
    )
    blood_sugar = _parse_required_float(
        _require_raw_value(form_data, "blood_sugar"),
        "blood_sugar",
        BLOOD_SUGAR_MIN,
        BLOOD_SUGAR_MAX,
    )
    fetal_movement = _parse_required_choice(
        _require_raw_value(form_data, "fetal_movement"),
        "fetal_movement",
        ALLOWED_FETAL_MOVEMENTS,
    )
    consciousness = _parse_required_choice(
        _require_raw_value(form_data, "consciousness"),
        "consciousness",
        ALLOWED_CONSCIOUSNESS_STATES,
    )

    patient = PatientInfo(
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        pregnancy_weeks=pregnancy_weeks,
        gravida=gravida,
        parity=parity,
        previous_c_section=previous_c_section,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        respiratory_rate=respiratory_rate,
        body_temperature=body_temperature,
        spo2=spo2,
        blood_sugar=blood_sugar,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )
    validate_patient_information(patient)
    return patient


def validate_patient_information(patient: PatientInfo) -> None:
    """Validate that the patient fields are clinically reasonable."""
    _parse_required_int(patient.age, "age", AGE_MIN, AGE_MAX)
    _parse_required_int(patient.height_cm, "height_cm", HEIGHT_CM_MIN, HEIGHT_CM_MAX)
    _parse_required_float(patient.weight_kg, "weight_kg", WEIGHT_KG_MIN, WEIGHT_KG_MAX)
    bmi = calculate_bmi(patient.height_cm, patient.weight_kg)
    if bmi < BMI_MIN or bmi > BMI_MAX:
        raise InvalidNumericValueError(
            f"BMI must be between {BMI_MIN:g} and {BMI_MAX:g}.",
            field="bmi",
        )
    _parse_required_int(
        patient.pregnancy_weeks,
        "pregnancy_weeks",
        PREGNANCY_WEEKS_MIN,
        PREGNANCY_WEEKS_MAX,
    )
    _parse_required_int(patient.gravida, "gravida", GRAVIDA_MIN, GRAVIDA_MAX)
    _parse_required_int(patient.parity, "parity", PARITY_MIN, PARITY_MAX)
    _validate_obstetric_counts(patient.gravida, patient.parity)
    _parse_required_int(patient.systolic_bp, "systolic_bp", SYSTOLIC_BP_MIN, SYSTOLIC_BP_MAX)
    _parse_required_int(patient.diastolic_bp, "diastolic_bp", DIASTOLIC_BP_MIN, DIASTOLIC_BP_MAX)
    _validate_blood_pressure(patient.systolic_bp, patient.diastolic_bp)
    _parse_required_int(patient.heart_rate, "heart_rate", HEART_RATE_MIN, HEART_RATE_MAX)
    _parse_required_int(
        patient.respiratory_rate,
        "respiratory_rate",
        RESPIRATORY_RATE_MIN,
        RESPIRATORY_RATE_MAX,
    )
    _parse_required_float(
        patient.body_temperature,
        "body_temperature",
        BODY_TEMPERATURE_MIN,
        BODY_TEMPERATURE_MAX,
    )
    _parse_required_int(patient.spo2, "spo2", SPO2_MIN, SPO2_MAX)
    _parse_required_float(patient.blood_sugar, "blood_sugar", BLOOD_SUGAR_MIN, BLOOD_SUGAR_MAX)

    if not isinstance(patient.previous_c_section, bool):
        raise InvalidYesNoError("Previous C-Section must be Yes or No.", field="previous_c_section")
    if not isinstance(patient.heavy_bleeding, bool):
        raise InvalidYesNoError("Heavy Bleeding must be Yes or No.", field="heavy_bleeding")
    if not isinstance(patient.severe_abdominal_pain, bool):
        raise InvalidYesNoError(
            "Severe Abdominal Pain must be Yes or No.",
            field="severe_abdominal_pain",
        )

    _parse_required_choice(patient.fetal_movement, "fetal_movement", ALLOWED_FETAL_MOVEMENTS)
    _parse_required_choice(patient.consciousness, "consciousness", ALLOWED_CONSCIOUSNESS_STATES)
