"""Unit tests for the core obstetric emergency decision support modules."""

import os
import tempfile
import unittest

from core import (
    InvalidChoiceError,
    InvalidNumericValueError,
    InvalidYesNoError,
    MissingValueError,
    ModelRiskEngine,
    PatientInfo,
    RuleBasedRiskEngine,
    parse_patient_from_form_data,
    validate_patient_information,
)
from core.explanation import generate_shap_explanation
from predict import predict_from_form_data


VALID_FORM_DATA = {
    "age": "28",
    "height_cm": "165",
    "weight_kg": "68",
    "pregnancy_weeks": "32",
    "gravida": "2",
    "parity": "1",
    "previous_c_section": "false",
    "heavy_bleeding": "false",
    "severe_abdominal_pain": "false",
    "blood_pressure": "120",
    "body_temperature": "36.8",
    "fetal_movement": "Normal",
    "consciousness": "Alert",
}


def make_patient(**overrides) -> PatientInfo:
    """Build a valid patient object for tests with optional overrides."""
    defaults = {
        "age": 28,
        "height_cm": 165,
        "weight_kg": 68.0,
        "pregnancy_weeks": 32,
        "gravida": 2,
        "parity": 1,
        "previous_c_section": False,
        "heavy_bleeding": False,
        "severe_abdominal_pain": False,
        "blood_pressure": 120,
        "body_temperature": 36.8,
        "fetal_movement": "Normal",
        "consciousness": "Alert",
    }
    defaults.update(overrides)
    return PatientInfo(**defaults)


class ValidationTests(unittest.TestCase):
    """Validate production-grade input parsing and error handling."""

    def test_parse_valid_form_data(self) -> None:
        """Valid form input should produce a normalized patient object."""
        patient = parse_patient_from_form_data(VALID_FORM_DATA)

        self.assertEqual(patient.age, 28)
        self.assertEqual(patient.height_cm, 165)
        self.assertEqual(patient.weight_kg, 68.0)
        self.assertEqual(patient.bmi, 25.0)
        self.assertEqual(patient.pregnancy_weeks, 32)
        self.assertEqual(patient.gravida, 2)
        self.assertEqual(patient.parity, 1)
        self.assertFalse(patient.previous_c_section)
        self.assertFalse(patient.heavy_bleeding)
        self.assertFalse(patient.severe_abdominal_pain)
        self.assertEqual(patient.blood_pressure, 120)
        self.assertEqual(patient.body_temperature, 36.8)
        self.assertEqual(patient.fetal_movement, "Normal")
        self.assertEqual(patient.consciousness, "Alert")

    def test_parse_accepts_boolean_and_numeric_api_payloads(self) -> None:
        """JSON API payloads should accept native booleans and numeric types."""
        patient = parse_patient_from_form_data(
            {
                "age": 30,
                "height_cm": 160,
                "weight_kg": 72.5,
                "pregnancy_weeks": 30,
                "gravida": 3,
                "parity": 2,
                "previous_c_section": True,
                "heavy_bleeding": True,
                "severe_abdominal_pain": False,
                "blood_pressure": 95,
                "body_temperature": 38.1,
                "fetal_movement": "reduced",
                "consciousness": "drowsy",
            }
        )

        self.assertEqual(patient.bmi, 28.3)
        self.assertTrue(patient.heavy_bleeding)
        self.assertTrue(patient.previous_c_section)
        self.assertFalse(patient.severe_abdominal_pain)
        self.assertEqual(patient.fetal_movement, "Reduced")
        self.assertEqual(patient.consciousness, "Drowsy")

    def test_missing_required_field_raises_meaningful_error(self) -> None:
        """Missing fields should raise a dedicated exception with a clear message."""
        payload = dict(VALID_FORM_DATA)
        del payload["age"]

        with self.assertRaises(MissingValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "age")
        self.assertIn("Age is required", str(context.exception))

    def test_empty_string_raises_missing_value_error(self) -> None:
        """Blank submitted values should be treated as missing input."""
        payload = dict(VALID_FORM_DATA)
        payload["blood_pressure"] = "   "

        with self.assertRaises(MissingValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "blood_pressure")
        self.assertIn("Blood Pressure is required", str(context.exception))

    def test_invalid_number_raises_meaningful_error(self) -> None:
        """Non-numeric values should be rejected with a field-specific message."""
        payload = dict(VALID_FORM_DATA)
        payload["age"] = "twenty-eight"

        with self.assertRaises(InvalidNumericValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "age")
        self.assertIn("whole number", str(context.exception))

    def test_decimal_age_raises_meaningful_error(self) -> None:
        """Age must be a whole number."""
        payload = dict(VALID_FORM_DATA)
        payload["age"] = "28.5"

        with self.assertRaises(InvalidNumericValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "age")
        self.assertIn("whole number", str(context.exception))

    def test_out_of_range_number_raises_meaningful_error(self) -> None:
        """Out-of-range numeric values should include the allowed range."""
        payload = dict(VALID_FORM_DATA)
        payload["pregnancy_weeks"] = "50"

        with self.assertRaises(InvalidNumericValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "pregnancy_weeks")
        self.assertIn("between 1 and 45", str(context.exception))

    def test_invalid_yes_no_raises_meaningful_error(self) -> None:
        """Unrecognized Yes/No values should raise InvalidYesNoError."""
        payload = dict(VALID_FORM_DATA)
        payload["heavy_bleeding"] = "maybe"

        with self.assertRaises(InvalidYesNoError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "heavy_bleeding")
        self.assertIn("Yes or No", str(context.exception))

    def test_invalid_choice_raises_meaningful_error(self) -> None:
        """Invalid dropdown values should raise InvalidChoiceError."""
        payload = dict(VALID_FORM_DATA)
        payload["consciousness"] = "Sleeping"

        with self.assertRaises(InvalidChoiceError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "consciousness")
        self.assertIn("Alert, Drowsy, Unconscious", str(context.exception))

    def test_parity_must_be_less_than_gravida(self) -> None:
        """Parity should be clinically consistent with gravida."""
        payload = dict(VALID_FORM_DATA)
        payload["gravida"] = "2"
        payload["parity"] = "2"

        with self.assertRaises(InvalidNumericValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "parity")

    def test_predict_from_form_data_rejects_invalid_input(self) -> None:
        """The prediction entry point should surface validation failures."""
        with self.assertRaises(InvalidNumericValueError):
            predict_from_form_data({"age": "abc"})


class RiskEngineTests(unittest.TestCase):
    """Validate the rule-based risk engine behavior."""

    def test_low_risk_result(self) -> None:
        """A stable patient should receive a low-risk assessment."""
        patient = make_patient()
        result = RuleBasedRiskEngine().assess_risk(patient)

        self.assertEqual(result.risk_level, "Low Risk")
        self.assertGreaterEqual(result.risk_score, 0)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertIn("Low Risk", result.risk_level)

    def test_critical_risk_result(self) -> None:
        """Severe symptoms should produce a critical-risk assessment."""
        patient = make_patient(
            age=30,
            pregnancy_weeks=30,
            heavy_bleeding=True,
            severe_abdominal_pain=True,
            blood_pressure=90,
            body_temperature=39.0,
            fetal_movement="Absent",
            consciousness="Unconscious",
        )

        result = RuleBasedRiskEngine().assess_risk(patient)

        self.assertEqual(result.risk_level, "Critical Risk")
        self.assertGreaterEqual(result.risk_score, 9)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertIn("immediate", result.recommended_action.lower())

    def test_validation_rejects_invalid_patient(self) -> None:
        """Invalid data should be rejected by the validator."""
        patient = make_patient(age=0)

        with self.assertRaises(InvalidNumericValueError) as context:
            validate_patient_information(patient)

        self.assertEqual(context.exception.field, "age")
        self.assertIn("between 12 and 55", str(context.exception))

    def test_validation_rejects_missing_information(self) -> None:
        """Missing or empty values should raise a dedicated exception."""
        patient = make_patient(fetal_movement="")

        with self.assertRaises(MissingValueError) as context:
            validate_patient_information(patient)

        self.assertEqual(context.exception.field, "fetal_movement")

    def test_model_engine_falls_back_to_rule_based_when_model_missing(self) -> None:
        """The model-backed engine should degrade gracefully if the model artifact is absent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = os.path.join(temp_dir, "missing_model.pkl")
            engine = ModelRiskEngine(model_path=missing_path)
            patient = make_patient()
            result = engine.assess_risk(patient)
            self.assertIn(result.risk_level, {"Low Risk", "Moderate Risk", "High Risk", "Critical Risk"})

    def test_generate_shap_explanation_creates_visuals(self) -> None:
        """An assessment should produce SHAP-based explanation artifacts for clinicians."""
        patient = make_patient(
            age=30,
            pregnancy_weeks=30,
            heavy_bleeding=True,
            severe_abdominal_pain=True,
            blood_pressure=95,
            body_temperature=38.1,
            fetal_movement="Reduced",
            consciousness="Drowsy",
        )
        result = ModelRiskEngine().assess_risk(patient)
        explanation = generate_shap_explanation(patient, result)
        self.assertTrue(explanation["feature_importance_path"].endswith(".svg"))
        self.assertTrue(explanation["waterfall_path"].endswith(".svg"))
        self.assertTrue(os.path.exists(explanation["feature_importance_path"]))
        self.assertTrue(os.path.exists(explanation["waterfall_path"]))
        self.assertGreater(len(explanation["feature_names"]), 0)


if __name__ == "__main__":
    unittest.main()
