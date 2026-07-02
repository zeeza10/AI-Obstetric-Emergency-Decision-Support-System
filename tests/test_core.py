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
    "heavy_vaginal_bleeding": "false",
    "bleeding_severity": "None",
    "abdominal_pain": "false",
    "pain_score": "0",
    "fetal_movement": "Normal",
    "loss_of_consciousness": "false",
    "convulsions": "false",
    "headache": "false",
    "blurred_vision": "false",
    "difficulty_breathing": "false",
    "chest_pain": "false",
    "vomiting": "false",
    "systolic_bp": "120",
    "diastolic_bp": "80",
    "heart_rate": "78",
    "respiratory_rate": "16",
    "body_temperature": "36.8",
    "spo2": "98",
    "blood_sugar": "95",
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
        "heavy_vaginal_bleeding": False,
        "bleeding_severity": "None",
        "abdominal_pain": False,
        "pain_score": 0,
        "fetal_movement": "Normal",
        "loss_of_consciousness": False,
        "convulsions": False,
        "headache": False,
        "blurred_vision": False,
        "difficulty_breathing": False,
        "chest_pain": False,
        "vomiting": False,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "heart_rate": 78,
        "respiratory_rate": 16,
        "body_temperature": 36.8,
        "spo2": 98,
        "blood_sugar": 95.0,
    }
    defaults.update(overrides)
    return PatientInfo(**defaults)


class ValidationTests(unittest.TestCase):
    """Validate production-grade input parsing and error handling."""

    def test_parse_valid_form_data(self) -> None:
        """Valid form input should produce a normalized patient object."""
        patient = parse_patient_from_form_data(VALID_FORM_DATA)

        self.assertEqual(patient.age, 28)
        self.assertFalse(patient.heavy_vaginal_bleeding)
        self.assertEqual(patient.bleeding_severity, "None")
        self.assertEqual(patient.pain_score, 0)
        self.assertFalse(patient.loss_of_consciousness)

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
                "heavy_vaginal_bleeding": True,
                "bleeding_severity": "Heavy",
                "abdominal_pain": True,
                "pain_score": 7,
                "fetal_movement": "reduced",
                "loss_of_consciousness": False,
                "convulsions": False,
                "headache": True,
                "blurred_vision": False,
                "difficulty_breathing": False,
                "chest_pain": False,
                "vomiting": True,
                "systolic_bp": 95,
                "diastolic_bp": 60,
                "heart_rate": 110,
                "respiratory_rate": 22,
                "body_temperature": 38.1,
                "spo2": 93,
                "blood_sugar": 145,
            }
        )

        self.assertTrue(patient.heavy_vaginal_bleeding)
        self.assertEqual(patient.bleeding_severity, "Heavy")
        self.assertEqual(patient.pain_score, 7)
        self.assertTrue(patient.headache)
        self.assertEqual(patient.fetal_movement, "Reduced")

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
        payload["systolic_bp"] = "   "

        with self.assertRaises(MissingValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "systolic_bp")
        self.assertIn("Systolic BP is required", str(context.exception))

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
        payload["convulsions"] = "maybe"

        with self.assertRaises(InvalidYesNoError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "convulsions")
        self.assertIn("Yes or No", str(context.exception))

    def test_invalid_choice_raises_meaningful_error(self) -> None:
        """Invalid dropdown values should raise InvalidChoiceError."""
        payload = dict(VALID_FORM_DATA)
        payload["bleeding_severity"] = "Extreme"

        with self.assertRaises(InvalidChoiceError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "bleeding_severity")

    def test_parity_must_be_less_than_gravida(self) -> None:
        """Parity should be clinically consistent with gravida."""
        payload = dict(VALID_FORM_DATA)
        payload["gravida"] = "2"
        payload["parity"] = "2"

        with self.assertRaises(InvalidNumericValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "parity")

    def test_diastolic_bp_must_be_lower_than_systolic(self) -> None:
        """Diastolic BP must be clinically lower than systolic BP."""
        payload = dict(VALID_FORM_DATA)
        payload["systolic_bp"] = "120"
        payload["diastolic_bp"] = "125"

        with self.assertRaises(InvalidNumericValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "diastolic_bp")

    def test_bleeding_severity_requires_bleeding_when_reported(self) -> None:
        """Bleeding severity must match reported vaginal bleeding."""
        payload = dict(VALID_FORM_DATA)
        payload["heavy_vaginal_bleeding"] = "true"
        payload["bleeding_severity"] = "None"

        with self.assertRaises(InvalidChoiceError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "bleeding_severity")

    def test_pain_score_requires_abdominal_pain(self) -> None:
        """Pain score must be consistent with abdominal pain."""
        payload = dict(VALID_FORM_DATA)
        payload["abdominal_pain"] = "true"
        payload["pain_score"] = "0"

        with self.assertRaises(InvalidNumericValueError) as context:
            parse_patient_from_form_data(payload)

        self.assertEqual(context.exception.field, "pain_score")

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
            heavy_vaginal_bleeding=True,
            bleeding_severity="Severe",
            abdominal_pain=True,
            pain_score=9,
            systolic_bp=90,
            diastolic_bp=55,
            body_temperature=39.0,
            fetal_movement="Absent",
            loss_of_consciousness=True,
            convulsions=True,
        )

        result = RuleBasedRiskEngine().assess_risk(patient)

        self.assertEqual(result.risk_level, "Critical Risk")
        self.assertGreaterEqual(result.risk_score, 9)
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
            heavy_vaginal_bleeding=True,
            bleeding_severity="Heavy",
            abdominal_pain=True,
            pain_score=8,
            systolic_bp=95,
            diastolic_bp=60,
            heart_rate=110,
            respiratory_rate=22,
            body_temperature=38.1,
            spo2=93,
            blood_sugar=145.0,
            fetal_movement="Reduced",
            headache=True,
            blurred_vision=True,
            difficulty_breathing=True,
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
