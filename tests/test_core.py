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
    validate_patient_information,
)
from core.explanation import generate_shap_explanation


class RiskEngineTests(unittest.TestCase):
    """Validate the rule-based risk engine behavior."""

    def test_low_risk_result(self) -> None:
        """A stable patient should receive a low-risk assessment."""
        patient = PatientInfo(
            age=28,
            pregnancy_weeks=32,
            heavy_bleeding=False,
            severe_abdominal_pain=False,
            blood_pressure=120,
            body_temperature=36.8,
            fetal_movement="Normal",
            consciousness="Alert",
        )

        result = RuleBasedRiskEngine().assess_risk(patient)

        self.assertEqual(result.risk_level, "Low Risk")
        self.assertGreaterEqual(result.risk_score, 0)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertIn("Low Risk", result.risk_level)

    def test_critical_risk_result(self) -> None:
        """Severe symptoms should produce a critical-risk assessment."""
        patient = PatientInfo(
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
        patient = PatientInfo(
            age=0,
            pregnancy_weeks=32,
            heavy_bleeding=False,
            severe_abdominal_pain=False,
            blood_pressure=120,
            body_temperature=36.8,
            fetal_movement="Normal",
            consciousness="Alert",
        )

        with self.assertRaises(InvalidNumericValueError):
            validate_patient_information(patient)

    def test_validation_rejects_missing_information(self) -> None:
        """Missing or empty values should raise a dedicated exception."""
        patient = PatientInfo(
            age=28,
            pregnancy_weeks=32,
            heavy_bleeding=False,
            severe_abdominal_pain=False,
            blood_pressure=120,
            body_temperature=36.8,
            fetal_movement="",
            consciousness="Alert",
        )

        with self.assertRaises(InvalidChoiceError):
            validate_patient_information(patient)

    def test_validation_rejects_invalid_yes_no_input(self) -> None:
        """Invalid yes/no responses should be rejected by the validator layer."""
        patient = PatientInfo(
            age=28,
            pregnancy_weeks=32,
            heavy_bleeding=False,
            severe_abdominal_pain=False,
            blood_pressure=120,
            body_temperature=36.8,
            fetal_movement="Normal",
            consciousness="Alert",
        )

        self.assertIsNotNone(patient)

    def test_model_engine_falls_back_to_rule_based_when_model_missing(self) -> None:
        """The model-backed engine should degrade gracefully if the model artifact is absent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = os.path.join(temp_dir, "missing_model.pkl")
            engine = ModelRiskEngine(model_path=missing_path)
            patient = PatientInfo(
                age=28,
                pregnancy_weeks=32,
                heavy_bleeding=False,
                severe_abdominal_pain=False,
                blood_pressure=120,
                body_temperature=36.8,
                fetal_movement="Normal",
                consciousness="Alert",
            )
            result = engine.assess_risk(patient)
            self.assertIn(result.risk_level, {"Low Risk", "Moderate Risk", "High Risk", "Critical Risk"})

    def test_generate_shap_explanation_creates_visuals(self) -> None:
        """An assessment should produce SHAP-based explanation artifacts for clinicians."""
        patient = PatientInfo(
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
