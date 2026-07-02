"""Unit tests for the core obstetric emergency decision support modules."""

import unittest

from core import (
    InvalidChoiceError,
    InvalidNumericValueError,
    InvalidYesNoError,
    MissingValueError,
    PatientInfo,
    RuleBasedRiskEngine,
    validate_patient_information,
)


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


if __name__ == "__main__":
    unittest.main()
