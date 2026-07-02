"""Console-based obstetric emergency decision support system.

This Version 1 implementation uses a transparent, rule-based engine to assess
obstetric emergency risk from patient-reported symptoms. It is intentionally
modular so that a machine learning model can replace the rule engine later
without changing the user-facing workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PatientInfo:
    """Represents the patient information collected from the console."""

    age: int
    pregnancy_weeks: int
    heavy_bleeding: bool
    severe_abdominal_pain: bool
    blood_pressure: int
    body_temperature: float
    fetal_movement: str
    consciousness: str


@dataclass
class AssessmentResult:
    """Represents the output of the risk assessment process."""

    risk_level: str
    risk_score: int
    clinical_explanation: str
    recommended_action: str


class RiskEngine(ABC):
    """Abstract base class for future risk assessment implementations."""

    @abstractmethod
    def assess_risk(self, patient: PatientInfo) -> AssessmentResult:
        """Assess risk for a patient and return a structured result."""


class RuleBasedRiskEngine(RiskEngine):
    """Transparent rule-based engine for Version 1."""

    def assess_risk(self, patient: PatientInfo) -> AssessmentResult:
        """Evaluate risk using a simple, explainable scoring approach."""
        score = 0
        reasons: List[str] = []

        if patient.age < 18 or patient.age > 35:
            score += 1
            reasons.append("Maternal age is outside the standard low-risk range.")

        if patient.pregnancy_weeks < 20 or patient.pregnancy_weeks > 42:
            score += 1
            reasons.append("Gestational age is outside the expected low-risk range.")

        if patient.heavy_bleeding:
            score += 3
            reasons.append("Heavy bleeding is a major warning sign.")

        if patient.severe_abdominal_pain:
            score += 2
            reasons.append("Severe abdominal pain suggests possible serious complications.")

        if patient.blood_pressure < 90 or patient.blood_pressure > 140:
            score += 2
            reasons.append("Blood pressure is outside the expected range.")

        if patient.body_temperature >= 38.0:
            score += 2
            reasons.append("Elevated temperature may indicate infection or inflammation.")

        if patient.fetal_movement == "Reduced":
            score += 2
            reasons.append("Fetal movement is reduced and needs urgent review.")
        elif patient.fetal_movement == "Absent":
            score += 4
            reasons.append("Absent fetal movement is a serious fetal concern.")

        if patient.consciousness == "Drowsy":
            score += 2
            reasons.append("The patient appears drowsy.")
        elif patient.consciousness == "Unconscious":
            score += 5
            reasons.append("The patient is unconscious.")

        if (
            patient.heavy_bleeding
            and patient.severe_abdominal_pain
            and patient.blood_pressure < 100
        ):
            score += 2
            reasons.append("Bleeding with abdominal pain and low blood pressure suggests instability.")

        risk_level, explanation, action = self._classify(score, patient)
        return AssessmentResult(
            risk_level=risk_level,
            risk_score=score,
            clinical_explanation=f"{explanation} {' '.join(reasons)}",
            recommended_action=action,
        )

    def _classify(self, score: int, patient: PatientInfo) -> tuple[str, str, str]:
        """Convert the computed score into a final risk category and recommendation."""
        if score >= 9 or (patient.heavy_bleeding and patient.consciousness == "Unconscious"):
            return (
                "Critical Risk",
                "The presentation suggests a severe obstetric emergency requiring immediate intervention.",
                "Call emergency services immediately and arrange urgent hospital transfer with continuous monitoring.",
            )

        if score >= 6:
            return (
                "High Risk",
                "The symptoms indicate a potentially dangerous obstetric condition that needs urgent clinical review.",
                "Seek urgent hospital assessment now and prepare for immediate medical intervention.",
            )

        if score >= 3:
            return (
                "Moderate Risk",
                "The patient has concerning symptoms that require prompt medical evaluation and close monitoring.",
                "Arrange prompt medical assessment and monitor the patient closely.",
            )

        return (
            "Low Risk",
            "The reported symptoms do not currently suggest an acute obstetric emergency.",
            "Continue routine monitoring and seek medical attention if symptoms worsen.",
        )


def prompt_int(prompt: str, minimum: int = 0) -> int:
    """Prompt the user until a valid integer is entered."""
    while True:
        raw_value = input(prompt).strip()
        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a valid whole number.")
            continue

        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue

        return value


def prompt_float(prompt: str, minimum: float = 0.0) -> float:
    """Prompt the user until a valid decimal number is entered."""
    while True:
        raw_value = input(prompt).strip()
        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue

        return value


def prompt_yes_no(prompt: str) -> bool:
    """Prompt the user until a Yes/No answer is provided."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with Yes or No.")


def prompt_choice(prompt: str, choices: List[str]) -> str:
    """Prompt the user until a valid choice is entered."""
    while True:
        answer = input(prompt).strip().title()
        if answer in choices:
            return answer
        print(f"Please choose from: {', '.join(choices)}")


def collect_patient_information() -> PatientInfo:
    """Collect patient information from the user via the console."""
    print("\nAI-Powered Obstetric Emergency Decision Support System")
    print("Please enter the following clinical information.\n")

    age = prompt_int("Patient age: ", minimum=1)
    pregnancy_weeks = prompt_int("Pregnancy weeks: ", minimum=1)
    heavy_bleeding = prompt_yes_no("Heavy bleeding (Yes/No): ")
    severe_abdominal_pain = prompt_yes_no("Severe abdominal pain (Yes/No): ")
    blood_pressure = prompt_int("Blood pressure (Systolic): ", minimum=1)
    body_temperature = prompt_float("Body temperature (°C): ", minimum=30.0)
    fetal_movement = prompt_choice(
        "Fetal movement (Normal/Reduced/Absent): ",
        ["Normal", "Reduced", "Absent"],
    )
    consciousness = prompt_choice(
        "Consciousness (Alert/Drowsy/Unconscious): ",
        ["Alert", "Drowsy", "Unconscious"],
    )

    return PatientInfo(
        age=age,
        pregnancy_weeks=pregnancy_weeks,
        heavy_bleeding=heavy_bleeding,
        severe_abdominal_pain=severe_abdominal_pain,
        blood_pressure=blood_pressure,
        body_temperature=body_temperature,
        fetal_movement=fetal_movement,
        consciousness=consciousness,
    )


def validate_patient_information(patient: PatientInfo) -> None:
    """Validate that the collected values are clinically reasonable."""
    if patient.age <= 0:
        raise ValueError("Age must be greater than zero.")
    if patient.pregnancy_weeks <= 0 or patient.pregnancy_weeks > 45:
        raise ValueError("Pregnancy weeks must be between 1 and 45.")
    if patient.blood_pressure <= 0:
        raise ValueError("Blood pressure must be greater than zero.")
    if patient.body_temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")


def assess_risk(patient: PatientInfo, engine: RiskEngine) -> AssessmentResult:
    """Assess the patient's risk using the selected engine."""
    validate_patient_information(patient)
    return engine.assess_risk(patient)


def generate_recommendations(result: AssessmentResult) -> Dict[str, str]:
    """Create a clear summary dictionary for display."""
    return {
        "Risk Level": result.risk_level,
        "Risk Score": str(result.risk_score),
        "Clinical Explanation": result.clinical_explanation,
        "Recommended Immediate Action": result.recommended_action,
    }


def display_results(result: AssessmentResult) -> None:
    """Show the assessment result to the user in a readable format."""
    print("\nAssessment Result")
    print("-" * 45)
    print(f"Risk Level: {result.risk_level}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Clinical Explanation: {result.clinical_explanation}")
    print(f"Recommended Immediate Action: {result.recommended_action}")


def main() -> None:
    """Run the console-based assessment workflow."""
    patient = collect_patient_information()
    engine = RuleBasedRiskEngine()
    result = assess_risk(patient, engine)
    display_results(result)


if __name__ == "__main__":
    main()
