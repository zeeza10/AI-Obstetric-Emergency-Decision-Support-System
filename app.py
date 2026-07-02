"""Console-based obstetric emergency decision support system.

This Version 1 implementation uses a transparent, rule-based engine to assess
obstetric emergency risk from patient-reported symptoms. It is intentionally
modular so that a machine learning model can replace the rule engine later
without changing the user-facing workflow.
"""

from __future__ import annotations

from typing import List

from core import (
    AssessmentResult,
    InvalidChoiceError,
    InvalidNumericValueError,
    InvalidYesNoError,
    MissingValueError,
    PatientInfo,
    ModelRiskEngine,
    RiskEngine,
    RuleBasedRiskEngine,
    build_recommendation_summary,
    setup_logger,
    validate_patient_information,
)

LOGGER = setup_logger(log_file="logs/app.log")


def prompt_int(prompt: str, minimum: int = 0) -> int:
    """Prompt the user until a valid integer is entered."""
    while True:
        raw_value = input(prompt).strip()
        try:
            if not raw_value:
                raise MissingValueError("A value is required.")
            value = int(raw_value)
            if value < minimum:
                raise InvalidNumericValueError(f"Value must be at least {minimum}.")
            return value
        except ValueError as exc:
            print("Please enter a valid whole number.")
        except (MissingValueError, InvalidNumericValueError) as exc:
            print(str(exc))


def prompt_float(prompt: str, minimum: float = 0.0) -> float:
    """Prompt the user until a valid decimal number is entered."""
    while True:
        raw_value = input(prompt).strip()
        try:
            if not raw_value:
                raise MissingValueError("A value is required.")
            value = float(raw_value)
            if value < minimum:
                raise InvalidNumericValueError(f"Value must be at least {minimum}.")
            return value
        except ValueError as exc:
            print("Please enter a valid number.")
        except (MissingValueError, InvalidNumericValueError) as exc:
            print(str(exc))


def prompt_yes_no(prompt: str) -> bool:
    """Prompt the user until a Yes/No answer is provided."""
    while True:
        answer = input(prompt).strip().lower()
        try:
            if not answer:
                raise MissingValueError("A value is required.")
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            raise InvalidYesNoError("Please answer with Yes or No.")
        except (MissingValueError, InvalidYesNoError) as exc:
            print(str(exc))


def prompt_choice(prompt: str, choices: List[str]) -> str:
    """Prompt the user until a valid choice is entered."""
    while True:
        answer = input(prompt).strip().title()
        try:
            if not answer:
                raise MissingValueError("A value is required.")
            if answer in choices:
                return answer
            raise InvalidChoiceError(f"Please choose from: {', '.join(choices)}")
        except InvalidChoiceError as exc:
            print(str(exc))
        except MissingValueError as exc:
            print(str(exc))


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


def assess_risk(patient: PatientInfo, engine: RiskEngine) -> AssessmentResult:
    """Assess the patient's risk using the selected engine."""
    try:
        validate_patient_information(patient)
        result = engine.assess_risk(patient)
        LOGGER.info(
            "Assessment completed: age=%s weeks=%s risk=%s score=%s",
            patient.age,
            patient.pregnancy_weeks,
            result.risk_level,
            result.risk_score,
        )
        return result
    except (InvalidNumericValueError, InvalidChoiceError, InvalidYesNoError, MissingValueError) as exc:
        LOGGER.warning("Assessment validation failed: %s", exc)
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Unexpected assessment error: %s", exc)
        raise


def generate_recommendations(result: AssessmentResult) -> dict[str, str]:
    """Create a clear summary dictionary for display."""
    return build_recommendation_summary(result)


def display_results(result: AssessmentResult) -> None:
    """Show the assessment result to the user in a readable format."""
    print("\nAssessment Result")
    print("-" * 45)
    print(f"Risk Level: {result.risk_level}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Clinical Explanation: {result.clinical_explanation}")
    print(f"Recommended Immediate Action: {result.recommended_action}")


def main() -> None:
    """Run the console-based assessment workflow."""
    try:
        patient = collect_patient_information()
        engine = ModelRiskEngine()
        result = assess_risk(patient, engine)
        display_results(result)
    except Exception as exc:
        LOGGER.error("Application error: %s", exc)
        raise


if __name__ == "__main__":
    main()
