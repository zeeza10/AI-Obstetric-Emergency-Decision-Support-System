"""Recommendation formatting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .patient import PatientInfo
    from .risk_engine import AssessmentResult


def get_clinical_guidance(risk_level: str) -> Tuple[str, str]:
    """Return a clinical explanation and recommended action for a given risk level."""
    guidance = {
        "Low Risk": (
            "The reported symptoms do not currently suggest an acute obstetric emergency.",
            "Continue routine monitoring and seek medical attention if symptoms worsen.",
        ),
        "Moderate Risk": (
            "The patient has concerning symptoms that require prompt medical evaluation and close monitoring.",
            "Arrange prompt medical assessment and monitor the patient closely.",
        ),
        "High Risk": (
            "The symptoms indicate a potentially dangerous obstetric condition that needs urgent clinical review.",
            "Seek urgent hospital assessment now and prepare for immediate medical intervention.",
        ),
        "Critical Risk": (
            "The presentation suggests a severe obstetric emergency requiring immediate intervention.",
            "Call emergency services immediately and arrange urgent hospital transfer with continuous monitoring.",
        ),
    }
    return guidance.get(risk_level, guidance["Moderate Risk"])


def build_recommendation_summary(result: AssessmentResult) -> Dict[str, str]:
    """Convert an assessment result into a user-friendly summary dictionary."""
    return {
        "Risk Level": result.risk_level,
        "Risk Percentage": f"{result.risk_percentage:.0f}%",
        "AI Confidence": f"{result.ai_confidence:.0f}%",
        "Possible Diagnosis": result.possible_diagnosis,
        "Emergency Level": result.emergency_level,
        "Clinical Recommendation": result.clinical_recommendation,
        "WHO Triage Category": result.who_triage_category,
        "Immediate Action Required": "Yes" if result.immediate_action_required else "No",
        "Hospital Admission Required": "Yes" if result.hospital_admission_required else "No",
        "Explainable AI": result.explainable_ai,
        "PDF Report": result.pdf_report,
    }


def enhance_clinical_decision_support(
    result: AssessmentResult,
    patient: PatientInfo,
) -> AssessmentResult:
    """Populate clinical decision-support fields derived from assessment output."""
    result.risk_percentage = _calculate_risk_percentage(result)
    result.ai_confidence = round(result.confidence_score * 100.0, 1)
    result.possible_diagnosis = _build_possible_diagnosis(patient, result)
    result.emergency_level = _map_emergency_level(result.risk_level)
    result.clinical_recommendation = result.recommended_action
    result.who_triage_category = _map_who_triage_category(result.risk_level)
    result.immediate_action_required = result.risk_level in {"High Risk", "Critical Risk"}
    result.hospital_admission_required = _requires_admission(result.risk_level, patient)
    result.explainable_ai = _build_explainable_ai_summary(result, patient)
    result.pdf_report = "Available from the Print Report button; choose Save as PDF in the print dialog."
    return result


def _calculate_risk_percentage(result: AssessmentResult) -> float:
    """Convert risk class and score into a bounded percentage for reporting."""
    if result.risk_score > 3:
        return round(min(100.0, max(0.0, result.risk_score / 18.0 * 100.0)), 1)

    percentage_by_level = {
        "Low Risk": 18.0,
        "Moderate Risk": 45.0,
        "High Risk": 72.0,
        "Critical Risk": 92.0,
    }
    return percentage_by_level.get(result.risk_level, 45.0)


def _map_emergency_level(risk_level: str) -> str:
    """Map risk level to a clinician-facing emergency level."""
    levels = {
        "Low Risk": "Level 4 - Routine",
        "Moderate Risk": "Level 3 - Urgent Review",
        "High Risk": "Level 2 - Emergency",
        "Critical Risk": "Level 1 - Immediate Resuscitation",
    }
    return levels.get(risk_level, levels["Moderate Risk"])


def _map_who_triage_category(risk_level: str) -> str:
    """Map risk level to WHO-style color triage terminology."""
    categories = {
        "Low Risk": "Green - Non-urgent",
        "Moderate Risk": "Yellow - Urgent",
        "High Risk": "Orange - Very urgent",
        "Critical Risk": "Red - Emergency",
    }
    return categories.get(risk_level, categories["Moderate Risk"])


def _requires_admission(risk_level: str, patient: PatientInfo) -> bool:
    """Return whether hospital admission or observation is recommended."""
    if risk_level in {"High Risk", "Critical Risk"}:
        return True
    return any(
        (
            patient.heavy_vaginal_bleeding,
            patient.convulsions,
            patient.loss_of_consciousness,
            patient.fetal_movement == "Absent",
            patient.urine_protein in {"3+", "4+"},
            patient.platelet_count < 100,
            patient.hemoglobin < 7.0,
            patient.spo2 < 92,
        )
    )


def _build_possible_diagnosis(patient: PatientInfo, result: AssessmentResult) -> str:
    """Infer likely clinical syndromes from the submitted features."""
    diagnoses: List[str] = []

    if patient.convulsions or (
        patient.previous_preeclampsia
        and (patient.headache or patient.blurred_vision)
        and (patient.hypertension or patient.systolic_bp >= 140 or patient.diastolic_bp >= 90)
    ):
        diagnoses.append("Severe preeclampsia/eclampsia")

    if patient.urine_protein in {"2+", "3+", "4+"} and (
        patient.hypertension or patient.systolic_bp >= 140 or patient.diastolic_bp >= 90
    ):
        diagnoses.append("Preeclampsia")

    if patient.heavy_vaginal_bleeding or patient.bleeding_severity in {"Heavy", "Severe"}:
        diagnoses.append("Obstetric hemorrhage")

    if patient.abdominal_pain and patient.heavy_vaginal_bleeding:
        diagnoses.append("Placental abruption concern")

    if patient.fetal_movement in {"Reduced", "Absent"}:
        diagnoses.append("Fetal compromise")

    if patient.heart_disease and (patient.chest_pain or patient.difficulty_breathing or patient.spo2 < 95):
        diagnoses.append("Cardiopulmonary compromise")

    if patient.diabetes and (patient.blood_sugar > 180 or patient.urine_glucose in {"2+", "3+", "4+"}):
        diagnoses.append("Hyperglycemia in pregnancy")

    if patient.hemoglobin < 11.0 or patient.anemia:
        diagnoses.append("Maternal anemia")

    if patient.platelet_count < 150:
        diagnoses.append("Thrombocytopenia")

    if patient.body_temperature >= 38.0:
        diagnoses.append("Possible infection/sepsis")

    if not diagnoses:
        if result.risk_level == "Low Risk":
            return "No acute obstetric emergency pattern identified"
        return "Undifferentiated obstetric risk requiring clinician assessment"

    return "; ".join(diagnoses[:4])


def _build_explainable_ai_summary(result: AssessmentResult, patient: PatientInfo) -> str:
    """Create a compact explanation of the main risk drivers."""
    drivers: List[str] = []
    if patient.heavy_vaginal_bleeding:
        drivers.append("heavy vaginal bleeding")
    if patient.bleeding_severity in {"Heavy", "Severe"}:
        drivers.append(f"{patient.bleeding_severity.lower()} bleeding severity")
    if patient.systolic_bp >= 140 or patient.diastolic_bp >= 90 or patient.hypertension:
        drivers.append("hypertension/elevated blood pressure")
    if patient.urine_protein in {"2+", "3+", "4+"}:
        drivers.append("significant urine protein")
    if patient.platelet_count < 150:
        drivers.append("low platelet count")
    if patient.hemoglobin < 11:
        drivers.append("low hemoglobin")
    if patient.fetal_movement in {"Reduced", "Absent"}:
        drivers.append(f"{patient.fetal_movement.lower()} fetal movement")
    if patient.loss_of_consciousness:
        drivers.append("loss of consciousness")
    if patient.convulsions:
        drivers.append("convulsions")
    if patient.chest_pain or patient.difficulty_breathing or patient.spo2 < 95:
        drivers.append("cardiopulmonary symptoms")

    if not drivers:
        drivers.append("stable submitted vitals and absence of major warning signs")

    return (
        f"The AI classified this case as {result.risk_level} "
        f"({result.risk_percentage:.0f}% risk, {result.ai_confidence:.0f}% confidence) "
        f"based mainly on {', '.join(drivers[:5])}."
    )
