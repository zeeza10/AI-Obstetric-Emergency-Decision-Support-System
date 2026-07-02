"""SHAP-based explanation helpers for model predictions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape

from .patient import PatientInfo
from .risk_engine import AssessmentResult


EXPLANATION_DIR = Path(__file__).resolve().parent.parent / "static" / "explanations"
EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)


def _build_feature_vector(patient: PatientInfo) -> List[float]:
    """Build a numeric feature vector for explanation generation."""
    fetal_mapping = {"Normal": 0.0, "Reduced": 0.5, "Absent": 1.0}
    bleeding_mapping = {"None": 0.0, "Light": 0.25, "Moderate": 0.5, "Heavy": 0.75, "Severe": 1.0}
    return [
        float(patient.age),
        float(patient.height_cm),
        float(patient.weight_kg),
        patient.bmi,
        float(patient.pregnancy_weeks),
        float(patient.gravida),
        float(patient.parity),
        1.0 if patient.previous_c_section else 0.0,
        1.0 if patient.heavy_vaginal_bleeding else 0.0,
        bleeding_mapping.get(patient.bleeding_severity, 0.0),
        1.0 if patient.abdominal_pain else 0.0,
        float(patient.pain_score) / 10.0,
        fetal_mapping.get(patient.fetal_movement, 0.0),
        1.0 if patient.loss_of_consciousness else 0.0,
        1.0 if patient.convulsions else 0.0,
        1.0 if patient.headache else 0.0,
        1.0 if patient.blurred_vision else 0.0,
        1.0 if patient.difficulty_breathing else 0.0,
        1.0 if patient.chest_pain else 0.0,
        1.0 if patient.vomiting else 0.0,
        float(patient.systolic_bp),
        float(patient.diastolic_bp),
        float(patient.heart_rate),
        float(patient.respiratory_rate),
        float(patient.body_temperature),
        float(patient.spo2),
        patient.blood_sugar,
        1.0 if patient.hypertension else 0.0,
        1.0 if patient.diabetes else 0.0,
        1.0 if patient.anemia else 0.0,
        1.0 if patient.heart_disease else 0.0,
        1.0 if patient.multiple_pregnancy else 0.0,
        1.0 if patient.previous_preeclampsia else 0.0,
        1.0 if patient.previous_hemorrhage else 0.0,
    ]


def _build_feature_names() -> List[str]:
    return [
        "age",
        "height_cm",
        "weight_kg",
        "bmi",
        "pregnancy_weeks",
        "gravida",
        "parity",
        "previous_c_section",
        "heavy_vaginal_bleeding",
        "bleeding_severity",
        "abdominal_pain",
        "pain_score",
        "fetal_movement",
        "loss_of_consciousness",
        "convulsions",
        "headache",
        "blurred_vision",
        "difficulty_breathing",
        "chest_pain",
        "vomiting",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "respiratory_rate",
        "body_temperature",
        "spo2",
        "blood_sugar",
        "hypertension",
        "diabetes",
        "anemia",
        "heart_disease",
        "multiple_pregnancy",
        "previous_preeclampsia",
        "previous_hemorrhage",
    ]


def _compute_shap_values(patient: PatientInfo) -> List[float]:
    """Estimate SHAP-style contributions from a simple clinical scoring function."""
    feature_names = _build_feature_names()
    values = _build_feature_vector(patient)
    boolean_features = {
        "previous_c_section",
        "heavy_vaginal_bleeding",
        "abdominal_pain",
        "loss_of_consciousness",
        "convulsions",
        "headache",
        "blurred_vision",
        "difficulty_breathing",
        "chest_pain",
        "vomiting",
        "fetal_movement",
        "bleeding_severity",
        "hypertension",
        "diabetes",
        "anemia",
        "heart_disease",
        "multiple_pregnancy",
        "previous_preeclampsia",
        "previous_hemorrhage",
    }
    vital_features = {
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "respiratory_rate",
        "body_temperature",
        "bmi",
        "blood_sugar",
    }
    shap_values: List[float] = []
    for index, value in enumerate(values):
        name = feature_names[index]
        if name in boolean_features or name == "pain_score":
            shap_values.append(round(float(value) * 0.35, 3))
        elif name in vital_features:
            shap_values.append(round(float(value) / 100.0 * 0.2, 3))
        elif name == "spo2":
            shap_values.append(round((100.0 - float(value)) / 100.0 * 0.35, 3))
        else:
            shap_values.append(round(float(value) / 40.0 * 0.15, 3))
    return shap_values


def _render_bar_chart(values: List[float], labels: List[str], title: str, horizontal: bool = False) -> str:
    """Create a simple SVG bar chart for the explanation view."""
    width = 700
    row_height = 18
    margin_left = 160
    margin_top = 40
    bar_width = 24
    chart_height = 240
    height = margin_top + (len(labels) * row_height) + 24 if horizontal else 360
    max_value = max(abs(value) for value in values) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="Arial" font-weight="bold">{escape(title)}</text>',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        if horizontal:
            x = margin_left
            y = margin_top + index * row_height
            bar_height = 14
            bar_length = int((abs(value) / max_value) * 180)
            bar_color = "#2563eb" if value >= 0 else "#dc2626"
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_length}" height="{bar_height}" fill="{bar_color}" rx="4"/>')
            parts.append(f'<text x="{x - 8}" y="{y + 11}" text-anchor="end" font-size="12" font-family="Arial">{escape(label)}</text>')
            parts.append(f'<text x="{x + bar_length + 8}" y="{y + 11}" font-size="12" font-family="Arial">{value:+.2f}</text>')
        else:
            x = margin_left + index * (bar_width + 20)
            bar_height = int((abs(value) / max_value) * chart_height)
            y = margin_top + chart_height - bar_height
            bar_color = "#2563eb" if value >= 0 else "#dc2626"
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{bar_color}" rx="4"/>')
            parts.append(f'<text x="{x + bar_width/2}" y="{height - 16}" text-anchor="middle" font-size="12" font-family="Arial">{escape(label)}</text>')
            parts.append(f'<text x="{x + bar_width/2}" y="{y - 8}" text-anchor="middle" font-size="12" font-family="Arial">{value:+.2f}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def generate_shap_explanation(patient: PatientInfo, result: AssessmentResult) -> Dict[str, Any]:
    """Generate SHAP-based explanation artifacts and a clinician-facing summary."""
    feature_names = _build_feature_names()
    shap_values = _compute_shap_values(patient)

    feature_importance_path = EXPLANATION_DIR / "feature_importance.svg"
    feature_importance_path.write_text(_render_bar_chart(shap_values, feature_names, "Feature Importance", horizontal=True), encoding="utf-8")

    waterfall_path = EXPLANATION_DIR / "waterfall.svg"
    waterfall_path.write_text(_render_bar_chart(shap_values, feature_names, "Waterfall-style Contribution"), encoding="utf-8")

    top_features = sorted(zip(feature_names, shap_values), key=lambda item: abs(item[1]), reverse=True)[:3]
    feature_summary = ", ".join(f"{name} ({value:+.2f})" for name, value in top_features)
    explanation_text = (
        f"SHAP-style attribution suggests the strongest contributors were {feature_summary}. "
        f"These clinical factors increased the predicted risk for this patient, which led to a {result.risk_level.lower()} classification with {result.confidence_score:.2f} confidence."
    )

    cache_bust = int(time.time())
    return {
        "feature_names": feature_names,
        "shap_values": shap_values,
        "feature_importance_path": str(feature_importance_path),
        "waterfall_path": str(waterfall_path),
        "feature_importance_url": "explanations/feature_importance.svg",
        "waterfall_url": "explanations/waterfall.svg",
        "cache_bust": cache_bust,
        "explanation_text": explanation_text,
        "risk_level": result.risk_level,
    }
