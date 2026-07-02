"""Flask web application for obstetric emergency risk assessment."""

from __future__ import annotations

from flask import Flask, render_template, request

from core import (
    InvalidChoiceError,
    InvalidNumericValueError,
    InvalidYesNoError,
    MissingValueError,
)
from core.explanation import generate_shap_explanation
from core.patient import PatientInfo
from predict import predict_risk

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/", methods=["GET"])
def index() -> str:
    """Render the patient assessment form."""
    return render_template("index.html", error_message=None)


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Process the patient form and render the assessment result."""
    try:
        age = int(request.form.get("age", "0"))
        pregnancy_weeks = int(request.form.get("pregnancy_weeks", "0"))
        heavy_bleeding = request.form.get("heavy_bleeding") == "true"
        severe_abdominal_pain = request.form.get("severe_abdominal_pain") == "true"
        blood_pressure = int(request.form.get("blood_pressure", "0"))
        body_temperature = float(request.form.get("body_temperature", "0"))
        fetal_movement = request.form.get("fetal_movement", "Normal")
        consciousness = request.form.get("consciousness", "Alert")

        result = predict_risk(
            age=age,
            pregnancy_weeks=pregnancy_weeks,
            heavy_bleeding=heavy_bleeding,
            severe_abdominal_pain=severe_abdominal_pain,
            blood_pressure=blood_pressure,
            body_temperature=body_temperature,
            fetal_movement=fetal_movement,
            consciousness=consciousness,
        )
        patient = PatientInfo(
            age=age,
            pregnancy_weeks=pregnancy_weeks,
            heavy_bleeding=heavy_bleeding,
            severe_abdominal_pain=severe_abdominal_pain,
            blood_pressure=blood_pressure,
            body_temperature=body_temperature,
            fetal_movement=fetal_movement,
            consciousness=consciousness,
        )
        explanation = generate_shap_explanation(patient, result)
        risk_class = result.risk_level.lower().replace(" ", "-")
        return render_template(
            "result.html",
            result=result,
            risk_class=risk_class,
            explanation=explanation,
        )
    except (ValueError, TypeError, MissingValueError, InvalidNumericValueError, InvalidChoiceError, InvalidYesNoError) as exc:
        return render_template("index.html", error_message=str(exc)), 400


@app.route("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
