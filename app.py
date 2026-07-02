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
from predict import predict_from_form_data

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/", methods=["GET"])
def index() -> str:
    """Render the patient assessment form."""
    return render_template("index.html", error_message=None)


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Process the patient form and render the assessment result."""
    try:
        result, patient = predict_from_form_data(request.form)
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
