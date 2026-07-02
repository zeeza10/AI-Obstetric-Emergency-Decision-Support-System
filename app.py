"""Flask web application for obstetric emergency risk assessment."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from config import get_config
from core import ValidationError
from core.audit import log_assessment_audit, setup_audit_logger
from core.explanation import generate_shap_explanation
from core.logger import setup_logger
from core.recommendation import build_recommendation_summary
from predict import predict_from_form_data

config = get_config()
logger = setup_logger(log_file="logs/app.log")
audit_logger = setup_audit_logger(log_file=config.audit_log_file)
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = config.secret_key
app.config["APP_NAME"] = config.app_name
app.config["APP_VERSION"] = config.version
app.config["COMPANY_NAME"] = config.company_name
app.config["FOOTER_TEXT"] = config.footer_text


def _build_assessment_response(result, patient, explanation, source: str):
    """Run shared post-processing for web and API assessment flows."""
    risk_class = result.risk_level.lower().replace(" ", "-")
    logger.info(
        "Assessment completed | source=%s | risk=%s | score=%.2f | confidence=%.2f | age=%s | weeks=%s | bleeding=%s | pain=%s",
        source,
        result.risk_level,
        result.risk_score,
        result.confidence_score,
        patient.age,
        patient.pregnancy_weeks,
        patient.heavy_vaginal_bleeding,
        patient.abdominal_pain,
    )
    log_assessment_audit(audit_logger, patient, result, source=source)
    return {
        "result": result,
        "patient": patient,
        "risk_class": risk_class,
        "explanation": explanation,
        "patient_summary": patient.to_summary_dict(),
        "recommendation_summary": build_recommendation_summary(result),
    }


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
        payload = _build_assessment_response(result, patient, explanation, source="web")
        return render_template(
            "result.html",
            result=payload["result"],
            patient=payload["patient"],
            patient_summary=payload["patient_summary"],
            risk_class=payload["risk_class"],
            explanation=payload["explanation"],
        )
    except (ValidationError, ValueError, TypeError) as exc:
        logger.exception("Assessment request failed: %s", exc)
        return render_template("index.html", error_message=str(exc)), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Return a JSON assessment result for programmatic integrations."""
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        result, patient = predict_from_form_data(payload)
        explanation = generate_shap_explanation(patient, result)
        assessment = _build_assessment_response(result, patient, explanation, source="api")
        return jsonify(
            {
                "status": "ok",
                "risk_level": result.risk_level,
                "risk_score": result.risk_score,
                "confidence_score": result.confidence_score,
                "clinical_explanation": result.clinical_explanation,
                "recommended_action": result.recommended_action,
                "patient_summary": assessment["patient_summary"],
                "recommendation_summary": assessment["recommendation_summary"],
                "explanation": {
                    "text": explanation["explanation_text"],
                    "feature_names": explanation["feature_names"],
                    "shap_values": explanation["shap_values"],
                    "feature_importance_url": explanation["feature_importance_url"],
                    "waterfall_url": explanation["waterfall_url"],
                },
            }
        ), 200
    except (ValidationError, ValueError, TypeError) as exc:
        logger.exception("API assessment request failed: %s", exc)
        response = {"status": "error", "message": str(exc)}
        if isinstance(exc, ValidationError) and exc.field:
            response["field"] = exc.field
        return jsonify(response), 400


@app.route("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "version": config.version, "engine": config.engine_type}


def _wants_json_response() -> bool:
    """Return True when the client expects a JSON error payload."""
    return request.path.startswith("/api") or request.accept_mimetypes.best == "application/json"


@app.errorhandler(404)
def not_found(error: HTTPException):
    """Render a professional 404 page or JSON error for API clients."""
    logger.warning("404 Not Found: %s", request.path)
    if _wants_json_response():
        return jsonify({"status": "error", "message": "The requested resource was not found."}), 404
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error: HTTPException):
    """Render a professional 500 page or JSON error for API clients."""
    logger.exception("500 Internal Server Error: %s", request.path)
    if _wants_json_response():
        return jsonify({"status": "error", "message": "An unexpected server error occurred."}), 500
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=config.debug, host="0.0.0.0", port=5000)
