from core.explanation import generate_shap_explanation
from core.patient import PatientInfo
from core.risk_engine import AssessmentResult

patient = PatientInfo(
    age=30,
    pregnancy_weeks=30,
    heavy_bleeding=True,
    severe_abdominal_pain=True,
    blood_pressure=95,
    body_temperature=38.1,
    fetal_movement='Reduced',
    consciousness='Drowsy',
)
result = AssessmentResult(
    risk_level='High Risk',
    risk_score=2.0,
    confidence_score=0.92,
    clinical_explanation='demo',
    recommended_action='demo',
)
explanation = generate_shap_explanation(patient, result)
print(explanation['explanation_text'])
print(explanation['feature_importance_path'])
print(explanation['waterfall_path'])
