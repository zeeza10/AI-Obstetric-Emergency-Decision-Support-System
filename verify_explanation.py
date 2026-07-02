from core.explanation import generate_shap_explanation
from core.patient import PatientInfo
from core.risk_engine import AssessmentResult

patient = PatientInfo(
    age=30,
    height_cm=160,
    weight_kg=72.5,
    pregnancy_weeks=30,
    gravida=3,
    parity=2,
    previous_c_section=True,
    heavy_vaginal_bleeding=True,
    bleeding_severity='Heavy',
    abdominal_pain=True,
    pain_score=8,
    fetal_movement='Reduced',
    loss_of_consciousness=False,
    convulsions=False,
    headache=True,
    blurred_vision=True,
    difficulty_breathing=True,
    chest_pain=False,
    vomiting=True,
    systolic_bp=95,
    diastolic_bp=60,
    heart_rate=110,
    respiratory_rate=22,
    body_temperature=38.1,
    spo2=93,
    blood_sugar=145.0,
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
