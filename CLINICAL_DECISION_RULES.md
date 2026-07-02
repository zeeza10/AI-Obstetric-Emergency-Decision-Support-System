# Clinical Decision Rules

This prototype classifies obstetric emergency risk using transparent clinical rules. It is intended for academic decision-support use and does not replace professional clinical judgment.

## Risk Levels

- Low Risk: total rule score 0-2, with no acute emergency pattern.
- Moderate Risk: total rule score 3-5, indicating concerning symptoms or history requiring prompt assessment.
- High Risk: total rule score 6-8, indicating potentially dangerous obstetric conditions requiring urgent hospital review.
- Critical Risk: total rule score 9 or higher, or heavy vaginal bleeding with loss of consciousness, indicating immediate emergency intervention.

## Added Checklist Risk Factors

| Factor | Rule Weight | Risk Meaning |
| --- | ---: | --- |
| Hypertension | +2 | Strong contributor because of hypertensive obstetric emergency risk. |
| Diabetes | +1 | Moderate contributor because of maternal and fetal complication risk. |
| Anemia | +1 | Moderate contributor because reduced physiologic reserve can worsen bleeding or systemic illness. |
| Heart Disease | +2 | Strong contributor because of maternal decompensation risk. |
| Multiple Pregnancy | +1 | Moderate contributor because of higher obstetric complication risk. |
| Previous Preeclampsia | +2 | Strong contributor because of recurrence and hypertensive emergency risk. |
| Previous Hemorrhage | +2 | Strong contributor because of recurrent bleeding risk. |

## Combination Rules

- Hypertension plus systolic BP at least 140 mmHg or diastolic BP at least 90 mmHg adds +1.
- Diabetes plus blood sugar below 70 mg/dL or above 180 mg/dL adds +1.
- Previous preeclampsia plus headache, blurred vision, or hypertension adds +1.
- Previous hemorrhage plus current heavy vaginal bleeding adds +1.
- Heart disease plus chest pain, difficulty breathing, or SpO2 below 95% adds +2.

## UI Checklist Options

The web assessment form includes Yes/No checklist options for:

- Hypertension
- Diabetes
- Anemia
- Heart Disease
- Multiple Pregnancy
- Previous Preeclampsia
- Previous Hemorrhage
