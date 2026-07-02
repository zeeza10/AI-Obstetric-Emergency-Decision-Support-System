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

## Added Lab Findings

| Finding | Rule Weight | Risk Meaning |
| --- | ---: | --- |
| Hemoglobin below 11 g/dL | +1 | Low hemoglobin increases concern for anemia and reduced reserve. |
| Hemoglobin below 7 g/dL | +3 | Severe anemia is treated as a critical physiologic risk marker. |
| Platelet count 100-149 x10^9/L | +1 | Mild thrombocytopenia increases clinical concern. |
| Platelet count 50-99 x10^9/L | +2 | Low platelets require urgent evaluation. |
| Platelet count below 50 x10^9/L | +3 | Very low platelets indicate high bleeding or severe disease risk. |
| Urine protein 1+ | +1 | Positive protein requires clinical review. |
| Urine protein 2+ | +2 | Moderate proteinuria increases preeclampsia concern. |
| Urine protein 3+ or 4+ | +3 | Marked proteinuria is a major preeclampsia warning sign. |
| Urine glucose 1+ to 4+ | +1 | Positive glucose may indicate poor glycemic control. |

## Combination Rules

- Hypertension plus systolic BP at least 140 mmHg or diastolic BP at least 90 mmHg adds +1.
- Diabetes plus blood sugar below 70 mg/dL or above 180 mg/dL adds +1.
- Diabetes plus urine glucose 2+ or higher adds +1.
- Previous preeclampsia plus headache, blurred vision, or hypertension adds +1.
- Urine protein 2+ or higher plus hypertension, systolic BP at least 140 mmHg, or diastolic BP at least 90 mmHg adds +2.
- Previous hemorrhage plus current heavy vaginal bleeding adds +1.
- Current heavy vaginal bleeding plus hemoglobin below 11 g/dL adds +1.
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

The web assessment form also includes lab inputs for:

- Hemoglobin
- Urine Protein
- Platelet Count
- Urine Glucose
