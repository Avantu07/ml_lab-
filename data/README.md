# Data

`heart-disease-dataset.csv` - real patient records (1,190 rows).

| Column | Description |
|---|---|
| age | Age in years |
| sex | 0 = female, 1 = male |
| chest_pain_type | 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic |
| resting_bp_s | Resting blood pressure (mm Hg) |
| cholesterol | Serum cholesterol (mg/dl) |
| fasting_blood_sugar | 1 if > 120 mg/dl, else 0 |
| resting_ecg | 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy |
| max_heart_rate | Maximum heart rate achieved (bpm) |
| exercise_angina | 1 = exercise-induced angina |
| oldpeak | ST depression induced by exercise relative to rest |
| st_slope | 1 upsloping, 2 flat, 3 downsloping |
| target | 0 = no disease, 1 = disease |

Cleaning: duplicate rows are dropped; cholesterol / resting BP of 0 and
st_slope of 0 are impossible or undefined and are treated as missing, then
imputed inside the model pipeline (training data only).
