"""Shared constants, paths and feature metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
RESULT_DIR = ROOT / "reports" / "results"

DATA_FILE = DATA_DIR / "heart-disease-dataset.csv"
# Values of 0 that are physiologically impossible (recorded for "unknown").
ZERO_MEANS_MISSING = ["resting_bp_s", "cholesterol"]
# Rows missing more than this many of the 11 inputs are dropped.
MAX_MISSING_FEATURES = 3

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CLUSTER_COUNT = 3

COLUMNS = [
    "age", "sex", "chest_pain_type", "resting_bp_s", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "oldpeak", "st_slope", "target",
]
TARGET = "target"
FEATURES = COLUMNS[:-1]

NUMERIC_FEATURES = [
    "age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak",
]
ORDINAL_FEATURES = []
BINARY_FEATURES = ["sex", "fasting_blood_sugar", "exercise_angina"]
NOMINAL_FEATURES = ["chest_pain_type", "resting_ecg", "st_slope"]

FEATURE_INFO = {
    "age": {"label": "Age (years)", "min": 1, "max": 120, "default": 54},
    "sex": {"label": "Sex", "choices": {0: "Female", 1: "Male"}},
    "chest_pain_type": {"label": "Chest pain type", "choices": {
        1: "1 - Typical angina", 2: "2 - Atypical angina",
        3: "3 - Non-anginal pain", 4: "4 - Asymptomatic"}},
    "resting_bp_s": {"label": "Resting blood pressure (mm Hg)",
                     "min": 50, "max": 250, "default": 130},
    "cholesterol": {"label": "Serum cholesterol (mg/dl)",
                    "min": 50, "max": 700, "default": 240},
    "fasting_blood_sugar": {"label": "Fasting blood sugar > 120 mg/dl",
                            "choices": {0: "No", 1: "Yes"}},
    "resting_ecg": {"label": "Resting ECG result", "choices": {
        0: "0 - Normal", 1: "1 - ST-T abnormality",
        2: "2 - Left ventricular hypertrophy"}},
    "max_heart_rate": {"label": "Maximum heart rate achieved (bpm)",
                       "min": 40, "max": 250, "default": 150},
    "exercise_angina": {"label": "Exercise-induced angina",
                        "choices": {0: "No", 1: "Yes"}},
    "oldpeak": {"label": "ST depression (exercise vs rest)",
                "min": -3.0, "max": 10.0, "default": 1.0},
    "st_slope": {"label": "Slope of peak-exercise ST segment", "choices": {
        1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}},
}

CLUSTER_MODEL_FILE = "agglomerative_clustering.joblib"
CLUSTER_TABLE = TABLE_DIR / "cluster_summary.csv"

MODEL_FILES = {
    "k-NN": "knn.joblib",
    "ID3 Decision Tree": "id3_decision_tree.joblib",
    "SVM": "svm.joblib",
    "Linear Regression": "linear_regression.joblib",
}


def ensure_dirs():
    for folder in (DATA_DIR, MODELS_DIR, FIG_DIR, TABLE_DIR, RESULT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
