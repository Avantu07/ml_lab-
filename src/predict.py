"""Model loading, input validation and supervised prediction."""
import json
import joblib
import pandas as pd

from .utils import FEATURE_INFO, FEATURES, MODEL_FILES, MODELS_DIR, RESULT_DIR


def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python app.py` to train models."
            )
        models[name] = joblib.load(path)
    return models


def load_results():
    return json.loads((RESULT_DIR / "metrics.json").read_text())


def load_clustering():
    path = MODELS_DIR / "agglomerative_clustering.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python app.py` to train clustering."
        )
    return joblib.load(path)


def validate_input(patient):
    missing = [
        f for f in FEATURES
        if f not in patient or patient[f] is None
    ]
    if missing:
        raise ValueError(f"Missing input values: {missing}")

    for name in FEATURES:
        info = FEATURE_INFO[name]
        try:
            value = float(patient[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric") from None
        if value != value:
            raise ValueError(f"{name} must not be NaN")
        if "choices" in info and value not in info["choices"]:
            raise ValueError(
                f"{name} must be one of {list(info['choices'])}"
            )
        if "min" in info and not info["min"] <= value <= info["max"]:
            raise ValueError(
                f"{name} must be between {info['min']} and {info['max']}"
            )

    return pd.DataFrame(
        [[float(patient[f]) for f in FEATURES]],
        columns=FEATURES,
    )


def predict_heart_disease(patient, model):
    """Return binary prediction and positive-class probability."""
    frame = patient if isinstance(patient, pd.DataFrame) else validate_input(patient)
    frame = frame[FEATURES]
    label = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    return label, probability
