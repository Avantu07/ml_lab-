"""Project tests for the syllabus-aligned ML pipeline."""
import numpy as np
import pytest

from src.data_loader import clean_data, load_raw_data, split_data
from src.predict import load_models, predict_heart_disease, validate_input
from src.preprocessing import build_preprocessor
from src.utils import FEATURES, MODEL_FILES, MODELS_DIR, ROOT

VALID = {
    "age": 54, "sex": 1, "chest_pain_type": 4, "resting_bp_s": 130,
    "cholesterol": 240, "fasting_blood_sugar": 0, "resting_ecg": 0,
    "max_heart_rate": 150, "exercise_angina": 1, "oldpeak": 1.0,
    "st_slope": 2,
}


@pytest.fixture(scope="module")
def clean_df():
    return clean_data(load_raw_data())[0]


@pytest.fixture(scope="module")
def models():
    if not all((MODELS_DIR / f).exists() for f in MODEL_FILES.values()):
        from src.train import run
        run(verbose=False)
    return load_models()


def test_dataset_loading():
    raw = load_raw_data()
    assert raw.shape[1] == 12       # 11 features + target
    assert len(raw) == 1190


def test_cleaning(clean_df):
    assert clean_df[FEATURES].isna().sum(axis=1).max() <= 3
    assert clean_df.duplicated().sum() == 0
    assert (clean_df[["resting_bp_s", "cholesterol"]].dropna() > 0).all().all()
    assert len(clean_df) > 850
    assert set(clean_df["target"].unique()) == {0, 1}
    assert "target" not in FEATURES


def test_stratified_split(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)
    assert abs(len(X_test) / len(clean_df) - 0.2) < 0.01
    assert abs(y_train.mean() - y_test.mean()) < 0.03
    assert not set(X_train.index) & set(X_test.index)


def test_preprocessor_fits_on_train_only(clean_df):
    X_train, X_test, _, _ = split_data(clean_df)
    pre = build_preprocessor().fit(X_train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_[0], X_train["age"].mean())
    assert not np.isclose(scaler.mean_[0], clean_df["age"].mean())
    out = pre.transform(X_test)
    assert out.shape[0] == len(X_test)
    assert not np.isnan(np.asarray(out)).any()   # imputed inside pipeline


def test_model_files_load(models):
    assert set(models) == set(MODEL_FILES)
    assert set(models) == {
        "k-NN", "ID3 Decision Tree", "SVM", "Linear Regression"
    }


def test_prediction_is_binary_and_probability_valid(models):
    for model in models.values():
        label, prob = predict_heart_disease(VALID, model)
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0


def test_feature_order_is_enforced(models):
    shuffled = validate_input(VALID)[FEATURES[::-1]]
    model = models["k-NN"]
    assert predict_heart_disease(shuffled, model) ==         predict_heart_disease(VALID, model)
    assert list(validate_input(VALID).columns) == FEATURES


def test_missing_input_rejected():
    bad = dict(VALID)
    bad.pop("age")
    with pytest.raises(ValueError):
        validate_input(bad)
    with pytest.raises(ValueError):
        validate_input(dict(VALID, cholesterol=None))


@pytest.mark.parametrize("field,value", [
    ("age", 500), ("sex", 2), ("chest_pain_type", 9), ("st_slope", 0), ("oldpeak", -10),
    ("cholesterol", "abc"), ("resting_bp_s", float("nan"))
])
def test_invalid_input_rejected(field, value):
    with pytest.raises(ValueError):
        validate_input(dict(VALID, **{field: value}))


def test_clustering_artifacts():
    from src.clustering import run_clustering
    clean = clean_data(load_raw_data())[0]
    clustered, summary, _, _ = run_clustering(clean)
    assert "target" in clustered.columns
    assert "cluster" in clustered.columns
    assert clustered["cluster"].nunique() == 3
    assert "target" not in summary.columns




def test_linear_regression_outputs(models):
    from src.evaluate import coefficient_table, regression_metrics
    from src.data_loader import split_data
    model = models["Linear Regression"]
    clean = clean_data(load_raw_data())[0]
    _, X_test, _, y_test = split_data(clean)
    reg = regression_metrics(model, X_test, y_test)
    assert set(reg) == {"R2", "MAE", "RMSE"}
    assert reg["RMSE"] >= reg["MAE"] >= 0
    coefs = coefficient_table(model)
    assert list(coefs.columns) == ["Feature", "Coefficient"]
    assert len(coefs) > 0


def test_impossible_values_become_missing():
    """Zero cholesterol / BP and st_slope 0 are treated as missing."""
    import pandas as pd
    df = pd.DataFrame([
        [32, 1, 1, 95, 0, 0, 0, 127, 0, 0.7, 0, 1],
        [54, 1, 4, 140, 239, 0, 0, 160, 0, 1.0, 1, 0],
    ], columns=["age", "sex", "chest_pain_type", "resting_bp_s",
                "cholesterol", "fasting_blood_sugar", "resting_ecg",
                "max_heart_rate", "exercise_angina", "oldpeak",
                "st_slope", "target"])
    clean, report = clean_data(df)
    assert len(clean) == 2
    assert np.isnan(clean.loc[0, "cholesterol"])
    assert np.isnan(clean.loc[0, "st_slope"])
    assert report["impossible_zeros_set_to_missing"] == 1
    assert report["undefined_st_slope_set_to_missing"] == 1
    assert set(clean["target"]) == {0, 1}
