"""Loading, inspecting, cleaning and splitting the heart-disease data.

The data come from ``data/heart-disease-dataset.csv`` (real patient records,
11 clinical inputs plus a binary ``target``). No records are generated.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import (
    COLUMNS, DATA_FILE, FEATURE_INFO, FEATURES, MAX_MISSING_FEATURES,
    RANDOM_STATE, TARGET, TEST_SIZE, ZERO_MEANS_MISSING, ensure_dirs,
)


def load_raw_data(download=False):
    """Read the local CSV file into a raw DataFrame.

    ``download`` is accepted only for backward compatibility and is ignored.
    """
    ensure_dirs()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found.")
    df = pd.read_csv(DATA_FILE)
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")
    return df[COLUMNS].apply(pd.to_numeric, errors="coerce")


def missing_value_table(df):
    """Feature | Missing Count | Missing Percentage."""
    return pd.DataFrame({
        "Feature": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing Percentage": (df.isna().mean() * 100).round(2).values,
    })


def find_invalid_values(df):
    """Count values outside medically plausible ranges / allowed categories."""
    rows = []
    for name, info in FEATURE_INFO.items():
        col = df[name].dropna()
        if "choices" in info:
            bad = (~col.isin(list(info["choices"]))).sum()
        else:
            bad = ((col < info["min"]) | (col > info["max"])).sum()
        rows.append({"Feature": name, "Invalid Count": int(bad)})
    return pd.DataFrame(rows)


def iqr_outlier_table(df, columns):
    """Count IQR-rule outliers (1.5 x IQR) for numeric columns."""
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({
            "Feature": col, "Q1": q1, "Q3": q3,
            "Lower Bound": round(low, 2), "Upper Bound": round(high, 2),
            "Outlier Count": count,
            "Outlier %": round(100 * count / len(df), 2),
        })
    return pd.DataFrame(rows)


def clean_data(df):
    """Clean the raw data and return (clean_df, report).

    1. impossible zeros (resting BP, cholesterol) and the undefined
       st_slope value 0 are treated as missing,
    2. exact duplicate rows are dropped,
    3. rows missing more than MAX_MISSING_FEATURES inputs are dropped,
    4. remaining gaps are kept as NaN and imputed INSIDE the model pipeline
       (fitted on training data only, so there is no leakage),
    5. target is already binary: 0 = no disease, 1 = disease.
    Outliers are analysed but NOT removed.
    """
    report = {
        "rows_raw": len(df), "columns_raw": df.shape[1],
        "duplicates_before": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "original_target_counts": {
            int(k): int(v)
            for k, v in df[TARGET].value_counts().sort_index().items()},
    }
    work = df.copy()
    # st_slope 0 is not a defined category (valid values are 1-3).
    bad_slope = work["st_slope"] == 0
    report["undefined_st_slope_set_to_missing"] = int(bad_slope.sum())
    work.loc[bad_slope, "st_slope"] = np.nan
    zero_fixed = 0
    for col in ZERO_MEANS_MISSING:
        mask = work[col] == 0
        zero_fixed += int(mask.sum())
        work.loc[mask, col] = np.nan
    report["impossible_zeros_set_to_missing"] = zero_fixed

    work = work.dropna(subset=[TARGET]).drop_duplicates()
    too_sparse = work[FEATURES].isna().sum(axis=1) > MAX_MISSING_FEATURES
    report["rows_dropped_too_many_missing"] = int(too_sparse.sum())
    clean = work[~too_sparse].reset_index(drop=True)

    clean[FEATURES] = clean[FEATURES].astype(float)
    clean[TARGET] = clean[TARGET].astype(int)
    report.update({
        "rows_clean": len(clean),
        "duplicates_after": int(clean.duplicated().sum()),
        "missing_after_cleaning": int(clean[FEATURES].isna().sum().sum()),
        "binary_target_counts": {
            int(k): int(v)
            for k, v in clean[TARGET].value_counts().sort_index().items()},
    })
    return clean, report


def split_data(df):
    """Stratified 80:20 split of features and target (before any fitting)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)
