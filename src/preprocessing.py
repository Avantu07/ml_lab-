"""Leakage-free preprocessing built with scikit-learn ColumnTransformer."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import (BINARY_FEATURES, NOMINAL_FEATURES, NUMERIC_FEATURES,
                    ORDINAL_FEATURES)


def build_preprocessor(scale=True):
    """Return an unfitted ColumnTransformer.

    * continuous features -> median imputation + StandardScaler,
    * nominal features -> most-frequent imputation + OneHotEncoder
      (chest_pain_type, resting_ecg, st_slope have no natural order),
    * binary features -> most-frequent imputation.

    The transformer lives inside a Pipeline, so imputation values and scaling
    statistics are learned only from the training folds during
    cross-validation and from the training set for the final model. Test data
    is only transformed, never fitted.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + ORDINAL_FEATURES),
        ("cat", nominal, NOMINAL_FEATURES),
        ("bin", binary, BINARY_FEATURES),
    ])
