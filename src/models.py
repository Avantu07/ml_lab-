"""ML model definitions used by the project.

Supervised models: k-NN, ID3 Decision Tree, SVM and Linear Regression.
Linear Regression is a regression algorithm, so it is wrapped in a small
estimator that thresholds its continuous output (a "linear probability
model") to make it directly comparable with the classifiers.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor
from .utils import RANDOM_STATE


class LinearRegressionClassifier(ClassifierMixin, BaseEstimator):
    """Ordinary least-squares regression on the 0/1 target, thresholded.

    * ``predict_continuous`` returns the raw regression output,
    * ``predict_proba`` clips that output to [0, 1],
    * ``predict`` returns 1 when the raw output >= ``threshold``.
    """

    def __init__(self, threshold=0.5, fit_intercept=True):
        self.threshold = threshold
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.regressor_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.regressor_.fit(X, np.asarray(y, dtype=float))
        self.coef_ = self.regressor_.coef_
        self.intercept_ = self.regressor_.intercept_
        return self

    def predict_continuous(self, X):
        return self.regressor_.predict(X)

    def predict_proba(self, X):
        p = np.clip(self.predict_continuous(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_continuous(X) >= self.threshold).astype(int)


def make_pipeline(estimator, scale=True):
    """Create a leakage-safe preprocessing + estimator pipeline."""
    return Pipeline([
        ("preprocess", build_preprocessor(scale=scale)),
        ("clf", estimator),
    ])


def get_model_configs():
    """Return the supervised models and their hyperparameter grids."""
    return {
        "k-NN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        ),
        "ID3 Decision Tree": (
            # ID3 uses entropy/information gain. Gini is deliberately excluded.
            DecisionTreeClassifier(
                criterion="entropy", random_state=RANDOM_STATE
            ),
            {
                "clf__max_depth": [2, 3, 5, 7, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            [
                {
                    "clf__C": [0.1, 1, 10, 100],
                    "clf__kernel": ["linear"],
                },
                {
                    "clf__C": [0.1, 1, 10, 100],
                    "clf__kernel": ["rbf"],
                    "clf__gamma": ["scale", "auto", 0.01],
                },
            ],
        ),
        "Linear Regression": (
            LinearRegressionClassifier(),
            {
                # Only the decision threshold is tuned (on training folds).
                "clf__threshold": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
            },
        ),
    }
