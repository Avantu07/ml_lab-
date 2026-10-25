"""Evaluation metrics and the single combined confusion-matrix figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score,
)
from .utils import FIG_DIR

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1"]


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No disease", "Disease signal"],
            zero_division=0
        ),
    }


def regression_metrics(model, X_test, y_test):
    """R2 / MAE / RMSE of the raw linear-regression output on the test set."""
    raw = model[:-1].transform(X_test)
    pred = model.named_steps["clf"].predict_continuous(raw)
    return {
        "R2": float(r2_score(y_test, pred)),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def coefficient_table(model):
    """Standardised linear-regression coefficients, sorted by magnitude."""
    names = [
        n.split("__", 1)[-1]
        for n in model[:-1].get_feature_names_out()
    ]
    coefs = model.named_steps["clf"].coef_
    df = pd.DataFrame({"Feature": names, "Coefficient": coefs})
    return df.reindex(
        df["Coefficient"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def plot_all_confusion_matrices(results):
    """One 2x2 figure replaces the per-model confusion-matrix images."""
    n = len(results)
    cols = 2 if n > 2 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, res) in zip(axes, results.items()):
        cm = np.asarray(res["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["No disease", "Disease"], fontsize=8)
        ax.set_yticks([0, 1], ["No disease", "Disease"], fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices_all.png", dpi=170)
    plt.close(fig)
