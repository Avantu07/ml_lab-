"""Training pipeline: clean -> split -> tune -> test -> cluster -> save."""
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from . import evaluate as ev
from .clustering import run_clustering
from .data_loader import (
    clean_data, find_invalid_values, iqr_outlier_table, load_raw_data,
    missing_value_table, split_data,
)
from .models import get_model_configs, make_pipeline
from .utils import (
    CV_FOLDS, DATA_FILE, FIG_DIR, MODEL_FILES, MODELS_DIR, NUMERIC_FEATURES,
    RANDOM_STATE, RESULT_DIR, TABLE_DIR, ensure_dirs,
)

# F1 is a taught classification metric and balances precision and recall.
TUNING_SCORING = "f1"
CV_SCORING = ["accuracy", "precision", "recall", "f1"]


def run(verbose=True):
    ensure_dirs()
    log = print if verbose else (lambda *a, **k: None)

    raw = load_raw_data()
    missing_value_table(raw).to_csv(TABLE_DIR / "missing_values.csv", index=False)
    find_invalid_values(raw).to_csv(TABLE_DIR / "invalid_values.csv", index=False)
    clean, report = clean_data(raw)
    iqr_outlier_table(clean, NUMERIC_FEATURES).to_csv(TABLE_DIR / "iqr_outliers.csv", index=False)

    X_train, X_test, y_train, y_test = split_data(clean)
    report.update({
        "data_file": DATA_FILE.name,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "split": "80:20 stratified",
        "random_state": RANDOM_STATE,
    })
    (RESULT_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2)
    )
    log(f"Data: {DATA_FILE.name} | raw {len(raw)} | clean {len(clean)} | "
        f"train {len(X_train)} | test {len(X_test)}")

    cv = StratifiedKFold(
        CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    results, best_rows, cv_rows, baseline_rows = {}, [], [], []

    for name, (estimator, grid) in get_model_configs().items():
        log(f"\n=== {name} ===")
        pipe = make_pipeline(estimator)

        base = clone(pipe).fit(X_train, y_train)
        base_metrics = ev.compute_metrics(base, X_test, y_test)

        search = GridSearchCV(
            pipe, grid, scoring=TUNING_SCORING, cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        cv_res = cross_validate(
            clone(best), X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1
        )
        cv_summary = {"Model": name}
        for metric in CV_SCORING:
            scores = cv_res[f"test_{metric}"]
            cv_summary[f"CV {metric} mean"] = float(scores.mean())
            cv_summary[f"CV {metric} std"] = float(scores.std())
        cv_rows.append(cv_summary)

        metrics = ev.compute_metrics(best, X_test, y_test)
        metrics["best_params"] = {
            k.replace("clf__", ""): v for k, v in search.best_params_.items()
        }
        metrics["cv_best_score"] = float(search.best_score_)
        metrics["cv_summary"] = cv_summary
        results[name] = metrics

        best_rows.append({
            "Model": name, **metrics["best_params"],
            f"CV {TUNING_SCORING}": search.best_score_,
        })
        baseline_rows.append({
            "Model": name,
            "Baseline Accuracy": base_metrics["Accuracy"],
            "Tuned Accuracy": metrics["Accuracy"],
            "Baseline F1": base_metrics["F1"],
            "Tuned F1": metrics["F1"],
        })

        if name == "Linear Regression":
            reg = ev.regression_metrics(best, X_test, y_test)
            metrics["regression_metrics"] = reg
            pd.DataFrame([reg]).round(4).to_csv(
                TABLE_DIR / "linear_regression_metrics.csv", index=False
            )
            ev.coefficient_table(best).round(4).to_csv(
                TABLE_DIR / "linear_regression_coefficients.csv", index=False
            )

        joblib.dump(best, MODELS_DIR / MODEL_FILES[name])

    # Required SVM practical comparisons: kernels and C values.
    svm_kernel_rows, svm_c_rows = svm_practical_comparisons(
        X_train, X_test, y_train, y_test
    )
    pd.DataFrame(svm_kernel_rows).round(4).to_csv(
        TABLE_DIR / "svm_kernel_comparison.csv", index=False
    )
    pd.DataFrame(svm_c_rows).round(4).to_csv(
        TABLE_DIR / "svm_c_comparison.csv", index=False
    )

    # Separate unsupervised component; target is not passed into clustering.
    cluster_df, cluster_summary, _, _ = run_clustering(clean)

    table = pd.DataFrame({
        n: {m: r[m] for m in ev.METRIC_NAMES + ["Specificity"]}
        for n, r in results.items()
    }).T
    table.index.name = "Model"

    table.round(4).to_csv(TABLE_DIR / "model_comparison.csv")
    pd.DataFrame(cv_rows).round(4).to_csv(
        TABLE_DIR / "cross_validation.csv", index=False
    ) if cv_rows else None
    pd.DataFrame(best_rows).to_csv(
        TABLE_DIR / "best_hyperparameters.csv", index=False
    )
    pd.DataFrame(baseline_rows).round(4).to_csv(
        TABLE_DIR / "baseline_vs_tuned.csv", index=False
    )
    (RESULT_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2, default=float)
    )

    ev.plot_all_confusion_matrices(results)

    log("\nSupervised results:\n", table.round(4).to_string())
    log("\nCluster sizes:\n", cluster_summary[["Patient Count"]].to_string())
    return table


def svm_practical_comparisons(X_train, X_test, y_train, y_test):
    """Compare SVM kernels and C values using CV on training data only.

    The held-out test set is intentionally not used to select a kernel or C.
    """
    rows_kernel, rows_c = [], []
    cv = StratifiedKFold(
        CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    for kernel in ["linear", "rbf"]:
        estimator, _ = get_model_configs()["SVM"]
        estimator.set_params(kernel=kernel, C=1)
        pipe = make_pipeline(estimator)
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=CV_SCORING, n_jobs=-1
        )
        rows_kernel.append({
            "Kernel": kernel, "C": 1,
            "CV Accuracy": cv_res["test_accuracy"].mean(),
            "CV Precision": cv_res["test_precision"].mean(),
            "CV Recall": cv_res["test_recall"].mean(),
            "CV F1": cv_res["test_f1"].mean(),
        })

    for c_value in [0.1, 1, 10, 100]:
        estimator, _ = get_model_configs()["SVM"]
        estimator.set_params(kernel="linear", C=c_value)
        pipe = make_pipeline(estimator)
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=CV_SCORING, n_jobs=-1
        )
        rows_c.append({
            "Kernel": "linear", "C": c_value,
            "CV Accuracy": cv_res["test_accuracy"].mean(),
            "CV Precision": cv_res["test_precision"].mean(),
            "CV Recall": cv_res["test_recall"].mean(),
            "CV F1": cv_res["test_f1"].mean(),
        })
    return rows_kernel, rows_c

if __name__ == "__main__":
    run()
