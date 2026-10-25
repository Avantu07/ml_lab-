"""Streamlit interface for the Heart Disease ML project."""
import json

import pandas as pd
import streamlit as st

from src.data_loader import clean_data, load_raw_data, missing_value_table
from src.predict import (
    load_models, load_results, predict_heart_disease, validate_input,
)
from src.utils import (
    FEATURE_INFO, FEATURES, FIG_DIR, RESULT_DIR, TABLE_DIR, TARGET,
)

DISCLAIMER = (
    "Educational project only. Model output is not a medical diagnosis "
    "and must not be used for clinical decisions."
)

st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;
                      max-width: 1180px;}
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    .hero {
        background: linear-gradient(120deg, #0b2545 0%, #13518a 100%);
        border-radius: 14px; padding: 2rem 2.2rem; margin-bottom: 1.4rem;
        color: #ffffff;
    }
    .hero h1 {margin: 0; font-size: 2rem; font-weight: 700; color: #fff;
              letter-spacing: -0.01em;}
    .hero p {margin: .45rem 0 0 0; font-size: 1rem; color: #cfe0f3;
             max-width: 760px; line-height: 1.5;}
    .hero .badge {display: inline-block; font-size: .72rem; font-weight: 600;
                  letter-spacing: .08em; text-transform: uppercase;
                  background: rgba(255,255,255,.14); border-radius: 999px;
                  padding: .2rem .7rem; margin-bottom: .8rem;}
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e3e8ef; border-radius: 12px;
        padding: 1rem 1.2rem; box-shadow: 0 1px 2px rgba(16,24,40,.04);
    }
    div[data-testid="stMetricLabel"] p {color: #5b6b7f; font-size: .82rem;
        font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}
    div[data-testid="stMetricValue"] {color: #0b2545; font-weight: 700;}
    .stTabs [data-baseweb="tab-list"] {gap: .25rem;
        border-bottom: 1px solid #e3e8ef;}
    .stTabs [data-baseweb="tab"] {padding: .6rem 1rem; font-weight: 600;
        color: #5b6b7f;}
    .stTabs [aria-selected="true"] {color: #13518a;}
    h2, h3 {color: #0b2545; font-weight: 700; letter-spacing: -0.01em;}
    .section-note {color: #5b6b7f; font-size: .92rem; margin: -.4rem 0 1rem;}
    .result-card {border-radius: 12px; padding: 1.2rem 1.4rem;
        border: 1px solid; margin: .6rem 0 1rem;}
    .result-pos {background: #fff4e5; border-color: #f5c27a; color: #7a4a00;}
    .result-neg {background: #e9f7ef; border-color: #9fd8b5; color: #14532d;}
    .result-card h4 {margin: 0 0 .2rem 0; font-size: 1.1rem; color: inherit;}
    .footer-note {color: #7a8798; font-size: .8rem; margin-top: 2rem;
        border-top: 1px solid #e3e8ef; padding-top: .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data / model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return load_models(), load_results()


@st.cache_data
def get_data():
    raw = load_raw_data()
    clean, _ = clean_data(raw)
    report = json.loads((RESULT_DIR / "cleaning_report.json").read_text())
    return raw, clean, report


def read_table(name, **kwargs):
    return pd.read_csv(TABLE_DIR / name, **kwargs)


def section(title, note=None):
    st.subheader(title)
    if note:
        st.markdown(f'<p class="section-note">{note}</p>',
                    unsafe_allow_html=True)


models, results = get_models()
raw_df, clean_df, report = get_data()
comparison = read_table("model_comparison.csv", index_col=0)
best_model = comparison["F1"].idxmax()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="badge">Machine Learning Project</div>
      <h1>Heart Disease Analytics</h1>
      <p>Compare four machine-learning algorithms on the heart-disease
      dataset, explore patient segments, and run
      predictions from a single dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Overview", "Data Explorer", "Model Performance",
    "Model Insights", "Risk Prediction", "Patient Segments",
])

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Patient records", f"{report['rows_clean']:,}",
        help="Unique patient records remaining after cleaning.",
    )
    c2.metric("Input features", len(FEATURES))
    c3.metric("Predictive models", len(models))
    c4.metric(
        "Best F1 score", f"{comparison['F1'].max():.1%}",
        help=f"Achieved by {best_model} on the held-out test set.",
    )

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        section("Methodology")
        st.markdown(
            """
            1. **Data preparation:** missing-value removal, duplicate
               check, IQR outlier review, encoding and scaling.
            2. **Stratified 80:20 split** with imputation, encoding and
               scaling fitted on training data only (no leakage).
            3. **Model training:** k-NN, ID3 Decision Tree, SVM and
               Linear Regression, tuned with 5-fold cross-validation.
            4. **Evaluation** on the untouched test set using Accuracy,
               Precision, Recall and F1.
            5. **Patient segmentation** with Agglomerative Hierarchical
               Clustering (unsupervised, reported separately).
            """
        )
    with right:
        section("Dataset")
        st.markdown(
            """
            **Heart-disease dataset** (`heart-disease-dataset.csv`):
            11 clinical inputs and a binary target,
            `0` no disease, `1` disease. All records are real patient
            data; none are generated.
            """
        )
        st.info(
            f"Top performer by F1: **{best_model}** "
            f"({comparison.loc[best_model, 'F1']:.1%}).",
        )
    st.caption(DISCLAIMER)

# --------------------------------------------------------------------------
# Data Explorer
# --------------------------------------------------------------------------
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw records", report["rows_raw"])
    c2.metric("Clean patients", report["rows_clean"])
    c3.metric("Duplicates removed", report["duplicates_before"])
    c4.metric("Train / test", f"{report['train_rows']} / {report['test_rows']}")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        section("Missing values")
        st.dataframe(
            missing_value_table(raw_df), hide_index=True,
            use_container_width=True,
        )
    with right:
        section("Target distribution")
        dist = clean_df[TARGET].map({
            0: "No disease", 1: "Disease signal"
        }).value_counts()
        st.bar_chart(dist)

    section("Sample records")
    st.dataframe(raw_df.head(10), use_container_width=True)

    section(
        "Exploratory analysis",
        "Key figures only; duplicate views were removed.",
    )
    eda_tabs = st.tabs([
        "Distributions", "Features by outcome", "Categorical",
        "Correlation",
    ])
    eda_figs = [
        "eda_numeric_distributions.png", "eda_boxplots_by_target.png",
        "eda_categorical_vs_target.png", "eda_correlation_heatmap.png",
    ]
    for tab, fig in zip(eda_tabs, eda_figs):
        with tab:
            path = FIG_DIR / fig
            if path.exists():
                st.image(str(path), use_container_width=True)

    with st.expander("Outlier analysis (IQR rule)"):
        st.dataframe(
            read_table("iqr_outliers.csv"), hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Outliers were identified but retained because the values "
            "were not automatically treated as invalid clinical records."
        )

# --------------------------------------------------------------------------
# Model Performance
# --------------------------------------------------------------------------
with tabs[2]:
    section(
        "Model comparison",
        f"Metrics are calculated on {report['test_rows']} untouched "
        "held-out test patients.",
    )
    st.dataframe(
        comparison.round(3).style.highlight_max(
            axis=0, color="#d9ecff"
        ),
        use_container_width=True,
    )
    st.bar_chart(comparison[["Accuracy", "Precision", "Recall", "F1"]])

    section("Confusion matrices")
    st.image(
        str(FIG_DIR / "confusion_matrices_all.png"),
        use_container_width=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.expander("Best hyperparameters"):
            st.dataframe(
                read_table("best_hyperparameters.csv").dropna(
                    axis=1, how="all"
                ),
                hide_index=True, use_container_width=True,
            )
    with right:
        with st.expander("Baseline vs tuned"):
            st.dataframe(
                read_table("baseline_vs_tuned.csv"),
                hide_index=True, use_container_width=True,
            )

# --------------------------------------------------------------------------
# Model Insights (Linear Regression + SVM)
# --------------------------------------------------------------------------
with tabs[3]:
    lin_tab, svm_tab = st.tabs(["Linear Regression", "SVM"])

    with lin_tab:
        section(
            "Linear Regression",
            "Least-squares regression on the 0/1 target. Predictions at "
            "or above the tuned threshold are classified as disease.",
        )
        reg = results["Linear Regression"]["regression_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{reg['R2']:.3f}")
        c2.metric("MAE", f"{reg['MAE']:.3f}")
        c3.metric("RMSE", f"{reg['RMSE']:.3f}")
        c4.metric(
            "Decision threshold",
            f"{results['Linear Regression']['best_params']['threshold']:.2f}",
        )
        coefs = read_table("linear_regression_coefficients.csv")
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("**Standardised coefficients (largest effect first)**")
            st.bar_chart(coefs.head(10).set_index("Feature")["Coefficient"])
        with right:
            st.markdown("**How to read this**")
            st.write(
                "Positive coefficients push the predicted value toward "
                "disease, negative ones toward no disease. Numeric "
                "features are standardised, so magnitudes are comparable."
            )
            with st.expander("All coefficients"):
                st.dataframe(
                    coefs, hide_index=True, use_container_width=True
                )

    with svm_tab:
        section(
            "SVM analysis",
            "Kernel and C comparisons use cross-validation on the "
            "training set only.",
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Kernel comparison (C = 1)**")
            st.dataframe(
                read_table("svm_kernel_comparison.csv"),
                hide_index=True, use_container_width=True,
            )
        with right:
            st.markdown("**C-value comparison (linear kernel)**")
            st.dataframe(
                read_table("svm_c_comparison.csv"),
                hide_index=True, use_container_width=True,
            )
        st.markdown("**Selected configuration**")
        st.json(results["SVM"]["best_params"])

# --------------------------------------------------------------------------
# Risk Prediction
# --------------------------------------------------------------------------
with tabs[4]:
    section("Patient risk prediction", DISCLAIMER)
    with st.form("patient_form"):
        values = {}
        columns = st.columns(3)
        for i, feature in enumerate(FEATURES):
            info = FEATURE_INFO[feature]
            with columns[i % 3]:
                if "choices" in info:
                    options = list(info["choices"])
                    values[feature] = st.selectbox(
                        info["label"], options,
                        format_func=lambda v, c=info["choices"]: c[v],
                        key=f"in_{feature}",
                    )
                else:
                    values[feature] = st.number_input(
                        info["label"],
                        min_value=info["min"],
                        max_value=info["max"],
                        value=info["default"],
                        step=0.1 if isinstance(info["min"], float) else 1,
                        key=f"in_{feature}",
                    )
        model_name = st.selectbox(
            "Model", list(models), index=list(models).index(best_model),
        )
        submitted = st.form_submit_button(
            "Run prediction", type="primary"
        )

    if submitted:
        try:
            patient = validate_input(values)
            label, prob = predict_heart_disease(
                patient, models[model_name]
            )
            all_preds = {
                name: predict_heart_disease(patient, mdl)
                for name, mdl in models.items()
            }
        except ValueError as error:
            st.error(f"Invalid input: {error}")
        else:
            if label == 1:
                st.markdown(
                    f'<div class="result-card result-pos">'
                    f"<h4>Positive disease signal</h4>"
                    f"{model_name} estimates {prob:.1%} probability."
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-card result-neg">'
                    f"<h4>No disease signal</h4>"
                    f"{model_name} estimates {prob:.1%} probability of "
                    f"the positive class.</div>",
                    unsafe_allow_html=True,
                )
            st.progress(min(max(prob, 0.0), 1.0))
            st.markdown("**All models for this patient**")
            st.dataframe(
                pd.DataFrame({
                    "Model": list(all_preds),
                    "Prediction": [
                        "Disease signal" if lab == 1 else "No disease"
                        for lab, _ in all_preds.values()
                    ],
                    "Positive probability": [
                        f"{p:.1%}" for _, p in all_preds.values()
                    ],
                }),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "Model output for the supplied attributes, not a "
                "clinical risk score."
            )

# --------------------------------------------------------------------------
# Patient Segments
# --------------------------------------------------------------------------
with tabs[5]:
    cluster_report = json.loads(
        (RESULT_DIR / "clustering_report.json").read_text()
    )
    section(
        "Agglomerative Hierarchical Clustering",
        "Unsupervised: the heart-disease target is not used to build "
        "the clusters.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", cluster_report["n_clusters"])
    c2.metric("Linkage", cluster_report["linkage"].title())
    c3.metric(
        "Patients", sum(cluster_report["cluster_sizes"].values())
    )

    st.markdown("**Cluster sizes and average feature values**")
    st.dataframe(
        read_table("cluster_summary.csv", index_col=0),
        use_container_width=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.image(
            str(FIG_DIR / "agglomerative_dendrogram.png"),
            caption="Dendrogram of hierarchical merging",
            use_container_width=True,
        )
    with right:
        st.image(
            str(FIG_DIR / "patient_clusters_age_chol.png"),
            caption="Clusters by age and cholesterol",
            use_container_width=True,
        )
    st.info(
        "Cluster numbers are group identifiers, not diagnoses. "
        "Interpretation is exploratory."
    )

st.markdown(
    f'<div class="footer-note">{DISCLAIMER}</div>',
    unsafe_allow_html=True,
)
