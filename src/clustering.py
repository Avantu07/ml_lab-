"""Agglomerative Hierarchical Clustering for patient segmentation.

The target column is never used by this module. Clustering is exploratory and
does not predict the heart-disease target.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from .preprocessing import build_preprocessor
from .utils import (
    CLUSTER_COUNT, CLUSTER_MODEL_FILE, CLUSTER_TABLE, FIG_DIR, FEATURES,
    RANDOM_STATE, RESULT_DIR, ensure_dirs,
)


def run_clustering(clean_df):
    """Fit agglomerative clustering on all cleaned patient features."""
    ensure_dirs()
    X = clean_df[FEATURES].copy()

    # This transformer is fit only on the data supplied to clustering and
    # never receives the target column.
    preprocessor = build_preprocessor(scale=True)
    X_scaled = preprocessor.fit_transform(X)
    if hasattr(X_scaled, "toarray"):
        X_scaled = X_scaled.toarray()

    model = AgglomerativeClustering(
        n_clusters=CLUSTER_COUNT,
        linkage="ward",
    )
    labels = model.fit_predict(X_scaled)

    result = clean_df.copy()
    result["cluster"] = labels + 1

    # Cluster sizes and interpretable feature summaries.
    sizes = result["cluster"].value_counts().sort_index()
    summary = result.groupby("cluster")[FEATURES].mean().round(2)
    summary.insert(0, "Patient Count", sizes)
    summary.to_csv(CLUSTER_TABLE)

    # Save labels and summary as a compact result artifact.
    result[FEATURES + ["cluster"]].to_csv(
        RESULT_DIR / "patient_clusters.csv", index=False
    )
    (RESULT_DIR / "clustering_report.json").write_text(
        json.dumps({
            "algorithm": "Agglomerative Hierarchical Clustering",
            "linkage": "ward",
            "n_clusters": CLUSTER_COUNT,
            "features_used": FEATURES,
            "target_used": False,
            "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        }, indent=2)
    )

    # Dendrogram demonstrates hierarchical merging. The full cleaned dataset
    # is small enough to visualize directly.
    Z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, no_labels=True, ax=ax, color_threshold=None)
    ax.set_title("Agglomerative Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Patient observations")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "agglomerative_dendrogram.png", dpi=180)
    plt.close(fig)

    # Simple 2-feature view using original, interpretable features.
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id in sorted(result["cluster"].unique()):
        subset = result[result["cluster"] == cluster_id]
        ax.scatter(subset["age"], subset["cholesterol"], label=f"Cluster {cluster_id}",
                   alpha=0.75)
    ax.set_xlabel("Age")
    ax.set_ylabel("Cholesterol")
    ax.set_title("Patient Clusters: Age vs Cholesterol")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "patient_clusters_age_chol.png", dpi=180)
    plt.close(fig)

    import joblib
    joblib.dump(
        {"preprocessor": preprocessor, "model": model, "features": FEATURES},
        __import__("pathlib").Path(__file__).resolve().parents[1] / "models" / CLUSTER_MODEL_FILE,
    )
    return result, summary, preprocessor, model


if __name__ == "__main__":
    from .data_loader import clean_data, load_raw_data
    run_clustering(clean_data(load_raw_data())[0])
