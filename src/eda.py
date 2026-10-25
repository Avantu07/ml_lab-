"""Exploratory data analysis: tables and figures for the report.

Run with:  python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import (clean_data, find_invalid_values, iqr_outlier_table,
                          load_raw_data, missing_value_table)
from .utils import (BINARY_FEATURES, FIG_DIR, FEATURE_INFO, NOMINAL_FEATURES,
                    NUMERIC_FEATURES, TABLE_DIR, TARGET, ensure_dirs)

sns.set_theme(style="whitegrid", context="notebook")
COLORS = {0: "#2ca02c", 1: "#d62728"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)


def run():
    ensure_dirs()
    raw = load_raw_data()
    clean, report = clean_data(raw)

    # --- text/table outputs -------------------------------------------
    print("Shape (raw):", raw.shape)
    print(raw.head(), "\n", raw.tail(), "\n")
    print(raw.dtypes, "\n")
    print(missing_value_table(raw), "\n")
    print("Duplicates before/after:", report["duplicates_before"],
          report["duplicates_after"])
    print(find_invalid_values(raw), "\n")
    raw.describe().T.round(2).to_csv(TABLE_DIR / "describe_raw.csv")
    clean.describe().T.round(2).to_csv(TABLE_DIR / "describe_clean.csv")
    raw.dtypes.astype(str).rename("dtype").to_csv(TABLE_DIR / "dtypes.csv")
    outliers = iqr_outlier_table(clean, NUMERIC_FEATURES)
    outliers.to_csv(TABLE_DIR / "iqr_outliers.csv", index=False)
    print(outliers, "\n")

    # --- numeric distributions by target ------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=clean, x=col, hue=TARGET, kde=True, bins=20,
                     palette=COLORS, ax=ax, stat="density", common_norm=False)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    save(fig, "eda_numeric_distributions.png")

    # --- boxplots by target (also shows outliers) ----------------------
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=clean, x=TARGET, y=col, hue=TARGET, ax=ax,
                    palette=COLORS, legend=False)
        ax.set_title(f"{col} by target")
    save(fig, "eda_boxplots_by_target.png")

    # --- categorical features vs target -------------------------------
    cat_cols = NOMINAL_FEATURES + BINARY_FEATURES
    plot_df = clean.copy()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), cat_cols):
        sub = plot_df.dropna(subset=[col]).copy()   # skip missing values
        sub[col] = sub[col].astype(int)
        sns.countplot(data=sub, x=col, hue=TARGET, palette=COLORS, ax=ax)
        ax.set_title(FEATURE_INFO[col]["label"], fontsize=9)
        ax.legend(title="target", fontsize=7)
    save(fig, "eda_categorical_vs_target.png")

    # --- correlation heatmap ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(clean.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap (Pearson)")
    save(fig, "eda_correlation_heatmap.png")
    clean.corr()[TARGET].drop(TARGET).sort_values().round(3).to_csv(
        TABLE_DIR / "target_correlations.csv", header=["correlation"])
    print(clean.corr()[TARGET].drop(TARGET).sort_values().round(3))


if __name__ == "__main__":
    run()
