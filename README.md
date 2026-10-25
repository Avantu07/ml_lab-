# Heart Disease Machine Learning Project

A Heart Disease ML mini-project using the Heart Disease
dataset, with a professional Streamlit dashboard.

> **Disclaimer:** This is an academic prototype. Model outputs are not a
> medical diagnosis and must not be used for clinical decisions.

## 1. Final algorithms

| Algorithm | Practical | Learning type | Purpose |
|---|---|---|---|
| k-NN Classification | Practical 4 | Supervised | Heart-disease classification |
| ID3 Decision Tree | Practical 5 | Supervised | Heart-disease classification |
| Agglomerative Hierarchical Clustering | Practical 6 | Unsupervised | Patient segmentation |
| SVM | Practical 7 | Supervised | Heart-disease classification |
| Linear Regression | Added extension | Supervised (regression) | Linear probability model on the 0/1 target, thresholded to classify |

Linear Regression is fitted on the 0/1 target; its continuous output is
thresholded (threshold tuned by cross-validation on the training set only) so
it can be compared with the classifiers. Besides the classification metrics it
also reports R², MAE and RMSE and a standardised coefficient table.

### Important distinction

The supervised comparison is **k-NN vs ID3 vs SVM vs Linear Regression**. Agglomerative
Hierarchical Clustering is a separate unsupervised analysis. Cluster numbers
are not heart-disease predictions.

## 2. Dataset

`data/heart-disease-dataset.csv` (real patient records, nothing synthetic):

- 1,190 raw rows, 12 columns: 11 input features + `target`
- `target`: `0` = no disease, `1` = disease
- Features: `age`, `sex`, `chest_pain_type`, `resting_bp_s`, `cholesterol`,
  `fasting_blood_sugar`, `resting_ecg`, `max_heart_rate`,
  `exercise_angina`, `oldpeak`, `st_slope`
- No null cells, but 272 exact duplicate rows (removed), 173 impossible
  values (cholesterol or resting BP recorded as 0, and one undefined
  `st_slope` of 0) which are set to missing and imputed **inside the
  pipeline** (fitted on training data only)
- After cleaning: 918 unique patients (410 no disease / 508 disease)

To use a different file, replace the CSV in `data/` (same column names) and
run `python app.py` to retrain everything.

## 3. Preprocessing and EDA

The project follows the supplied Practical 1 requirements:

1. `head()`, `tail()`, `describe()` and `shape`
2. Missing/null identification and handling (impossible zeros -> missing,
   rows with more than 3 missing inputs dropped, the rest imputed in-pipeline)
3. Duplicate detection and handling
4. Outlier identification using the IQR rule
5. Data encoding
6. Univariate, bivariate and multivariate analysis
7. Feature scaling
8. Stratified 80:20 train-test split

One duplicate row and one nearly empty row are removed. IQR outliers are
reported but not automatically deleted.

Categorical variables are one-hot encoded where appropriate. Scaling is
performed inside the supervised model pipeline so that the scaler is fitted
on training data only.

## 4. Supervised learning workflow

```text
Raw Dataset
   ↓
EDA + Data Cleaning
   ↓
Preprocessing
   ↓
Stratified 80:20 Split
   ↓
Training Set ──→ k-NN
             ├→ ID3 Decision Tree
             ├→ SVM
             └→ Linear Regression
   ↓
5-fold CV / Hyperparameter Tuning
   ↓
Untouched Test Set
   ↓
Accuracy, Precision, Recall, F1, Confusion Matrix
```

The same held-out test set is used for the final comparison of the three
supervised classifiers. The test set is not used for hyperparameter selection.

### k-NN

The grid compares different values of `k`, weighting strategies and distance
metrics. Scaling is required because k-NN is distance based.

### ID3 Decision Tree

The implementation uses an entropy-based `DecisionTreeClassifier`, matching
the ID3/information-gain approach. Gini criterion is deliberately excluded.

### Linear Regression

Ordinary least squares on the standardised/one-hot features. Only the decision
threshold is tuned. See `reports/tables/linear_regression_*.csv`.

### SVM

The practical analysis includes:

- kernel comparison: linear vs RBF
- C-value comparison for the linear kernel
- final hyperparameter selection using cross-validation on the training set

## 5. Unsupervised learning

Agglomerative Hierarchical Clustering is applied separately to the cleaned
patient features.

```text
Patient Features
      ↓
Encoding + Scaling
      ↓
Agglomerative Hierarchical Clustering
      ↓
Patient Clusters
      ↓
Cluster Summary + Dendrogram + Visualization
```

The target column is never supplied to the clustering algorithm.

Three exploratory clusters are produced using Ward linkage. Cluster
characteristics are summarized using patient count and feature averages.
Cluster labels should be interpreted as group identifiers, not diagnoses.

## 6. Project structure

```text
heart-disease-ml-prediction/
├── data/
│   ├── heart-disease-dataset.csv
│   └── README.md
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── clustering.py
│   └── utils.py
├── models/
│   ├── knn.joblib
│   ├── id3_decision_tree.joblib
│   ├── svm.joblib
│   ├── linear_regression.joblib
│   └── agglomerative_clustering.joblib
├── reports/
│   ├── figures/
│   ├── tables/
│   └── results/
├── tests/
│   └── test_pipeline.py
├── app.py
├── ui.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 7. Run

```bash
python -m src.eda
python app.py
streamlit run ui.py
```

Run tests with:

```bash
pip install -r requirements-dev.txt
python -m pytest tests -v
```

## 8. Reproducibility

- `random_state=42`
- stratified 80:20 split
- 5-fold stratified cross-validation for supervised tuning
- preprocessing is contained in the supervised model pipeline
- held-out test data is not used for tuning
- generated metrics are calculated from model predictions, not hard-coded

## 9. Generated outputs

`reports/tables/` contains the dataset audit, model comparison,
cross-validation, hyperparameter and SVM comparison tables.

`reports/figures/` contains four EDA figures, one combined confusion-matrix
figure, the hierarchical dendrogram and the cluster visualization. Redundant
charts (per-metric bars, per-model confusion matrices, duplicate boxplots,
target-distribution image, workflow image) were removed; the dashboard draws
simple charts natively.

`reports/results/` contains the cleaning, model and clustering results.
