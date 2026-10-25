# Saved Models

`python app.py` creates the four fitted supervised pipelines:

- `knn.joblib`
- `id3_decision_tree.joblib`
- `svm.joblib`
- `linear_regression.joblib`

It also saves the agglomerative clustering preprocessor/model bundle:

- `agglomerative_clustering.joblib`

These are generated artifacts and are tied to the scikit-learn/scipy
versions used for training.
