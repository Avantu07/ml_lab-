"""Command-line entry point.

``python app.py``  trains, tunes, evaluates and saves every model
(generating models/*.joblib and reports/*). The Streamlit UI (ui.py)
only loads the saved results. The original helper name
``predict_heart_disease`` is re-exported for backward compatibility.
"""
from src.predict import predict_heart_disease  # noqa: F401
from src.train import run

if __name__ == "__main__":
    run()
