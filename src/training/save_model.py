import os
import joblib
from sklearn.pipeline import Pipeline

def save_model_local(pipeline: Pipeline, model_path: str) -> None:
    """Saves the trained model pipeline locally to disk.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline to save.
        model_path (str): Target path on disk.

    Returns:
        None

    Dependencies:
        - os.makedirs
        - joblib.dump
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving model local copy to {model_path}...")
    joblib.dump(pipeline, model_path)
