from typing import Dict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates the trained pipeline on the test split and returns calculated metrics.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target set.

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, f1, and roc_auc.

    Dependencies:
        - sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    return metrics
