import argparse
import sys
import os

# Add the project root to the python path to support running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.sklearn

from src.training.load_data import load_and_split_data
from src.training.build_pipeline import build_pipeline
from src.training.evaluate import evaluate_model
from src.training.save_model import save_model_local

def run_training_pipeline(data_path: str, model_path: str, n_estimators: int, max_depth: int, random_state: int) -> None:
    """Orchestrates the complete training pipeline and registers the results with MLflow.

    Args:
        data_path (str): Path to the processed dataset.
        model_path (str): Target path for local model serialization.
        n_estimators (int): Number of RandomForest estimators.
        max_depth (int): RandomForest max depth.
        random_state (int): Random seed for reproducibility.

    Returns:
        None

    Dependencies:
        - src.training.load_data.load_and_split_data
        - src.training.build_pipeline.build_pipeline
        - src.training.evaluate.evaluate_model
        - src.training.save_model.save_model_local
        - mlflow
    """
    # Load and split
    X_train, X_test, y_train, y_test = load_and_split_data(data_path, random_state)
    
    # Configure MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Flight_Delay_Prediction")
    
    with mlflow.start_run() as run:
        print("Training model...")
        pipeline = build_pipeline(n_estimators, max_depth, random_state)
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        print("Evaluating model...")
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        print("Metrics:")
        for name, value in metrics.items():
            print(f"  {name.capitalize()}: {value:.4f}")
            
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
        # Save locally
        save_model_local(pipeline, model_path)
        
        # Log and register model in MLflow Model Registry
        model_name = f"FlightDelayRandomForest_est{n_estimators}_depth{max_depth}_rs{random_state}"
        print(f"Logging model artifact to MLflow and registering as '{model_name}'...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Training run tracking completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train flight delay prediction model.")
    parser.add_argument('--data-path', type=str, default='data/processed/processed_flights.csv',
                        help='Path to the cleaned csv file')
    parser.add_argument('--model-path', type=str, default='artifacts/model.joblib',
                        help='Path where the model joblib will be saved')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in RandomForest')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Max depth of trees in RandomForest')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    run_training_pipeline(
        data_path=args.data_path,
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
