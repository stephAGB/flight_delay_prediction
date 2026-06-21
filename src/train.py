import argparse
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

def train_model(data_path: str, model_path: str, n_estimators: int, max_depth: int, random_state: int):
    print(f"Loading processed data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Define features and target
    features = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 'DEPARTURE_DELAY']
    target = 'delayed'
    
    X = df[features]
    y = df[target]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Setup MLflow tracking URI and experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Flight_Delay_Prediction")
    
    with mlflow.start_run() as run:
        print("Training model...")
        
        # Define preprocessing for numerical and categorical features
        numerical_cols = ['MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'DEPARTURE_DELAY']
        categorical_cols = ['AIRLINE']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Define the complete model pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Save model local copy
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Saving model local copy to {model_path}...")
        joblib.dump(pipeline, model_path)
        
        # Log model to MLflow
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
    
    train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
