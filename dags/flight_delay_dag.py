import os
import sys
# Ensure Airflow can resolve packages from the project root directory
sys.path.append('/opt/airflow')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# Set paths
DATA_RAW = '/opt/airflow/data/raw/flights.csv'
DATA_PROCESSED = '/opt/airflow/data/processed/processed_flights.csv'
X_TRAIN_PATH = '/opt/airflow/data/processed/X_train.csv'
X_TEST_PATH = '/opt/airflow/data/processed/X_test.csv'
Y_TRAIN_PATH = '/opt/airflow/data/processed/y_train.csv'
Y_TEST_PATH = '/opt/airflow/data/processed/y_test.csv'
MODEL_TEMP_PATH = '/opt/airflow/artifacts/model_temp.joblib'
MODEL_FINAL_PATH = '/opt/airflow/artifacts/model.joblib'

def on_failure_clean_up(context):
    """Callback function to clean up temporary artifacts if a task fails."""
    import os
    task_id = context.get('task_instance').task_id
    print(f"Task {task_id} failed. Starting clean-up...")
    if os.path.exists(MODEL_TEMP_PATH):
        try:
            os.remove(MODEL_TEMP_PATH)
            print(f"Cleaned up temporary model file: {MODEL_TEMP_PATH}")
        except Exception as e:
            print(f"Failed to delete temporary model file: {e}")

# Default arguments for DAG
default_args = {
    'owner': 'steph_ynov',
    'depends_on_past': False,
    'start_date': datetime(2026, 6, 20),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'on_failure_callback': on_failure_clean_up,
}

def run_preprocess(**kwargs):
    """Task 1: Preprocesses the raw flight delay data."""
    import os
    from airflow.exceptions import AirflowFailException
    from src.preprocessing.reduce_data import preprocess_data
    
    print(f"Checking if raw data file exists at {DATA_RAW}...")
    if not os.path.exists(DATA_RAW):
        raise AirflowFailException(f"Raw data file not found at: {DATA_RAW}")
        
    print("Starting preprocessing...")
    preprocess_data(DATA_RAW, DATA_PROCESSED, sample_size=100000)
    print("Preprocessing completed.")

def run_load_split(**kwargs):
    """Task 2: Loads processed data, splits it, and saves train/test sets to disk."""
    from src.training.load_data import load_and_split_data
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PROCESSED, random_state=42)
    
    # Save datasets locally so subsequent tasks can read them
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    pd.DataFrame(y_train).to_csv(Y_TRAIN_PATH, index=False)
    pd.DataFrame(y_test).to_csv(Y_TEST_PATH, index=False)
    print("Load and split completed.")

def run_validate_data(**kwargs):
    """Task: Performs quality controls on processed and split data before training."""
    from airflow.exceptions import AirflowFailException
    import os
    import pandas as pd
    
    print("Starting data validation checks...")
    
    # 1. Check file existence and non-emptiness
    paths = {
        "processed_data": DATA_PROCESSED,
        "X_train": X_TRAIN_PATH,
        "X_test": X_TEST_PATH,
        "y_train": Y_TRAIN_PATH,
        "y_test": Y_TEST_PATH
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            raise AirflowFailException(f"Validation failure: File for {name} does not exist at {path}")
        if os.path.getsize(path) == 0:
            raise AirflowFailException(f"Validation failure: File for {name} at {path} is empty")
            
    # 2. Load data for checks
    try:
        df_processed = pd.read_csv(DATA_PROCESSED)
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)
    except Exception as e:
        raise AirflowFailException(f"Validation failure: Failed to load CSV files: {e}")
        
    # 3. Check for minimum rows
    min_rows = 1000
    if len(df_processed) < min_rows:
        raise AirflowFailException(f"Validation failure: Processed dataset has too few rows ({len(df_processed)} < {min_rows})")
        
    # 4. Column validation
    expected_cols = {'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 'DEPARTURE_DELAY', 'delayed'}
    missing_cols = expected_cols - set(df_processed.columns)
    if missing_cols:
        raise AirflowFailException(f"Validation failure: Processed dataset is missing columns: {missing_cols}")
        
    # 5. Null checks
    null_counts = df_processed.isnull().sum().sum()
    if null_counts > 0:
        raise AirflowFailException(f"Validation failure: Processed dataset contains {null_counts} null values")
        
    # 6. Target class balance check
    target_col = 'delayed'
    class_counts = df_processed[target_col].value_counts()
    if len(class_counts) < 2:
        raise AirflowFailException(f"Validation failure: Target variable '{target_col}' contains only class {class_counts.index[0]}")
    
    # 7. Alignment checks (shape and size compatibility)
    if len(X_train) != len(y_train):
        raise AirflowFailException(f"Validation failure: Train set features and labels length mismatch: {len(X_train)} vs {len(y_train)}")
        
    if len(X_test) != len(y_test):
        raise AirflowFailException(f"Validation failure: Test set features and labels length mismatch: {len(X_test)} vs {len(y_test)}")
        
    # Check that train set isn't empty and has realistic proportion
    total_split_len = len(X_train) + len(X_test)
    if total_split_len == 0:
        raise AirflowFailException("Validation failure: Total split size is 0")
        
    test_ratio = len(X_test) / total_split_len
    print(f"Data split ratio: {len(X_train)} train, {len(X_test)} test (test ratio: {test_ratio:.2%})")
    
    if not (0.10 <= test_ratio <= 0.30):
        raise AirflowFailException(f"Validation failure: Test split ratio is abnormal: {test_ratio:.2%}")
        
    print("All data validation checks passed successfully!")

def run_train(**kwargs):
    """Task 3: Builds the scikit-learn pipeline, fits it, and logs parameters to MLflow."""
    from src.training.build_pipeline import build_pipeline
    
    # Read training data
    print("Loading train dataset...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]
    
    n_estimators = 50
    max_depth = 8
    random_state = 42
    
    # Build pipeline
    pipeline = build_pipeline(n_estimators, max_depth, random_state)
    
    # Train
    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Configure MLflow and start run
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Flight_Delay_Prediction")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Started MLflow Run: {run_id}")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        
        # Save temp model for evaluation and final save
        joblib.dump(pipeline, MODEL_TEMP_PATH)
        
        # Pass MLflow Run ID to downstream tasks via XCom
        kwargs['ti'].xcom_push(key='mlflow_run_id', value=run_id)
        print("Training and parameters logging completed.")

def run_evaluate(**kwargs):
    """Task 4: Evaluates model on test data and logs metrics to MLflow."""
    from src.training.evaluate import evaluate_model
    
    # Load test dataset and model
    print("Loading test dataset and model...")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]
    pipeline = joblib.load(MODEL_TEMP_PATH)
    
    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Resume MLflow run
    run_id = kwargs['ti'].xcom_pull(task_ids='train_task', key='mlflow_run_id')
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    
    with mlflow.start_run(run_id=run_id):
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"Logged metric {name}: {value:.4f}")
            
    print("Model evaluation metrics logging completed.")

def run_save_model(**kwargs):
    """Task 5: Promotes the temporary model to the final artifacts/model.joblib path."""
    from src.training.save_model import save_model_local
    print("Promoting model to final path...")
    pipeline = joblib.load(MODEL_TEMP_PATH)
    save_model_local(pipeline, MODEL_FINAL_PATH)
    
    # Clean up temp file
    if os.path.exists(MODEL_TEMP_PATH):
        os.remove(MODEL_TEMP_PATH)
        
    print("Model saved locally successfully.")

def run_register_model(**kwargs):
    """Task 6: Registers the model in the MLflow Model Registry."""
    run_id = kwargs['ti'].xcom_pull(task_ids='train_task', key='mlflow_run_id')
    
    n_estimators = 50
    max_depth = 8
    random_state = 42
    model_name = f"FlightDelayRandomForest_est{n_estimators}_depth{max_depth}_rs{random_state}"
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    pipeline = joblib.load(MODEL_FINAL_PATH)
    
    with mlflow.start_run(run_id=run_id):
        print(f"Logging and registering model: {model_name}...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=model_name
        )
    print("Model registered successfully in registry.")

# Define the DAG
with DAG(
    'flight_delay_prediction_pipeline',
    default_args=default_args,
    description='Modular flight delay prediction training pipeline',
    schedule_interval=None, # Triggered manually or by orchestration
    catchup=False,
) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess_task',
        python_callable=run_preprocess,
    )

    load_split_task = PythonOperator(
        task_id='load_split_task',
        python_callable=run_load_split,
    )

    train_task = PythonOperator(
        task_id='train_task',
        python_callable=run_train,
    )

    validate_data_task = PythonOperator(
        task_id='validate_data_task',
        python_callable=run_validate_data,
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_task',
        python_callable=run_evaluate,
    )

    save_model_task = PythonOperator(
        task_id='save_model_task',
        python_callable=run_save_model,
    )

    register_model_task = PythonOperator(
        task_id='register_model_task',
        python_callable=run_register_model,
    )

    # Enchaînement des tâches avec validation intermédiaire
    preprocess_task >> load_split_task >> validate_data_task >> train_task >> evaluate_task >> save_model_task >> register_model_task
