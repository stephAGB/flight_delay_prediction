import sys
from unittest.mock import MagicMock

# Mock the airflow package to prevent ModuleNotFoundError
class MockDAG:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        pass

mock_airflow = MagicMock()
mock_airflow.DAG = MockDAG

# Define a custom exception mock for AirflowFailException
class MockAirflowFailException(Exception):
    pass

mock_airflow.exceptions.AirflowFailException = MockAirflowFailException

sys.modules['airflow'] = mock_airflow
sys.modules['airflow.exceptions'] = mock_airflow.exceptions
sys.modules['airflow.operators'] = mock_airflow.operators
sys.modules['airflow.operators.python'] = mock_airflow.operators.python

import os
import pytest
import pandas as pd
from unittest.mock import patch
from dags.flight_delay_dag import run_validate_data, run_preprocess

def test_run_validate_data_success(tmp_path):
    # Create valid mock datasets with at least 1000 rows
    data_processed = pd.DataFrame({
        'MONTH': [1] * 1000,
        'DAY_OF_WEEK': [1] * 1000,
        'AIRLINE': ['AA'] * 1000,
        'DISTANCE': [100.0] * 1000,
        'DEPARTURE_DELAY': [10.0] * 1000,
        'delayed': [0] * 500 + [1] * 500
    })
    X_train = data_processed[['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 'DEPARTURE_DELAY']].iloc[:800]
    y_train = data_processed['delayed'].iloc[:800]
    X_test = data_processed[['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 'DEPARTURE_DELAY']].iloc[800:]
    y_test = data_processed['delayed'].iloc[800:]
    
    # Save to temp paths
    proc_path = tmp_path / "processed.csv"
    x_tr_path = tmp_path / "X_train.csv"
    x_te_path = tmp_path / "X_test.csv"
    y_tr_path = tmp_path / "y_train.csv"
    y_te_path = tmp_path / "y_test.csv"
    
    data_processed.to_csv(proc_path, index=False)
    X_train.to_csv(x_tr_path, index=False)
    X_test.to_csv(x_te_path, index=False)
    y_train.to_csv(y_tr_path, index=False)
    y_test.to_csv(y_te_path, index=False)
    
    with patch('dags.flight_delay_dag.DATA_PROCESSED', str(proc_path)), \
         patch('dags.flight_delay_dag.X_TRAIN_PATH', str(x_tr_path)), \
         patch('dags.flight_delay_dag.X_TEST_PATH', str(x_te_path)), \
         patch('dags.flight_delay_dag.Y_TRAIN_PATH', str(y_tr_path)), \
         patch('dags.flight_delay_dag.Y_TEST_PATH', str(y_te_path)):
        # Should complete successfully without raising exceptions
        run_validate_data()

def test_run_validate_data_fails_missing_file(tmp_path):
    proc_path = tmp_path / "processed_missing.csv"
    
    with patch('dags.flight_delay_dag.DATA_PROCESSED', str(proc_path)):
        with pytest.raises(MockAirflowFailException) as exc_info:
            run_validate_data()
        assert "does not exist" in str(exc_info.value)

def test_run_validate_data_fails_missing_column(tmp_path):
    # missing 'delayed' target column
    data_processed = pd.DataFrame({
        'MONTH': [1] * 1000,
        'DAY_OF_WEEK': [1] * 1000,
        'AIRLINE': ['AA'] * 1000,
        'DISTANCE': [100.0] * 1000,
        'DEPARTURE_DELAY': [10.0] * 1000
    })
    X_train = data_processed.iloc[:800]
    y_train = pd.Series([0] * 800)
    X_test = data_processed.iloc[800:]
    y_test = pd.Series([0] * 200)
    
    proc_path = tmp_path / "processed.csv"
    x_tr_path = tmp_path / "X_train.csv"
    x_te_path = tmp_path / "X_test.csv"
    y_tr_path = tmp_path / "y_train.csv"
    y_te_path = tmp_path / "y_test.csv"
    
    data_processed.to_csv(proc_path, index=False)
    X_train.to_csv(x_tr_path, index=False)
    X_test.to_csv(x_te_path, index=False)
    y_train.to_csv(y_tr_path, index=False)
    y_test.to_csv(y_te_path, index=False)
    
    with patch('dags.flight_delay_dag.DATA_PROCESSED', str(proc_path)), \
         patch('dags.flight_delay_dag.X_TRAIN_PATH', str(x_tr_path)), \
         patch('dags.flight_delay_dag.X_TEST_PATH', str(x_te_path)), \
         patch('dags.flight_delay_dag.Y_TRAIN_PATH', str(y_tr_path)), \
         patch('dags.flight_delay_dag.Y_TEST_PATH', str(y_te_path)):
        with pytest.raises(MockAirflowFailException) as exc_info:
            run_validate_data()
        assert "missing columns" in str(exc_info.value)

def test_run_preprocess_fails_missing_raw_file(tmp_path):
    raw_path = tmp_path / "flights_missing.csv"
    with patch('dags.flight_delay_dag.DATA_RAW', str(raw_path)):
        with pytest.raises(MockAirflowFailException) as exc_info:
            run_preprocess()
        assert "Raw data file not found" in str(exc_info.value)

def test_run_preprocess_success(tmp_path):
    raw_path = tmp_path / "flights.csv"
    proc_path = tmp_path / "processed.csv"
    
    raw_data = pd.DataFrame({
        'MONTH': [1],
        'DAY_OF_WEEK': [1],
        'AIRLINE': ['AA'],
        'DISTANCE': [100.0],
        'ARRIVAL_DELAY': [10.0],
        'DEPARTURE_DELAY': [5.0],
        'CANCELLED': [0],
        'DIVERTED': [0]
    })
    raw_data.to_csv(raw_path, index=False)
    
    with patch('dags.flight_delay_dag.DATA_RAW', str(raw_path)), \
         patch('dags.flight_delay_dag.DATA_PROCESSED', str(proc_path)):
        run_preprocess()
        assert os.path.exists(proc_path)
