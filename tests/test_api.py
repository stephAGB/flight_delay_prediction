from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import src.api

@pytest.fixture
def mock_model():
    model = MagicMock()
    # predict returns an array of predictions, e.g. [1] for delayed
    model.predict.return_value = [1]
    # predict_proba returns class probabilities, e.g. [[0.2, 0.8]]
    model.predict_proba.return_value = [[0.2, 0.8]]
    return model

@pytest.fixture
def client(mock_model):
    # Mock os.path.exists to return True and joblib.load to return mock_model
    with patch("os.path.exists", return_value=True), patch("joblib.load", return_value=mock_model):
        with TestClient(src.api.app) as c:
            yield c
    # Clean up global state after test
    src.api.model = None

def test_health_check_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_health_check_unhealthy():
    # Mock os.path.exists to return False so startup does not load any model
    with patch("os.path.exists", return_value=False):
        with TestClient(src.api.app) as client_unhealthy:
            # Ensure model is None
            src.api.model = None
            response = client_unhealthy.get("/health")
            assert response.status_code == 200
            assert response.json() == {
                "status": "unhealthy",
                "model_loaded": False,
                "message": "Model not loaded yet."
            }

def test_predict_success(client, mock_model):
    payload = {
        "MONTH": 6,
        "DAY_OF_WEEK": 1,
        "AIRLINE": "AA",
        "DISTANCE": 1200.0,
        "DEPARTURE_DELAY": 45.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["prediction"] == 1
    assert json_data["probability_delayed"] == 0.8
    assert json_data["delayed"] is True
    
    # Verify mock model methods were called with correct input
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()

def test_predict_validation_error(client):
    # Send invalid payload (missing month and invalid day of week)
    payload = {
        "DAY_OF_WEEK": 8, # Should be 1-7
        "AIRLINE": "AA",
        "DISTANCE": -10.0, # Should be >= 0
        "DEPARTURE_DELAY": 10.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity

