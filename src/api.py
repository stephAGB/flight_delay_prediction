import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Flight Delay Prediction API",
    description="FastAPI API to predict whether a flight will be delayed > 15 minutes at arrival.",
    version="1.0.0"
)

# Global variables for model
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
model = None

# Input data model for request validation
class FlightInput(BaseModel):
    MONTH: int = Field(..., ge=1, le=12, description="Month of flight (1-12)", example=6)
    DAY_OF_WEEK: int = Field(..., ge=1, le=7, description="Day of the week (1-7)", example=1)
    AIRLINE: str = Field(..., description="Airline IATA code (e.g., AA, DL, UA)", example="AA")
    DISTANCE: float = Field(..., ge=0, description="Flight distance in miles", example=1000.0)
    DEPARTURE_DELAY: float = Field(..., description="Departure delay in minutes", example=10.0)

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model successfully loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will return 503 until model is available.")

@app.get("/health")
def health():
    if model is None:
        return {"status": "unhealthy", "model_loaded": False, "message": "Model not loaded yet."}
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(flight: FlightInput):
    global model
    # Reload model if it was not loaded yet but now exists
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="Model not loaded and not found on disk.")
            
    # Format input data as DataFrame
    input_dict = {
        'MONTH': [flight.MONTH],
        'DAY_OF_WEEK': [flight.DAY_OF_WEEK],
        'AIRLINE': [flight.AIRLINE],
        'DISTANCE': [flight.DISTANCE],
        'DEPARTURE_DELAY': [flight.DEPARTURE_DELAY]
    }
    input_df = pd.DataFrame(input_dict)
    
    try:
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        return {
            "prediction": prediction,
            "probability_delayed": probability,
            "delayed": prediction == 1
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
