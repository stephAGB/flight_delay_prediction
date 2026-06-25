import os
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Global variables for model
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model successfully loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will return 503 until model is available.")
    yield

app = FastAPI(
    title="Flight Delay Prediction API",
    description="FastAPI API to predict whether a flight will be delayed > 15 minutes at arrival.",
    version="1.0.0",
    lifespan=lifespan
)

# Input data model for request validation
class FlightInput(BaseModel):
    MONTH: int = Field(..., ge=1, le=12, description="Month of flight (1-12)", examples=[6])
    DAY_OF_WEEK: int = Field(..., ge=1, le=7, description="Day of the week (1-7)", examples=[1])
    AIRLINE: str = Field(..., description="Airline IATA code (e.g., AA, DL, UA)", examples=["AA"])
    DISTANCE: float = Field(..., ge=0, description="Flight distance in miles", examples=[1000.0])
    DEPARTURE_DELAY: float = Field(..., description="Departure delay in minutes", examples=[10.0])

# Swagger documentation response models
class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status ('healthy' or 'unhealthy')", examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether the prediction model is loaded", examples=[True])
    message: Optional[str] = Field(None, description="Detailed status message", examples=["Model successfully loaded."])

class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction (1 for delayed, 0 for on-time)", examples=[1])
    probability_delayed: float = Field(..., description="Probability of delay", examples=[0.8])
    delayed: bool = Field(..., description="Boolean prediction indicating if delay is > 15 minutes", examples=[True])

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error classification", examples=["HTTP Error"])
    message: str = Field(..., description="Detailed error description", examples=["Model not loaded and not found on disk."])

class ValidationErrorDetail(BaseModel):
    loc: List[str] = Field(..., description="Location of error in request (e.g. ['body', 'MONTH'])", examples=[["body", "MONTH"]])
    msg: str = Field(..., description="Error message detail", examples=["value is not a valid integer"])
    type: str = Field(..., description="Validation error type", examples=["type_error.integer"])

class ValidationErrorResponse(BaseModel):
    error: str = Field("Validation Error", description="Error classification")
    detail: List[ValidationErrorDetail] = Field(..., description="List of validation errors")

# Custom Exception Handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            message=str(exc.detail)
        ).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        details.append(
            ValidationErrorDetail(
                loc=[str(loc) for loc in error.get("loc", [])],
                msg=error.get("msg", ""),
                type=error.get("type", "")
            )
        )
    return JSONResponse(
        status_code=422,
        content=ValidationErrorResponse(
            detail=details
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc)
        ).model_dump()
    )

@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {
            "model": HealthResponse,
            "description": "Health status of the API and model loading state."
        }
    },
    summary="Check API and Model Health",
    description="Returns the loading status of the ML model and the overall health of the API.",
    tags=["Monitoring"]
)
def health():
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message="Model not loaded yet."
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="Model successfully loaded."
    )

@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        200: {
            "model": PredictResponse,
            "description": "Prediction result containing delayed status and class probability."
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad Request. Prediction or preprocessing failed."
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation Error. The input features are invalid."
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error."
        },
        503: {
            "model": ErrorResponse,
            "description": "Service Unavailable. Prediction model is not loaded yet."
        }
    },
    summary="Predict Flight Delay",
    description="Predicts whether a flight will be delayed by more than 15 minutes at arrival based on features.",
    tags=["Prediction"]
)
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
        
        return PredictResponse(
            prediction=prediction,
            probability_delayed=probability,
            delayed=(prediction == 1)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


