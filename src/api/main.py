import os
import sys
import logging
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing import get_feature_engineering_pipeline, add_is_high_risk, AggregateFeatures, DateTimeFeatures
from .pydantic_models import (
    PredictionRequest, PredictionResponse, HealthResponse,
    BatchPredictionRequest, BatchPredictionResponse, CustomerData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and pipeline
model = None
model_name = "credit_risk_random_forest"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown."""
    global model, model_name
    
    # Always load the local fitted pipeline
    pipeline_path = os.path.join(os.path.dirname(__file__), '..', '..', 'fitted_pipeline.joblib')
    if os.path.exists(pipeline_path):
        model = joblib.load(pipeline_path)
        print(f"Loaded model type: {type(model)}")
        logger.info(f"Fitted pipeline loaded from {pipeline_path}")
    else:
        logger.error("No fitted pipeline found. API will run in test mode without model predictions.")
        model = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk based on customer transaction data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_customer_data(customer_data: CustomerData) -> pd.DataFrame:
    """Convert customer data to DataFrame for prediction."""
    df = pd.DataFrame([customer_data.dict()])
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    return df.drop(columns=id_cols)


def get_risk_category(probability: float) -> str:
    """Convert probability to risk category."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def calculate_confidence_score(probability: float) -> float:
    """Calculate confidence score based on probability distance from 0.5."""
    return abs(probability - 0.5) * 2


def is_high_risk_flag(probability: float, threshold: float = 0.5) -> bool:
    """Return True if risk_probability >= threshold."""
    return probability >= threshold


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X_input = preprocess_customer_data(request.customer_data)
        risk_probability = model.predict_proba(X_input)[0, 1]
        risk_category = get_risk_category(risk_probability)
        confidence_score = calculate_confidence_score(risk_probability)
        high_risk = is_high_risk_flag(risk_probability, threshold)
        return {
            "customer_id": request.customer_data.CustomerId,
            "risk_probability": float(risk_probability),
            "risk_category": risk_category,
            "confidence_score": float(confidence_score),
            "is_high_risk": high_risk
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_risk_batch(request: BatchPredictionRequest):
    """Predict credit risk for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for customer_data in request.customers:
            # Preprocess customer data
            X_transformed = preprocess_customer_data(customer_data)
            
            # Make prediction
            risk_probability = model.predict_proba(X_transformed)[0, 1]
            
            # Get risk category and confidence
            risk_category = get_risk_category(risk_probability)
            confidence_score = calculate_confidence_score(risk_probability)
            
            predictions.append(PredictionResponse(
                customer_id=customer_data.CustomerId,
                risk_probability=float(risk_probability),
                risk_category=risk_category,
                confidence_score=float(confidence_score)
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "features": getattr(model, 'feature_names_in_', 'Unknown'),
        "loaded_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
