from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np


class CustomerData(BaseModel):
    """Input data model for customer transaction."""
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: int


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    customer_data: CustomerData


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    customer_id: str
    risk_probability: float
    risk_category: str
    confidence_score: float
    is_high_risk: bool


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Current timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    customers: List[CustomerData] = Field(..., description="List of customer data")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total number of customers processed")
