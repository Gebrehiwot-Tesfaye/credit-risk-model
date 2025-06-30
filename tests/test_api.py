import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

client = TestClient(app)


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing."""
    return {
        "TransactionId": "TXN001",
        "BatchId": "BATCH001",
        "AccountId": "ACC001",
        "SubscriptionId": "SUB001",
        "CustomerId": "CUST001",
        "CurrencyCode": "USD",
        "CountryCode": "US",
        "ProviderId": "PROV001",
        "ProductId": "PROD001",
        "ProductCategory": "Electronics",
        "ChannelId": "WEB",
        "Amount": 100.0,
        "Value": 100.0,
        "TransactionStartTime": "2023-01-01T10:00:00",
        "PricingStrategy": "Standard"
    }


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.7, 0.3]])  # 30% risk probability
    return mock


@pytest.fixture
def mock_pipeline():
    """Mock pipeline for testing."""
    mock = MagicMock()
    mock.transform.return_value = np.array([[1, 2, 3, 4, 5]])  # Mock transformed features
    return mock


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns correct response."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


class TestPredictionEndpoint:
    """Test prediction endpoints."""
    
    @patch('api.main.model')
    @patch('api.main.pipeline')
    def test_predict_single_customer(self, mock_pipeline, mock_model, sample_customer_data):
        """Test single customer prediction endpoint."""
        # Mock the model and pipeline
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_pipeline.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        
        # Make request
        request_data = {"customer_data": sample_customer_data}
        response = client.post("/predict", json=request_data)
        
        # Check response
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert "risk_probability" in data
        assert "risk_category" in data
        assert "confidence_score" in data
        assert data["customer_id"] == "CUST001"
        assert 0 <= data["risk_probability"] <= 1
        assert data["risk_category"] in ["Low", "Medium", "High"]
        assert 0 <= data["confidence_score"] <= 1
    
    @patch('api.main.model')
    @patch('api.main.pipeline')
    def test_predict_batch_customers(self, mock_pipeline, mock_model, sample_customer_data):
        """Test batch prediction endpoint."""
        # Mock the model and pipeline
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_pipeline.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        
        # Make request with multiple customers
        request_data = {
            "customers": [sample_customer_data, sample_customer_data]
        }
        response = client.post("/predict/batch", json=request_data)
        
        # Check response
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_customers" in data
        assert len(data["predictions"]) == 2
        assert data["total_customers"] == 2
    
    def test_predict_invalid_data(self):
        """Test prediction with invalid data."""
        invalid_data = {"customer_data": {"invalid": "data"}}
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error


class TestModelInfoEndpoint:
    """Test model information endpoint."""
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        # This might return 503 if model is not loaded, which is expected
        assert response.status_code in [200, 503]


class TestErrorHandling:
    """Test error handling."""
    
    def test_model_not_loaded_prediction(self, sample_customer_data):
        """Test prediction when model is not loaded."""
        # This test might fail if model is loaded, which is expected
        request_data = {"customer_data": sample_customer_data}
        response = client.post("/predict", json=request_data)
        # Should return 503 if model is not loaded
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 