"""
Integration tests for the FastAPI application.
"""

import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'api'))


@pytest.fixture
def api_client():
    """Create test client for API."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self, api_client):
        """Test health endpoint returns OK."""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_includes_version(self, api_client):
        """Test health endpoint includes version info."""
        response = api_client.get("/health")
        data = response.json()
        
        assert "version" in data
        assert "uptime_seconds" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""
    
    def test_predict_valid_ticker(self, api_client):
        """Test prediction with valid ticker."""
        response = api_client.post("/predict", json={
            "ticker": "AAPL",
            "forecast_horizon": 7
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ticker" in data
        assert "predictions" in data
        assert len(data["predictions"]) == 7
    
    def test_predict_invalid_ticker(self, api_client):
        """Test prediction with invalid ticker returns error."""
        response = api_client.post("/predict", json={
            "ticker": "INVALID_TICKER_12345",
            "forecast_horizon": 7
        })
        
        # Should return 404 or 500 for invalid ticker
        assert response.status_code in [404, 500]
    
    def test_predict_custom_horizon(self, api_client):
        """Test prediction with custom forecast horizon."""
        response = api_client.post("/predict", json={
            "ticker": "MSFT",
            "forecast_horizon": 14
        })
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 14


class TestPerformanceEndpoint:
    """Tests for performance metrics endpoint."""
    
    def test_performance_returns_metrics(self, api_client):
        """Test performance endpoint returns metrics."""
        response = api_client.get("/performance?ticker=AAPL")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert "rmse" in data["metrics"]
        assert "mae" in data["metrics"]


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
