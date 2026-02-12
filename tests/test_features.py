"""
Tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngineer


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # Generate realistic price data
    close = 150 + np.cumsum(np.random.randn(n_days) * 2)
    high = close + np.abs(np.random.randn(n_days) * 2)
    low = close - np.abs(np.random.randn(n_days) * 2)
    open_price = close + np.random.randn(n_days) * 1
    volume = np.random.randint(1000000, 10000000, n_days)
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_add_all_features(self, sample_stock_data):
        """Test that all features are generated."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        
        # Should have more columns than original
        assert len(result.columns) > len(sample_stock_data.columns)
        
        # Check key features exist
        assert 'RSI_14' in result.columns
        assert 'MACD' in result.columns
        assert 'BB_Upper' in result.columns
        assert 'ATR' in result.columns
        assert 'OBV' in result.columns
    
    def test_rsi_values(self, sample_stock_data):
        """Test RSI is within valid range."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        result_clean = engineer.handle_missing_values(result, method='drop')
        
        assert result_clean['RSI_14'].min() >= 0
        assert result_clean['RSI_14'].max() <= 100
    
    def test_bollinger_bands(self, sample_stock_data):
        """Test Bollinger Bands relationship."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        result_clean = engineer.handle_missing_values(result, method='drop')
        
        # Upper band should be above lower band
        assert (result_clean['BB_Upper'] >= result_clean['BB_Lower']).all()
    
    def test_handle_missing_values_drop(self, sample_stock_data):
        """Test missing value handling with drop method."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        clean = engineer.handle_missing_values(result, method='drop')
        
        assert clean.isnull().sum().sum() == 0
    
    def test_handle_missing_values_ffill(self, sample_stock_data):
        """Test missing value handling with forward fill."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        clean = engineer.handle_missing_values(result, method='ffill')
        
        # Should have same number of rows (forward fill doesn't drop)
        assert len(clean) == len(result)
    
    def test_feature_columns_tracked(self, sample_stock_data):
        """Test that feature columns are properly tracked."""
        engineer = FeatureEngineer()
        engineer.add_all_features(sample_stock_data)
        
        assert len(engineer.feature_columns) > 0
        assert 'Close' not in engineer.feature_columns  # Original columns excluded


class TestTechnicalIndicators:
    """Tests for individual technical indicators."""
    
    def test_macd_calculation(self, sample_stock_data):
        """Test MACD calculation."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        result_clean = engineer.handle_missing_values(result, method='drop')
        
        # MACD should be difference of EMAs
        assert 'MACD' in result_clean.columns
        assert 'MACD_Signal' in result_clean.columns
        assert 'MACD_Histogram' in result_clean.columns
        
        # Histogram = MACD - Signal
        np.testing.assert_array_almost_equal(
            result_clean['MACD_Histogram'].values,
            (result_clean['MACD'] - result_clean['MACD_Signal']).values,
            decimal=5
        )
    
    def test_atr_positive(self, sample_stock_data):
        """Test ATR is always positive."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        result_clean = engineer.handle_missing_values(result, method='drop')
        
        assert (result_clean['ATR'] >= 0).all()
    
    def test_volume_indicators(self, sample_stock_data):
        """Test volume-based indicators."""
        engineer = FeatureEngineer()
        result = engineer.add_all_features(sample_stock_data)
        
        assert 'OBV' in result.columns
        assert 'VWAP' in result.columns
        assert 'MFI' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
