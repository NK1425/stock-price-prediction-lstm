"""
Ensemble Model: LSTM + XGBoost/LightGBM with Uncertainty Quantification.

This module implements a production-grade ensemble that combines:
1. LSTM-Attention for sequential pattern recognition
2. XGBoost/LightGBM for feature-based predictions
3. Quantile regression for uncertainty estimation (prediction intervals)

This is what real production ML systems use - not just a single model.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import gradient boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")


@dataclass
class PredictionResult:
    """Container for prediction results with uncertainty."""
    point_estimate: np.ndarray  # Shape: (samples, horizons)
    lower_bound: np.ndarray     # Lower confidence interval
    upper_bound: np.ndarray     # Upper confidence interval
    confidence_level: float     # e.g., 0.95 for 95% CI
    model_contributions: Dict[str, np.ndarray]  # Per-model predictions


class QuantileRegressor:
    """
    Quantile regression wrapper for uncertainty estimation.
    
    Uses gradient boosting to predict multiple quantiles,
    enabling prediction intervals.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_lightgbm: bool = True
    ):
        """
        Initialize quantile regressor.
        
        Args:
            quantiles: Quantiles to predict (e.g., [0.05, 0.5, 0.95] for 90% CI)
            n_estimators: Number of boosting iterations
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            use_lightgbm: Use LightGBM if True, else XGBoost
        """
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        
        self.models = {}
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """
        Fit quantile models.
        
        Args:
            X: Features of shape (samples, features)
            y: Targets of shape (samples,)
        
        Returns:
            Self for chaining
        """
        for q in self.quantiles:
            if self.use_lightgbm:
                model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    verbosity=-1,
                    force_col_wise=True
                )
            elif HAS_XGBOOST:
                model = xgb.XGBRegressor(
                    objective='reg:quantileerror',
                    quantile_alpha=q,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    verbosity=0
                )
            else:
                # Fallback: use standard regression with scaling
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    loss='quantile' if q != 0.5 else 'squared_error',
                    alpha=q if q != 0.5 else 0.5,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate
                )
            
            model.fit(X, y)
            self.models[q] = model
            logger.debug(f"Fitted quantile model for q={q}")
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict all quantiles.
        
        Args:
            X: Features of shape (samples, features)
        
        Returns:
            Dictionary mapping quantile to predictions
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return {q: model.predict(X) for q, model in self.models.items()}


class GradientBoostingPredictor:
    """
    XGBoost/LightGBM predictor for tabular features.
    
    This complements the LSTM by capturing feature interactions
    that sequential models might miss.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        use_lightgbm: bool = True,
        enable_quantiles: bool = True
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        self.enable_quantiles = enable_quantiles
        
        self.point_model = None
        self.quantile_model = None
        self.feature_scaler = StandardScaler()
        self.feature_names = None
        self._fitted = False
    
    def _create_features(
        self,
        X_sequences: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Create tabular features from sequences for gradient boosting.
        
        Extracts summary statistics from each sequence that capture
        patterns gradient boosting can learn from.
        
        Args:
            X_sequences: Shape (samples, seq_length, n_features)
            feature_names: Names of original features
        
        Returns:
            Tabular features of shape (samples, n_derived_features)
        """
        n_samples, seq_length, n_features = X_sequences.shape
        
        # For each feature, compute:
        # - Last value (most recent)
        # - Mean over sequence
        # - Std over sequence
        # - Min/Max
        # - Trend (slope of linear fit)
        # - Change from first to last
        
        feature_list = []
        names = []
        
        for f_idx in range(n_features):
            f_name = feature_names[f_idx] if feature_names else f"f{f_idx}"
            feature_seq = X_sequences[:, :, f_idx]
            
            # Last value
            feature_list.append(feature_seq[:, -1])
            names.append(f"{f_name}_last")
            
            # Recent values (last 5 days)
            for i in range(1, min(6, seq_length)):
                feature_list.append(feature_seq[:, -i-1])
                names.append(f"{f_name}_lag{i}")
            
            # Statistics
            feature_list.append(np.mean(feature_seq, axis=1))
            names.append(f"{f_name}_mean")
            
            feature_list.append(np.std(feature_seq, axis=1))
            names.append(f"{f_name}_std")
            
            feature_list.append(np.min(feature_seq, axis=1))
            names.append(f"{f_name}_min")
            
            feature_list.append(np.max(feature_seq, axis=1))
            names.append(f"{f_name}_max")
            
            # Change from start to end
            change = feature_seq[:, -1] - feature_seq[:, 0]
            feature_list.append(change)
            names.append(f"{f_name}_change")
            
            # Trend (simplified: mean of second half - mean of first half)
            half = seq_length // 2
            trend = np.mean(feature_seq[:, half:], axis=1) - np.mean(feature_seq[:, :half], axis=1)
            feature_list.append(trend)
            names.append(f"{f_name}_trend")
        
        self.feature_names = names
        return np.column_stack(feature_list)
    
    def fit(
        self,
        X_sequences: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'GradientBoostingPredictor':
        """
        Fit gradient boosting model on sequence data.
        
        Args:
            X_sequences: Shape (samples, seq_length, n_features)
            y: Target values (first horizon only for now)
            feature_names: Names of sequence features
        
        Returns:
            Self for chaining
        """
        # Handle multi-horizon targets
        if len(y.shape) > 1:
            y = y[:, 0]  # Use first horizon for primary prediction
        
        # Create tabular features
        X_tabular = self._create_features(X_sequences, feature_names)
        X_scaled = self.feature_scaler.fit_transform(X_tabular)
        
        logger.info(f"Created {X_scaled.shape[1]} tabular features for gradient boosting")
        
        # Fit point prediction model
        if self.use_lightgbm:
            self.point_model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                verbosity=-1,
                force_col_wise=True
            )
        elif HAS_XGBOOST:
            self.point_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                verbosity=0
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.point_model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate
            )
        
        self.point_model.fit(X_scaled, y)
        
        # Fit quantile models for uncertainty
        if self.enable_quantiles:
            self.quantile_model = QuantileRegressor(
                quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                n_estimators=self.n_estimators // 2,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                use_lightgbm=self.use_lightgbm
            )
            self.quantile_model.fit(X_scaled, y)
        
        self._fitted = True
        logger.info("Gradient boosting model fitted successfully")
        
        return self
    
    def predict(
        self,
        X_sequences: np.ndarray,
        return_quantiles: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[float, np.ndarray]]]:
        """
        Make predictions with optional quantiles.
        
        Args:
            X_sequences: Shape (samples, seq_length, n_features)
            return_quantiles: Whether to return quantile predictions
        
        Returns:
            Tuple of (point predictions, quantile dict or None)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_tabular = self._create_features(X_sequences, self.feature_names[:X_sequences.shape[2] * 12] if self.feature_names else None)
        X_scaled = self.feature_scaler.transform(X_tabular)
        
        point_pred = self.point_model.predict(X_scaled)
        
        quantiles = None
        if return_quantiles and self.quantile_model is not None:
            quantiles = self.quantile_model.predict(X_scaled)
        
        return point_pred, quantiles
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances."""
        if not self._fitted:
            raise ValueError("Model not fitted.")
        
        if hasattr(self.point_model, 'feature_importances_'):
            importance = self.point_model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_names))
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)


class EnsembleStockPredictor(BaseEstimator, RegressorMixin):
    """
    Production-grade ensemble combining LSTM and Gradient Boosting.
    
    This ensemble:
    1. Uses LSTM-Attention for sequential pattern recognition
    2. Uses XGBoost/LightGBM for feature interactions
    3. Provides prediction intervals via quantile regression
    4. Supports multiple forecast horizons
    
    Architecture:
        Input Sequences
              │
        ┌─────┴─────┐
        ▼           ▼
      LSTM      GradBoost
        │           │
        └─────┬─────┘
              ▼
         Weighted
          Average
              │
              ▼
        Point Estimate
        + Confidence
          Intervals
    """
    
    def __init__(
        self,
        lstm_model=None,
        attention_model=None,
        sequence_builder=None,
        lstm_weight: float = 0.6,
        gb_weight: float = 0.4,
        confidence_level: float = 0.90,
        enable_uncertainty: bool = True
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            lstm_model: Pre-trained LSTM model
            attention_model: LSTM model that outputs attention weights
            sequence_builder: SequenceBuilder for inverse transforms
            lstm_weight: Weight for LSTM predictions (0-1)
            gb_weight: Weight for gradient boosting predictions (0-1)
            confidence_level: Confidence level for intervals (e.g., 0.90)
            enable_uncertainty: Whether to compute prediction intervals
        """
        self.lstm_model = lstm_model
        self.attention_model = attention_model
        self.sequence_builder = sequence_builder
        self.lstm_weight = lstm_weight
        self.gb_weight = gb_weight
        self.confidence_level = confidence_level
        self.enable_uncertainty = enable_uncertainty
        
        self.gb_predictor = GradientBoostingPredictor(
            enable_quantiles=enable_uncertainty
        )
        
        # For multi-horizon: train separate GB models per horizon
        self.gb_models_per_horizon = {}
        
        self._fitted = False
        self.feature_names = None
        self.forecast_horizon = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> 'EnsembleStockPredictor':
        """
        Fit the ensemble model.
        
        If LSTM model is not provided, only gradient boosting is used.
        If LSTM is provided, it's assumed to be pre-trained.
        
        Args:
            X_train: Training sequences (samples, seq_length, features)
            y_train: Training targets (samples, horizons)
            X_val: Validation sequences
            y_val: Validation targets
            feature_names: Names of features
            epochs: Training epochs for LSTM
            batch_size: Batch size for LSTM
        
        Returns:
            Self for chaining
        """
        self.feature_names = feature_names
        self.forecast_horizon = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Fit gradient boosting for each horizon
        logger.info("Fitting gradient boosting models for each horizon...")
        for h in range(self.forecast_horizon):
            y_h = y_train[:, h] if len(y_train.shape) > 1 else y_train
            
            gb = GradientBoostingPredictor(enable_quantiles=self.enable_uncertainty)
            gb.fit(X_train, y_h, feature_names)
            self.gb_models_per_horizon[h] = gb
            
            logger.info(f"  Horizon {h+1}/{self.forecast_horizon} fitted")
        
        self._fitted = True
        logger.info("Ensemble model fitting complete")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_all: bool = False
    ) -> PredictionResult:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            X: Input sequences (samples, seq_length, features)
            return_all: Whether to return all model contributions
        
        Returns:
            PredictionResult with point estimates and confidence intervals
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.forecast_horizon))
        lower_bounds = np.zeros_like(predictions)
        upper_bounds = np.zeros_like(predictions)
        
        lstm_preds = None
        gb_preds = np.zeros_like(predictions)
        
        # Get LSTM predictions if available
        if self.lstm_model is not None:
            lstm_preds = self.lstm_model.predict(X, verbose=0)
            if self.sequence_builder is not None:
                lstm_preds = self.sequence_builder.inverse_transform_predictions(lstm_preds)
        
        # Get gradient boosting predictions for each horizon
        for h in range(self.forecast_horizon):
            gb_pred, quantiles = self.gb_models_per_horizon[h].predict(X, return_quantiles=True)
            
            if self.sequence_builder is not None:
                # Inverse transform single horizon prediction
                gb_pred_reshaped = gb_pred.reshape(-1, 1)
                gb_pred = self.sequence_builder.inverse_transform_predictions(gb_pred_reshaped).flatten()
            
            gb_preds[:, h] = gb_pred
            
            # Confidence intervals from quantiles
            if quantiles is not None:
                alpha = (1 - self.confidence_level) / 2
                lower_q = alpha
                upper_q = 1 - alpha
                
                # Find closest quantiles
                available_qs = sorted(quantiles.keys())
                lower_key = min(available_qs, key=lambda x: abs(x - lower_q))
                upper_key = min(available_qs, key=lambda x: abs(x - upper_q))
                
                lower_pred = quantiles[lower_key]
                upper_pred = quantiles[upper_key]
                
                if self.sequence_builder is not None:
                    lower_pred = self.sequence_builder.inverse_transform_predictions(
                        lower_pred.reshape(-1, 1)
                    ).flatten()
                    upper_pred = self.sequence_builder.inverse_transform_predictions(
                        upper_pred.reshape(-1, 1)
                    ).flatten()
                
                lower_bounds[:, h] = lower_pred
                upper_bounds[:, h] = upper_pred
        
        # Combine predictions
        if lstm_preds is not None:
            predictions = self.lstm_weight * lstm_preds + self.gb_weight * gb_preds
        else:
            predictions = gb_preds
        
        # Adjust bounds for ensemble
        if lstm_preds is not None:
            # Widen bounds slightly to account for model uncertainty
            uncertainty_factor = 1.1
            center = predictions
            lower_bounds = center - (center - lower_bounds) * uncertainty_factor
            upper_bounds = center + (upper_bounds - center) * uncertainty_factor
        
        model_contributions = {'gradient_boosting': gb_preds}
        if lstm_preds is not None:
            model_contributions['lstm'] = lstm_preds
        
        return PredictionResult(
            point_estimate=predictions,
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            confidence_level=self.confidence_level,
            model_contributions=model_contributions
        )
    
    def predict_with_confidence(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method returning arrays directly.
        
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        result = self.predict(X)
        return result.point_estimate, result.lower_bound, result.upper_bound
    
    def save(self, path: str):
        """Save ensemble model to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save gradient boosting models
        for h, gb in self.gb_models_per_horizon.items():
            joblib.dump(gb, os.path.join(path, f'gb_horizon_{h}.joblib'))
        
        # Save config
        config = {
            'lstm_weight': self.lstm_weight,
            'gb_weight': self.gb_weight,
            'confidence_level': self.confidence_level,
            'forecast_horizon': self.forecast_horizon,
            'feature_names': self.feature_names
        }
        joblib.dump(config, os.path.join(path, 'ensemble_config.joblib'))
        
        logger.info(f"Ensemble model saved to {path}")
    
    def load(self, path: str):
        """Load ensemble model from disk."""
        import os
        
        # Load config
        config = joblib.load(os.path.join(path, 'ensemble_config.joblib'))
        self.lstm_weight = config['lstm_weight']
        self.gb_weight = config['gb_weight']
        self.confidence_level = config['confidence_level']
        self.forecast_horizon = config['forecast_horizon']
        self.feature_names = config['feature_names']
        
        # Load gradient boosting models
        self.gb_models_per_horizon = {}
        for h in range(self.forecast_horizon):
            self.gb_models_per_horizon[h] = joblib.load(
                os.path.join(path, f'gb_horizon_{h}.joblib')
            )
        
        self._fitted = True
        logger.info(f"Ensemble model loaded from {path}")


def create_ensemble_predictor(
    lstm_model=None,
    attention_model=None,
    sequence_builder=None,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    feature_names: List[str] = None,
    lstm_weight: float = 0.6
) -> EnsembleStockPredictor:
    """
    Factory function to create and optionally train ensemble predictor.
    
    Args:
        lstm_model: Pre-trained LSTM model (optional)
        attention_model: LSTM with attention output (optional)
        sequence_builder: For inverse transforms
        X_train: Training data to fit gradient boosting
        y_train: Training targets
        feature_names: Feature names
        lstm_weight: Weight for LSTM in ensemble
    
    Returns:
        Configured EnsembleStockPredictor
    """
    ensemble = EnsembleStockPredictor(
        lstm_model=lstm_model,
        attention_model=attention_model,
        sequence_builder=sequence_builder,
        lstm_weight=lstm_weight,
        gb_weight=1.0 - lstm_weight,
        enable_uncertainty=True
    )
    
    if X_train is not None and y_train is not None:
        ensemble.fit(X_train, y_train, feature_names=feature_names)
    
    return ensemble


if __name__ == "__main__":
    # Test ensemble with synthetic data
    np.random.seed(42)
    
    # Synthetic sequences
    n_samples = 200
    seq_length = 60
    n_features = 10
    forecast_horizon = 7
    
    X = np.random.randn(n_samples, seq_length, n_features)
    y = np.random.randn(n_samples, forecast_horizon) + 100
    
    # Create and fit ensemble
    ensemble = EnsembleStockPredictor(
        lstm_model=None,  # No LSTM for this test
        enable_uncertainty=True
    )
    
    ensemble.fit(X[:160], y[:160], feature_names=[f"feature_{i}" for i in range(n_features)])
    
    # Predict
    result = ensemble.predict(X[160:])
    
    print("Ensemble Prediction Test")
    print(f"  Point estimates shape: {result.point_estimate.shape}")
    print(f"  Lower bounds shape: {result.lower_bound.shape}")
    print(f"  Upper bounds shape: {result.upper_bound.shape}")
    print(f"  Confidence level: {result.confidence_level}")
    print(f"\nSample prediction for first sample:")
    print(f"  Point: {result.point_estimate[0]}")
    print(f"  Lower: {result.lower_bound[0]}")
    print(f"  Upper: {result.upper_bound[0]}")
