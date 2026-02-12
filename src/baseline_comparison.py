"""
Baseline Comparison and Proper Evaluation Module.

This module proves our model is actually better than simple approaches.
A senior ML engineer's first question: "How does this compare to baselines?"

Implements:
1. Naive forecasts (last value, random walk)
2. Moving average forecasts
3. ARIMA/SARIMA baseline
4. Walk-forward validation with proper time-series splits
5. Trading metrics (Sharpe ratio, max drawdown, directional accuracy)
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed. Install with: pip install statsmodels")


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    rmse: float
    mae: float
    mape: float
    smape: float
    r2: float
    directional_accuracy: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class BaselineResult:
    """Result from a baseline model."""
    name: str
    predictions: np.ndarray
    metrics: EvaluationMetrics
    horizon_metrics: Dict[int, EvaluationMetrics]


class BaselineForecaster:
    """
    Collection of baseline forecasting methods.
    
    These are the "dumb" models we need to beat to prove our ML model is useful.
    If you can't beat these, your ML model is useless.
    """
    
    @staticmethod
    def naive_forecast(
        y_true: np.ndarray,
        X: np.ndarray,
        forecast_horizon: int
    ) -> np.ndarray:
        """
        Naive forecast: predict the last observed value for all horizons.
        
        This is the simplest possible baseline. If your model can't beat this,
        it's not learning anything useful.
        
        Args:
            y_true: Actual values (not used, just for interface consistency)
            X: Input sequences (samples, seq_length, features)
            forecast_horizon: Number of steps to forecast
        
        Returns:
            Predictions array (samples, forecast_horizon)
        """
        # Last value of the Close price (assuming it's first feature or we use the target)
        last_values = X[:, -1, 0]  # Shape: (samples,)
        
        # Repeat for each horizon
        predictions = np.tile(last_values.reshape(-1, 1), (1, forecast_horizon))
        
        return predictions
    
    @staticmethod
    def random_walk_forecast(
        y_true: np.ndarray,
        X: np.ndarray,
        forecast_horizon: int,
        drift: bool = True
    ) -> np.ndarray:
        """
        Random walk forecast: last value + drift (if enabled).
        
        Random walk is notoriously hard to beat for stock prices.
        This is based on the Efficient Market Hypothesis.
        
        Args:
            y_true: Actual values (not used)
            X: Input sequences
            forecast_horizon: Forecast horizon
            drift: Whether to include drift term (average historical change)
        
        Returns:
            Predictions array
        """
        n_samples = X.shape[0]
        seq_length = X.shape[1]
        
        # Get historical close prices from sequence
        close_sequence = X[:, :, 0]  # Assuming Close is first feature
        last_values = close_sequence[:, -1]
        
        predictions = np.zeros((n_samples, forecast_horizon))
        
        for i in range(n_samples):
            if drift:
                # Calculate historical daily drift
                daily_returns = np.diff(close_sequence[i]) / close_sequence[i, :-1]
                avg_drift = np.mean(daily_returns)
            else:
                avg_drift = 0
            
            # Propagate with drift
            current = last_values[i]
            for h in range(forecast_horizon):
                current = current * (1 + avg_drift)
                predictions[i, h] = current
        
        return predictions
    
    @staticmethod
    def moving_average_forecast(
        y_true: np.ndarray,
        X: np.ndarray,
        forecast_horizon: int,
        window: int = 5
    ) -> np.ndarray:
        """
        Simple moving average forecast.
        
        Uses the average of the last `window` days to predict future values.
        
        Args:
            y_true: Actual values (not used)
            X: Input sequences
            forecast_horizon: Forecast horizon
            window: Moving average window size
        
        Returns:
            Predictions array
        """
        close_sequence = X[:, :, 0]
        ma_values = np.mean(close_sequence[:, -window:], axis=1)
        
        # Repeat for each horizon
        predictions = np.tile(ma_values.reshape(-1, 1), (1, forecast_horizon))
        
        return predictions
    
    @staticmethod
    def exponential_smoothing_forecast(
        y_true: np.ndarray,
        X: np.ndarray,
        forecast_horizon: int,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Simple exponential smoothing forecast.
        
        Gives more weight to recent observations.
        
        Args:
            y_true: Actual values (not used)
            X: Input sequences
            forecast_horizon: Forecast horizon
            alpha: Smoothing parameter (0-1)
        
        Returns:
            Predictions array
        """
        close_sequence = X[:, :, 0]
        n_samples, seq_length = close_sequence.shape
        
        # Calculate exponentially weighted average
        weights = np.array([(1 - alpha) ** i for i in range(seq_length)])
        weights = weights[::-1]  # Most recent gets highest weight
        weights = weights / weights.sum()
        
        smoothed_values = np.sum(close_sequence * weights, axis=1)
        
        # Repeat for each horizon
        predictions = np.tile(smoothed_values.reshape(-1, 1), (1, forecast_horizon))
        
        return predictions


class ARIMABaseline:
    """
    ARIMA baseline model.
    
    ARIMA is the gold standard for time-series forecasting.
    If we can't beat ARIMA, our deep learning model isn't adding value.
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize ARIMA baseline.
        
        Args:
            order: (p, d, q) ARIMA parameters
            seasonal_order: (P, D, Q, s) seasonal parameters (optional)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
    
    def forecast(
        self,
        history: np.ndarray,
        forecast_horizon: int
    ) -> np.ndarray:
        """
        Make ARIMA forecast.
        
        Args:
            history: Historical values
            forecast_horizon: Number of steps to forecast
        
        Returns:
            Forecast array
        """
        if not HAS_STATSMODELS:
            # Fallback to naive forecast
            return np.full(forecast_horizon, history[-1])
        
        try:
            if self.seasonal_order:
                model = SARIMAX(
                    history,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(
                    history,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=forecast_horizon)
            
            return np.array(forecast)
            
        except Exception as e:
            logger.debug(f"ARIMA failed: {e}, falling back to naive")
            return np.full(forecast_horizon, history[-1])
    
    def forecast_batch(
        self,
        X: np.ndarray,
        forecast_horizon: int
    ) -> np.ndarray:
        """
        Make ARIMA forecasts for multiple sequences.
        
        Args:
            X: Input sequences (samples, seq_length, features)
            forecast_horizon: Forecast horizon
        
        Returns:
            Predictions array (samples, forecast_horizon)
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, forecast_horizon))
        
        for i in range(n_samples):
            history = X[i, :, 0]  # Close prices
            predictions[i] = self.forecast(history, forecast_horizon)
            
            if (i + 1) % 50 == 0:
                logger.debug(f"ARIMA: processed {i+1}/{n_samples} sequences")
        
        return predictions


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            prices: Price series for trading metrics (optional)
        
        Returns:
            EvaluationMetrics instance
        """
        # Flatten if needed
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # MAPE
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + epsilon))) * 100
        
        # SMAPE (Symmetric MAPE)
        smape = np.mean(
            2 * np.abs(y_true_flat - y_pred_flat) / 
            (np.abs(y_true_flat) + np.abs(y_pred_flat) + epsilon)
        ) * 100
        
        # R-squared
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))
        
        # Directional accuracy
        if len(y_true_flat) > 1:
            true_direction = np.sign(np.diff(y_true_flat))
            pred_direction = np.sign(np.diff(y_pred_flat))
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 50.0
        
        # Trading metrics (if prices provided)
        sharpe = None
        max_dd = None
        win_rate = None
        profit_factor = None
        
        if prices is not None and len(prices) > 1:
            sharpe, max_dd, win_rate, profit_factor = \
                MetricsCalculator._calculate_trading_metrics(y_true, y_pred, prices)
        
        return EvaluationMetrics(
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape),
            smape=float(smape),
            r2=float(r2),
            directional_accuracy=float(directional_accuracy),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
    
    @staticmethod
    def _calculate_trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Calculate trading-specific metrics.
        
        Simulates a simple trading strategy:
        - Go long if predicted price > current price
        - Go short (or cash) if predicted price < current price
        
        Returns:
            Tuple of (sharpe_ratio, max_drawdown, win_rate, profit_factor)
        """
        # Use first horizon predictions for simplicity
        if len(y_true.shape) > 1:
            y_true = y_true[:, 0]
            y_pred = y_pred[:, 0]
        
        # Predicted direction (1 = long, -1 = short)
        pred_signal = np.sign(y_pred[1:] - prices[:-1])
        
        # Actual returns
        actual_returns = (y_true[1:] - prices[:-1]) / prices[:-1]
        
        # Strategy returns
        strategy_returns = pred_signal * actual_returns
        
        # Sharpe Ratio (annualized)
        if np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(np.min(drawdowns)) * 100
        
        # Win Rate
        wins = np.sum(strategy_returns > 0)
        total_trades = len(strategy_returns)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 50.0
        
        # Profit Factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 1.0
        
        return float(sharpe), float(max_drawdown), float(win_rate), float(profit_factor)


class BaselineComparator:
    """
    Compare model against baselines with proper walk-forward validation.
    
    This is the key class that proves your model is worth deploying.
    """
    
    def __init__(
        self,
        forecast_horizon: int = 7,
        include_arima: bool = True
    ):
        """
        Initialize comparator.
        
        Args:
            forecast_horizon: Number of steps to forecast
            include_arima: Whether to include ARIMA (slower but important)
        """
        self.forecast_horizon = forecast_horizon
        self.include_arima = include_arima and HAS_STATSMODELS
        self.metrics_calculator = MetricsCalculator()
        
        self.baselines = {
            'naive': BaselineForecaster.naive_forecast,
            'random_walk': lambda y, X, h: BaselineForecaster.random_walk_forecast(y, X, h, drift=True),
            'moving_average_5': lambda y, X, h: BaselineForecaster.moving_average_forecast(y, X, h, window=5),
            'moving_average_20': lambda y, X, h: BaselineForecaster.moving_average_forecast(y, X, h, window=20),
            'exp_smoothing': BaselineForecaster.exponential_smoothing_forecast
        }
        
        if self.include_arima:
            self.arima = ARIMABaseline(order=(5, 1, 0))
    
    def compare(
        self,
        model_predictions: np.ndarray,
        y_true: np.ndarray,
        X: np.ndarray,
        prices: Optional[np.ndarray] = None,
        model_name: str = "LSTM-Attention"
    ) -> Dict[str, BaselineResult]:
        """
        Compare model predictions against all baselines.
        
        Args:
            model_predictions: Model's predictions (samples, horizons)
            y_true: Actual values (samples, horizons)
            X: Input sequences (for baselines)
            prices: Initial prices for trading metrics
            model_name: Name for the main model
        
        Returns:
            Dictionary of baseline results
        """
        results = {}
        
        # Evaluate main model
        logger.info(f"Evaluating {model_name}...")
        model_metrics = self.metrics_calculator.calculate_metrics(y_true, model_predictions, prices)
        horizon_metrics = {}
        for h in range(self.forecast_horizon):
            horizon_metrics[h] = self.metrics_calculator.calculate_metrics(
                y_true[:, h], model_predictions[:, h], prices
            )
        
        results[model_name] = BaselineResult(
            name=model_name,
            predictions=model_predictions,
            metrics=model_metrics,
            horizon_metrics=horizon_metrics
        )
        
        # Evaluate baselines
        for baseline_name, baseline_func in self.baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}...")
            
            try:
                baseline_preds = baseline_func(y_true, X, self.forecast_horizon)
                baseline_metrics = self.metrics_calculator.calculate_metrics(
                    y_true, baseline_preds, prices
                )
                
                horizon_metrics = {}
                for h in range(self.forecast_horizon):
                    horizon_metrics[h] = self.metrics_calculator.calculate_metrics(
                        y_true[:, h], baseline_preds[:, h], prices
                    )
                
                results[baseline_name] = BaselineResult(
                    name=baseline_name,
                    predictions=baseline_preds,
                    metrics=baseline_metrics,
                    horizon_metrics=horizon_metrics
                )
            except Exception as e:
                logger.warning(f"Baseline {baseline_name} failed: {e}")
        
        # ARIMA baseline
        if self.include_arima:
            logger.info("Evaluating baseline: ARIMA...")
            try:
                arima_preds = self.arima.forecast_batch(X, self.forecast_horizon)
                arima_metrics = self.metrics_calculator.calculate_metrics(
                    y_true, arima_preds, prices
                )
                
                horizon_metrics = {}
                for h in range(self.forecast_horizon):
                    horizon_metrics[h] = self.metrics_calculator.calculate_metrics(
                        y_true[:, h], arima_preds[:, h], prices
                    )
                
                results['arima'] = BaselineResult(
                    name='ARIMA(5,1,0)',
                    predictions=arima_preds,
                    metrics=arima_metrics,
                    horizon_metrics=horizon_metrics
                )
            except Exception as e:
                logger.warning(f"ARIMA failed: {e}")
        
        return results
    
    def generate_comparison_report(
        self,
        results: Dict[str, BaselineResult],
        model_name: str = "LSTM-Attention"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report.
        
        Args:
            results: Results from compare()
            model_name: Name of the main model
        
        Returns:
            Report dictionary
        """
        report = {
            'model_name': model_name,
            'summary': {},
            'detailed_metrics': {},
            'horizon_comparison': {},
            'improvement_over_baselines': {}
        }
        
        # Summary table
        summary_rows = []
        for name, result in results.items():
            summary_rows.append({
                'Model': name,
                'RMSE': result.metrics.rmse,
                'MAE': result.metrics.mae,
                'MAPE (%)': result.metrics.mape,
                'Directional Acc (%)': result.metrics.directional_accuracy,
                'Sharpe': result.metrics.sharpe_ratio or 'N/A'
            })
        
        report['summary'] = summary_rows
        
        # Detailed metrics
        for name, result in results.items():
            report['detailed_metrics'][name] = result.metrics.to_dict()
        
        # Improvement over baselines
        if model_name in results:
            model_rmse = results[model_name].metrics.rmse
            model_mape = results[model_name].metrics.mape
            
            for name, result in results.items():
                if name != model_name:
                    rmse_improvement = (result.metrics.rmse - model_rmse) / result.metrics.rmse * 100
                    mape_improvement = (result.metrics.mape - model_mape) / result.metrics.mape * 100
                    
                    report['improvement_over_baselines'][name] = {
                        'rmse_improvement_pct': rmse_improvement,
                        'mape_improvement_pct': mape_improvement,
                        'beats_baseline': model_rmse < result.metrics.rmse
                    }
        
        # Horizon comparison
        for h in range(self.forecast_horizon):
            horizon_data = {}
            for name, result in results.items():
                if h in result.horizon_metrics:
                    horizon_data[name] = {
                        'rmse': result.horizon_metrics[h].rmse,
                        'mae': result.horizon_metrics[h].mae,
                        'mape': result.horizon_metrics[h].mape
                    }
            report['horizon_comparison'][f'day_{h+1}'] = horizon_data
        
        return report
    
    def print_comparison(self, results: Dict[str, BaselineResult], model_name: str = "LSTM-Attention"):
        """Print a formatted comparison table."""
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON RESULTS")
        print("=" * 80)
        
        # Sort by RMSE
        sorted_results = sorted(results.items(), key=lambda x: x[1].metrics.rmse)
        
        print(f"\n{'Model':<25} {'RMSE':>10} {'MAE':>10} {'MAPE %':>10} {'Dir Acc %':>12} {'Sharpe':>10}")
        print("-" * 80)
        
        for name, result in sorted_results:
            sharpe = f"{result.metrics.sharpe_ratio:.2f}" if result.metrics.sharpe_ratio else "N/A"
            marker = " ***" if name == model_name else ""
            
            print(f"{name:<25} {result.metrics.rmse:>10.2f} {result.metrics.mae:>10.2f} "
                  f"{result.metrics.mape:>10.2f} {result.metrics.directional_accuracy:>12.1f} "
                  f"{sharpe:>10}{marker}")
        
        print("\n*** = Main model")
        
        # Check if model beats baselines
        if model_name in results:
            model_rmse = results[model_name].metrics.rmse
            beaten = sum(1 for n, r in results.items() 
                        if n != model_name and r.metrics.rmse > model_rmse)
            total = len(results) - 1
            
            print(f"\n{model_name} beats {beaten}/{total} baselines by RMSE")
            
            if beaten == total:
                print("✅ Model outperforms ALL baselines!")
            elif beaten >= total / 2:
                print("⚠️ Model outperforms most baselines, but not all.")
            else:
                print("❌ Model does NOT outperform majority of baselines!")


class WalkForwardEvaluator:
    """
    Walk-forward evaluation for proper time-series model assessment.
    
    This is the correct way to evaluate time-series models.
    Simple train/test splits can lead to overly optimistic results.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 60,
        gap: int = 0,
        expanding: bool = True
    ):
        """
        Initialize walk-forward evaluator.
        
        Args:
            n_splits: Number of train/test splits
            test_size: Size of each test set
            gap: Gap between train and test (to prevent leakage)
            expanding: If True, training set expands; if False, sliding window
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
    
    def evaluate(
        self,
        model_factory,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform walk-forward evaluation.
        
        Args:
            model_factory: Callable that returns a fresh model instance
            X: Full feature data
            y: Full target data
            feature_names: Names of features
            verbose: Print progress
        
        Returns:
            Evaluation results dictionary
        """
        n_samples = len(X)
        fold_metrics = []
        all_predictions = []
        all_actuals = []
        
        # Calculate split points
        min_train_size = n_samples - self.n_splits * (self.test_size + self.gap)
        if min_train_size < self.test_size:
            raise ValueError("Not enough data for specified number of splits")
        
        for fold in range(self.n_splits):
            if self.expanding:
                train_end = min_train_size + fold * self.test_size
                train_start = 0
            else:
                train_end = min_train_size + fold * self.test_size
                train_start = fold * self.test_size
            
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            # Split data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            if verbose:
                logger.info(f"Fold {fold+1}: Train {train_start}-{train_end}, "
                           f"Test {test_start}-{test_end}")
            
            # Train model
            model = model_factory()
            
            # Handle different model interfaces
            if hasattr(model, 'fit'):
                try:
                    model.fit(X_train, y_train, feature_names=feature_names)
                except TypeError:
                    model.fit(X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
                if hasattr(predictions, 'point_estimate'):
                    predictions = predictions.point_estimate
            else:
                predictions = model(X_test)
            
            # Calculate metrics
            calculator = MetricsCalculator()
            metrics = calculator.calculate_metrics(y_test, predictions)
            fold_metrics.append(metrics)
            
            all_predictions.append(predictions)
            all_actuals.append(y_test)
            
            if verbose:
                logger.info(f"  Fold {fold+1} RMSE: {metrics.rmse:.4f}, "
                           f"MAE: {metrics.mae:.4f}, MAPE: {metrics.mape:.2f}%")
        
        # Aggregate metrics
        aggregated = {
            'n_folds': len(fold_metrics),
            'metrics': {
                'rmse_mean': np.mean([m.rmse for m in fold_metrics]),
                'rmse_std': np.std([m.rmse for m in fold_metrics]),
                'mae_mean': np.mean([m.mae for m in fold_metrics]),
                'mae_std': np.std([m.mae for m in fold_metrics]),
                'mape_mean': np.mean([m.mape for m in fold_metrics]),
                'mape_std': np.std([m.mape for m in fold_metrics]),
                'directional_accuracy_mean': np.mean([m.directional_accuracy for m in fold_metrics]),
            },
            'fold_metrics': [m.to_dict() for m in fold_metrics]
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("WALK-FORWARD EVALUATION SUMMARY")
            print("=" * 60)
            print(f"Number of folds: {aggregated['n_folds']}")
            print(f"RMSE: {aggregated['metrics']['rmse_mean']:.4f} "
                  f"(±{aggregated['metrics']['rmse_std']:.4f})")
            print(f"MAE: {aggregated['metrics']['mae_mean']:.4f} "
                  f"(±{aggregated['metrics']['mae_std']:.4f})")
            print(f"MAPE: {aggregated['metrics']['mape_mean']:.2f}% "
                  f"(±{aggregated['metrics']['mape_std']:.2f}%)")
            print(f"Directional Accuracy: {aggregated['metrics']['directional_accuracy_mean']:.1f}%")
        
        return aggregated


def run_full_comparison(
    model_predictions: np.ndarray,
    y_true: np.ndarray,
    X: np.ndarray,
    prices: Optional[np.ndarray] = None,
    model_name: str = "LSTM-Attention-Ensemble",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full baseline comparison and generate report.
    
    Args:
        model_predictions: Model's predictions
        y_true: Actual values
        X: Input sequences
        prices: Initial prices for trading metrics
        model_name: Name of the main model
        save_path: Path to save report JSON
    
    Returns:
        Comparison report dictionary
    """
    comparator = BaselineComparator(
        forecast_horizon=y_true.shape[1] if len(y_true.shape) > 1 else 1,
        include_arima=True
    )
    
    results = comparator.compare(model_predictions, y_true, X, prices, model_name)
    comparator.print_comparison(results, model_name)
    
    report = comparator.generate_comparison_report(results, model_name)
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Comparison report saved to {save_path}")
    
    return report


if __name__ == "__main__":
    # Test baseline comparison with synthetic data
    np.random.seed(42)
    
    # Synthetic data
    n_samples = 100
    seq_length = 60
    n_features = 10
    forecast_horizon = 7
    
    # Create realistic-looking price sequences
    base_price = 150
    X = np.zeros((n_samples, seq_length, n_features))
    for i in range(n_samples):
        prices = base_price + np.cumsum(np.random.randn(seq_length) * 2)
        X[i, :, 0] = prices  # Close price
        for f in range(1, n_features):
            X[i, :, f] = prices + np.random.randn(seq_length) * 5
    
    # True values
    y_true = np.zeros((n_samples, forecast_horizon))
    for i in range(n_samples):
        last_price = X[i, -1, 0]
        y_true[i] = last_price + np.cumsum(np.random.randn(forecast_horizon) * 2)
    
    # Simulated model predictions (slightly better than naive)
    model_predictions = y_true + np.random.randn(n_samples, forecast_horizon) * 1.5
    
    # Run comparison
    report = run_full_comparison(
        model_predictions=model_predictions,
        y_true=y_true,
        X=X,
        model_name="Test-Model"
    )
    
    print("\n\nTest completed successfully!")
