"""
Model Monitoring and Drift Detection.

Production ML models need continuous monitoring for:
1. Model drift: When predictions diverge from actuals
2. Data quality: Anomalies in input data
3. Performance degradation: Metrics getting worse over time
4. Input distribution shift: Features changing over time

This is what separates production ML from notebook ML.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import os

import numpy as np
import pandas as pd
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert for detected drift or anomaly."""
    alert_type: str  # 'model_drift', 'data_quality', 'performance', 'input_shift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PredictionMonitor:
    """
    Monitor predictions vs actuals for model drift.
    
    Tracks:
    - Rolling RMSE/MAE
    - Directional accuracy over time
    - Prediction bias
    - Residual distribution changes
    """
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.2,  # 20% increase in error
        alert_cooldown_hours: int = 24
    ):
        """
        Initialize prediction monitor.
        
        Args:
            window_size: Rolling window for metrics
            drift_threshold: Relative change to trigger alert
            alert_cooldown_hours: Hours between repeat alerts
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.alert_cooldown_hours = alert_cooldown_hours
        
        # Rolling buffers
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Baseline metrics (set during initialization)
        self.baseline_rmse = None
        self.baseline_mae = None
        self.baseline_directional_accuracy = None
        
        # Alert tracking
        self.alerts = []
        self.last_alert_time = {}
    
    def set_baseline(
        self,
        rmse: float,
        mae: float,
        directional_accuracy: float
    ):
        """Set baseline metrics from training evaluation."""
        self.baseline_rmse = rmse
        self.baseline_mae = mae
        self.baseline_directional_accuracy = directional_accuracy
        logger.info(f"Baseline set - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    def record_prediction(
        self,
        prediction: float,
        actual: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a prediction-actual pair.
        
        Args:
            prediction: Predicted value
            actual: Actual value
            timestamp: Time of prediction
        """
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.now())
        
        # Check for drift after enough data
        if len(self.predictions) >= self.window_size // 2:
            self._check_drift()
    
    def record_batch(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ):
        """Record batch of predictions."""
        if timestamps is None:
            timestamps = [datetime.now()] * len(predictions)
        
        for pred, actual, ts in zip(predictions.flatten(), actuals.flatten(), timestamps):
            self.record_prediction(pred, actual, ts)
    
    def _check_drift(self):
        """Check for model drift and create alerts."""
        if self.baseline_rmse is None:
            return
        
        current_rmse = self._calculate_rmse()
        current_mae = self._calculate_mae()
        current_da = self._calculate_directional_accuracy()
        
        # Check RMSE drift
        if self.baseline_rmse > 0:
            rmse_change = (current_rmse - self.baseline_rmse) / self.baseline_rmse
            if rmse_change > self.drift_threshold:
                self._create_alert(
                    alert_type='model_drift',
                    severity=self._get_severity(rmse_change),
                    metric_name='rmse',
                    current_value=current_rmse,
                    threshold=self.baseline_rmse * (1 + self.drift_threshold),
                    message=f"RMSE increased by {rmse_change*100:.1f}% from baseline"
                )
        
        # Check directional accuracy drift
        if self.baseline_directional_accuracy is not None and self.baseline_directional_accuracy > 0:
            da_change = (self.baseline_directional_accuracy - current_da) / self.baseline_directional_accuracy
            if da_change > self.drift_threshold:
                self._create_alert(
                    alert_type='performance',
                    severity=self._get_severity(da_change),
                    metric_name='directional_accuracy',
                    current_value=current_da,
                    threshold=self.baseline_directional_accuracy * (1 - self.drift_threshold),
                    message=f"Directional accuracy dropped by {da_change*100:.1f}%"
                )
    
    def _calculate_rmse(self) -> float:
        """Calculate current rolling RMSE."""
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        return np.sqrt(np.mean((preds - acts) ** 2))
    
    def _calculate_mae(self) -> float:
        """Calculate current rolling MAE."""
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        return np.mean(np.abs(preds - acts))
    
    def _calculate_directional_accuracy(self) -> float:
        """Calculate current rolling directional accuracy."""
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        if len(preds) < 2:
            return 50.0
        
        pred_direction = np.sign(np.diff(preds))
        actual_direction = np.sign(np.diff(acts))
        
        return np.mean(pred_direction == actual_direction) * 100
    
    def _get_severity(self, change: float) -> str:
        """Get alert severity based on change magnitude."""
        if change > 0.5:
            return 'critical'
        elif change > 0.3:
            return 'high'
        elif change > 0.2:
            return 'medium'
        return 'low'
    
    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        message: str
    ):
        """Create an alert if cooldown allows."""
        alert_key = f"{alert_type}_{metric_name}"
        
        # Check cooldown
        if alert_key in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[alert_key]
            if time_since_last < timedelta(hours=self.alert_cooldown_hours):
                return
        
        alert = DriftAlert(
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        
        self.alerts.append(alert)
        self.last_alert_time[alert_key] = datetime.now()
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current rolling metrics."""
        if len(self.predictions) < 2:
            return {}
        
        return {
            'rolling_rmse': self._calculate_rmse(),
            'rolling_mae': self._calculate_mae(),
            'rolling_directional_accuracy': self._calculate_directional_accuracy(),
            'sample_count': len(self.predictions),
            'baseline_rmse': self.baseline_rmse,
            'baseline_mae': self.baseline_mae
        }
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[DriftAlert]:
        """Get alerts filtered by severity and/or time."""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts 
                     if datetime.fromisoformat(a.timestamp) > since]
        
        return alerts


class DataQualityMonitor:
    """
    Monitor input data quality for anomalies.
    
    Checks:
    - Missing values
    - Outliers
    - Value range changes
    - Volume anomalies
    """
    
    def __init__(
        self,
        outlier_std_threshold: float = 3.0,
        missing_threshold: float = 0.1
    ):
        """
        Initialize data quality monitor.
        
        Args:
            outlier_std_threshold: Std deviations for outlier detection
            missing_threshold: Fraction of missing values to alert
        """
        self.outlier_std_threshold = outlier_std_threshold
        self.missing_threshold = missing_threshold
        
        # Reference statistics (set during training)
        self.feature_stats = {}
        self.alerts = []
    
    def set_reference_stats(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Set reference statistics from training data.
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to monitor
        """
        for col in feature_columns:
            if col in df.columns:
                self.feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75)
                }
        
        logger.info(f"Set reference stats for {len(self.feature_stats)} features")
    
    def check_data_quality(
        self,
        df: pd.DataFrame,
        raise_on_critical: bool = False
    ) -> List[DriftAlert]:
        """
        Check data quality and return alerts.
        
        Args:
            df: DataFrame to check
            raise_on_critical: Raise exception on critical issues
        
        Returns:
            List of alerts
        """
        alerts = []
        
        # Check missing values
        missing_alerts = self._check_missing_values(df)
        alerts.extend(missing_alerts)
        
        # Check outliers
        outlier_alerts = self._check_outliers(df)
        alerts.extend(outlier_alerts)
        
        # Check distribution shift
        shift_alerts = self._check_distribution_shift(df)
        alerts.extend(shift_alerts)
        
        self.alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            level = 'error' if alert.severity in ['critical', 'high'] else 'warning'
            getattr(logger, level)(f"Data Quality [{alert.severity}]: {alert.message}")
        
        if raise_on_critical:
            critical = [a for a in alerts if a.severity == 'critical']
            if critical:
                raise ValueError(f"Critical data quality issues: {[a.message for a in critical]}")
        
        return alerts
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[DriftAlert]:
        """Check for missing values."""
        alerts = []
        
        for col in self.feature_stats.keys():
            if col not in df.columns:
                continue
            
            missing_frac = df[col].isnull().mean()
            if missing_frac > self.missing_threshold:
                alerts.append(DriftAlert(
                    alert_type='data_quality',
                    severity='high' if missing_frac > 0.3 else 'medium',
                    metric_name=f'{col}_missing',
                    current_value=missing_frac,
                    threshold=self.missing_threshold,
                    message=f"{col} has {missing_frac*100:.1f}% missing values",
                    timestamp=datetime.now().isoformat()
                ))
        
        return alerts
    
    def _check_outliers(self, df: pd.DataFrame) -> List[DriftAlert]:
        """Check for outliers using IQR method."""
        alerts = []
        
        for col, stats in self.feature_stats.items():
            if col not in df.columns:
                continue
            
            iqr = stats['q3'] - stats['q1']
            lower_bound = stats['q1'] - 1.5 * iqr
            upper_bound = stats['q3'] + 1.5 * iqr
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).mean()
            
            if outliers > 0.1:  # More than 10% outliers
                alerts.append(DriftAlert(
                    alert_type='data_quality',
                    severity='medium',
                    metric_name=f'{col}_outliers',
                    current_value=outliers,
                    threshold=0.1,
                    message=f"{col} has {outliers*100:.1f}% outliers",
                    timestamp=datetime.now().isoformat()
                ))
        
        return alerts
    
    def _check_distribution_shift(self, df: pd.DataFrame) -> List[DriftAlert]:
        """Check for distribution shift from reference."""
        alerts = []
        
        for col, stats in self.feature_stats.items():
            if col not in df.columns or stats['std'] == 0:
                continue
            
            current_mean = df[col].mean()
            current_std = df[col].std()
            
            # Check mean shift (in terms of std)
            mean_shift = abs(current_mean - stats['mean']) / stats['std']
            
            if mean_shift > 2.0:  # Mean shifted by more than 2 std
                alerts.append(DriftAlert(
                    alert_type='input_shift',
                    severity='high' if mean_shift > 3.0 else 'medium',
                    metric_name=f'{col}_mean_shift',
                    current_value=mean_shift,
                    threshold=2.0,
                    message=f"{col} mean shifted by {mean_shift:.1f} std deviations",
                    timestamp=datetime.now().isoformat()
                ))
            
            # Check std change
            std_change = abs(current_std - stats['std']) / stats['std']
            
            if std_change > 0.5:  # Std changed by more than 50%
                alerts.append(DriftAlert(
                    alert_type='input_shift',
                    severity='medium',
                    metric_name=f'{col}_std_change',
                    current_value=std_change,
                    threshold=0.5,
                    message=f"{col} std changed by {std_change*100:.1f}%",
                    timestamp=datetime.now().isoformat()
                ))
        
        return alerts


class ModelHealthDashboard:
    """
    Aggregate monitoring information into a health dashboard.
    """
    
    def __init__(
        self,
        prediction_monitor: PredictionMonitor,
        data_quality_monitor: DataQualityMonitor
    ):
        self.prediction_monitor = prediction_monitor
        self.data_quality_monitor = data_quality_monitor
        self.last_update = None
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall model health status.
        
        Returns:
            Health status dictionary
        """
        pred_metrics = self.prediction_monitor.get_current_metrics()
        recent_alerts = self.prediction_monitor.get_alerts(
            since=datetime.now() - timedelta(hours=24)
        )
        recent_alerts.extend(
            self.data_quality_monitor.alerts[-10:]  # Last 10 alerts
        )
        
        # Determine overall status
        critical_count = sum(1 for a in recent_alerts if a.severity == 'critical')
        high_count = sum(1 for a in recent_alerts if a.severity == 'high')
        
        if critical_count > 0:
            status = 'critical'
            status_color = 'red'
        elif high_count > 0:
            status = 'degraded'
            status_color = 'orange'
        elif len(recent_alerts) > 5:
            status = 'warning'
            status_color = 'yellow'
        else:
            status = 'healthy'
            status_color = 'green'
        
        return {
            'status': status,
            'status_color': status_color,
            'metrics': pred_metrics,
            'alert_counts': {
                'critical': critical_count,
                'high': high_count,
                'medium': sum(1 for a in recent_alerts if a.severity == 'medium'),
                'low': sum(1 for a in recent_alerts if a.severity == 'low')
            },
            'recent_alerts': [a.to_dict() for a in recent_alerts[:5]],
            'last_update': datetime.now().isoformat()
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate a monitoring report.
        
        Args:
            output_path: Path to save report JSON
        
        Returns:
            Report dictionary
        """
        health = self.get_health_status()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'health_status': health['status'],
            'current_metrics': health['metrics'],
            'alerts_24h': health['alert_counts'],
            'all_recent_alerts': [a.to_dict() for a in 
                                  self.prediction_monitor.get_alerts(
                                      since=datetime.now() - timedelta(hours=24)
                                  )],
            'recommendations': self._generate_recommendations(health)
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Monitoring report saved to {output_path}")
        
        return report
    
    def _generate_recommendations(self, health: Dict) -> List[str]:
        """Generate recommendations based on health status."""
        recommendations = []
        
        if health['status'] == 'critical':
            recommendations.append("URGENT: Model requires immediate attention. Consider rolling back to previous version.")
            recommendations.append("Investigate recent data for anomalies or market regime changes.")
        
        if health['alert_counts']['high'] > 0:
            recommendations.append("Review high-severity alerts and consider model retraining.")
        
        metrics = health.get('metrics', {})
        if metrics.get('rolling_rmse', 0) > metrics.get('baseline_rmse', 0) * 1.2:
            recommendations.append("RMSE has increased significantly. Schedule model retraining.")
        
        if not recommendations:
            recommendations.append("Model is performing within expected parameters. Continue monitoring.")
        
        return recommendations


def create_monitoring_system(
    baseline_rmse: float,
    baseline_mae: float,
    baseline_da: float,
    training_data: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[PredictionMonitor, DataQualityMonitor, ModelHealthDashboard]:
    """
    Create and configure a complete monitoring system.
    
    Args:
        baseline_rmse: Baseline RMSE from training
        baseline_mae: Baseline MAE from training
        baseline_da: Baseline directional accuracy
        training_data: Training DataFrame for reference stats
        feature_columns: Feature columns to monitor
    
    Returns:
        Tuple of (prediction_monitor, data_quality_monitor, dashboard)
    """
    pred_monitor = PredictionMonitor()
    pred_monitor.set_baseline(baseline_rmse, baseline_mae, baseline_da)
    
    data_monitor = DataQualityMonitor()
    data_monitor.set_reference_stats(training_data, feature_columns)
    
    dashboard = ModelHealthDashboard(pred_monitor, data_monitor)
    
    return pred_monitor, data_monitor, dashboard


if __name__ == "__main__":
    # Test monitoring system
    print("Testing Monitoring System...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    
    # Synthetic training data
    train_data = pd.DataFrame({
        'Close': 150 + np.cumsum(np.random.randn(n_samples) * 2),
        'Volume': np.random.randint(1000000, 10000000, n_samples),
        'RSI_14': 50 + np.random.randn(n_samples) * 10
    })
    
    # Create monitoring system
    pred_monitor, data_monitor, dashboard = create_monitoring_system(
        baseline_rmse=3.0,
        baseline_mae=2.5,
        baseline_da=55.0,
        training_data=train_data,
        feature_columns=['Close', 'Volume', 'RSI_14']
    )
    
    # Simulate predictions with drift
    for i in range(100):
        # Add increasing error to simulate drift
        drift_factor = 1.0 + (i / 200)
        prediction = 150 + np.random.randn() * 2 * drift_factor
        actual = 150 + np.random.randn() * 2
        
        pred_monitor.record_prediction(prediction, actual)
    
    # Get health status
    health = dashboard.get_health_status()
    print(f"\nHealth Status: {health['status']}")
    print(f"Current Metrics: {health['metrics']}")
    print(f"Alert Counts: {health['alert_counts']}")
    
    # Generate report
    report = dashboard.generate_report('monitoring_report.json')
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Clean up
    os.remove('monitoring_report.json')
    print("\nMonitoring system test completed!")
