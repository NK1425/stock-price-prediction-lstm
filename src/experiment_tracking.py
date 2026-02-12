"""
MLflow Experiment Tracking for Stock Price Prediction.

This module provides experiment tracking capabilities:
1. Log all training runs with parameters and metrics
2. Track model versions
3. Compare experiments
4. Model registry integration

MLflow is the industry standard for ML experiment tracking.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import mlflow.tensorflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class ExperimentTracker:
    """
    MLflow experiment tracking wrapper.
    
    Provides a clean interface for:
    - Creating/managing experiments
    - Logging parameters, metrics, and artifacts
    - Model versioning and registration
    - Experiment comparison
    """
    
    def __init__(
        self,
        experiment_name: str = "stock-price-prediction",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Location for storing artifacts
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', './mlruns')
        self.artifact_location = artifact_location
        
        self.experiment_id = None
        self.active_run = None
        
        if HAS_MLFLOW:
            self._setup_mlflow()
        else:
            logger.warning("MLflow not available. Tracking disabled.")
    
    def _setup_mlflow(self):
        """Configure MLflow settings."""
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location
            )
            logger.info(f"Created experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to attach to the run
        
        Returns:
            Run ID if successful
        """
        if not HAS_MLFLOW:
            return None
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started run: {run_name} (ID: {self.active_run.info.run_id})")
        
        return self.active_run.info.run_id
    
    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if HAS_MLFLOW and self.active_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended run with status: {status}")
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if HAS_MLFLOW:
            for key, value in params.items():
                mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if HAS_MLFLOW:
            for key, value in metrics.items():
                if value is not None and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        if HAS_MLFLOW:
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, data: Dict, filename: str):
        """Log a dictionary as a JSON artifact."""
        if HAS_MLFLOW:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name
            mlflow.log_artifact(temp_path, artifact_path="data")
            os.unlink(temp_path)
    
    def log_figure(self, figure, filename: str):
        """Log a matplotlib figure."""
        if HAS_MLFLOW:
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            figure.savefig(temp_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(temp_path, artifact_path="figures")
            os.unlink(temp_path)
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_name: Optional[str] = None
    ):
        """
        Log a trained model.
        
        Args:
            model: Trained model (Keras, sklearn, etc.)
            artifact_path: Path within artifacts to store model
            registered_name: Name to register in model registry
        """
        if not HAS_MLFLOW:
            return
        
        # Detect model type and log accordingly
        model_type = type(model).__module__
        
        if 'tensorflow' in model_type or 'keras' in model_type:
            mlflow.tensorflow.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_name
            )
        elif 'sklearn' in model_type or 'xgboost' in model_type or 'lightgbm' in model_type:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_name
            )
        else:
            # Generic Python model
            import mlflow.pyfunc
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=registered_name
            )
        
        logger.info(f"Logged model to {artifact_path}")
    
    def log_training_run(
        self,
        ticker: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        baseline_comparison: Optional[Dict] = None,
        model=None,
        figures: Optional[List] = None
    ):
        """
        Log a complete training run.
        
        Convenience method that logs everything for a training run.
        
        Args:
            ticker: Stock ticker
            params: Training parameters
            metrics: Evaluation metrics
            baseline_comparison: Comparison with baselines
            model: Trained model
            figures: List of (figure, filename) tuples
        """
        if not HAS_MLFLOW:
            return
        
        run_name = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.start_run(
            run_name=run_name,
            tags={
                "ticker": ticker,
                "task": "stock_prediction"
            }
        )
        
        try:
            # Log parameters
            self.log_params({
                "ticker": ticker,
                **params
            })
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log baseline comparison
            if baseline_comparison:
                self.log_dict(baseline_comparison, "baseline_comparison.json")
                
                # Log improvement metrics
                for baseline, comparison in baseline_comparison.get('improvement_over_baselines', {}).items():
                    if isinstance(comparison, dict):
                        self.log_metrics({
                            f"improvement_vs_{baseline}": comparison.get('rmse_improvement_pct', 0)
                        })
            
            # Log model
            if model:
                self.log_model(model, "model")
            
            # Log figures
            if figures:
                for fig, filename in figures:
                    self.log_figure(fig, filename)
            
            self.end_run()
            
        except Exception as e:
            logger.error(f"Error logging run: {e}")
            self.end_run(status="FAILED")
    
    def get_best_run(
        self,
        metric_name: str = "rmse",
        maximize: bool = False
    ) -> Optional[Dict]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Metric to optimize
            maximize: If True, find max; if False, find min
        
        Returns:
            Best run info dictionary
        """
        if not HAS_MLFLOW:
            return None
        
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "start_time": run.info.start_time
            }
        
        return None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str] = ["rmse", "mae", "mape"]
    ) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to include in comparison
        
        Returns:
            DataFrame with comparison
        """
        if not HAS_MLFLOW:
            return pd.DataFrame()
        
        client = MlflowClient()
        comparison_data = []
        
        for run_id in run_ids:
            run = client.get_run(run_id)
            row = {
                "run_id": run_id,
                "run_name": run.info.run_name,
                **{m: run.data.metrics.get(m) for m in metrics}
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


def track_experiment(
    func=None,
    experiment_name: str = "stock-price-prediction"
):
    """
    Decorator for tracking experiments.
    
    Usage:
        @track_experiment(experiment_name="my-experiment")
        def train_model(ticker, **params):
            # Training code
            return metrics
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            tracker = ExperimentTracker(experiment_name=experiment_name)
            tracker.start_run()
            
            try:
                result = f(*args, **kwargs)
                
                # If result is a dict with metrics, log them
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() 
                              if isinstance(v, (int, float))}
                    tracker.log_metrics(metrics)
                
                tracker.end_run()
                return result
                
            except Exception as e:
                tracker.end_run(status="FAILED")
                raise
        
        return wrapper
    
    if func:
        return decorator(func)
    return decorator


if __name__ == "__main__":
    # Test experiment tracking
    print("Testing Experiment Tracking...")
    
    if HAS_MLFLOW:
        tracker = ExperimentTracker(experiment_name="test-experiment")
        
        # Simulate a training run
        tracker.start_run(run_name="test-run")
        
        tracker.log_params({
            "ticker": "AAPL",
            "sequence_length": 60,
            "forecast_horizon": 7,
            "lstm_units": [128, 64],
            "learning_rate": 0.001
        })
        
        tracker.log_metrics({
            "rmse": 3.45,
            "mae": 2.67,
            "mape": 1.82,
            "directional_accuracy": 56.3
        })
        
        tracker.end_run()
        
        print("Test run logged successfully!")
        print(f"View results at: {tracker.tracking_uri}")
    else:
        print("MLflow not available. Install with: pip install mlflow")
