"""
Evaluation and Visualization Module for Stock Price Prediction.

This module provides comprehensive evaluation metrics and visualization
tools for analyzing model performance.
"""

import os
import json
import logging
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.

    This class provides methods for evaluating prediction performance
    and creating visualizations including actual vs predicted plots,
    attention weight visualizations, and error analysis.

    Attributes:
        results_dir (str): Directory to save results.
        figures_dir (str): Directory to save figures.

    Example:
        >>> evaluator = ModelEvaluator(results_dir='./results')
        >>> evaluator.plot_predictions(predictions, actuals, dates)
        >>> evaluator.plot_attention_weights(attention, feature_names)
    """

    def __init__(
        self,
        results_dir: str = './results',
        ticker: str = 'STOCK'
    ):
        """
        Initialize the evaluator.

        Args:
            results_dir: Directory to save results.
            ticker: Stock ticker for labeling.
        """
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        self.metrics_dir = os.path.join(results_dir, 'metrics')
        self.ticker = ticker

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        horizon: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            predictions: Predicted values.
            actuals: Actual values.
            horizon: Specific horizon to evaluate (None for all).

        Returns:
            Dictionary of metric names to values.
        """
        if horizon is not None:
            pred = predictions[:, horizon]
            actual = actuals[:, horizon]
        else:
            pred = predictions.flatten()
            actual = actuals.flatten()

        # Basic metrics
        rmse = np.sqrt(np.mean((pred - actual)**2))
        mae = np.mean(np.abs(pred - actual))
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100

        # Additional metrics
        mse = np.mean((pred - actual)**2)

        # R-squared
        ss_res = np.sum((actual - pred)**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # Directional accuracy (for daily returns)
        if len(pred) > 1:
            pred_direction = np.sign(np.diff(pred))
            actual_direction = np.sign(np.diff(actual))
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
        else:
            directional_accuracy = np.nan

        # Symmetric MAPE
        smape = np.mean(
            2 * np.abs(pred - actual) / (np.abs(pred) + np.abs(actual) + 1e-8)
        ) * 100

        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'mse': float(mse),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'smape': float(smape)
        }

    def calculate_rolling_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate rolling evaluation metrics.

        Args:
            predictions: Predicted values (first horizon).
            actuals: Actual values (first horizon).
            window: Rolling window size.

        Returns:
            DataFrame with rolling metrics.
        """
        # Use first prediction horizon
        if len(predictions.shape) > 1:
            pred = predictions[:, 0]
            actual = actuals[:, 0]
        else:
            pred = predictions
            actual = actuals

        # Create DataFrame
        df = pd.DataFrame({
            'predicted': pred,
            'actual': actual,
            'error': pred - actual,
            'abs_error': np.abs(pred - actual),
            'squared_error': (pred - actual)**2
        })

        # Rolling metrics
        df['rolling_rmse'] = np.sqrt(df['squared_error'].rolling(window=window).mean())
        df['rolling_mae'] = df['abs_error'].rolling(window=window).mean()
        df['rolling_bias'] = df['error'].rolling(window=window).mean()

        return df

    def plot_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: Optional[List] = None,
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot actual vs predicted prices.

        Args:
            predictions: Predicted values.
            actuals: Actual values.
            dates: Dates for x-axis.
            title: Plot title.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib figure.
        """
        # Use first prediction horizon
        pred = predictions[:, 0] if len(predictions.shape) > 1 else predictions
        actual = actuals[:, 0] if len(actuals.shape) > 1 else actuals

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Main plot
        ax1 = axes[0]
        x = dates if dates is not None else range(len(pred))

        ax1.plot(x, actual, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
        ax1.plot(x, pred, label='Predicted', color='#E94F37', linewidth=2, alpha=0.8)
        ax1.fill_between(x, actual, pred, alpha=0.2, color='gray')

        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(
            title or f'{self.ticker} Stock Price: Actual vs Predicted',
            fontsize=14, fontweight='bold'
        )
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        if dates is not None:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Error plot
        ax2 = axes[1]
        errors = pred - actual
        colors = ['#E94F37' if e > 0 else '#2E86AB' for e in errors]
        ax2.bar(x, errors, color=colors, alpha=0.6, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Error ($)', fontsize=12)
        ax2.set_title('Prediction Error', fontsize=12)
        ax2.grid(True, alpha=0.3)

        if dates is not None:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_dir, f'{self.ticker}_predictions.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {path}")

        if show:
            plt.show()

        return fig

    def plot_rolling_rmse(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: Optional[List] = None,
        window: int = 20,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot rolling RMSE over time.

        Args:
            predictions: Predicted values.
            actuals: Actual values.
            dates: Dates for x-axis.
            window: Rolling window size.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib figure.
        """
        rolling_df = self.calculate_rolling_metrics(predictions, actuals, window)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        x = dates if dates is not None else range(len(rolling_df))

        # Rolling RMSE
        ax1 = axes[0]
        ax1.plot(x, rolling_df['rolling_rmse'], color='#E94F37', linewidth=2)
        ax1.fill_between(x, 0, rolling_df['rolling_rmse'], alpha=0.3, color='#E94F37')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Rolling RMSE ($)', fontsize=12)
        ax1.set_title(f'{self.ticker} Rolling RMSE (Window={window})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Rolling Bias
        ax2 = axes[1]
        bias = rolling_df['rolling_bias']
        colors = ['#E94F37' if b > 0 else '#2E86AB' for b in bias.fillna(0)]
        ax2.fill_between(x, 0, bias, alpha=0.5,
                        where=bias >= 0, color='#E94F37', label='Over-prediction')
        ax2.fill_between(x, 0, bias, alpha=0.5,
                        where=bias < 0, color='#2E86AB', label='Under-prediction')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Rolling Bias ($)', fontsize=12)
        ax2.set_title('Rolling Prediction Bias', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if dates is not None:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_dir, f'{self.ticker}_rolling_rmse.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved rolling RMSE plot to {path}")

        if show:
            plt.show()

        return fig

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 5,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Visualize attention weights.

        Args:
            attention_weights: Attention weights of shape (samples, sequence_length).
            sample_indices: Specific samples to visualize.
            feature_names: Names of features (not used for time attention).
            n_samples: Number of samples to show.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib figure.
        """
        if sample_indices is None:
            # Select evenly spaced samples
            total = len(attention_weights)
            sample_indices = np.linspace(0, total-1, n_samples, dtype=int)

        n_samples = len(sample_indices)
        seq_length = attention_weights.shape[1]

        fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3*n_samples))
        if n_samples == 1:
            axes = [axes]

        for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
            weights = attention_weights[sample_idx]

            # Create heatmap-like bar plot
            colors = plt.cm.YlOrRd(weights / weights.max())
            bars = ax.bar(range(seq_length), weights, color=colors, edgecolor='none')

            ax.set_xlabel('Time Step (days ago)', fontsize=10)
            ax.set_ylabel('Attention Weight', fontsize=10)
            ax.set_title(f'Sample {sample_idx}: Attention over Time Steps', fontsize=11)

            # Invert x-axis labels to show "days ago"
            ax.set_xticks(range(0, seq_length, 10))
            ax.set_xticklabels([seq_length - t for t in range(0, seq_length, 10)])

            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_dir, f'{self.ticker}_attention_weights.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention weights plot to {path}")

        if show:
            plt.show()

        return fig

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        dates: Optional[List] = None,
        n_samples: int = 50,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot attention weights as a heatmap.

        Args:
            attention_weights: Attention weights of shape (samples, sequence_length).
            dates: Dates for y-axis labels.
            n_samples: Number of samples to show.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib figure.
        """
        # Select subset of samples
        step = max(1, len(attention_weights) // n_samples)
        weights_subset = attention_weights[::step][:n_samples]

        fig, ax = plt.subplots(figsize=(14, 8))

        seq_length = weights_subset.shape[1]

        # Create heatmap
        im = ax.imshow(weights_subset, aspect='auto', cmap='YlOrRd')

        ax.set_xlabel('Time Step in Sequence (0 = oldest)', fontsize=12)
        ax.set_ylabel('Prediction Sample', fontsize=12)
        ax.set_title(f'{self.ticker} Attention Weight Heatmap', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=11)

        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_dir, f'{self.ticker}_attention_heatmap.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {path}")

        if show:
            plt.show()

        return fig

    def plot_horizon_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot performance metrics across forecast horizons.

        Args:
            predictions: Predictions of shape (samples, horizons).
            actuals: Actuals of shape (samples, horizons).
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib figure.
        """
        n_horizons = predictions.shape[1]
        horizons = range(1, n_horizons + 1)

        metrics = {'rmse': [], 'mae': [], 'mape': []}

        for h in range(n_horizons):
            m = self.calculate_metrics(predictions, actuals, horizon=h)
            metrics['rmse'].append(m['rmse'])
            metrics['mae'].append(m['mae'])
            metrics['mape'].append(m['mape'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RMSE
        axes[0].bar(horizons, metrics['rmse'], color='#E94F37', alpha=0.8)
        axes[0].set_xlabel('Forecast Horizon (Days)', fontsize=12)
        axes[0].set_ylabel('RMSE ($)', fontsize=12)
        axes[0].set_title('RMSE by Horizon', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # MAE
        axes[1].bar(horizons, metrics['mae'], color='#2E86AB', alpha=0.8)
        axes[1].set_xlabel('Forecast Horizon (Days)', fontsize=12)
        axes[1].set_ylabel('MAE ($)', fontsize=12)
        axes[1].set_title('MAE by Horizon', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # MAPE
        axes[2].bar(horizons, metrics['mape'], color='#A23B72', alpha=0.8)
        axes[2].set_xlabel('Forecast Horizon (Days)', fontsize=12)
        axes[2].set_ylabel('MAPE (%)', fontsize=12)
        axes[2].set_title('MAPE by Horizon', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_dir, f'{self.ticker}_horizon_performance.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved horizon performance plot to {path}")

        if show:
            plt.show()

        return fig

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot error distribution analysis.

        Args:
            predictions: Predicted values.
            actuals: Actual values.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib figure.
        """
        pred = predictions[:, 0] if len(predictions.shape) > 1 else predictions
        actual = actuals[:, 0] if len(actuals.shape) > 1 else actuals
        errors = pred - actual

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)

        # Error histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(errors, bins=50, color='#2E86AB', alpha=0.7, edgecolor='white')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(errors):.2f}')
        ax1.set_xlabel('Prediction Error ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Error Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Residuals vs Predicted
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(pred, errors, alpha=0.5, color='#E94F37', s=20)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Predicted Price ($)', fontsize=12)
        ax3.set_ylabel('Residual ($)', fontsize=12)
        ax3.set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Actual vs Predicted scatter
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(actual, pred, alpha=0.5, color='#2E86AB', s=20)
        min_val = min(actual.min(), pred.min())
        max_val = max(actual.max(), pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect Prediction')
        ax4.set_xlabel('Actual Price ($)', fontsize=12)
        ax4.set_ylabel('Predicted Price ($)', fontsize=12)
        ax4.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_dir, f'{self.ticker}_error_analysis.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved error analysis plot to {path}")

        if show:
            plt.show()

        return fig

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot training history (loss and metrics over epochs).

        Args:
            history: Dictionary with training history.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history.get('loss', [])) + 1)

        # Loss
        ax1 = axes[0]
        ax1.plot(epochs, history.get('loss', []), label='Training Loss',
                color='#2E86AB', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], label='Validation Loss',
                    color='#E94F37', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE
        ax2 = axes[1]
        if 'mae' in history:
            ax2.plot(epochs, history['mae'], label='Training MAE',
                    color='#2E86AB', linewidth=2)
        if 'val_mae' in history:
            ax2.plot(epochs, history['val_mae'], label='Validation MAE',
                    color='#E94F37', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title('Training and Validation MAE', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_dir, f'{self.ticker}_training_history.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training history plot to {path}")

        if show:
            plt.show()

        return fig

    def generate_report(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
        dates: Optional[List] = None,
        history: Optional[Dict] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Args:
            predictions: Predicted values.
            actuals: Actual values.
            attention_weights: Attention weights (optional).
            dates: Dates for x-axis (optional).
            history: Training history (optional).
            save: Whether to save figures and metrics.

        Returns:
            Dictionary with all metrics and figure paths.
        """
        logger.info(f"Generating evaluation report for {self.ticker}...")

        report = {
            'ticker': self.ticker,
            'metrics': {},
            'figures': []
        }

        # Calculate metrics
        report['metrics']['overall'] = self.calculate_metrics(predictions, actuals)

        # Per-horizon metrics
        n_horizons = predictions.shape[1]
        report['metrics']['by_horizon'] = {}
        for h in range(n_horizons):
            report['metrics']['by_horizon'][f'day_{h+1}'] = \
                self.calculate_metrics(predictions, actuals, horizon=h)

        # Generate all plots
        fig = self.plot_predictions(predictions, actuals, dates, save=save)
        report['figures'].append('predictions.png')
        plt.close(fig)

        fig = self.plot_rolling_rmse(predictions, actuals, dates, save=save)
        report['figures'].append('rolling_rmse.png')
        plt.close(fig)

        fig = self.plot_horizon_performance(predictions, actuals, save=save)
        report['figures'].append('horizon_performance.png')
        plt.close(fig)

        fig = self.plot_error_distribution(predictions, actuals, save=save)
        report['figures'].append('error_analysis.png')
        plt.close(fig)

        if attention_weights is not None:
            fig = self.plot_attention_weights(attention_weights, save=save)
            report['figures'].append('attention_weights.png')
            plt.close(fig)

            fig = self.plot_attention_heatmap(attention_weights, save=save)
            report['figures'].append('attention_heatmap.png')
            plt.close(fig)

        if history is not None:
            fig = self.plot_training_history(history, save=save)
            report['figures'].append('training_history.png')
            plt.close(fig)

        # Save report
        if save:
            report_path = os.path.join(self.metrics_dir, f'{self.ticker}_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Saved report to {report_path}")

        logger.info("Report generation complete")
        return report


def evaluate_model(
    trainer,
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a trained model.

    Args:
        trainer: Trained StockPriceTrainer instance.
        show_plots: Whether to display plots.

    Returns:
        Evaluation report dictionary.
    """
    # Get predictions with attention
    predictions, actuals, attention = trainer.predict()

    # Create evaluator
    evaluator = ModelEvaluator(
        results_dir=trainer.results_dir,
        ticker=trainer.ticker
    )

    # Get training history if available
    history = None
    if trainer.history is not None:
        history = {k: [float(v) for v in vals]
                  for k, vals in trainer.history.history.items()}

    # Generate report
    report = evaluator.generate_report(
        predictions=predictions,
        actuals=actuals,
        attention_weights=attention,
        dates=trainer.test_dates,
        history=history,
        save=True
    )

    if show_plots:
        # Show key plots
        evaluator.plot_predictions(predictions, actuals, trainer.test_dates, save=False, show=True)

    return report


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 200
    n_horizons = 7
    seq_length = 60

    # Simulated predictions and actuals
    base_price = 150
    actuals = base_price + np.cumsum(np.random.randn(n_samples) * 2)
    actuals = np.tile(actuals.reshape(-1, 1), (1, n_horizons))
    actuals = actuals + np.random.randn(n_samples, n_horizons) * 3

    predictions = actuals + np.random.randn(n_samples, n_horizons) * 5

    # Simulated attention weights
    attention = np.random.dirichlet(np.ones(seq_length), size=n_samples)

    # Dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='B').tolist()

    # Training history
    history = {
        'loss': list(np.exp(-np.linspace(0, 2, 50)) + np.random.rand(50) * 0.1),
        'val_loss': list(np.exp(-np.linspace(0, 1.8, 50)) + np.random.rand(50) * 0.15),
        'mae': list(np.exp(-np.linspace(0, 1.5, 50)) * 10 + np.random.rand(50)),
        'val_mae': list(np.exp(-np.linspace(0, 1.3, 50)) * 10 + np.random.rand(50) * 1.2)
    }

    # Create evaluator and generate report
    evaluator = ModelEvaluator(results_dir='./results', ticker='TEST')
    report = evaluator.generate_report(
        predictions=predictions,
        actuals=actuals,
        attention_weights=attention,
        dates=dates,
        history=history,
        save=True
    )

    print("\n=== Evaluation Report ===")
    print(f"Overall Metrics: {report['metrics']['overall']}")
    print(f"Generated Figures: {report['figures']}")
