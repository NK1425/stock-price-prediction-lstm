"""
Training Pipeline for Stock Price Prediction.

This module provides a complete training pipeline with walk-forward
validation, proper data splitting, and comprehensive logging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data_loader import StockDataLoader, create_train_test_split
from feature_engineering import FeatureEngineer, create_target_variable
from sequence_builder import (
    SequenceBuilder,
    WalkForwardValidator,
    create_train_val_test_sequences
)
from model import (
    build_lstm_attention_model,
    build_attention_model_with_weights,
    get_callbacks,
    LSTMAttentionPredictor,
    AttentionLayer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class StockPriceTrainer:
    """
    Complete training pipeline for stock price prediction.

    This class orchestrates the entire training process including
    data loading, feature engineering, model training, and validation.

    Attributes:
        ticker (str): Stock ticker symbol.
        sequence_length (int): Number of time steps in input sequence.
        forecast_horizon (int): Number of days to predict.
        model_dir (str): Directory to save models.
        results_dir (str): Directory to save results.

    Example:
        >>> trainer = StockPriceTrainer('AAPL', sequence_length=60, forecast_horizon=7)
        >>> trainer.prepare_data('2019-01-01', '2024-01-01')
        >>> history = trainer.train(epochs=100)
        >>> metrics = trainer.evaluate()
    """

    def __init__(
        self,
        ticker: str = 'AAPL',
        sequence_length: int = 60,
        forecast_horizon: int = 7,
        model_dir: str = './models',
        results_dir: str = './results',
        data_dir: str = './data'
    ):
        """
        Initialize the trainer.

        Args:
            ticker: Stock ticker symbol.
            sequence_length: Number of time steps for LSTM.
            forecast_horizon: Number of days to predict.
            model_dir: Directory to save trained models.
            results_dir: Directory to save results and metrics.
            data_dir: Directory for data caching.
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.data_dir = data_dir

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)

        # Initialize components
        self.data_loader = StockDataLoader(cache_dir=data_dir)
        self.feature_engineer = FeatureEngineer()
        self.sequence_builder = None
        self.model = None
        self.attention_model = None

        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.test_dates = None

        # Training history
        self.history = None
        self.metrics = {}

        # Feature columns to use
        self.feature_columns = None

    def prepare_data(
        self,
        start_date: str,
        end_date: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> None:
        """
        Prepare data for training.

        This method downloads data, generates features, handles missing values,
        and creates train/val/test sequences.

        Args:
            start_date: Start date for data download.
            end_date: End date for data download.
            train_ratio: Proportion for training.
            val_ratio: Proportion for validation.
            test_ratio: Proportion for testing.
        """
        logger.info(f"Preparing data for {self.ticker}...")

        # Download data
        self.raw_data = self.data_loader.download_stock_data(
            self.ticker, start_date, end_date
        )
        logger.info(f"Downloaded {len(self.raw_data)} days of data")

        # Generate features
        self.processed_data = self.feature_engineer.add_all_features(self.raw_data)
        logger.info(f"Generated {len(self.feature_engineer.feature_columns)} features")

        # Handle missing values
        self.processed_data = self.feature_engineer.handle_missing_values(
            self.processed_data, method='drop'
        )

        # Define feature columns
        self.feature_columns = self._get_feature_columns()
        logger.info(f"Using {len(self.feature_columns)} features for modeling")

        # Initialize sequence builder
        self.sequence_builder = SequenceBuilder(
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            feature_columns=self.feature_columns,
            target_column='Close'
        )

        # Create sequences with time-based splitting
        (self.X_train, self.y_train), \
        (self.X_val, self.y_val), \
        (self.X_test, self.y_test), \
        self.test_dates = create_train_val_test_sequences(
            self.processed_data,
            self.sequence_builder,
            train_ratio, val_ratio, test_ratio
        )

        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Validation set: {self.X_val.shape}")
        logger.info(f"Test set: {self.X_test.shape}")

    def _get_feature_columns(self) -> List[str]:
        """Get the feature columns to use for modeling."""
        # Core features that should be included
        core_features = [
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]

        # Technical indicators
        technical_features = [
            'Returns', 'Log_Returns',
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
            'RSI_14', 'RSI_7',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Percent',
            'ATR', 'ATR_Percent',
            'Volatility_20d',
            'Momentum_10', 'Momentum_20',
            'OBV', 'OBV_Trend',
            'VWAP', 'Price_VWAP_Ratio',
            'Stoch_K', 'Stoch_D',
            'Williams_R', 'CCI',
            'ROC_10', 'MFI'
        ]

        # Combine and filter for available columns
        all_features = core_features + technical_features
        available_features = [
            f for f in all_features
            if f in self.processed_data.columns
        ]

        return available_features

    def build_model(
        self,
        lstm_units: List[int] = [128, 64, 32],
        attention_units: int = 64,
        dense_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ) -> None:
        """
        Build the LSTM-Attention model.

        Args:
            lstm_units: Units for each LSTM layer.
            attention_units: Units in attention layer.
            dense_units: Units for dense layers.
            dropout_rate: Dropout rate.
            learning_rate: Learning rate.
        """
        n_features = self.X_train.shape[2]
        input_shape = (self.sequence_length, n_features)

        self.model, self.attention_model = build_attention_model_with_weights(
            input_shape=input_shape,
            output_steps=self.forecast_horizon,
            lstm_units=lstm_units,
            attention_units=attention_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )

        logger.info("Model built successfully")
        self.model.summary(print_fn=logger.info)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            epochs: Maximum number of epochs.
            batch_size: Batch size.
            patience: Early stopping patience.

        Returns:
            Training history.
        """
        if self.model is None:
            self.build_model()

        logger.info("Starting training...")

        # Callbacks
        model_path = os.path.join(
            self.model_dir,
            f'{self.ticker}_lstm_attention.keras'
        )
        callbacks = get_callbacks(
            model_path=model_path,
            patience=patience
        )

        # Train
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")

        # Save training history
        self._save_training_history()

        return self.history

    def train_walk_forward(
        self,
        n_splits: int = 5,
        epochs_per_split: int = 50,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train using walk-forward validation.

        This method trains multiple models using expanding window
        walk-forward validation to prevent data leakage.

        Args:
            n_splits: Number of validation splits.
            epochs_per_split: Epochs per split.
            batch_size: Batch size.

        Returns:
            Dictionary of metrics across splits.
        """
        logger.info(f"Starting walk-forward validation with {n_splits} splits...")

        validator = WalkForwardValidator(
            n_splits=n_splits,
            test_size=60,
            expanding=True
        )

        all_metrics = {
            'rmse': [],
            'mae': [],
            'mape': []
        }
        all_predictions = []
        all_actuals = []

        splits = validator.split(self.processed_data)

        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Split {i+1}/{n_splits} ---")

            # Get data for this split
            train_data = self.processed_data.iloc[train_idx]
            test_data = self.processed_data.iloc[test_idx]

            # Create new sequence builder for this split
            split_builder = SequenceBuilder(
                sequence_length=self.sequence_length,
                forecast_horizon=self.forecast_horizon,
                feature_columns=self.feature_columns,
                target_column='Close'
            )

            # Create sequences
            X_train_split, y_train_split = split_builder.create_sequences(
                train_data, fit=True
            )
            X_test_split, y_test_split = split_builder.transform_sequences(test_data)

            if len(X_train_split) == 0 or len(X_test_split) == 0:
                logger.warning(f"Split {i+1} has insufficient data, skipping")
                continue

            # Build fresh model
            n_features = X_train_split.shape[2]
            model = build_lstm_attention_model(
                input_shape=(self.sequence_length, n_features),
                output_steps=self.forecast_horizon,
                lstm_units=[64, 32],
                dropout_rate=0.2
            )

            # Train with early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # Use last 20% of training for validation
            val_size = int(len(X_train_split) * 0.2)
            X_train_inner = X_train_split[:-val_size]
            y_train_inner = y_train_split[:-val_size]
            X_val_inner = X_train_split[-val_size:]
            y_val_inner = y_train_split[-val_size:]

            model.fit(
                X_train_inner, y_train_inner,
                validation_data=(X_val_inner, y_val_inner),
                epochs=epochs_per_split,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )

            # Evaluate on test
            predictions = model.predict(X_test_split, verbose=0)

            # Inverse transform
            pred_unscaled = split_builder.inverse_transform_predictions(predictions)
            actual_unscaled = split_builder.inverse_transform_predictions(y_test_split)

            # Calculate metrics for first prediction step
            rmse = np.sqrt(np.mean((pred_unscaled[:, 0] - actual_unscaled[:, 0])**2))
            mae = np.mean(np.abs(pred_unscaled[:, 0] - actual_unscaled[:, 0]))
            mape = np.mean(np.abs((actual_unscaled[:, 0] - pred_unscaled[:, 0]) /
                                  (actual_unscaled[:, 0] + 1e-8))) * 100

            all_metrics['rmse'].append(rmse)
            all_metrics['mae'].append(mae)
            all_metrics['mape'].append(mape)

            all_predictions.extend(pred_unscaled[:, 0].tolist())
            all_actuals.extend(actual_unscaled[:, 0].tolist())

            logger.info(f"Split {i+1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        # Summary
        logger.info("\n=== Walk-Forward Validation Summary ===")
        for metric, values in all_metrics.items():
            logger.info(f"{metric.upper()}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")

        self.metrics['walk_forward'] = all_metrics

        return all_metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("Evaluating model on test set...")

        # Get predictions
        predictions = self.model.predict(self.X_test, verbose=0)

        # Inverse transform
        pred_unscaled = self.sequence_builder.inverse_transform_predictions(predictions)
        actual_unscaled = self.sequence_builder.inverse_transform_predictions(self.y_test)

        # Calculate metrics for each forecast horizon
        metrics = {}
        for h in range(self.forecast_horizon):
            pred_h = pred_unscaled[:, h]
            actual_h = actual_unscaled[:, h]

            rmse = np.sqrt(np.mean((pred_h - actual_h)**2))
            mae = np.mean(np.abs(pred_h - actual_h))
            mape = np.mean(np.abs((actual_h - pred_h) / (actual_h + 1e-8))) * 100

            metrics[f'day_{h+1}'] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape)
            }

        # Overall metrics (average across horizons)
        all_rmse = np.sqrt(np.mean((pred_unscaled - actual_unscaled)**2))
        all_mae = np.mean(np.abs(pred_unscaled - actual_unscaled))
        all_mape = np.mean(np.abs((actual_unscaled - pred_unscaled) /
                                  (actual_unscaled + 1e-8))) * 100

        metrics['overall'] = {
            'rmse': float(all_rmse),
            'mae': float(all_mae),
            'mape': float(all_mape)
        }

        self.metrics['test'] = metrics

        logger.info(f"Overall Test RMSE: {all_rmse:.4f}")
        logger.info(f"Overall Test MAE: {all_mae:.4f}")
        logger.info(f"Overall Test MAPE: {all_mape:.2f}%")

        # Save metrics
        self._save_metrics()

        return metrics

    def predict(
        self,
        n_days: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on test data.

        Args:
            n_days: Number of samples to predict (None for all).

        Returns:
            Tuple of (predictions, actuals, attention_weights).
        """
        if n_days is not None:
            X = self.X_test[:n_days]
            y = self.y_test[:n_days]
        else:
            X = self.X_test
            y = self.y_test

        predictions, attention = self.attention_model.predict(X, verbose=0)

        # Inverse transform
        pred_unscaled = self.sequence_builder.inverse_transform_predictions(predictions)
        actual_unscaled = self.sequence_builder.inverse_transform_predictions(y)

        return pred_unscaled, actual_unscaled, attention

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model.

        Args:
            path: Custom path to save the model.

        Returns:
            Path where the model was saved.
        """
        if path is None:
            path = os.path.join(
                self.model_dir,
                f'{self.ticker}_lstm_attention_final.keras'
            )

        self.model.save(path)
        logger.info(f"Model saved to {path}")

        # Also save the sequence builder parameters
        builder_params = {
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'feature_columns': self.feature_columns,
            'target_column': 'Close'
        }
        params_path = path.replace('.keras', '_params.json')
        with open(params_path, 'w') as f:
            json.dump(builder_params, f, indent=2)

        return path

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.model = keras.models.load_model(
            path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"Model loaded from {path}")

    def _save_training_history(self) -> None:
        """Save training history to file."""
        if self.history is None:
            return

        history_dict = {k: [float(v) for v in vals]
                       for k, vals in self.history.history.items()}

        path = os.path.join(
            self.results_dir, 'metrics',
            f'{self.ticker}_training_history.json'
        )
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Training history saved to {path}")

    def _save_metrics(self) -> None:
        """Save evaluation metrics to file."""
        path = os.path.join(
            self.results_dir, 'metrics',
            f'{self.ticker}_metrics.json'
        )
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Metrics saved to {path}")


def train_single_stock(
    ticker: str = 'AAPL',
    start_date: str = '2019-01-01',
    end_date: str = '2024-01-01',
    sequence_length: int = 60,
    forecast_horizon: int = 7,
    epochs: int = 100,
    batch_size: int = 32,
    use_walk_forward: bool = False
) -> StockPriceTrainer:
    """
    Train a model for a single stock.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for data.
        end_date: End date for data.
        sequence_length: LSTM sequence length.
        forecast_horizon: Days to predict.
        epochs: Training epochs.
        batch_size: Batch size.
        use_walk_forward: Whether to use walk-forward validation.

    Returns:
        Trained StockPriceTrainer instance.
    """
    logger.info(f"=== Training model for {ticker} ===")

    trainer = StockPriceTrainer(
        ticker=ticker,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )

    # Prepare data
    trainer.prepare_data(start_date, end_date)

    # Build and train model
    trainer.build_model()

    if use_walk_forward:
        trainer.train_walk_forward(n_splits=5, epochs_per_split=50)
    else:
        trainer.train(epochs=epochs, batch_size=batch_size)

    # Evaluate
    trainer.evaluate()

    # Save model
    trainer.save_model()

    return trainer


if __name__ == "__main__":
    # Example: Train model for AAPL
    trainer = train_single_stock(
        ticker='AAPL',
        start_date='2019-01-01',
        end_date='2024-01-01',
        sequence_length=60,
        forecast_horizon=7,
        epochs=100,
        use_walk_forward=False
    )

    print("\n=== Training Complete ===")
    print(f"Test Metrics: {trainer.metrics.get('test', {}).get('overall', {})}")
