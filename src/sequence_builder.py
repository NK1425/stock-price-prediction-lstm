"""
Sequence Builder Module for Time-Series Forecasting.

This module provides utilities for creating sliding window sequences
suitable for LSTM and other recurrent neural network architectures.
It supports multi-step forecasting and proper train/test splitting.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    A class for building sequences from time-series data.

    This class creates sliding window sequences for LSTM models, handling
    proper scaling, multi-step forecasting, and train/val/test splitting
    with no data leakage.

    Attributes:
        sequence_length (int): Number of time steps in each input sequence.
        forecast_horizon (int): Number of time steps to forecast.
        feature_columns (List[str]): Columns to use as features.
        target_column (str): Column to use as target.
        feature_scaler: Fitted scaler for features.
        target_scaler: Fitted scaler for target.

    Example:
        >>> builder = SequenceBuilder(sequence_length=60, forecast_horizon=7)
        >>> X_train, y_train = builder.create_sequences(train_data)
        >>> X_test, y_test = builder.transform_sequences(test_data)
    """

    def __init__(
        self,
        sequence_length: int = 60,
        forecast_horizon: int = 7,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'Close',
        scaler_type: str = 'minmax'
    ):
        """
        Initialize the SequenceBuilder.

        Args:
            sequence_length: Number of time steps to look back.
            forecast_horizon: Number of time steps to predict.
            feature_columns: List of columns to use as features.
            target_column: Column to predict.
            scaler_type: Type of scaler ('minmax' or 'standard').
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler_type = scaler_type

        self.feature_scaler = None
        self.target_scaler = None
        self._fitted = False

    def fit(self, data: pd.DataFrame) -> 'SequenceBuilder':
        """
        Fit scalers on the training data.

        This method should be called only on training data to prevent
        data leakage.

        Args:
            data: Training data DataFrame.

        Returns:
            Self for method chaining.
        """
        # Determine feature columns if not specified
        if self.feature_columns is None:
            # Exclude non-numeric and target columns
            self.feature_columns = [
                col for col in data.columns
                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']
                and col != self.target_column
                and col != 'Target'
            ]

        # Initialize scalers
        if self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

        # Fit scalers
        feature_data = data[self.feature_columns].values
        target_data = data[[self.target_column]].values

        self.feature_scaler.fit(feature_data)
        self.target_scaler.fit(target_data)

        self._fitted = True
        logger.info(f"Fitted scalers on {len(self.feature_columns)} features")

        return self

    def create_sequences(
        self,
        data: pd.DataFrame,
        fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from the data.

        Args:
            data: DataFrame with features and target.
            fit: Whether to fit scalers on this data.

        Returns:
            Tuple of (X sequences, y targets) as numpy arrays.
            X shape: (samples, sequence_length, n_features)
            y shape: (samples, forecast_horizon)
        """
        if fit:
            self.fit(data)

        if not self._fitted:
            raise ValueError("Scalers not fitted. Call fit() or set fit=True")

        # Scale features
        feature_data = data[self.feature_columns].values
        target_data = data[[self.target_column]].values

        scaled_features = self.feature_scaler.transform(feature_data)
        scaled_target = self.target_scaler.transform(target_data).flatten()

        # Create sequences
        X, y = [], []
        total_length = self.sequence_length + self.forecast_horizon

        for i in range(len(scaled_features) - total_length + 1):
            # Input sequence
            X.append(scaled_features[i:i + self.sequence_length])

            # Target sequence (next forecast_horizon values)
            y.append(scaled_target[i + self.sequence_length:i + total_length])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences with shape X={X.shape}, y={y.shape}")

        return X, y

    def transform_sequences(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data into sequences using fitted scalers.

        Args:
            data: DataFrame to transform.

        Returns:
            Tuple of (X sequences, y targets).
        """
        return self.create_sequences(data, fit=False)

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform scaled predictions to original scale.

        Args:
            predictions: Scaled predictions of shape (samples, forecast_horizon).

        Returns:
            Predictions in original scale.
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted")

        # Reshape for inverse transform
        original_shape = predictions.shape
        predictions_flat = predictions.reshape(-1, 1)
        unscaled = self.target_scaler.inverse_transform(predictions_flat)

        return unscaled.reshape(original_shape)

    def inverse_transform_features(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform scaled features to original scale.

        Args:
            features: Scaled features.

        Returns:
            Features in original scale.
        """
        if self.feature_scaler is None:
            raise ValueError("Feature scaler not fitted")

        original_shape = features.shape
        features_2d = features.reshape(-1, len(self.feature_columns))
        unscaled = self.feature_scaler.inverse_transform(features_2d)

        return unscaled.reshape(original_shape)

    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        return self.feature_columns.copy() if self.feature_columns else []


class WalkForwardValidator:
    """
    Walk-forward validation for time-series data.

    This class implements walk-forward (rolling window) validation to prevent
    data leakage and properly evaluate time-series models.

    Attributes:
        n_splits (int): Number of validation splits.
        train_size (int): Minimum training set size.
        test_size (int): Size of each test set.
        gap (int): Gap between train and test to prevent leakage.

    Example:
        >>> validator = WalkForwardValidator(n_splits=5, test_size=30)
        >>> for train_idx, test_idx in validator.split(data):
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: int = 30,
        gap: int = 0,
        expanding: bool = True
    ):
        """
        Initialize the WalkForwardValidator.

        Args:
            n_splits: Number of train/test splits.
            train_size: Minimum training size. If None, uses expanding window.
            test_size: Size of each test set.
            gap: Number of samples to skip between train and test.
            expanding: If True, training set expands. If False, uses sliding window.
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding

    def split(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.

        Args:
            data: Data to split.

        Yields:
            Tuple of (train_indices, test_indices).
        """
        n_samples = len(data)

        # Calculate split points
        if self.train_size is None:
            # Expanding window: reserve space for n_splits test sets
            min_train = n_samples - (self.n_splits * (self.test_size + self.gap))
            if min_train < self.test_size:
                raise ValueError("Not enough data for specified splits")
            self.train_size = min_train

        splits = []
        for i in range(self.n_splits):
            if self.expanding:
                train_start = 0
                train_end = self.train_size + i * self.test_size
            else:
                train_start = i * self.test_size
                train_end = train_start + self.train_size

            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        return splits

    def get_split_info(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> List[dict]:
        """
        Get detailed information about each split.

        Args:
            data: Data being split.

        Returns:
            List of dictionaries with split information.
        """
        splits = self.split(data)
        info = []

        for i, (train_idx, test_idx) in enumerate(splits):
            split_info = {
                'split': i + 1,
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'train_size': len(train_idx),
                'test_start': test_idx[0],
                'test_end': test_idx[-1],
                'test_size': len(test_idx)
            }

            if isinstance(data, pd.DataFrame):
                split_info['train_start_date'] = data.index[train_idx[0]]
                split_info['train_end_date'] = data.index[train_idx[-1]]
                split_info['test_start_date'] = data.index[test_idx[0]]
                split_info['test_end_date'] = data.index[test_idx[-1]]

            info.append(split_info)

        return info


def create_train_val_test_sequences(
    data: pd.DataFrame,
    sequence_builder: SequenceBuilder,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray],
           List[pd.Timestamp]]:
    """
    Create train, validation, and test sequences with proper time-based splitting.

    Args:
        data: DataFrame with features.
        sequence_builder: SequenceBuilder instance.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test), test_dates).
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split data chronologically
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()

    logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Fit on training data and create sequences
    X_train, y_train = sequence_builder.create_sequences(train_data, fit=True)
    X_val, y_val = sequence_builder.transform_sequences(val_data)
    X_test, y_test = sequence_builder.transform_sequences(test_data)

    # Get test dates for plotting
    seq_length = sequence_builder.sequence_length
    forecast_horizon = sequence_builder.forecast_horizon
    test_dates = test_data.index[seq_length:len(test_data) - forecast_horizon + 1].tolist()

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), test_dates


def create_single_prediction_sequence(
    data: pd.DataFrame,
    sequence_builder: SequenceBuilder
) -> np.ndarray:
    """
    Create a single sequence for making a prediction.

    Takes the last sequence_length points from the data to create
    an input sequence for predicting the next forecast_horizon values.

    Args:
        data: DataFrame with at least sequence_length rows.
        sequence_builder: Fitted SequenceBuilder instance.

    Returns:
        Numpy array of shape (1, sequence_length, n_features).
    """
    if len(data) < sequence_builder.sequence_length:
        raise ValueError(
            f"Data must have at least {sequence_builder.sequence_length} rows"
        )

    # Get the last sequence_length rows
    recent_data = data.iloc[-sequence_builder.sequence_length:]

    # Scale features
    feature_data = recent_data[sequence_builder.feature_columns].values
    scaled_features = sequence_builder.feature_scaler.transform(feature_data)

    # Reshape to (1, sequence_length, n_features)
    return scaled_features.reshape(1, sequence_builder.sequence_length, -1)


if __name__ == "__main__":
    # Example usage
    from data_loader import StockDataLoader
    from feature_engineering import FeatureEngineer

    # Load and prepare data
    loader = StockDataLoader(cache_dir='./data')
    data = loader.download_stock_data('AAPL', '2020-01-01', '2024-01-01')

    engineer = FeatureEngineer()
    data_with_features = engineer.add_all_features(data)
    data_clean = engineer.handle_missing_values(data_with_features, method='drop')

    # Define feature columns
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI_14', 'MACD', 'MACD_Signal',
        'BB_Upper', 'BB_Lower', 'BB_Width',
        'ATR', 'SMA_20', 'EMA_20',
        'Momentum_10', 'OBV', 'Returns'
    ]
    feature_cols = [c for c in feature_cols if c in data_clean.columns]

    # Create sequence builder
    builder = SequenceBuilder(
        sequence_length=60,
        forecast_horizon=7,
        feature_columns=feature_cols,
        target_column='Close'
    )

    # Create sequences with splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test), test_dates = \
        create_train_val_test_sequences(data_clean, builder)

    print(f"\nSequence shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Test inverse transform
    sample_pred = y_test[:5]
    unscaled = builder.inverse_transform_predictions(sample_pred)
    print(f"\nSample unscaled predictions: {unscaled[0]}")

    # Walk-forward validation example
    print("\n--- Walk-Forward Validation ---")
    validator = WalkForwardValidator(n_splits=5, test_size=60)
    split_info = validator.get_split_info(data_clean)
    for info in split_info:
        print(f"Split {info['split']}: Train {info['train_size']} samples, "
              f"Test {info['test_size']} samples")
