"""
Feature Engineering Module for Stock Price Prediction.

This module generates technical indicators and features for stock price
prediction. It implements 15+ technical indicators including RSI, MACD,
Bollinger Bands, and various momentum and volatility indicators.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class for generating technical indicators and features.

    This class provides methods to calculate various technical indicators
    used in stock price prediction, including trend, momentum, volatility,
    and volume-based indicators.

    Attributes:
        feature_columns (List[str]): List of generated feature column names.
        scalers (dict): Dictionary of fitted scalers for each feature group.

    Example:
        >>> engineer = FeatureEngineer()
        >>> data_with_features = engineer.add_all_features(stock_data)
        >>> print(data_with_features.columns.tolist())
    """

    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.feature_columns: List[str] = []
        self.scalers = {}
        self.feature_stats = {}

    def add_all_features(
        self,
        data: pd.DataFrame,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Add all technical indicators to the dataset.

        Args:
            data: Stock data DataFrame with OHLCV columns.
            include_target: Whether to add the target variable (next day close).

        Returns:
            DataFrame with all features added.
        """
        df = data.copy()

        # Store original columns
        original_cols = df.columns.tolist()

        # Add all feature groups
        df = self._add_price_features(df)
        df = self._add_returns(df)
        df = self._add_moving_averages(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_atr(df)
        df = self._add_volatility(df)
        df = self._add_momentum(df)
        df = self._add_obv(df)
        df = self._add_vwap(df)
        df = self._add_stochastic(df)
        df = self._add_williams_r(df)
        df = self._add_cci(df)
        df = self._add_roc(df)
        df = self._add_mfi(df)

        # Track feature columns (excluding original and target)
        self.feature_columns = [
            col for col in df.columns
            if col not in original_cols and col != 'Target'
        ]

        logger.info(f"Generated {len(self.feature_columns)} features")

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Price range
        df['Price_Range'] = df['High'] - df['Low']

        # Price position within day's range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)

        # Gap (overnight change)
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = df['Gap'] / df['Close'].shift(1)

        # Body size (for candlestick analysis)
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Body_Direction'] = np.sign(df['Close'] - df['Open'])

        # Upper and lower shadows
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        # Simple returns
        df['Returns'] = df['Close'].pct_change()

        # Log returns
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Multi-period returns
        for period in [5, 10, 20]:
            df[f'Returns_{period}d'] = df['Close'].pct_change(periods=period)

        # Cumulative returns
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # Price relative to moving averages
        df['Price_SMA_20_Ratio'] = df['Close'] / df['SMA_20']
        df['Price_SMA_50_Ratio'] = df['Close'] / df['SMA_50']
        df['Price_EMA_20_Ratio'] = df['Close'] / df['EMA_20']

        # Moving average crossovers
        df['SMA_5_20_Cross'] = (df['SMA_5'] > df['SMA_20']).astype(int)
        df['EMA_5_20_Cross'] = (df['EMA_5'] > df['EMA_20']).astype(int)

        # Golden/Death cross indicator
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)

        return df

    def _add_rsi(
        self,
        df: pd.DataFrame,
        periods: List[int] = [14, 7, 21]
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions.
        """
        for period in periods:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            # Use EMA for smoother RSI
            avg_gain_ema = gain.ewm(span=period, adjust=False).mean()
            avg_loss_ema = loss.ewm(span=period, adjust=False).mean()

            rs = avg_gain_ema / (avg_loss_ema + 1e-8)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        return df

    def _add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD).

        MACD is a trend-following momentum indicator that shows the
        relationship between two moving averages.
        """
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # MACD crossover signals
        df['MACD_Cross'] = (df['MACD'] > df['MACD_Signal']).astype(int)

        return df

    def _add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) and upper/lower bands
        based on standard deviations.
        """
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()

        df['BB_Middle'] = sma
        df['BB_Upper'] = sma + (std_dev * std)
        df['BB_Lower'] = sma - (std_dev * std)

        # Bandwidth and %B
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)

        return df

    def _add_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range
        of an asset price for a period.
        """
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()

        # Normalized ATR
        df['ATR_Percent'] = df['ATR'] / df['Close'] * 100

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Historical volatility (annualized)
        for period in [10, 20, 30]:
            df[f'Volatility_{period}d'] = df['Returns'].rolling(
                window=period
            ).std() * np.sqrt(252)

        # Parkinson volatility (uses high-low range)
        df['Parkinson_Vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            (np.log(df['High'] / df['Low']) ** 2).rolling(window=20).mean()
        ) * np.sqrt(252)

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # Simple momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'Momentum_Pct_{period}'] = df['Close'] / df['Close'].shift(period) - 1

        # Acceleration (momentum of momentum)
        df['Acceleration'] = df['Momentum_10'] - df['Momentum_10'].shift(5)

        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV).

        OBV uses volume flow to predict changes in stock price.
        """
        obv = np.where(
            df['Close'] > df['Close'].shift(1),
            df['Volume'],
            np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)
        )
        df['OBV'] = pd.Series(obv, index=df.index).cumsum()

        # OBV moving average and divergence
        df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
        df['OBV_Trend'] = df['OBV'] - df['OBV_SMA']

        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Volume Weighted Average Price (VWAP).

        VWAP gives the average price weighted by volume.
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

        # Rolling VWAP
        for period in [10, 20]:
            cumulative_tp_vol = (typical_price * df['Volume']).rolling(window=period).sum()
            cumulative_vol = df['Volume'].rolling(window=period).sum()
            df[f'VWAP_{period}'] = cumulative_tp_vol / (cumulative_vol + 1e-8)

        # Price relative to VWAP
        df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP']

        return df

    def _add_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.

        Compares closing price to price range over a period.
        """
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()

        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-8)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()

        # Stochastic crossover
        df['Stoch_Cross'] = (df['Stoch_K'] > df['Stoch_D']).astype(int)

        return df

    def _add_williams_r(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Williams %R.

        Williams %R is a momentum indicator measuring overbought/oversold levels.
        """
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()

        df['Williams_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low + 1e-8)

        return df

    def _add_cci(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.DataFrame:
        """
        Add Commodity Channel Index (CCI).

        CCI measures the difference between current price and historical average.
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        df['CCI'] = (typical_price - sma) / (0.015 * mean_deviation + 1e-8)

        return df

    def _add_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Rate of Change (ROC).

        ROC measures the percentage change between current price and past price.
        """
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = (
                (df['Close'] - df['Close'].shift(period)) /
                df['Close'].shift(period) * 100
            )

        return df

    def _add_mfi(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Money Flow Index (MFI).

        MFI is a volume-weighted RSI that measures buying and selling pressure.
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        money_ratio = positive_mf / (negative_mf + 1e-8)
        df['MFI'] = 100 - (100 / (1 + money_ratio))

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'drop',
        fill_value: float = 0.0
    ) -> pd.DataFrame:
        """
        Handle missing values in the feature set.

        Args:
            df: DataFrame with features.
            method: Method to handle missing values ('drop', 'ffill', 'bfill', 'fill').
            fill_value: Value to use when method is 'fill'.

        Returns:
            DataFrame with missing values handled.
        """
        initial_len = len(df)

        if method == 'drop':
            df = df.dropna()
        elif method == 'ffill':
            df = df.ffill().bfill()  # Forward fill, then backward fill remaining
        elif method == 'bfill':
            df = df.bfill().ffill()
        elif method == 'fill':
            df = df.fillna(fill_value)
        else:
            raise ValueError(f"Unknown method: {method}")

        final_len = len(df)
        if initial_len != final_len:
            logger.info(f"Removed {initial_len - final_len} rows with missing values")

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        scaler_type: str = 'minmax',
        fit: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Scale features using specified scaler.

        Args:
            df: DataFrame with features to scale.
            feature_columns: Columns to scale. If None, uses self.feature_columns.
            scaler_type: Type of scaler ('minmax', 'standard', 'robust').
            fit: Whether to fit the scaler or use existing.

        Returns:
            Tuple of (scaled DataFrame, scaler dictionary).
        """
        cols_to_scale = feature_columns or self.feature_columns
        cols_to_scale = [c for c in cols_to_scale if c in df.columns]

        if not cols_to_scale:
            logger.warning("No columns to scale")
            return df, {}

        df_scaled = df.copy()

        if scaler_type == 'minmax':
            scaler_class = MinMaxScaler
        elif scaler_type == 'standard':
            scaler_class = StandardScaler
        elif scaler_type == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        if fit:
            self.scalers = {}
            for col in cols_to_scale:
                scaler = scaler_class()
                df_scaled[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
        else:
            for col in cols_to_scale:
                if col in self.scalers:
                    df_scaled[col] = self.scalers[col].transform(df[[col]])
                else:
                    logger.warning(f"No scaler found for {col}, skipping")

        return df_scaled, self.scalers

    def get_feature_importance_columns(self) -> List[str]:
        """
        Get the most important feature columns for modeling.

        Returns:
            List of important feature column names.
        """
        important_features = [
            # Price-based
            'Close', 'Returns', 'Log_Returns',

            # Trend indicators
            'SMA_20', 'SMA_50', 'EMA_20',
            'Price_SMA_20_Ratio', 'Price_SMA_50_Ratio',

            # Momentum
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'Momentum_10', 'ROC_10',

            # Volatility
            'BB_Width', 'BB_Percent', 'ATR', 'Volatility_20d',

            # Volume
            'OBV', 'MFI', 'VWAP',

            # Oscillators
            'Stoch_K', 'Williams_R', 'CCI'
        ]

        return [f for f in important_features if f in self.feature_columns or f in ['Close', 'Returns', 'Log_Returns']]


def create_target_variable(
    df: pd.DataFrame,
    target_col: str = 'Close',
    horizon: int = 1,
    target_type: str = 'price'
) -> pd.DataFrame:
    """
    Create target variable for prediction.

    Args:
        df: DataFrame with stock data.
        target_col: Column to use as target.
        horizon: Number of days ahead to predict.
        target_type: Type of target ('price', 'return', 'direction').

    Returns:
        DataFrame with target variable added.
    """
    df = df.copy()

    if target_type == 'price':
        df['Target'] = df[target_col].shift(-horizon)
    elif target_type == 'return':
        df['Target'] = df[target_col].pct_change(periods=horizon).shift(-horizon)
    elif target_type == 'direction':
        future_return = df[target_col].pct_change(periods=horizon).shift(-horizon)
        df['Target'] = (future_return > 0).astype(int)
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    return df


if __name__ == "__main__":
    # Example usage
    from data_loader import StockDataLoader

    # Load data
    loader = StockDataLoader(cache_dir='./data')
    data = loader.download_stock_data('AAPL', '2020-01-01', '2024-01-01')

    # Generate features
    engineer = FeatureEngineer()
    data_with_features = engineer.add_all_features(data)

    print(f"\nOriginal columns: {len(data.columns)}")
    print(f"Total columns after features: {len(data_with_features.columns)}")
    print(f"Generated features: {len(engineer.feature_columns)}")
    print(f"\nFeature list:\n{engineer.feature_columns}")

    # Handle missing values and scale
    data_clean = engineer.handle_missing_values(data_with_features, method='drop')
    print(f"\nClean data shape: {data_clean.shape}")

    # Add target
    data_final = create_target_variable(data_clean, horizon=1)
    data_final = data_final.dropna()
    print(f"Final data shape: {data_final.shape}")
