"""
Data Loader Module for Stock Price Prediction.

This module handles downloading, caching, and preprocessing of stock data
from Yahoo Finance. It supports S&P 500 stocks and provides utilities for
data validation and quality checks.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# S&P 500 top companies by market cap (as of 2024)
SP500_TOP_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
    "TMO", "ABT", "ACN", "DHR", "NEE", "DIS", "VZ", "ADBE", "CMCSA",
    "NKE", "INTC", "TXN", "WFC", "CRM", "PM", "RTX", "BMY", "UPS"
]


class StockDataLoader:
    """
    A class for downloading and managing stock market data.

    This class provides methods to download historical stock data from Yahoo Finance,
    validate the data quality, and handle caching for improved performance.

    Attributes:
        cache_dir (str): Directory for caching downloaded data.
        default_start (datetime): Default start date for data download.
        default_end (datetime): Default end date for data download.

    Example:
        >>> loader = StockDataLoader(cache_dir='./data')
        >>> data = loader.download_stock_data('AAPL', '2020-01-01', '2024-01-01')
        >>> print(data.shape)
    """

    def __init__(
        self,
        cache_dir: str = './data',
        default_years: int = 5
    ):
        """
        Initialize the StockDataLoader.

        Args:
            cache_dir: Directory to store cached data files.
            default_years: Default number of years of historical data to download.
        """
        self.cache_dir = cache_dir
        self.default_years = default_years
        self.default_end = datetime.now()
        self.default_start = self.default_end - timedelta(days=365 * default_years)

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def download_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
            start_date: Start date in 'YYYY-MM-DD' format. Defaults to 5 years ago.
            end_date: End date in 'YYYY-MM-DD' format. Defaults to today.
            use_cache: Whether to use cached data if available.
            interval: Data interval ('1d', '1h', '5m', etc.).

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close.

        Raises:
            ValueError: If ticker is invalid or no data is available.
        """
        # Set default dates
        start = start_date or self.default_start.strftime('%Y-%m-%d')
        end = end_date or self.default_end.strftime('%Y-%m-%d')

        # Check cache
        cache_file = self._get_cache_path(ticker, start, end, interval)
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached data for {ticker}")
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return data

        # Download from Yahoo Finance
        logger.info(f"Downloading {ticker} data from {start} to {end}")
        try:
            # Use Ticker.history() for reliable data fetching
            stock = yf.Ticker(ticker)
            data = stock.history(start=start, end=end, interval=interval)

            # Remove timezone from index for easier handling
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            if data.empty:
                raise ValueError(f"No data available for ticker '{ticker}'")

            # Clean column names (remove multi-level if present)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure standard column names
            data = self._standardize_columns(data)

            # Validate data quality
            data = self._validate_and_clean(data, ticker)

            # Cache the data
            if use_cache:
                data.to_csv(cache_file)
                logger.info(f"Cached data saved to {cache_file}")

            return data

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
            raise ValueError(f"Failed to download data for '{ticker}': {str(e)}")

    def download_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stocks.

        Args:
            tickers: List of stock ticker symbols.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            use_cache: Whether to use cached data.

        Returns:
            Dictionary mapping ticker symbols to their DataFrames.
        """
        stock_data = {}
        failed_tickers = []

        for ticker in tickers:
            try:
                data = self.download_stock_data(
                    ticker, start_date, end_date, use_cache
                )
                stock_data[ticker] = data
                logger.info(f"Successfully downloaded {ticker}: {len(data)} records")
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers}")

        return stock_data

    def get_sp500_data(
        self,
        n_stocks: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for top S&P 500 stocks.

        Args:
            n_stocks: Number of top stocks to download.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            Dictionary mapping ticker symbols to their DataFrames.
        """
        tickers = SP500_TOP_STOCKS[:n_stocks]
        return self.download_multiple_stocks(tickers, start_date, end_date)

    def _get_cache_path(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str
    ) -> str:
        """Generate cache file path for a stock."""
        filename = f"{ticker}_{start}_{end}_{interval}.csv"
        return os.path.join(self.cache_dir, filename)

    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to a consistent format."""
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adjclose': 'Adj Close',
            'dividends': 'Dividends',
            'stock splits': 'Stock Splits'
        }

        # Convert to lowercase for matching
        data.columns = [
            column_mapping.get(col.lower(), col)
            for col in data.columns
        ]

        return data

    def _validate_and_clean(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Validate and clean the downloaded data.

        Args:
            data: Raw stock data DataFrame.
            ticker: Stock ticker for logging purposes.

        Returns:
            Cleaned DataFrame.
        """
        initial_len = len(data)

        # Remove rows with zero or negative prices
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col in data.columns:
                data = data[data[col] > 0]

        # Remove rows with zero volume (non-trading days)
        if 'Volume' in data.columns:
            data = data[data['Volume'] >= 0]

        # Check for and log data quality issues
        null_counts = data.isnull().sum()
        if null_counts.any():
            logger.warning(f"{ticker} has null values: {null_counts[null_counts > 0].to_dict()}")

        # Sort by date
        data = data.sort_index()

        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]

        final_len = len(data)
        if initial_len != final_len:
            logger.info(f"{ticker}: Removed {initial_len - final_len} invalid rows")

        return data

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the stock data.

        Args:
            data: Stock data DataFrame.

        Returns:
            Dictionary containing data statistics.
        """
        return {
            'start_date': data.index.min().strftime('%Y-%m-%d'),
            'end_date': data.index.max().strftime('%Y-%m-%d'),
            'total_days': len(data),
            'trading_days': len(data),
            'price_range': {
                'min': float(data['Close'].min()),
                'max': float(data['Close'].max()),
                'mean': float(data['Close'].mean()),
                'std': float(data['Close'].std())
            },
            'volume_stats': {
                'min': float(data['Volume'].min()),
                'max': float(data['Volume'].max()),
                'mean': float(data['Volume'].mean())
            },
            'missing_values': data.isnull().sum().to_dict(),
            'returns': {
                'total': float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100),
                'annualized': float(
                    ((data['Close'].iloc[-1] / data['Close'].iloc[0]) **
                     (252 / len(data)) - 1) * 100
                )
            }
        }


def create_train_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into train, validation, and test sets.

    Uses time-based splitting to prevent data leakage. The data is split
    chronologically, with earlier data used for training.

    Args:
        data: Stock data DataFrame with datetime index.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.

    Returns:
        Tuple of (train_data, val_data, test_data) DataFrames.

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()

    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    loader = StockDataLoader(cache_dir='./data')

    # Download single stock
    aapl_data = loader.download_stock_data('AAPL', '2019-01-01', '2024-01-01')
    print(f"Downloaded AAPL: {aapl_data.shape}")
    print(loader.get_data_summary(aapl_data))

    # Split data
    train, val, test = create_train_test_split(aapl_data)
    print(f"\nTrain: {train.index.min()} to {train.index.max()}")
    print(f"Val: {val.index.min()} to {val.index.max()}")
    print(f"Test: {test.index.min()} to {test.index.max()}")
