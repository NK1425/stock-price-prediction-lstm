"""
Advanced Feature Engineering: Sentiment Analysis and Market Regime Detection.

This module adds sophisticated features that differentiate this project
from basic tutorials:
1. Sentiment analysis from Yahoo Finance news headlines
2. Market regime detection using Hidden Markov Models / clustering
3. Correlation features with market indices and sector ETFs
4. Volatility regime classification
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    logger.warning("TextBlob not installed. Install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    logger.warning("VADER not installed. Install with: pip install vaderSentiment")

try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    logger.warning("hmmlearn not installed. Falling back to GMM for regime detection.")


class SentimentAnalyzer:
    """
    Analyze sentiment from financial news headlines.
    
    Uses VADER (Valence Aware Dictionary for Sentiment Reasoning)
    which is specifically tuned for social media/news text.
    Falls back to TextBlob if VADER is not available.
    """
    
    def __init__(self, use_vader: bool = True):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_vader: Whether to use VADER (recommended for financial text)
        """
        self.use_vader = use_vader and HAS_VADER
        
        if self.use_vader:
            self.analyzer = SentimentIntensityAnalyzer()
            logger.info("Using VADER sentiment analyzer")
        elif HAS_TEXTBLOB:
            self.analyzer = None  # TextBlob doesn't need initialization
            logger.info("Using TextBlob sentiment analyzer (VADER not available)")
        else:
            self.analyzer = None
            logger.warning("No sentiment analyzer available")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores:
            - compound: Overall sentiment (-1 to 1)
            - positive: Positive sentiment (0 to 1)
            - negative: Negative sentiment (0 to 1)
            - neutral: Neutral sentiment (0 to 1)
        """
        if not text or pd.isna(text):
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        if self.use_vader and self.analyzer:
            scores = self.analyzer.polarity_scores(str(text))
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        elif HAS_TEXTBLOB:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            return {
                'compound': polarity,
                'positive': max(0, polarity),
                'negative': abs(min(0, polarity)),
                'neutral': 1 - subjectivity
            }
        else:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def analyze_headlines(self, headlines: List[str]) -> Dict[str, float]:
        """
        Analyze multiple headlines and aggregate sentiment.
        
        Args:
            headlines: List of news headlines
        
        Returns:
            Aggregated sentiment scores
        """
        if not headlines:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        sentiments = [self.analyze_text(h) for h in headlines]
        
        return {
            'compound': np.mean([s['compound'] for s in sentiments]),
            'positive': np.mean([s['positive'] for s in sentiments]),
            'negative': np.mean([s['negative'] for s in sentiments]),
            'neutral': np.mean([s['neutral'] for s in sentiments])
        }


class YahooNewsSentiment:
    """
    Fetch and analyze sentiment from Yahoo Finance news.
    
    Uses yfinance's built-in news fetching capability.
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self._cache = {}
    
    def get_news_sentiment(
        self,
        ticker: str,
        lookback_days: int = 7
    ) -> Dict[str, float]:
        """
        Get aggregated sentiment from recent news for a stock.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back for news
        
        Returns:
            Sentiment scores dictionary
        """
        import yfinance as yf
        
        cache_key = f"{ticker}_{lookback_days}_{datetime.now().date()}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                logger.debug(f"No news found for {ticker}")
                return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            
            # Filter by date and extract headlines
            cutoff = datetime.now() - timedelta(days=lookback_days)
            headlines = []
            
            for item in news:
                # Yahoo Finance news items have different structures
                title = item.get('title', '')
                if title:
                    headlines.append(title)
            
            sentiment = self.sentiment_analyzer.analyze_headlines(headlines[:20])  # Limit to 20
            self._cache[cache_key] = sentiment
            
            logger.debug(f"Analyzed {len(headlines)} headlines for {ticker}")
            return sentiment
            
        except Exception as e:
            logger.warning(f"Error fetching news for {ticker}: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Add sentiment features to a DataFrame.
        
        Note: This adds current sentiment to all rows as a static feature
        since we can't get historical daily sentiment without an API.
        For production, you'd want a proper news API with historical data.
        
        Args:
            df: DataFrame with stock data
            ticker: Stock ticker
        
        Returns:
            DataFrame with sentiment features added
        """
        sentiment = self.get_news_sentiment(ticker)
        
        df = df.copy()
        df['Sentiment_Compound'] = sentiment['compound']
        df['Sentiment_Positive'] = sentiment['positive']
        df['Sentiment_Negative'] = sentiment['negative']
        
        # Create sentiment momentum (placeholder - would need historical sentiment)
        # In production, you'd track sentiment over time
        df['Sentiment_Bullish'] = (sentiment['compound'] > 0.1).astype(int)
        df['Sentiment_Bearish'] = (sentiment['compound'] < -0.1).astype(int)
        
        return df


class MarketRegimeDetector:
    """
    Detect market regimes (bull/bear/sideways) using statistical methods.
    
    Uses a combination of:
    1. Hidden Markov Models (if available) or Gaussian Mixture Models
    2. Volatility-based regime classification
    3. Trend-based regime classification
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        use_hmm: bool = True
    ):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            use_hmm: Whether to use HMM (True) or GMM (False)
        """
        self.n_regimes = n_regimes
        self.use_hmm = use_hmm and HAS_HMM
        self.model = None
        self.scaler = StandardScaler()
        self._fitted = False
        
        # Regime labels (will be assigned after fitting)
        self.regime_labels = {
            0: 'unknown',
            1: 'unknown',
            2: 'unknown'
        }
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for regime detection.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Feature array for regime detection
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features['returns'] = df['Close'].pct_change()
        
        # Volatility (20-day rolling std)
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Trend (20-day price momentum)
        features['momentum'] = df['Close'].pct_change(20)
        
        # Volume change
        if 'Volume' in df.columns:
            features['volume_change'] = df['Volume'].pct_change(5)
        
        # Drop NaN
        features = features.dropna()
        
        return features.values, features.index
    
    def fit(self, df: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit regime detection model.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Self for chaining
        """
        X, _ = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        if self.use_hmm:
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            self.model.fit(X_scaled)
        else:
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                n_init=5,
                random_state=42
            )
            self.model.fit(X_scaled)
        
        # Label regimes based on characteristics
        self._label_regimes(df, X_scaled)
        
        self._fitted = True
        logger.info(f"Fitted {'HMM' if self.use_hmm else 'GMM'} regime detector with {self.n_regimes} regimes")
        
        return self
    
    def _label_regimes(self, df: pd.DataFrame, X_scaled: np.ndarray):
        """
        Assign meaningful labels to regimes based on their characteristics.
        """
        if self.use_hmm:
            regimes = self.model.predict(X_scaled)
        else:
            regimes = self.model.predict(X_scaled)
        
        # Calculate average returns per regime
        returns = df['Close'].pct_change().dropna().values
        returns = returns[-len(regimes):]  # Align with regime predictions
        
        regime_returns = {}
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() > 0:
                regime_returns[r] = np.mean(returns[mask])
            else:
                regime_returns[r] = 0
        
        # Sort by average return
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        
        # Assign labels
        if self.n_regimes == 3:
            self.regime_labels[sorted_regimes[0][0]] = 'bear'
            self.regime_labels[sorted_regimes[1][0]] = 'sideways'
            self.regime_labels[sorted_regimes[2][0]] = 'bull'
        elif self.n_regimes == 2:
            self.regime_labels[sorted_regimes[0][0]] = 'bear'
            self.regime_labels[sorted_regimes[1][0]] = 'bull'
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regimes.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (regime indices, regime probabilities)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X, idx = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        if self.use_hmm:
            regimes = self.model.predict(X_scaled)
            probs = self.model.predict_proba(X_scaled)
        else:
            regimes = self.model.predict(X_scaled)
            probs = self.model.predict_proba(X_scaled)
        
        return regimes, probs
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime features to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with regime features added
        """
        df = df.copy()
        
        try:
            regimes, probs = self.predict(df)
            
            # Create aligned series
            X, idx = self._prepare_features(df)
            
            # Regime indicator
            regime_series = pd.Series(index=idx, data=regimes)
            df['Market_Regime'] = regime_series.reindex(df.index)
            
            # Regime probabilities
            for r in range(self.n_regimes):
                prob_series = pd.Series(index=idx, data=probs[:, r])
                label = self.regime_labels.get(r, f'regime_{r}')
                df[f'Regime_Prob_{label}'] = prob_series.reindex(df.index)
            
            # Binary regime indicators
            df['Is_Bull_Regime'] = (df['Market_Regime'] == self._get_regime_id('bull')).astype(int)
            df['Is_Bear_Regime'] = (df['Market_Regime'] == self._get_regime_id('bear')).astype(int)
            
            # Forward fill any NaN values
            regime_cols = [c for c in df.columns if 'Regime' in c or 'Market_Regime' in c]
            df[regime_cols] = df[regime_cols].ffill().fillna(0)
            
        except Exception as e:
            logger.warning(f"Error adding regime features: {e}")
            df['Market_Regime'] = 1  # Default to sideways
            df['Is_Bull_Regime'] = 0
            df['Is_Bear_Regime'] = 0
        
        return df
    
    def _get_regime_id(self, label: str) -> int:
        """Get regime ID for a given label."""
        for rid, rlabel in self.regime_labels.items():
            if rlabel == label:
                return rid
        return 1  # Default


class VolatilityRegimeClassifier:
    """
    Classify volatility regimes: low, normal, high, extreme.
    
    Uses rolling volatility percentiles to classify current volatility regime.
    """
    
    def __init__(
        self,
        lookback_period: int = 252,
        vol_window: int = 20
    ):
        """
        Initialize volatility classifier.
        
        Args:
            lookback_period: Period to calculate percentiles over
            vol_window: Window for volatility calculation
        """
        self.lookback_period = lookback_period
        self.vol_window = vol_window
    
    def add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility regime features.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with volatility regime features
        """
        df = df.copy()
        
        # Calculate rolling volatility
        returns = df['Close'].pct_change()
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)  # Annualized
        
        df['Volatility_Annualized'] = vol
        
        # Rolling percentile of volatility
        vol_percentile = vol.rolling(self.lookback_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )
        df['Volatility_Percentile'] = vol_percentile
        
        # Classify regime
        def classify_vol(p):
            if pd.isna(p):
                return 2  # Normal
            elif p < 0.2:
                return 1  # Low
            elif p < 0.8:
                return 2  # Normal
            elif p < 0.95:
                return 3  # High
            else:
                return 4  # Extreme
        
        df['Volatility_Regime'] = vol_percentile.apply(classify_vol)
        
        # Binary indicators
        df['Is_High_Volatility'] = (df['Volatility_Regime'] >= 3).astype(int)
        df['Is_Low_Volatility'] = (df['Volatility_Regime'] == 1).astype(int)
        
        # Volatility change
        df['Volatility_Change'] = vol.pct_change(5)
        
        # VIX-like feature (if we had VIX, we'd use it; this is a proxy)
        df['Volatility_Spike'] = (vol > vol.rolling(60).mean() * 1.5).astype(int)
        
        return df


class MarketCorrelationFeatures:
    """
    Add correlation features with market indices and sector ETFs.
    
    Helps the model understand systematic risk and sector relationships.
    """
    
    # Major indices and sector ETFs
    REFERENCE_TICKERS = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'IWM': 'Russell 2000',
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials'
    }
    
    def __init__(
        self,
        reference_tickers: Optional[List[str]] = None,
        correlation_window: int = 20
    ):
        """
        Initialize correlation feature generator.
        
        Args:
            reference_tickers: Tickers to calculate correlations with
            correlation_window: Rolling window for correlation
        """
        self.reference_tickers = reference_tickers or ['SPY', 'QQQ']
        self.correlation_window = correlation_window
        self._reference_data = {}
    
    def fetch_reference_data(
        self,
        start_date: str,
        end_date: str
    ) -> None:
        """
        Fetch reference ticker data.
        
        Args:
            start_date: Start date
            end_date: End date
        """
        import yfinance as yf
        
        for ticker in self.reference_tickers:
            try:
                data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                if not data.empty:
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    self._reference_data[ticker] = data['Close']
                    logger.debug(f"Fetched {ticker} data: {len(data)} rows")
            except Exception as e:
                logger.warning(f"Error fetching {ticker}: {e}")
    
    def add_correlation_features(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Add correlation features to DataFrame.
        
        Args:
            df: DataFrame with stock data
            ticker: Stock ticker (to skip if in reference list)
        
        Returns:
            DataFrame with correlation features added
        """
        df = df.copy()
        stock_returns = df['Close'].pct_change()
        
        # Fetch reference data if not already loaded
        if not self._reference_data:
            start = df.index.min().strftime('%Y-%m-%d')
            end = df.index.max().strftime('%Y-%m-%d')
            self.fetch_reference_data(start, end)
        
        for ref_ticker, ref_data in self._reference_data.items():
            if ref_ticker == ticker:
                continue
            
            try:
                # Align dates
                ref_returns = ref_data.pct_change()
                aligned_stock, aligned_ref = stock_returns.align(ref_returns, join='inner')
                
                if len(aligned_stock) < self.correlation_window:
                    continue
                
                # Rolling correlation
                correlation = aligned_stock.rolling(self.correlation_window).corr(aligned_ref)
                df[f'Corr_{ref_ticker}'] = correlation.reindex(df.index)
                
                # Rolling beta
                covariance = aligned_stock.rolling(self.correlation_window).cov(aligned_ref)
                ref_variance = aligned_ref.rolling(self.correlation_window).var()
                beta = covariance / ref_variance
                df[f'Beta_{ref_ticker}'] = beta.reindex(df.index)
                
                # Relative strength
                stock_return_20d = df['Close'].pct_change(20)
                ref_return_20d = ref_data.pct_change(20).reindex(df.index)
                df[f'RelStrength_{ref_ticker}'] = stock_return_20d - ref_return_20d
                
            except Exception as e:
                logger.debug(f"Error calculating correlation with {ref_ticker}: {e}")
        
        # Fill NaN values
        corr_cols = [c for c in df.columns if c.startswith(('Corr_', 'Beta_', 'RelStrength_'))]
        df[corr_cols] = df[corr_cols].ffill().fillna(0)
        
        return df


class AdvancedFeatureEngineer:
    """
    Combines all advanced feature engineering capabilities.
    
    This is the main interface for adding sophisticated features
    that go beyond basic technical indicators.
    """
    
    def __init__(
        self,
        enable_sentiment: bool = True,
        enable_regime: bool = True,
        enable_volatility: bool = True,
        enable_correlation: bool = True
    ):
        """
        Initialize advanced feature engineer.
        
        Args:
            enable_sentiment: Add sentiment features
            enable_regime: Add market regime features
            enable_volatility: Add volatility regime features
            enable_correlation: Add correlation features
        """
        self.enable_sentiment = enable_sentiment
        self.enable_regime = enable_regime
        self.enable_volatility = enable_volatility
        self.enable_correlation = enable_correlation
        
        self.sentiment_analyzer = YahooNewsSentiment() if enable_sentiment else None
        self.regime_detector = MarketRegimeDetector() if enable_regime else None
        self.volatility_classifier = VolatilityRegimeClassifier() if enable_volatility else None
        self.correlation_features = MarketCorrelationFeatures() if enable_correlation else None
        
        self.advanced_feature_names = []
    
    def add_all_advanced_features(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Add all advanced features to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with all advanced features
        """
        original_cols = set(df.columns)
        
        # Sentiment features
        if self.enable_sentiment and self.sentiment_analyzer:
            try:
                df = self.sentiment_analyzer.add_sentiment_features(df, ticker)
                logger.info("Added sentiment features")
            except Exception as e:
                logger.warning(f"Error adding sentiment features: {e}")
        
        # Market regime features
        if self.enable_regime and self.regime_detector:
            try:
                self.regime_detector.fit(df)
                df = self.regime_detector.add_regime_features(df)
                logger.info("Added market regime features")
            except Exception as e:
                logger.warning(f"Error adding regime features: {e}")
        
        # Volatility regime features
        if self.enable_volatility and self.volatility_classifier:
            try:
                df = self.volatility_classifier.add_volatility_regime_features(df)
                logger.info("Added volatility regime features")
            except Exception as e:
                logger.warning(f"Error adding volatility features: {e}")
        
        # Correlation features
        if self.enable_correlation and self.correlation_features:
            try:
                df = self.correlation_features.add_correlation_features(df, ticker)
                logger.info("Added correlation features")
            except Exception as e:
                logger.warning(f"Error adding correlation features: {e}")
        
        # Track new features
        self.advanced_feature_names = list(set(df.columns) - original_cols)
        logger.info(f"Total advanced features added: {len(self.advanced_feature_names)}")
        
        return df


def add_advanced_features(
    df: pd.DataFrame,
    ticker: str,
    enable_all: bool = True
) -> pd.DataFrame:
    """
    Convenience function to add all advanced features.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker
        enable_all: Enable all feature types
    
    Returns:
        DataFrame with advanced features
    """
    engineer = AdvancedFeatureEngineer(
        enable_sentiment=enable_all,
        enable_regime=enable_all,
        enable_volatility=enable_all,
        enable_correlation=enable_all
    )
    
    return engineer.add_all_advanced_features(df, ticker)


if __name__ == "__main__":
    # Test advanced features
    import yfinance as yf
    
    print("Testing Advanced Feature Engineering...")
    
    # Download test data
    ticker = 'AAPL'
    data = yf.Ticker(ticker).history(start='2023-01-01', end='2024-01-01')
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    print(f"Original data shape: {data.shape}")
    
    # Add advanced features
    engineer = AdvancedFeatureEngineer(
        enable_sentiment=True,
        enable_regime=True,
        enable_volatility=True,
        enable_correlation=True
    )
    
    data_advanced = engineer.add_all_advanced_features(data, ticker)
    
    print(f"Enhanced data shape: {data_advanced.shape}")
    print(f"\nNew features added ({len(engineer.advanced_feature_names)}):")
    for feat in sorted(engineer.advanced_feature_names):
        print(f"  - {feat}")
    
    # Show sample values
    print("\nSample values (last row):")
    for feat in engineer.advanced_feature_names[:10]:
        print(f"  {feat}: {data_advanced[feat].iloc[-1]:.4f}")
