"""
Stock Price Prediction with LSTM and Attention - Streamlit Web Interface

A production-quality deep learning application for predicting stock prices
using LSTM networks with attention mechanisms.
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction | LSTM + Attention",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .prediction-up {
        color: #00C853;
        font-weight: bold;
    }
    .prediction-down {
        color: #FF5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading
@st.cache_data(ttl=300)
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with caching."""
    from data_loader import StockDataLoader
    loader = StockDataLoader(cache_dir='./data')
    return loader.download_stock_data(ticker, start_date, end_date)


@st.cache_data(ttl=300)
def generate_features(data):
    """Generate features with caching."""
    from feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    data_with_features = engineer.add_all_features(data)
    data_clean = engineer.handle_missing_values(data_with_features, method='drop')
    return data_clean, engineer.feature_columns


def get_feature_columns():
    """Get feature columns for the model."""
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Returns', 'Log_Returns',
        'SMA_20', 'SMA_50', 'EMA_20',
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Percent',
        'ATR', 'Volatility_20d',
        'Momentum_10', 'OBV', 'VWAP',
        'Stoch_K', 'Williams_R', 'CCI', 'ROC_10', 'MFI'
    ]


def create_model(input_shape, forecast_horizon):
    """Create the LSTM-Attention model."""
    import tensorflow as tf
    tf.random.set_seed(42)

    from model import build_attention_model_with_weights

    model, attention_model = build_attention_model_with_weights(
        input_shape=input_shape,
        output_steps=forecast_horizon,
        lstm_units=[64, 32],
        attention_units=32,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    return model, attention_model


def plot_stock_history(data, ticker):
    """Plot stock price history."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Price chart
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], linewidth=1.5, color='#2E86AB')
    ax1.fill_between(data.index, data['Close'].min(), data['Close'], alpha=0.1, color='#2E86AB')
    ax1.set_title(f'{ticker} Stock Price History', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Volume chart
    ax2 = axes[1]
    colors = ['#00C853' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#FF5252'
              for i in range(len(data))]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7, width=1)
    ax2.set_ylabel('Volume', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_technical_indicators(data):
    """Plot technical indicators."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # RSI
    ax1 = axes[0]
    ax1.plot(data.index, data['RSI_14'], linewidth=1.5, color='#764ba2')
    ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax1.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax1.set_ylabel('RSI (14)', fontsize=11)
    ax1.set_title('Technical Indicators', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # MACD
    ax2 = axes[1]
    ax2.plot(data.index, data['MACD'], label='MACD', linewidth=1.5, color='#2E86AB')
    ax2.plot(data.index, data['MACD_Signal'], label='Signal', linewidth=1.5, color='#E94F37')
    colors = ['#00C853' if v >= 0 else '#FF5252' for v in data['MACD_Histogram']]
    ax2.bar(data.index, data['MACD_Histogram'], color=colors, alpha=0.5, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('MACD', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Bollinger Bands
    ax3 = axes[2]
    ax3.plot(data.index, data['Close'], label='Close', linewidth=1.5, color='#2E86AB')
    ax3.plot(data.index, data['BB_Upper'], label='Upper Band', linewidth=1, color='#E94F37', alpha=0.7)
    ax3.plot(data.index, data['BB_Lower'], label='Lower Band', linewidth=1, color='#00C853', alpha=0.7)
    ax3.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], alpha=0.1, color='gray')
    ax3.set_ylabel('Price ($)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_predictions(actual, predicted, dates=None, title="Actual vs Predicted"):
    """Plot actual vs predicted prices."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = dates if dates is not None else range(len(actual))

    ax.plot(x, actual, label='Actual', linewidth=2, color='#2E86AB')
    ax.plot(x, predicted, label='Predicted', linewidth=2, color='#E94F37', linestyle='--')
    ax.fill_between(x, actual, predicted, alpha=0.2, color='gray')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    if dates is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


def plot_attention_weights(attention_weights, sequence_length):
    """Plot attention weights heatmap."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Average attention across samples
    avg_attention = np.mean(attention_weights, axis=0)

    # Create heatmap
    im = ax.imshow(avg_attention.reshape(1, -1), aspect='auto', cmap='YlOrRd')

    ax.set_xlabel('Time Step (days ago)', fontsize=11)
    ax.set_title('Attention Weights - Which Days Matter Most', fontsize=14, fontweight='bold')
    ax.set_yticks([])

    # X-axis labels (days ago)
    tick_positions = np.arange(0, sequence_length, 10)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([sequence_length - t for t in tick_positions])

    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2, label='Attention Weight')
    plt.tight_layout()
    return fig


def plot_forecast(historical_prices, predicted_prices, ticker):
    """Plot forecast with historical context."""
    fig, ax = plt.subplots(figsize=(12, 6))

    n_hist = len(historical_prices)
    n_pred = len(predicted_prices)

    # Historical
    ax.plot(range(n_hist), historical_prices, label='Historical', linewidth=2, color='#2E86AB')

    # Predictions
    ax.plot(range(n_hist - 1, n_hist + n_pred),
            [historical_prices[-1]] + list(predicted_prices),
            label='Forecast', linewidth=2, color='#E94F37', linestyle='--', marker='o', markersize=6)

    # Confidence band (simple estimate)
    std = np.std(historical_prices[-30:]) if len(historical_prices) >= 30 else np.std(historical_prices)
    upper = [historical_prices[-1]] + [p + std * (i + 1) * 0.5 for i, p in enumerate(predicted_prices)]
    lower = [historical_prices[-1]] + [p - std * (i + 1) * 0.5 for i, p in enumerate(predicted_prices)]
    ax.fill_between(range(n_hist - 1, n_hist + n_pred), lower, upper, alpha=0.2, color='#E94F37')

    ax.axvline(x=n_hist - 1, color='gray', linestyle=':', linewidth=2)
    ax.text(n_hist - 1, ax.get_ylim()[1], ' Today', fontsize=10, va='top')

    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title(f'{ticker} Price Forecast - Next {n_pred} Days', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">📈 Stock Price Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LSTM Neural Network with Attention Mechanism</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Stock selection
    st.sidebar.subheader("Stock Selection")
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']

    ticker_option = st.sidebar.selectbox(
        "Select Stock",
        options=popular_stocks + ['Other'],
        index=0
    )

    if ticker_option == 'Other':
        ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
    else:
        ticker = ticker_option

    # Date range
    st.sidebar.subheader("Data Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*3),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider("Lookback Period (days)", 30, 120, 60)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 14, 7)

    # Training parameters
    st.sidebar.subheader("Training")
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 30)

    # Action button
    run_prediction = st.sidebar.button("🚀 Run Prediction", type="primary", use_container_width=True)

    # Main content
    if run_prediction:
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Load data
            status_text.text("📥 Downloading stock data...")
            progress_bar.progress(10)

            data = load_stock_data(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if data.empty or len(data) < sequence_length + forecast_horizon + 50:
                st.error(f"Insufficient data for {ticker}. Please select a larger date range.")
                return

            # Display stock info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            with col2:
                change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                st.metric("Daily Change", f"{change:+.2f}%")
            with col3:
                st.metric("52W High", f"${data['Close'].max():.2f}")
            with col4:
                st.metric("52W Low", f"${data['Close'].min():.2f}")

            # Step 2: Generate features
            status_text.text("🔧 Generating technical indicators...")
            progress_bar.progress(20)

            data_clean, all_features = generate_features(data)

            # Filter feature columns
            feature_cols = get_feature_columns()
            feature_cols = [c for c in feature_cols if c in data_clean.columns]

            # Step 3: Create sequences
            status_text.text("📊 Building sequences...")
            progress_bar.progress(30)

            from sequence_builder import SequenceBuilder, create_train_val_test_sequences

            builder = SequenceBuilder(
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon,
                feature_columns=feature_cols,
                target_column='Close'
            )

            (X_train, y_train), (X_val, y_val), (X_test, y_test), test_dates = \
                create_train_val_test_sequences(data_clean, builder, 0.7, 0.15, 0.15)

            # Step 4: Build model
            status_text.text("🧠 Building LSTM-Attention model...")
            progress_bar.progress(40)

            import tensorflow as tf
            tf.random.set_seed(42)
            np.random.seed(42)

            n_features = X_train.shape[2]
            model, attention_model = create_model(
                input_shape=(sequence_length, n_features),
                forecast_horizon=forecast_horizon
            )

            # Step 5: Train model
            status_text.text("🏋️ Training model...")

            # Custom callback for progress updates
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = 40 + int((epoch + 1) / epochs * 40)
                    progress_bar.progress(min(progress, 80))

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stop, ProgressCallback()],
                verbose=0
            )

            # Step 6: Make predictions
            status_text.text("🔮 Making predictions...")
            progress_bar.progress(85)

            # Test set predictions
            test_predictions, test_attention = attention_model.predict(X_test, verbose=0)
            test_pred_unscaled = builder.inverse_transform_predictions(test_predictions)
            test_actual_unscaled = builder.inverse_transform_predictions(y_test)

            # Future predictions
            from sequence_builder import create_single_prediction_sequence
            latest_sequence = create_single_prediction_sequence(data_clean, builder)
            future_pred, future_attention = attention_model.predict(latest_sequence, verbose=0)
            future_prices = builder.inverse_transform_predictions(future_pred)[0]

            # Step 7: Calculate metrics
            status_text.text("📈 Calculating metrics...")
            progress_bar.progress(95)

            # First day predictions for metrics
            pred_day1 = test_pred_unscaled[:, 0]
            actual_day1 = test_actual_unscaled[:, 0]

            rmse = np.sqrt(np.mean((pred_day1 - actual_day1)**2))
            mae = np.mean(np.abs(pred_day1 - actual_day1))
            mape = np.mean(np.abs((actual_day1 - pred_day1) / actual_day1)) * 100

            progress_bar.progress(100)
            status_text.text("✅ Complete!")

            # Display results
            st.markdown("---")
            st.header("📊 Results")

            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Forecast", "📉 Historical", "🎯 Model Performance",
                "🔍 Attention Analysis", "📋 Technical Indicators"
            ])

            with tab1:
                st.subheader(f"🔮 {forecast_horizon}-Day Price Forecast")

                # Forecast metrics
                last_price = data_clean['Close'].iloc[-1]

                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_pred = np.mean(future_prices)
                    change = (avg_pred - last_price) / last_price * 100
                    st.metric("Average Forecast", f"${avg_pred:.2f}", f"{change:+.2f}%")
                with col2:
                    final_change = (future_prices[-1] - last_price) / last_price * 100
                    st.metric(f"Day {forecast_horizon} Forecast", f"${future_prices[-1]:.2f}", f"{final_change:+.2f}%")
                with col3:
                    if final_change > 2:
                        trend = "🟢 Bullish"
                    elif final_change < -2:
                        trend = "🔴 Bearish"
                    else:
                        trend = "🟡 Neutral"
                    st.metric("Trend Signal", trend)

                # Forecast chart
                hist_prices = data_clean['Close'].iloc[-60:].values
                fig = plot_forecast(hist_prices, future_prices, ticker)
                st.pyplot(fig)
                plt.close(fig)

                # Forecast table
                st.subheader("Detailed Forecast")
                forecast_df = pd.DataFrame({
                    'Day': [f"Day {i+1}" for i in range(forecast_horizon)],
                    'Predicted Price': [f"${p:.2f}" for p in future_prices],
                    'Change from Today': [f"{(p - last_price) / last_price * 100:+.2f}%" for p in future_prices]
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            with tab2:
                st.subheader("Historical Stock Data")
                fig = plot_stock_history(data.tail(252), ticker)  # Last year
                st.pyplot(fig)
                plt.close(fig)

            with tab3:
                st.subheader("Model Performance Metrics")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with col2:
                    st.metric("MAE", f"${mae:.2f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")

                # Training history
                st.subheader("Training History")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                axes[0].plot(history.history['loss'], label='Training')
                axes[0].plot(history.history['val_loss'], label='Validation')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(history.history['mae'], label='Training')
                axes[1].plot(history.history['val_mae'], label='Validation')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].set_title('Mean Absolute Error')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Actual vs Predicted
                st.subheader("Test Set: Actual vs Predicted")
                fig = plot_predictions(actual_day1, pred_day1)
                st.pyplot(fig)
                plt.close(fig)

            with tab4:
                st.subheader("Attention Weight Analysis")
                st.markdown("""
                The attention mechanism shows which historical days the model considers most important
                when making predictions. Higher weights indicate more influential time steps.
                """)

                fig = plot_attention_weights(test_attention, sequence_length)
                st.pyplot(fig)
                plt.close(fig)

                # Top attended days
                avg_attention = np.mean(test_attention, axis=0)
                top_indices = np.argsort(avg_attention)[-5:][::-1]

                st.subheader("Most Important Days")
                for i, idx in enumerate(top_indices):
                    days_ago = sequence_length - idx
                    st.write(f"{i+1}. **{days_ago} days ago** - Attention: {avg_attention[idx]:.4f}")

            with tab5:
                st.subheader("Technical Indicators")
                recent_data = data_clean.tail(252)  # Last year
                fig = plot_technical_indicators(recent_data)
                st.pyplot(fig)
                plt.close(fig)

                # Current indicator values
                st.subheader("Current Indicator Values")
                latest = data_clean.iloc[-1]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    rsi = latest['RSI_14']
                    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI (14)", f"{rsi:.1f}", rsi_signal)
                with col2:
                    macd = latest['MACD']
                    signal = latest['MACD_Signal']
                    macd_signal = "Bullish" if macd > signal else "Bearish"
                    st.metric("MACD", f"{macd:.2f}", macd_signal)
                with col3:
                    bb_pct = latest['BB_Percent']
                    bb_signal = "Upper Band" if bb_pct > 0.8 else "Lower Band" if bb_pct < 0.2 else "Middle"
                    st.metric("BB %B", f"{bb_pct:.2f}", bb_signal)
                with col4:
                    atr = latest['ATR']
                    st.metric("ATR", f"${atr:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

    else:
        # Welcome screen
        st.info("👈 Configure parameters in the sidebar and click **Run Prediction** to start.")

        st.markdown("---")
        st.header("🎯 Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🧠 Deep Learning")
            st.markdown("""
            - Multi-layer LSTM architecture
            - Custom attention mechanism
            - Dropout & regularization
            - Early stopping
            """)

        with col2:
            st.markdown("### 📊 Technical Analysis")
            st.markdown("""
            - 15+ technical indicators
            - RSI, MACD, Bollinger Bands
            - ATR, OBV, VWAP
            - Momentum indicators
            """)

        with col3:
            st.markdown("### 📈 Predictions")
            st.markdown("""
            - Multi-step forecasting
            - Attention visualization
            - Performance metrics
            - Confidence analysis
            """)

        st.markdown("---")
        st.header("📚 How It Works")

        st.markdown("""
        1. **Data Collection**: Historical OHLCV data from Yahoo Finance
        2. **Feature Engineering**: 15+ technical indicators calculated
        3. **Sequence Building**: Sliding window sequences for LSTM
        4. **Model Training**: LSTM with attention mechanism learns patterns
        5. **Prediction**: Model forecasts next N days of closing prices
        6. **Analysis**: Attention weights show which days influenced predictions
        """)

        st.markdown("---")
        st.warning("""
        **⚠️ Disclaimer**: This tool is for educational purposes only.
        Stock predictions are inherently uncertain and should not be used as financial advice.
        Always consult with qualified financial advisors before making investment decisions.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Built with TensorFlow & Streamlit | "
        "<a href='https://github.com/NK1425/stock-price-prediction-lstm'>GitHub</a>"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
