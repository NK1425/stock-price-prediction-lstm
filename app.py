import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, start, end):
    """Fetch stock data with caching."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Stock Price Prediction using Neural Networks")
st.markdown("### Predict future stock prices using Machine Learning")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*5))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
prediction_days = st.sidebar.slider("Days to use for prediction", min_value=30, max_value=120, value=60)
future_days = st.sidebar.slider("Days to predict into future", min_value=1, max_value=30, value=7)

st.sidebar.subheader("Model Parameters")
hidden_layers = st.sidebar.selectbox("Model Complexity", ["Simple (Fast)", "Medium", "Complex (Slower)"], index=0)
max_iter = st.sidebar.slider("Training Iterations", min_value=100, max_value=1000, value=300)

# Map complexity to hidden layer sizes
layer_config = {
    "Simple (Fast)": (50,),
    "Medium": (100, 50),
    "Complex (Slower)": (100, 100, 50)
}

if st.sidebar.button("Train Model & Predict"):
    try:
        with st.spinner(f"Downloading {ticker} data..."):
            data = fetch_stock_data(ticker, start_date, end_date)
        
        if data.empty:
            st.error(f"No data found for ticker '{ticker}'.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("52W High", f"${data['Close'].max():.2f}")
            with col3:
                st.metric("52W Low", f"${data['Close'].min():.2f}")
            with col4:
                change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                st.metric("Overall Change", f"{change:.2f}%")
            
            st.subheader("Historical Stock Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], linewidth=2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{ticker} Stock Price History')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("Training Neural Network Model")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing data...")
            progress_bar.progress(10)
            
            close_prices = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            # Create sequences for training
            x_train, y_train = [], []
            for i in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[i-prediction_days:i, 0])
                y_train.append(scaled_data[i, 0])
            
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            
            progress_bar.progress(20)
            
            status_text.text("Training model...")
            progress_bar.progress(30)
            
            # Use MLPRegressor (Neural Network)
            model = MLPRegressor(
                hidden_layer_sizes=layer_config[hidden_layers],
                activation='relu',
                solver='adam',
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            model.fit(x_train, y_train)
            
            progress_bar.progress(70)
            status_text.text("Making predictions...")
            
            # Make future predictions
            predictions = []
            current_batch = scaled_data[-prediction_days:, 0].copy()
            
            for i in range(future_days):
                pred = model.predict(current_batch.reshape(1, -1))[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[1:], pred)
            
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            progress_bar.progress(90)
            
            last_date = data.index[-1]
            prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
            
            st.subheader("Price Predictions")
            fig, ax = plt.subplots(figsize=(12, 6))
            recent_data = data.tail(90)
            ax.plot(recent_data.index, recent_data['Close'], label='Historical', linewidth=2, color='blue')
            ax.plot(prediction_dates, predictions, label='Predicted', linewidth=2, color='red', linestyle='--', marker='o')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{ticker} Price Prediction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.subheader("Predicted Prices")
            pred_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': predictions.flatten()})
            pred_df['Date'] = pred_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(pred_df, use_container_width=True)
            
            progress_bar.progress(100)
            status_text.text("Complete! ✅")
            
            st.warning("⚠️ **Disclaimer**: Educational purposes only. Not financial advice.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("👈 Configure parameters and click 'Train Model & Predict'")
    st.markdown("""
    ### Features:
    - Real-time stock data from Yahoo Finance
    - Neural network predictions (MLPRegressor)
    - Interactive visualizations
    - Customizable parameters
    - Fast training and predictions
    """)

st.markdown("---")
st.markdown("Built with Streamlit • Powered by scikit-learn")
