import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Stock Price Prediction using LSTM")
st.markdown("### Predict future stock prices using Long Short-Term Memory neural networks")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*5))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
prediction_days = st.sidebar.slider("Days to use for prediction", min_value=30, max_value=120, value=60)
future_days = st.sidebar.slider("Days to predict into future", min_value=1, max_value=30, value=7)

st.sidebar.subheader("Model Parameters")
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=50)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=64, value=32)

if st.sidebar.button("Train Model & Predict"):
    try:
        with st.spinner(f"Downloading {ticker} data..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
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
            
            st.subheader("Training LSTM Model")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing data...")
            progress_bar.progress(10)
            
            close_prices = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            x_train, y_train = [], []
            for i in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[i-prediction_days:i, 0])
                y_train.append(scaled_data[i, 0])
            
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            progress_bar.progress(20)
            
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            status_text.text(f"Training model...")
            progress_bar.progress(30)
            
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
            
            progress_bar.progress(70)
            
            predictions = []
            current_batch = scaled_data[-prediction_days:].copy()
            
            for i in range(future_days):
                pred = model.predict(current_batch.reshape(1, prediction_days, 1), verbose=0)[0, 0]
                predictions.append(pred)
                current_batch = np.append(current_batch[1:], [[pred]], axis=0)
            
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
    - LSTM neural network predictions
    - Interactive visualizations
    - Customizable parameters
    """)

st.markdown("---")
st.markdown("Built with Streamlit • Powered by Keras 3")
