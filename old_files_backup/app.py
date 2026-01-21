import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Prediction using LSTM")
st.markdown("### Predict future stock prices using Long Short-Term Memory neural networks")

# Sidebar inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*5))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
prediction_days = st.sidebar.slider("Days to use for prediction", min_value=30, max_value=120, value=60)
future_days = st.sidebar.slider("Days to predict into future", min_value=1, max_value=30, value=7)

# Model parameters
st.sidebar.subheader("Model Parameters")
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=50)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=64, value=32)

if st.sidebar.button("Train Model & Predict"):
    try:
        # Download stock data
        with st.spinner(f"Downloading {ticker} data..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
        else:
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['Close'][-1]:.2f}")
            with col2:
                st.metric("52W High", f"${data['Close'].max():.2f}")
            with col3:
                st.metric("52W Low", f"${data['Close'].min():.2f}")
            with col4:
                change = ((data['Close'][-1] - data['Close'][0]) / data['Close'][0]) * 100
                st.metric("Overall Change", f"{change:.2f}%")
            
            # Plot historical data
            st.subheader("Historical Stock Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label='Close Price', linewidth=2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{ticker} Stock Price History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Prepare data for LSTM
            st.subheader("Training LSTM Model")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing data...")
            progress_bar.progress(10)
            
            # Use only Close price
            close_prices = data['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            # Create training data
            x_train = []
            y_train = []
            
            for i in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[i-prediction_days:i, 0])
                y_train.append(scaled_data[i, 0])
            
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            status_text.text("Building model architecture...")
            progress_bar.progress(20)
            
            # Build LSTM model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            status_text.text(f"Training model for {epochs} epochs...")
            progress_bar.progress(30)
            
            # Train the model
            history = model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.1
            )
            
            progress_bar.progress(70)
            status_text.text("Making predictions...")
            
            # Prepare test data
            test_data = scaled_data[-prediction_days:]
            x_test = []
            x_test.append(test_data)
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Predict future prices
            predictions = []
            current_batch = test_data.copy()
            
            for i in range(future_days):
                current_batch_reshaped = current_batch.reshape((1, prediction_days, 1))
                next_pred = model.predict(current_batch_reshaped, verbose=0)[0, 0]
                predictions.append(next_pred)
                current_batch = np.append(current_batch[1:], [[next_pred]], axis=0)
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            progress_bar.progress(90)
            status_text.text("Generating visualization...")
            
            # Create prediction dates
            last_date = data.index[-1]
            prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
            
            # Plot predictions
            st.subheader("Price Predictions")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data (last 90 days)
            recent_data = data.tail(90)
            ax.plot(recent_data.index, recent_data['Close'], label='Historical Price', linewidth=2, color='blue')
            
            # Plot predictions
            ax.plot(prediction_dates, predictions, label='Predicted Price', linewidth=2, color='red', linestyle='--', marker='o')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{ticker} Price Prediction for Next {future_days} Days')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Display predictions table
            st.subheader("Predicted Prices")
            pred_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted Price': predictions.flatten()
            })
            pred_df['Date'] = pred_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(pred_df, use_container_width=True)
            
            # Training metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Final Training Loss", f"{history.history['loss'][-1]:.6f}")
            with col2:
                st.metric("Final Validation Loss", f"{history.history['val_loss'][-1]:.6f}")
            
            # Plot training history
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Model Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            progress_bar.progress(100)
            status_text.text("Complete! ✅")
            
            # Disclaimer
            st.warning("⚠️ **Disclaimer**: This is for educational purposes only. Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your ticker symbol and date range, then try again.")

else:
    st.info("👈 Configure the parameters in the sidebar and click 'Train Model & Predict' to start.")

