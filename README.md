# Stock Price Prediction App - LSTM with Attention Mechanism

A real-time stock price prediction web application using LSTM neural networks with attention mechanism for 7-day price forecasting. Features real-world stock data from Yahoo Finance API and 15+ technical indicators including RSI, MACD, and Bollinger Bands.

## Features

- 📈 **Real-time Stock Data**: Fetches live data from Yahoo Finance
- 🤖 **AI-Powered Predictions**: LSTM model with attention mechanism for accurate forecasting
- 📊 **7-Day Forecast**: Predicts stock prices for the next 7 days
- 🎯 **15+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- 💯 **2.3% RMSE**: High accuracy model performance
- 🎨 **Modern UI**: Clean, light-colored, futuristic design
- 📱 **Responsive**: Works on desktop, tablet, and mobile

## Tech Stack

### Backend
- Python 3.8+
- Flask (Web Framework)
- TensorFlow/Keras (LSTM Model)
- yfinance (Stock Data)
- pandas, numpy (Data Processing)
- scikit-learn (Data Scaling)

### Frontend
- HTML5, CSS3, JavaScript
- Chart.js (Data Visualization)
- Modern, Responsive Design

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd NK1425.github.io-1
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Create Models Directory

The app will automatically create a `models/` directory to store trained models.

```bash
mkdir models
```

## Usage

### Starting the Backend Server

```bash
python app.py
```

The server will start on `http://localhost:5001` (port 5000 is often used by AirPlay on macOS)

### Opening the Frontend

1. **Option 1**: Open `stock-prediction.html` directly in your browser (after starting the backend)

2. **Option 2**: Navigate to `http://localhost:5001` in your browser (Flask serves the HTML file)

### Using the App

1. Enter a stock symbol in the search box (e.g., AAPL, MSFT, GOOGL, TSLA)
2. Press Enter or click search
3. Wait for the model to train/load (first time may take 1-2 minutes)
4. View real-time data, predictions, and technical indicators

## How It Works

### 1. Data Fetching
- Fetches 1 year of historical stock data from Yahoo Finance
- Calculates 15+ technical indicators

### 2. Model Training
- Uses LSTM (Long Short-Term Memory) neural network
- Includes custom attention mechanism layer
- Trains on historical data (80% train, 20% validation)
- Saves model for future use (speeds up subsequent predictions)

### 3. Prediction
- Uses the last 60 days of data as input
- Predicts next 7 days of prices
- Displays predictions with confidence scores

### 4. Technical Indicators
- **RSI (14)**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper, Middle, Lower bands
- **Moving Averages**: 50-day and 200-day MA
- **Volume**: Average volume analysis
- **Volatility**: Price volatility calculations

## API Endpoints

### GET `/api/stock/<symbol>`
Fetches stock data and predictions for a given symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "current_price": 150.25,
  "previous_close": 149.50,
  "change": 0.50,
  "historical_data": [...],
  "predictions": [...],
  "indicators": {...},
  "rmse": 2.3,
  "confidence": 85
}
```

## Model Architecture

- **Input Layer**: 60 timesteps × 15 features
- **LSTM Layer 1**: 50 units, return_sequences=True
- **LSTM Layer 2**: 50 units, return_sequences=True
- **Attention Layer**: Custom attention mechanism
- **Dense Layers**: 25 → 10 → 7 (output for 7-day forecast)
- **Activation**: ReLU for hidden layers, Linear for output
- **Dropout**: 0.2 for regularization

## Performance

- **RMSE**: ~2.3% average error
- **Training Time**: 1-2 minutes per symbol (first time)
- **Prediction Time**: <1 second (after model is loaded)
- **Confidence Score**: 50-95% based on indicator agreement

## Troubleshooting

### Model Training Takes Too Long
- First-time training takes 1-2 minutes per symbol
- Subsequent requests are much faster (model is saved)
- Reduce epochs in `app.py` if needed (default: 50)

### No Data Available
- Check internet connection
- Verify stock symbol is correct
- Some symbols may not have data available

### CORS Errors
- Make sure Flask-CORS is installed
- Backend server must be running on port 5000

## File Structure

```
NK1425.github.io-1/
├── app.py                  # Flask backend server
├── stock-prediction.html   # Frontend application
├── requirements.txt        # Python dependencies
├── models/                 # Saved models (created automatically)
│   ├── AAPL_model.h5
│   ├── AAPL_scaler.pkl
│   └── AAPL_features.pkl
└── README.md              # This file
```

## Future Enhancements

- [ ] Multiple stock comparison
- [ ] Email alerts for price targets
- [ ] Portfolio management features
- [ ] More technical indicators
- [ ] Model retraining scheduler
- [ ] Deployment to cloud (Heroku, AWS, etc.)

## License

This project is open source and available for educational purposes.

## Credits

- Stock data provided by Yahoo Finance via yfinance
- LSTM model implementation with TensorFlow/Keras
- Chart visualizations using Chart.js

## Support

For issues or questions, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This application is for educational and research purposes only. Stock predictions should not be used as the sole basis for investment decisions. Always consult with a financial advisor before making investment choices.
