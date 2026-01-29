# Stock Price Prediction with LSTM and Attention Mechanism

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-price-prediction-lstm.streamlit.app)

A production-quality deep learning pipeline for predicting stock prices using LSTM networks with attention mechanisms. This project implements an end-to-end machine learning workflow for multi-step time-series forecasting.

## Live Demo

**Try the app:** [https://stock-price-prediction-lstm.streamlit.app](https://stock-price-prediction-lstm.streamlit.app)

## Features

- **LSTM with Attention**: Multi-layer LSTM architecture with custom attention mechanism for improved temporal modeling
- **15+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, and more
- **Multi-Step Forecasting**: Predict the next 7 days of closing prices
- **Walk-Forward Validation**: Proper time-series cross-validation to prevent data leakage
- **Comprehensive Evaluation**: RMSE, MAE, MAPE metrics with visualization
- **Attention Visualization**: Understand which time steps the model focuses on

## Project Structure

```
stock-price-prediction-lstm/
├── app.py                     # Streamlit web interface
├── main.py                    # CLI entry point
├── data/                      # Cached stock data
├── models/                    # Saved trained models
├── notebooks/                 # Jupyter notebooks for exploration
│   └── exploration.ipynb
├── results/
│   ├── figures/              # Visualization outputs
│   └── metrics/              # Evaluation metrics (JSON)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data download and preprocessing
│   ├── feature_engineering.py # Technical indicator generation
│   ├── sequence_builder.py   # Sequence creation for LSTM
│   ├── model.py              # LSTM-Attention architecture
│   ├── train.py              # Training pipeline
│   └── evaluate.py           # Evaluation and visualization
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Web Interface (Recommended)

Run the Streamlit app locally:
```bash
streamlit run app.py
```

Or visit the live demo: [https://stock-price-prediction-lstm.streamlit.app](https://stock-price-prediction-lstm.streamlit.app)

### Command Line

```bash
# Train a model
python main.py --ticker AAPL --train --epochs 100

# Make predictions
python main.py --ticker AAPL --predict

# Evaluate model
python main.py --ticker AAPL --evaluate
```

### Training a Model (Python API)

```python
from src.train import train_single_stock

# Train model for Apple stock
trainer = train_single_stock(
    ticker='AAPL',
    start_date='2019-01-01',
    end_date='2024-01-01',
    sequence_length=60,      # Look back 60 days
    forecast_horizon=7,      # Predict next 7 days
    epochs=100
)

# Access metrics
print(trainer.metrics)
```

### Using Individual Components

```python
from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.sequence_builder import SequenceBuilder
from src.model import build_lstm_attention_model

# 1. Load data
loader = StockDataLoader(cache_dir='./data')
data = loader.download_stock_data('AAPL', '2019-01-01', '2024-01-01')

# 2. Generate features
engineer = FeatureEngineer()
data_with_features = engineer.add_all_features(data)
data_clean = engineer.handle_missing_values(data_with_features)

# 3. Build sequences
builder = SequenceBuilder(
    sequence_length=60,
    forecast_horizon=7,
    feature_columns=['Close', 'Volume', 'RSI_14', 'MACD', ...]
)
X_train, y_train = builder.create_sequences(train_data, fit=True)
X_test, y_test = builder.transform_sequences(test_data)

# 4. Build and train model
model = build_lstm_attention_model(
    input_shape=(60, n_features),
    output_steps=7,
    lstm_units=[128, 64, 32],
    attention_units=64
)
model.fit(X_train, y_train, epochs=100)
```

## Methodology

### Data Pipeline

1. **Data Acquisition**: Historical OHLCV data from Yahoo Finance via `yfinance`
2. **Feature Engineering**: 15+ technical indicators calculated from price and volume data
3. **Preprocessing**: Missing value handling and MinMax scaling
4. **Sequence Creation**: Sliding window approach for time-series input

### Technical Indicators

| Category | Indicators |
|----------|------------|
| Trend | SMA (5, 10, 20, 50, 200), EMA (5, 10, 20, 50) |
| Momentum | RSI (7, 14, 21), MACD, Stochastic, Williams %R, ROC |
| Volatility | Bollinger Bands, ATR, Historical Volatility |
| Volume | OBV, VWAP, MFI |
| Other | CCI, Price Position, Returns, Log Returns |

### Model Architecture

```
Input (60 timesteps, N features)
    │
    ▼
BatchNormalization
    │
    ▼
LSTM (128 units, return_sequences=True) + LayerNorm
    │
    ▼
LSTM (64 units, return_sequences=True) + LayerNorm
    │
    ▼
LSTM (32 units, return_sequences=True) + LayerNorm
    │
    ▼
Attention Layer (64 units)  ──► Attention Weights
    │
    ▼
Dense (64) + BatchNorm + Dropout
    │
    ▼
Dense (32) + BatchNorm + Dropout
    │
    ▼
Output (7 days forecast)
```

### Attention Mechanism

The custom attention layer computes importance weights for each time step:

```
score = tanh(W · hidden_states + b)
attention_weights = softmax(score · u)
context_vector = Σ(attention_weights · hidden_states)
```

This allows the model to focus on the most relevant historical patterns when making predictions.

### Training Strategy

- **Time-Based Splitting**: 70% train, 15% validation, 15% test (chronological order)
- **Walk-Forward Validation**: Expanding window approach to simulate real trading
- **Early Stopping**: Patience of 15 epochs monitoring validation loss
- **Learning Rate Scheduling**: Reduce LR by 0.5x on plateau
- **Regularization**: L2 regularization (0.001) and Dropout (0.2)

### Evaluation Metrics

- **RMSE**: Root Mean Squared Error (primary metric)
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct up/down predictions
- **R²**: Coefficient of determination

## Results

### Sample Performance (AAPL, 2019-2024)

| Metric | Day 1 | Day 3 | Day 7 |
|--------|-------|-------|-------|
| RMSE ($) | ~2.50 | ~3.20 | ~4.50 |
| MAE ($) | ~1.80 | ~2.40 | ~3.50 |
| MAPE (%) | ~1.2 | ~1.6 | ~2.3 |

*Note: Actual results vary based on market conditions and training period.*

### Visualizations

The evaluation module generates:

1. **Actual vs Predicted Plot**: Compare predicted and actual prices
2. **Rolling RMSE**: Track error over time
3. **Attention Heatmap**: Visualize which time steps influence predictions
4. **Error Distribution**: Analyze prediction residuals
5. **Horizon Performance**: Compare accuracy across forecast days

## Advanced Usage

### Walk-Forward Validation

```python
from src.train import StockPriceTrainer

trainer = StockPriceTrainer(ticker='AAPL')
trainer.prepare_data('2019-01-01', '2024-01-01')
trainer.build_model()

# Run walk-forward validation
metrics = trainer.train_walk_forward(
    n_splits=5,
    epochs_per_split=50
)
print(f"Average RMSE: {np.mean(metrics['rmse']):.4f}")
```

### Custom Model Configuration

```python
from src.model import build_lstm_attention_model

model = build_lstm_attention_model(
    input_shape=(60, 30),
    output_steps=7,
    lstm_units=[256, 128, 64],      # Deeper network
    attention_units=128,
    dense_units=[128, 64, 32],
    dropout_rate=0.3,
    use_bidirectional=True,          # Bidirectional LSTM
    use_multi_head_attention=True,   # Multi-head attention
    num_attention_heads=8,
    learning_rate=0.0005
)
```

### Generating Evaluation Report

```python
from src.evaluate import evaluate_model

report = evaluate_model(trainer, show_plots=True)
print(report['metrics']['overall'])
```

## File Descriptions

| File | Description |
|------|-------------|
| `data_loader.py` | Downloads and caches stock data from Yahoo Finance |
| `feature_engineering.py` | Generates 15+ technical indicators |
| `sequence_builder.py` | Creates sliding window sequences for LSTM |
| `model.py` | LSTM-Attention architecture with custom layers |
| `train.py` | Complete training pipeline with validation |
| `evaluate.py` | Metrics calculation and visualization |

## Reproducibility

To ensure reproducible results:

```python
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

All random operations use fixed seeds by default.

## Limitations & Disclaimer

- **Not Financial Advice**: This project is for educational purposes only
- **Market Efficiency**: Stock prices are influenced by many factors not captured in historical data
- **Overfitting Risk**: Past performance does not guarantee future results
- **Transaction Costs**: Real trading involves fees, slippage, and market impact

## Future Improvements

- [ ] Add transformer-based architecture (Temporal Fusion Transformer)
- [ ] Incorporate sentiment analysis from news/social media
- [ ] Add portfolio optimization layer
- [ ] Implement real-time prediction API
- [ ] Add GPU training optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data API
- [TensorFlow](https://www.tensorflow.org/) for deep learning framework
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) for attention mechanism inspiration

---

Built with Python, TensorFlow, and a passion for quantitative finance.
