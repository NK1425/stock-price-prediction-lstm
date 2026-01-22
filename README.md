# 📈 Stock Price Prediction using LSTM

A Streamlit web application that predicts future stock prices using Long Short-Term Memory (LSTM) neural networks.

## Features

- Real-time stock data from Yahoo Finance
- LSTM neural network for time series prediction
- Interactive Streamlit dashboard
- Customizable parameters
- Fast/simple model option for quick predictions
- Data caching for improved performance

## Local Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub and click "New app"
4. Select your repository and set main file to `app.py`
5. Click "Deploy"

**Note**: First deployment takes 3-5 minutes due to TensorFlow installation.

## Optimizations for Faster Deployment

- Uses `tensorflow-cpu` (smaller than full TensorFlow with GPU)
- Data caching with `@st.cache_data` 
- Optional simple model for faster training
- Reduced default epochs (25 instead of 50)

## Disclaimer

⚠️ Educational purposes only. Not financial advice.

---

Built with ❤️ using Streamlit and TensorFlow
