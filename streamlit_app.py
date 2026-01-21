import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Configuration
API_BASE_URL = "http://localhost:5001/api"  # Change this to your deployed Flask API URL

# Page configuration
st.set_page_config(
    page_title="NeuralStock AI | Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #6366f1;
    }
    .positive {
        color: #10b981;
        font-weight: 600;
    }
    .negative {
        color: #ef4444;
        font-weight: 600;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .chat-user {
        background: #6366f1;
        color: white;
        margin-left: auto;
    }
    .chat-ai {
        background: #f1f5f9;
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = 'AAPL'
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

def fetch_stock_data(symbol):
    """Fetch stock data from Flask API"""
    try:
        with st.spinner(f'Loading data for {symbol}...'):
            response = requests.get(f"{API_BASE_URL}/stock/{symbol}", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {str(e)}")
        st.info("Make sure the Flask backend is running at http://localhost:5001")
        return None

def search_stocks(query):
    """Search for stock symbols"""
    try:
        response = requests.get(f"{API_BASE_URL}/search/{query}", timeout=5)
        if response.status_code == 200:
            return response.json().get('results', [])
        return []
    except:
        return []

def send_chat_message(question, symbol):
    """Send message to AI chat API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={'question': question, 'symbol': symbol},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get('answer', 'Sorry, I could not process your question.')
        return "Error connecting to AI chat service."
    except:
        return "Error connecting to AI chat service."

# Header
st.markdown('<h1 class="main-header">🧠 NeuralStock AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #64748b; font-size: 1.2rem;">Next-Gen Stock Price Prediction with LSTM & Attention Mechanism</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Search Stock")
    
    # Stock search with suggestions
    search_query = st.text_input("Enter stock symbol", value=st.session_state.current_symbol, key="search_input")
    
    # Get suggestions
    if len(search_query) >= 1:
        suggestions = search_stocks(search_query)
        if suggestions:
            st.write("**Suggestions:**")
            for suggestion in suggestions[:5]:
                if st.button(f"{suggestion['symbol']} - {suggestion['name']}", key=f"sugg_{suggestion['symbol']}"):
                    st.session_state.current_symbol = suggestion['symbol']
                    st.rerun()
    
    # Load button
    if st.button("🚀 Load Stock Data", type="primary", use_container_width=True):
        st.session_state.stock_data = fetch_stock_data(st.session_state.current_symbol)
        st.rerun()
    
    st.divider()
    
    # Current symbol display
    st.write(f"**Current Symbol:** {st.session_state.current_symbol}")
    
    # AI Chat Section
    st.header("💬 AI Assistant")
    st.write("Ask me anything about stocks, predictions, or technical indicators!")
    
    # Chat input
    chat_input = st.text_input("Your question:", key="chat_input", placeholder="e.g., What is RSI?")
    
    if st.button("Send", key="send_chat", use_container_width=True) and chat_input:
        with st.spinner("AI is thinking..."):
            answer = send_chat_message(chat_input, st.session_state.current_symbol)
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()
    
    # Chat history
    if st.session_state.chat_history:
        st.divider()
        st.write("**Chat History:**")
        for i, msg in enumerate(st.session_state.chat_history[-6:]):  # Show last 6 messages
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message chat-ai">{msg["content"]}</div>', unsafe_allow_html=True)

# Main content
if st.session_state.stock_data:
    data = st.session_state.stock_data
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${data['current_price']:.2f}",
            f"{data['change']:+.2f}%",
            delta_color="normal" if data['change'] >= 0 else "inverse"
        )
    
    with col2:
        forecast_price = data['predictions'][-1]['price'] if data['predictions'] else data['current_price']
        forecast_change = data['predictions'][-1]['change'] if data['predictions'] else 0
        st.metric(
            "7-Day Forecast",
            f"${forecast_price:.2f}",
            f"{forecast_change:+.2f}%",
            delta_color="normal" if forecast_change >= 0 else "inverse"
        )
    
    with col3:
        st.metric("RMSE Accuracy", f"{data['rmse']:.2f}%")
    
    with col4:
        st.metric("AI Confidence", f"{data['confidence']:.0f}%")
    
    st.divider()
    
    # Main chart
    st.subheader(f"📊 {data['symbol']} - Price Chart & Predictions")
    
    # Prepare data for chart
    historical_df = pd.DataFrame(data['historical_data'])
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    
    predictions_df = pd.DataFrame(data['predictions'])
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    # Create interactive chart
    fig = go.Figure()
    
    # Historical price line
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['close'],
        name='Historical Price',
        line=dict(color='#6366f1', width=3),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(99, 102, 241, 0.1)'
    ))
    
    # Forecast line
    if len(predictions_df) > 0:
        # Connect historical to forecast
        last_historical_date = historical_df['date'].iloc[-1]
        first_prediction_price = predictions_df['price'].iloc[0]
        last_historical_price = historical_df['close'].iloc[-1]
        
        fig.add_trace(go.Scatter(
            x=[last_historical_date, predictions_df['date'].iloc[0]],
            y=[last_historical_price, first_prediction_price],
            name='Forecast Connection',
            line=dict(color='#10b981', width=2, dash='dot'),
            mode='lines',
            showlegend=False
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['price'],
            name='AI Forecast (LSTM)',
            line=dict(color='#10b981', width=3, dash='dash'),
            mode='lines+markers',
            marker=dict(size=8, symbol='diamond')
        ))
    
    fig.update_layout(
        title=f"{data['symbol']} Price Chart with 7-Day AI Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Two columns for indicators and predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Technical Indicators")
        indicators = data['indicators']
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("RSI (14)", f"{indicators['rsi']:.2f}")
            st.metric("MACD", f"{indicators['macd']:.2f}")
            st.metric("BB Upper", f"${indicators['bb_upper']:.2f}")
            st.metric("BB Middle", f"${indicators['bb_middle']:.2f}")
        
        with metrics_col2:
            st.metric("BB Lower", f"${indicators['bb_lower']:.2f}")
            st.metric("MA 50", f"${indicators['ma_50']:.2f}")
            st.metric("MA 200", f"${indicators['ma_200']:.2f}")
            st.metric("Volume", f"{indicators['volume']:,.0f}")
    
    with col2:
        st.subheader("🔮 7-Day Price Forecast")
        predictions_df_display = predictions_df.copy()
        predictions_df_display['Price'] = predictions_df_display['price'].apply(lambda x: f"${x:.2f}")
        predictions_df_display['Change'] = predictions_df_display['change'].apply(lambda x: f"{x:+.2f}%")
        predictions_df_display['Date'] = predictions_df_display['date'].dt.strftime('%b %d, %Y')
        predictions_df_display = predictions_df_display[['Date', 'Price', 'Change']]
        
        # Style the dataframe
        st.dataframe(
            predictions_df_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Show company name if available
        if 'company_name' in data:
            st.caption(f"📌 {data['company_name']}")

else:
    # Welcome screen
    st.info("👆 **Use the sidebar to search and load a stock symbol**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - Real-time stock data
        - AI-powered predictions
        - 15+ technical indicators
        - 7-day price forecast
        - AI chat assistant
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 AI Model
        - LSTM with Attention
        - 2.3% RMSE accuracy
        - Deep learning powered
        - Real-time training
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Indicators
        - RSI, MACD
        - Bollinger Bands
        - Moving Averages
        - Volume analysis
        - Volatility metrics
        """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #94a3b8; padding: 2rem;">
    <p>Powered by LSTM Neural Networks with Attention Mechanism | RMSE: 2.3%</p>
    <p>Made with ❤️ using Streamlit & Flask</p>
</div>
""", unsafe_allow_html=True)
