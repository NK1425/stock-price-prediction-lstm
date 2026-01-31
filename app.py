"""
Stock Price Prediction with LSTM and Attention - Modern Streamlit Interface

A futuristic, Apple-inspired deep learning application for predicting stock prices
using LSTM networks with attention mechanisms.
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="StockAI | Intelligent Price Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODERN CSS STYLING - Apple-inspired Design
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #0f172a;
        --dark-light: #1e293b;
        --gray: #64748b;
        --light: #f1f5f9;
        --white: #ffffff;
    }

    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: gradientShift 8s ease infinite;
        background-size: 200% 200%;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 20px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255,255,255,0.9);
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--dark) 0%, var(--dark-light) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transition: all 0.5s ease;
    }

    .metric-card:hover::after {
        transform: scale(1.5);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 0.875rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-change {
        font-size: 0.875rem;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }

    .metric-change.positive {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }

    .metric-change.negative {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--dark);
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .section-header .icon {
        font-size: 1.5rem;
    }

    /* Info badges */
    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        font-size: 0.875rem;
        color: #0369a1;
        border: 1px solid #bae6fd;
        transition: all 0.2s ease;
    }

    .info-badge:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.15);
    }

    /* Tooltip styling */
    .tooltip-container {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip-text {
        visibility: hidden;
        width: 250px;
        background: var(--dark);
        color: white;
        text-align: left;
        border-radius: 12px;
        padding: 1rem;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: all 0.3s ease;
        font-size: 0.8rem;
        line-height: 1.5;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }

    .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--light);
        padding: 0.5rem;
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        background-size: 200% 100%;
        animation: progressShimmer 2s ease infinite;
    }

    @keyframes progressShimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Stock info card */
    .stock-info-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }

    .stock-name {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .stock-ticker {
        font-size: 1rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .stock-price-large {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    /* Real-time indicator */
    .realtime-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
    }

    .prediction-card.bearish {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    .prediction-card.neutral {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    /* Abbreviation helper */
    .abbr-helper {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 6px;
        font-size: 0.75rem;
        color: var(--primary);
        cursor: help;
        transition: all 0.2s ease;
    }

    .abbr-helper:hover {
        background: rgba(99, 102, 241, 0.2);
    }

    /* Smooth animations */
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .slide-in {
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 10px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--light);
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# STOCK DATABASE - Comprehensive list
# ============================================================================
STOCK_DATABASE = {
    # Tech Giants
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software"},
    "GOOGL": {"name": "Alphabet Inc. (Google)", "sector": "Technology", "industry": "Internet Services"},
    "GOOG": {"name": "Alphabet Inc. Class C", "sector": "Technology", "industry": "Internet Services"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "industry": "E-Commerce"},
    "META": {"name": "Meta Platforms Inc.", "sector": "Technology", "industry": "Social Media"},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Cyclical", "industry": "Electric Vehicles"},
    "AMD": {"name": "Advanced Micro Devices", "sector": "Technology", "industry": "Semiconductors"},
    "INTC": {"name": "Intel Corporation", "sector": "Technology", "industry": "Semiconductors"},
    "CRM": {"name": "Salesforce Inc.", "sector": "Technology", "industry": "Cloud Software"},
    "ORCL": {"name": "Oracle Corporation", "sector": "Technology", "industry": "Enterprise Software"},
    "ADBE": {"name": "Adobe Inc.", "sector": "Technology", "industry": "Software"},
    "CSCO": {"name": "Cisco Systems Inc.", "sector": "Technology", "industry": "Networking"},
    "AVGO": {"name": "Broadcom Inc.", "sector": "Technology", "industry": "Semiconductors"},
    "QCOM": {"name": "Qualcomm Inc.", "sector": "Technology", "industry": "Semiconductors"},
    "TXN": {"name": "Texas Instruments", "sector": "Technology", "industry": "Semiconductors"},
    "IBM": {"name": "IBM Corporation", "sector": "Technology", "industry": "IT Services"},
    "NOW": {"name": "ServiceNow Inc.", "sector": "Technology", "industry": "Cloud Software"},
    "SHOP": {"name": "Shopify Inc.", "sector": "Technology", "industry": "E-Commerce"},
    "SQ": {"name": "Block Inc. (Square)", "sector": "Technology", "industry": "Fintech"},
    "PYPL": {"name": "PayPal Holdings", "sector": "Technology", "industry": "Fintech"},
    "UBER": {"name": "Uber Technologies", "sector": "Technology", "industry": "Ride-Sharing"},
    "LYFT": {"name": "Lyft Inc.", "sector": "Technology", "industry": "Ride-Sharing"},
    "SNAP": {"name": "Snap Inc.", "sector": "Technology", "industry": "Social Media"},
    "PINS": {"name": "Pinterest Inc.", "sector": "Technology", "industry": "Social Media"},
    "TWTR": {"name": "Twitter Inc.", "sector": "Technology", "industry": "Social Media"},
    "SPOT": {"name": "Spotify Technology", "sector": "Technology", "industry": "Streaming"},
    "NFLX": {"name": "Netflix Inc.", "sector": "Communication", "industry": "Streaming"},
    "DIS": {"name": "Walt Disney Company", "sector": "Communication", "industry": "Entertainment"},

    # Finance
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial", "industry": "Banking"},
    "BAC": {"name": "Bank of America", "sector": "Financial", "industry": "Banking"},
    "WFC": {"name": "Wells Fargo & Co.", "sector": "Financial", "industry": "Banking"},
    "GS": {"name": "Goldman Sachs", "sector": "Financial", "industry": "Investment Banking"},
    "MS": {"name": "Morgan Stanley", "sector": "Financial", "industry": "Investment Banking"},
    "C": {"name": "Citigroup Inc.", "sector": "Financial", "industry": "Banking"},
    "V": {"name": "Visa Inc.", "sector": "Financial", "industry": "Payment Processing"},
    "MA": {"name": "Mastercard Inc.", "sector": "Financial", "industry": "Payment Processing"},
    "AXP": {"name": "American Express", "sector": "Financial", "industry": "Credit Services"},
    "BLK": {"name": "BlackRock Inc.", "sector": "Financial", "industry": "Asset Management"},
    "SCHW": {"name": "Charles Schwab", "sector": "Financial", "industry": "Brokerage"},
    "COF": {"name": "Capital One Financial", "sector": "Financial", "industry": "Banking"},

    # Healthcare
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "UNH": {"name": "UnitedHealth Group", "sector": "Healthcare", "industry": "Health Insurance"},
    "PFE": {"name": "Pfizer Inc.", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "ABBV": {"name": "AbbVie Inc.", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "MRK": {"name": "Merck & Co.", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "LLY": {"name": "Eli Lilly & Co.", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "TMO": {"name": "Thermo Fisher Scientific", "sector": "Healthcare", "industry": "Life Sciences"},
    "ABT": {"name": "Abbott Laboratories", "sector": "Healthcare", "industry": "Medical Devices"},
    "DHR": {"name": "Danaher Corporation", "sector": "Healthcare", "industry": "Life Sciences"},
    "BMY": {"name": "Bristol-Myers Squibb", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "AMGN": {"name": "Amgen Inc.", "sector": "Healthcare", "industry": "Biotechnology"},
    "GILD": {"name": "Gilead Sciences", "sector": "Healthcare", "industry": "Biotechnology"},
    "MRNA": {"name": "Moderna Inc.", "sector": "Healthcare", "industry": "Biotechnology"},
    "BIIB": {"name": "Biogen Inc.", "sector": "Healthcare", "industry": "Biotechnology"},

    # Consumer
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer Defensive", "industry": "Retail"},
    "PG": {"name": "Procter & Gamble", "sector": "Consumer Defensive", "industry": "Consumer Goods"},
    "KO": {"name": "Coca-Cola Company", "sector": "Consumer Defensive", "industry": "Beverages"},
    "PEP": {"name": "PepsiCo Inc.", "sector": "Consumer Defensive", "industry": "Beverages"},
    "COST": {"name": "Costco Wholesale", "sector": "Consumer Defensive", "industry": "Retail"},
    "HD": {"name": "Home Depot Inc.", "sector": "Consumer Cyclical", "industry": "Home Improvement"},
    "LOW": {"name": "Lowe's Companies", "sector": "Consumer Cyclical", "industry": "Home Improvement"},
    "MCD": {"name": "McDonald's Corporation", "sector": "Consumer Cyclical", "industry": "Restaurants"},
    "SBUX": {"name": "Starbucks Corporation", "sector": "Consumer Cyclical", "industry": "Restaurants"},
    "NKE": {"name": "Nike Inc.", "sector": "Consumer Cyclical", "industry": "Apparel"},
    "TGT": {"name": "Target Corporation", "sector": "Consumer Defensive", "industry": "Retail"},
    "CVS": {"name": "CVS Health Corporation", "sector": "Healthcare", "industry": "Pharmacy"},
    "WBA": {"name": "Walgreens Boots Alliance", "sector": "Healthcare", "industry": "Pharmacy"},

    # Energy
    "XOM": {"name": "Exxon Mobil Corporation", "sector": "Energy", "industry": "Oil & Gas"},
    "CVX": {"name": "Chevron Corporation", "sector": "Energy", "industry": "Oil & Gas"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy", "industry": "Oil & Gas"},
    "SLB": {"name": "Schlumberger Limited", "sector": "Energy", "industry": "Oil Services"},
    "EOG": {"name": "EOG Resources", "sector": "Energy", "industry": "Oil & Gas"},
    "OXY": {"name": "Occidental Petroleum", "sector": "Energy", "industry": "Oil & Gas"},

    # Industrial
    "BA": {"name": "Boeing Company", "sector": "Industrial", "industry": "Aerospace"},
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrial", "industry": "Machinery"},
    "HON": {"name": "Honeywell International", "sector": "Industrial", "industry": "Conglomerate"},
    "UPS": {"name": "United Parcel Service", "sector": "Industrial", "industry": "Logistics"},
    "FDX": {"name": "FedEx Corporation", "sector": "Industrial", "industry": "Logistics"},
    "GE": {"name": "General Electric", "sector": "Industrial", "industry": "Conglomerate"},
    "MMM": {"name": "3M Company", "sector": "Industrial", "industry": "Conglomerate"},
    "LMT": {"name": "Lockheed Martin", "sector": "Industrial", "industry": "Defense"},
    "RTX": {"name": "Raytheon Technologies", "sector": "Industrial", "industry": "Defense"},
    "DE": {"name": "Deere & Company", "sector": "Industrial", "industry": "Machinery"},

    # Real Estate
    "AMT": {"name": "American Tower Corp", "sector": "Real Estate", "industry": "REITs"},
    "PLD": {"name": "Prologis Inc.", "sector": "Real Estate", "industry": "REITs"},
    "CCI": {"name": "Crown Castle Inc.", "sector": "Real Estate", "industry": "REITs"},
    "EQIX": {"name": "Equinix Inc.", "sector": "Real Estate", "industry": "Data Centers"},

    # Utilities
    "NEE": {"name": "NextEra Energy", "sector": "Utilities", "industry": "Electric Utilities"},
    "DUK": {"name": "Duke Energy Corp", "sector": "Utilities", "industry": "Electric Utilities"},
    "SO": {"name": "Southern Company", "sector": "Utilities", "industry": "Electric Utilities"},

    # Telecom
    "T": {"name": "AT&T Inc.", "sector": "Communication", "industry": "Telecom"},
    "VZ": {"name": "Verizon Communications", "sector": "Communication", "industry": "Telecom"},
    "TMUS": {"name": "T-Mobile US Inc.", "sector": "Communication", "industry": "Telecom"},

    # ETFs & Indices
    "SPY": {"name": "S&P 500 ETF", "sector": "ETF", "industry": "Index Fund"},
    "QQQ": {"name": "Nasdaq 100 ETF", "sector": "ETF", "industry": "Index Fund"},
    "DIA": {"name": "Dow Jones ETF", "sector": "ETF", "industry": "Index Fund"},
    "IWM": {"name": "Russell 2000 ETF", "sector": "ETF", "industry": "Index Fund"},
    "VOO": {"name": "Vanguard S&P 500 ETF", "sector": "ETF", "industry": "Index Fund"},
    "VTI": {"name": "Vanguard Total Market ETF", "sector": "ETF", "industry": "Index Fund"},
    "ARKK": {"name": "ARK Innovation ETF", "sector": "ETF", "industry": "Thematic"},
    "XLK": {"name": "Technology Select ETF", "sector": "ETF", "industry": "Sector"},
    "XLF": {"name": "Financial Select ETF", "sector": "ETF", "industry": "Sector"},
    "XLE": {"name": "Energy Select ETF", "sector": "ETF", "industry": "Sector"},

    # Crypto-related
    "COIN": {"name": "Coinbase Global", "sector": "Financial", "industry": "Cryptocurrency"},
    "MARA": {"name": "Marathon Digital", "sector": "Financial", "industry": "Crypto Mining"},
    "RIOT": {"name": "Riot Platforms", "sector": "Financial", "industry": "Crypto Mining"},
    "MSTR": {"name": "MicroStrategy Inc.", "sector": "Technology", "industry": "Bitcoin Holdings"},
}

# Technical Indicator Explanations
INDICATOR_EXPLANATIONS = {
    "RSI": {
        "full_name": "Relative Strength Index",
        "description": "Momentum oscillator measuring speed and magnitude of price movements. Values above 70 indicate overbought conditions, below 30 indicate oversold.",
        "range": "0-100"
    },
    "MACD": {
        "full_name": "Moving Average Convergence Divergence",
        "description": "Trend-following momentum indicator showing relationship between two moving averages. Bullish when MACD crosses above signal line.",
        "range": "Varies"
    },
    "SMA": {
        "full_name": "Simple Moving Average",
        "description": "Average price over a specified period. Helps identify trend direction and support/resistance levels.",
        "range": "Price-based"
    },
    "EMA": {
        "full_name": "Exponential Moving Average",
        "description": "Weighted moving average giving more importance to recent prices. More responsive to new information than SMA.",
        "range": "Price-based"
    },
    "BB": {
        "full_name": "Bollinger Bands",
        "description": "Volatility bands placed above and below a moving average. Price touching upper band may indicate overbought, lower band oversold.",
        "range": "Price-based"
    },
    "ATR": {
        "full_name": "Average True Range",
        "description": "Measures market volatility by decomposing the entire range of an asset price. Higher ATR indicates higher volatility.",
        "range": "Price-based"
    },
    "OBV": {
        "full_name": "On-Balance Volume",
        "description": "Uses volume flow to predict changes in stock price. Rising OBV indicates buying pressure.",
        "range": "Volume-based"
    },
    "VWAP": {
        "full_name": "Volume Weighted Average Price",
        "description": "Average price weighted by volume. Institutional traders use it as a benchmark for trade execution.",
        "range": "Price-based"
    },
    "MFI": {
        "full_name": "Money Flow Index",
        "description": "Volume-weighted RSI. Incorporates both price and volume to measure buying/selling pressure.",
        "range": "0-100"
    },
    "CCI": {
        "full_name": "Commodity Channel Index",
        "description": "Measures current price relative to average price over a period. Values above +100 suggest overbought, below -100 oversold.",
        "range": "-300 to +300"
    },
    "Stoch": {
        "full_name": "Stochastic Oscillator",
        "description": "Compares closing price to price range over a period. Shows momentum by measuring position of close relative to high-low range.",
        "range": "0-100"
    },
    "%R": {
        "full_name": "Williams %R",
        "description": "Momentum indicator measuring overbought/oversold levels. Similar to Stochastic but inverted scale.",
        "range": "-100 to 0"
    },
    "ROC": {
        "full_name": "Rate of Change",
        "description": "Measures percentage change in price over a specified period. Positive values indicate upward momentum.",
        "range": "Percentage"
    }
}

# Time period options
TIME_PERIODS = {
    "1W": {"days": 7, "label": "1 Week"},
    "1M": {"days": 30, "label": "1 Month"},
    "3M": {"days": 90, "label": "3 Months"},
    "6M": {"days": 180, "label": "6 Months"},
    "1Y": {"days": 365, "label": "1 Year"},
    "2Y": {"days": 730, "label": "2 Years"},
    "5Y": {"days": 1825, "label": "5 Years"},
    "MAX": {"days": 3650, "label": "Max"}
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def search_stocks(query: str) -> List[Dict]:
    """Search stocks by ticker or name."""
    query = query.upper().strip()
    results = []

    for ticker, info in STOCK_DATABASE.items():
        if (query in ticker or
            query.lower() in info["name"].lower() or
            query.lower() in info["sector"].lower() or
            query.lower() in info["industry"].lower()):
            results.append({"ticker": ticker, **info})

    # Sort by relevance (exact ticker match first)
    results.sort(key=lambda x: (0 if x["ticker"] == query else 1, x["ticker"]))
    return results[:20]


@st.cache_data(ttl=60)
def get_realtime_price(ticker: str) -> Dict:
    """Get real-time stock price and info."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2d")

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        return {
            "price": current_price,
            "change": change,
            "change_pct": change_pct,
            "volume": hist['Volume'].iloc[-1],
            "high": hist['High'].iloc[-1],
            "low": hist['Low'].iloc[-1],
            "open": hist['Open'].iloc[-1],
            "prev_close": prev_close,
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "eps": info.get("trailingEps", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "beta": info.get("beta", 0),
            "description": info.get("longBusinessSummary", ""),
            "website": info.get("website", ""),
            "employees": info.get("fullTimeEmployees", 0),
            "headquarters": f"{info.get('city', '')}, {info.get('state', '')}",
        }
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def load_stock_data(ticker: str, days: int) -> pd.DataFrame:
    """Load stock data for specified period."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'))

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        return data
    except:
        return pd.DataFrame()


def create_price_chart(data: pd.DataFrame, ticker: str, chart_type: str = "area") -> go.Figure:
    """Create interactive price chart."""
    fig = go.Figure()

    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker,
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))
    elif chart_type == "area":
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=ticker,
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
    else:  # line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=ticker,
            line=dict(color='#6366f1', width=2)
        ))

    fig.update_layout(
        template='plotly_white',
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            side='right'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Inter'
        )
    )

    return fig


def create_volume_chart(data: pd.DataFrame) -> go.Figure:
    """Create volume bar chart."""
    colors = ['#10b981' if data['Close'].iloc[i] >= data['Open'].iloc[i]
              else '#ef4444' for i in range(len(data))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        opacity=0.7,
        name='Volume'
    ))

    fig.update_layout(
        template='plotly_white',
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', side='right'),
        showlegend=False
    )

    return fig


def create_technical_chart(data: pd.DataFrame, indicators: List[str]) -> go.Figure:
    """Create multi-indicator technical chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25]
    )

    # Price with MAs
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        name='Price', line=dict(color='#1e293b', width=2)
    ), row=1, col=1)

    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'],
            name='SMA 20', line=dict(color='#6366f1', width=1, dash='dot')
        ), row=1, col=1)

    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_50'],
            name='SMA 50', line=dict(color='#8b5cf6', width=1, dash='dot')
        ), row=1, col=1)

    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'],
            name='BB Upper', line=dict(color='#94a3b8', width=1),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'],
            name='BB Lower', line=dict(color='#94a3b8', width=1),
            fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)',
            showlegend=False
        ), row=1, col=1)

    # RSI
    if 'RSI_14' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI_14'],
            name='RSI', line=dict(color='#8b5cf6', width=2)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'],
            name='MACD', line=dict(color='#6366f1', width=2)
        ), row=3, col=1)
        if 'MACD_Signal' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MACD_Signal'],
                name='Signal', line=dict(color='#f59e0b', width=2)
            ), row=3, col=1)
        if 'MACD_Histogram' in data.columns:
            colors = ['#10b981' if v >= 0 else '#ef4444' for v in data['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=data.index, y=data['MACD_Histogram'],
                name='Histogram', marker_color=colors, opacity=0.5
            ), row=3, col=1)

    fig.update_layout(
        template='plotly_white',
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, side='right')
    fig.update_yaxes(title_text="RSI", row=2, col=1, side='right')
    fig.update_yaxes(title_text="MACD", row=3, col=1, side='right')

    return fig


def create_prediction_chart(historical: np.ndarray, predicted: np.ndarray,
                           dates: List, forecast_dates: List) -> go.Figure:
    """Create prediction visualization."""
    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=historical,
        mode='lines',
        name='Historical',
        line=dict(color='#1e293b', width=2)
    ))

    # Predicted prices
    all_pred_dates = [dates[-1]] + forecast_dates
    all_pred_values = [historical[-1]] + list(predicted)

    fig.add_trace(go.Scatter(
        x=all_pred_dates,
        y=all_pred_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#6366f1', width=3, dash='dash'),
        marker=dict(size=10, symbol='circle')
    ))

    # Confidence band
    std = np.std(historical[-30:]) if len(historical) >= 30 else np.std(historical)
    upper = [historical[-1]] + [p + std * (i + 1) * 0.3 for i, p in enumerate(predicted)]
    lower = [historical[-1]] + [p - std * (i + 1) * 0.3 for i, p in enumerate(predicted)]

    fig.add_trace(go.Scatter(
        x=all_pred_dates + all_pred_dates[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band',
        showlegend=True
    ))

    # Add vertical line at prediction start
    fig.add_vline(x=dates[-1], line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        template='plotly_white',
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )

    return fig


def create_attention_chart(attention_weights: np.ndarray, seq_length: int) -> go.Figure:
    """Create attention weights visualization."""
    avg_attention = np.mean(attention_weights, axis=0)
    days_ago = list(range(seq_length, 0, -1))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=days_ago,
        y=avg_attention,
        marker=dict(
            color=avg_attention,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Weight')
        ),
        hovertemplate='%{x} days ago<br>Weight: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        template='plotly_white',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(title='Days Ago', showgrid=False),
        yaxis=dict(title='Attention Weight', showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )

    return fig


def format_number(num: float, prefix: str = "", suffix: str = "") -> str:
    """Format large numbers with K, M, B suffixes."""
    if num is None or num == 0:
        return "N/A"

    if abs(num) >= 1e12:
        return f"{prefix}{num/1e12:.2f}T{suffix}"
    elif abs(num) >= 1e9:
        return f"{prefix}{num/1e9:.2f}B{suffix}"
    elif abs(num) >= 1e6:
        return f"{prefix}{num/1e6:.2f}M{suffix}"
    elif abs(num) >= 1e3:
        return f"{prefix}{num/1e3:.2f}K{suffix}"
    else:
        return f"{prefix}{num:.2f}{suffix}"


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


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🚀 StockAI</div>
        <div class="hero-subtitle">Intelligent Stock Price Prediction powered by LSTM & Attention Mechanism</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Stock Search")

        # Search input
        search_query = st.text_input(
            "Search stocks",
            placeholder="Enter ticker or company name...",
            help="Search by ticker symbol (e.g., AAPL) or company name (e.g., Apple)"
        )

        # Show search results
        if search_query:
            results = search_stocks(search_query)
            if results:
                st.markdown("**Search Results:**")
                options = [f"{r['ticker']} - {r['name']}" for r in results]
                selected = st.selectbox("Select a stock", options, label_visibility="collapsed")
                ticker = selected.split(" - ")[0] if selected else "AAPL"
            else:
                st.warning("No stocks found. Try a different search term.")
                ticker = "AAPL"
        else:
            # Popular stocks
            st.markdown("**Popular Stocks:**")
            popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "SPY"]
            ticker = st.selectbox("Select a stock", popular, label_visibility="collapsed")

        st.markdown("---")

        # Time period selector
        st.markdown("### 📅 Time Period")
        period_cols = st.columns(4)
        selected_period = "1Y"

        for i, (key, period) in enumerate(TIME_PERIODS.items()):
            col_idx = i % 4
            with period_cols[col_idx]:
                if st.button(key, key=f"period_{key}", use_container_width=True):
                    selected_period = key

        # Store selected period in session state
        if 'selected_period' not in st.session_state:
            st.session_state.selected_period = "1Y"

        period_options = list(TIME_PERIODS.keys())
        selected_period = st.radio(
            "Select period",
            period_options,
            index=period_options.index(st.session_state.selected_period),
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.selected_period = selected_period

        st.markdown("---")

        # Chart type
        st.markdown("### 📊 Chart Settings")
        chart_type = st.selectbox(
            "Chart Type",
            ["area", "candlestick", "line"],
            format_func=lambda x: {"area": "Area Chart", "candlestick": "Candlestick", "line": "Line Chart"}[x]
        )

        show_volume = st.checkbox("Show Volume", value=True)
        show_indicators = st.checkbox("Show Technical Indicators", value=False)

        st.markdown("---")

        # Prediction settings
        st.markdown("### 🔮 Prediction Settings")
        forecast_horizon = st.slider("Forecast Days", 1, 14, 7)
        sequence_length = st.slider("Lookback Period", 30, 120, 60)

        run_prediction = st.button("🚀 Run AI Prediction", type="primary", use_container_width=True)

    # Main content
    # Get stock info
    stock_info = STOCK_DATABASE.get(ticker, {"name": ticker, "sector": "Unknown", "industry": "Unknown"})
    realtime_data = get_realtime_price(ticker)

    # Auto-generate prediction for selected stock
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_stock_prediction(ticker: str, forecast_days: int = 7):
        """Get cached prediction for a stock."""
        try:
            import tensorflow as tf
            tf.random.set_seed(42)
            np.random.seed(42)

            from feature_engineering import FeatureEngineer
            from sequence_builder import SequenceBuilder, create_train_val_test_sequences, create_single_prediction_sequence
            from model import build_attention_model_with_weights

            # Load 3 years of data
            train_data = load_stock_data(ticker, 1095)
            if train_data.empty or len(train_data) < 200:
                return None

            engineer = FeatureEngineer()
            data_with_features = engineer.add_all_features(train_data.copy())
            data_clean = engineer.handle_missing_values(data_with_features, method='ffill')

            feature_cols = get_feature_columns()
            feature_cols = [c for c in feature_cols if c in data_clean.columns]

            seq_length = 60
            builder = SequenceBuilder(
                sequence_length=seq_length,
                forecast_horizon=forecast_days,
                feature_columns=feature_cols,
                target_column='Close'
            )

            (X_train, y_train), (X_val, y_val), _, _ = \
                create_train_val_test_sequences(data_clean, builder, 0.7, 0.15, 0.15)

            n_features = X_train.shape[2]
            model, attention_model = build_attention_model_with_weights(
                input_shape=(seq_length, n_features),
                output_steps=forecast_days,
                lstm_units=[64, 32],
                attention_units=32,
                dropout_rate=0.2,
                learning_rate=0.001
            )

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=8, restore_best_weights=True
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            latest_sequence = create_single_prediction_sequence(data_clean, builder)
            future_pred, attention = attention_model.predict(latest_sequence, verbose=0)
            future_prices = builder.inverse_transform_predictions(future_pred)[0]

            last_price = data_clean['Close'].iloc[-1]
            last_date = data_clean.index[-1]

            return {
                'prices': future_prices,
                'last_price': last_price,
                'last_date': last_date,
                'attention': attention,
                'forecast_days': forecast_days
            }
        except Exception as e:
            return None

    if realtime_data:
        # Stock Header Card
        col1, col2 = st.columns([2, 1])

        with col1:
            change_class = "positive" if realtime_data['change'] >= 0 else "negative"
            change_icon = "▲" if realtime_data['change'] >= 0 else "▼"

            st.markdown(f"""
            <div class="stock-info-card fade-in">
                <div class="stock-ticker">{ticker}</div>
                <div class="stock-name">{stock_info['name']}</div>
                <div class="stock-price-large">${realtime_data['price']:.2f}</div>
                <span class="metric-change {change_class}">
                    {change_icon} ${abs(realtime_data['change']):.2f} ({realtime_data['change_pct']:+.2f}%)
                </span>
                <div style="margin-top: 1rem; color: rgba(255,255,255,0.6); font-size: 0.875rem;">
                    <span class="realtime-dot"></span> Real-time data
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("##### Quick Stats")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Open", f"${realtime_data['open']:.2f}")
                st.metric("High", f"${realtime_data['high']:.2f}")
                st.metric("Low", f"${realtime_data['low']:.2f}")
            with stats_col2:
                st.metric("Volume", format_number(realtime_data['volume']))
                st.metric("52W High", f"${realtime_data['52w_high']:.2f}")
                st.metric("52W Low", f"${realtime_data['52w_low']:.2f}")

    # ==================== PREDICTION SECTION ====================
    st.markdown("---")
    st.markdown("## 🔮 AI Price Prediction")

    with st.spinner(f"🧠 Generating AI prediction for {ticker}..."):
        prediction = get_stock_prediction(ticker, forecast_horizon)

    if prediction:
        last_price = prediction['last_price']
        future_prices = prediction['prices']
        tomorrow_price = future_prices[0]
        week_price = future_prices[-1]

        tomorrow_change = (tomorrow_price - last_price) / last_price * 100
        week_change = (week_price - last_price) / last_price * 100

        # Determine trend
        if week_change > 2:
            trend_color = "#10b981"
            trend_bg = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
            trend_text = "BULLISH"
            trend_icon = "📈"
        elif week_change < -2:
            trend_color = "#ef4444"
            trend_bg = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
            trend_text = "BEARISH"
            trend_icon = "📉"
        else:
            trend_color = "#f59e0b"
            trend_bg = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
            trend_text = "NEUTRAL"
            trend_icon = "➡️"

        # Main prediction cards
        pred_cols = st.columns([1, 1, 1, 1])

        with pred_cols[0]:
            st.markdown(f"""
            <div style="background: {trend_bg}; border-radius: 16px; padding: 1.5rem; color: white; text-align: center;">
                <div style="font-size: 0.8rem; opacity: 0.9; margin-bottom: 0.5rem;">AI SIGNAL</div>
                <div style="font-size: 2rem; margin-bottom: 0.25rem;">{trend_icon}</div>
                <div style="font-size: 1.25rem; font-weight: 700;">{trend_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with pred_cols[1]:
            tomorrow_icon = "▲" if tomorrow_change >= 0 else "▼"
            tomorrow_color = "#10b981" if tomorrow_change >= 0 else "#ef4444"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.5rem; color: white;">
                <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.5rem;">TOMORROW'S PRICE</div>
                <div style="font-size: 1.75rem; font-weight: 700;">${tomorrow_price:.2f}</div>
                <div style="color: {tomorrow_color}; font-size: 0.9rem; margin-top: 0.5rem;">
                    {tomorrow_icon} {tomorrow_change:+.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with pred_cols[2]:
            week_icon = "▲" if week_change >= 0 else "▼"
            week_color = "#10b981" if week_change >= 0 else "#ef4444"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.5rem; color: white;">
                <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.5rem;">{forecast_horizon}-DAY FORECAST</div>
                <div style="font-size: 1.75rem; font-weight: 700;">${week_price:.2f}</div>
                <div style="color: {week_color}; font-size: 0.9rem; margin-top: 0.5rem;">
                    {week_icon} {week_change:+.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with pred_cols[3]:
            avg_price = np.mean(future_prices)
            avg_change = (avg_price - last_price) / last_price * 100
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 16px; padding: 1.5rem; color: white;">
                <div style="font-size: 0.8rem; opacity: 0.9; margin-bottom: 0.5rem;">AVG PREDICTED</div>
                <div style="font-size: 1.75rem; font-weight: 700;">${avg_price:.2f}</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                    {avg_change:+.2f}% expected
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Detailed daily predictions
        st.markdown("##### 📅 Daily Price Forecast")

        forecast_dates = pd.date_range(
            start=prediction['last_date'] + timedelta(days=1),
            periods=forecast_horizon,
            freq='B'
        )

        # Create forecast table with styling
        forecast_data = []
        for i, (date, price) in enumerate(zip(forecast_dates, future_prices)):
            change = (price - last_price) / last_price * 100
            day_label = "Tomorrow" if i == 0 else date.strftime('%a, %b %d')
            forecast_data.append({
                "Day": day_label,
                "Date": date.strftime('%Y-%m-%d'),
                "Predicted Price": f"${price:.2f}",
                "Change from Today": f"{change:+.2f}%",
                "Direction": "🟢 Up" if change > 0 else "🔴 Down" if change < 0 else "🟡 Flat"
            })

        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # Mini prediction chart
        st.markdown("##### 📊 Forecast Visualization")

        hist_data = load_stock_data(ticker, 60)
        if not hist_data.empty:
            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='#64748b', width=2)
            ))

            # Prediction line
            all_dates = [hist_data.index[-1]] + list(forecast_dates)
            all_prices = [last_price] + list(future_prices)

            fig.add_trace(go.Scatter(
                x=all_dates,
                y=all_prices,
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='#6366f1', width=3, dash='dash'),
                marker=dict(size=8, color='#6366f1')
            ))

            # Add shaded prediction area
            fig.add_vrect(
                x0=hist_data.index[-1],
                x1=forecast_dates[-1],
                fillcolor="rgba(99, 102, 241, 0.1)",
                layer="below",
                line_width=0
            )

            fig.update_layout(
                template='plotly_white',
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', side='right', title='Price ($)'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Prediction confidence note
        st.markdown("""
        <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong>⚠️ Important:</strong> AI predictions are based on historical patterns and technical analysis.
            Stock markets are inherently unpredictable. Use this as one of many tools in your research, not as financial advice.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning(f"Unable to generate prediction for {ticker}. This may be due to insufficient historical data.")

    # Load historical data
    days = TIME_PERIODS[selected_period]["days"]
    data = load_stock_data(ticker, days)

    if data.empty:
        st.error(f"Unable to load data for {ticker}. Please try another stock.")
        return

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Chart", "📊 Technical Analysis", "🔮 AI Prediction", "📋 Company Info", "📚 Learn"
    ])

    with tab1:
        st.markdown(f"### {stock_info['name']} ({ticker}) - {TIME_PERIODS[selected_period]['label']}")

        # Price chart
        fig = create_price_chart(data, ticker, chart_type)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Volume chart
        if show_volume:
            st.markdown("##### Trading Volume")
            vol_fig = create_volume_chart(data)
            st.plotly_chart(vol_fig, use_container_width=True, config={'displayModeBar': False})

        # Key metrics
        st.markdown("##### Key Metrics")
        metric_cols = st.columns(6)

        metrics_data = [
            ("Market Cap", format_number(realtime_data.get('market_cap', 0), "$")),
            ("P/E Ratio", f"{realtime_data.get('pe_ratio', 0):.2f}" if realtime_data.get('pe_ratio') else "N/A"),
            ("EPS", f"${realtime_data.get('eps', 0):.2f}" if realtime_data.get('eps') else "N/A"),
            ("Beta", f"{realtime_data.get('beta', 0):.2f}" if realtime_data.get('beta') else "N/A"),
            ("Avg Volume", format_number(realtime_data.get('avg_volume', 0))),
            ("Dividend", f"{realtime_data.get('dividend_yield', 0)*100:.2f}%" if realtime_data.get('dividend_yield') else "N/A")
        ]

        for col, (label, value) in zip(metric_cols, metrics_data):
            with col:
                st.metric(label, value)

    with tab2:
        st.markdown("### Technical Analysis")

        # Generate technical indicators
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        data_with_indicators = engineer.add_all_features(data.copy())
        data_clean = engineer.handle_missing_values(data_with_indicators, method='ffill')

        # Technical chart
        fig = create_technical_chart(data_clean, ['SMA', 'BB', 'RSI', 'MACD'])
        st.plotly_chart(fig, use_container_width=True)

        # Indicator explanations
        st.markdown("##### Indicator Values & Explanations")

        ind_cols = st.columns(4)

        latest = data_clean.iloc[-1]

        indicators_display = [
            ("RSI (14)", f"{latest.get('RSI_14', 0):.1f}", "RSI"),
            ("MACD", f"{latest.get('MACD', 0):.2f}", "MACD"),
            ("Stochastic %K", f"{latest.get('Stoch_K', 0):.1f}", "Stoch"),
            ("Williams %R", f"{latest.get('Williams_R', 0):.1f}", "%R"),
        ]

        for col, (name, value, key) in zip(ind_cols, indicators_display):
            with col:
                explanation = INDICATOR_EXPLANATIONS.get(key, {})
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.5rem;">
                        {name}
                        <span class="abbr-helper" title="{explanation.get('full_name', '')}">ℹ️</span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{value}</div>
                    <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.5rem;">
                        Range: {explanation.get('range', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Signal summary
        st.markdown("##### Signal Summary")

        rsi = latest.get('RSI_14', 50)
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)

        signals = []
        if rsi > 70:
            signals.append(("RSI", "Overbought", "🔴", "Consider selling"))
        elif rsi < 30:
            signals.append(("RSI", "Oversold", "🟢", "Consider buying"))
        else:
            signals.append(("RSI", "Neutral", "🟡", "Hold"))

        if macd > macd_signal:
            signals.append(("MACD", "Bullish", "🟢", "Upward momentum"))
        else:
            signals.append(("MACD", "Bearish", "🔴", "Downward momentum"))

        sig_cols = st.columns(len(signals))
        for col, (ind, status, icon, desc) in zip(sig_cols, signals):
            with col:
                st.info(f"{icon} **{ind}**: {status}\n\n{desc}")

    with tab3:
        st.markdown("### 🔮 AI-Powered Price Prediction - Deep Analysis")

        st.info("📊 **Quick predictions are shown above!** This tab provides deeper analysis and model insights.")

        if run_prediction and prediction:
            try:
                import tensorflow as tf
                tf.random.set_seed(42)
                np.random.seed(42)

                from sequence_builder import SequenceBuilder, create_train_val_test_sequences
                from model import build_attention_model_with_weights

                progress = st.progress(0)
                status = st.empty()

                # Prepare data
                status.text("📊 Preparing data and generating features...")
                progress.progress(10)

                # Get more historical data for training
                train_data = load_stock_data(ticker, 1095)  # 3 years
                if train_data.empty or len(train_data) < sequence_length + forecast_horizon + 100:
                    st.error("Insufficient data for prediction. Please try a different stock.")
                    return

                data_with_features = engineer.add_all_features(train_data.copy())
                data_clean = engineer.handle_missing_values(data_with_features, method='ffill')

                feature_cols = get_feature_columns()
                feature_cols = [c for c in feature_cols if c in data_clean.columns]

                progress.progress(20)
                status.text("🔧 Building sequences...")

                builder = SequenceBuilder(
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                    feature_columns=feature_cols,
                    target_column='Close'
                )

                (X_train, y_train), (X_val, y_val), (X_test, y_test), test_dates = \
                    create_train_val_test_sequences(data_clean, builder, 0.7, 0.15, 0.15)

                progress.progress(30)
                status.text("🧠 Building LSTM-Attention model...")

                n_features = X_train.shape[2]
                model, attention_model = build_attention_model_with_weights(
                    input_shape=(sequence_length, n_features),
                    output_steps=forecast_horizon,
                    lstm_units=[64, 32],
                    attention_units=32,
                    dropout_rate=0.2,
                    learning_rate=0.001
                )

                progress.progress(40)
                status.text("🏋️ Training model...")

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0
                )

                progress.progress(80)
                status.text("🔮 Generating predictions...")

                # Get predictions
                from sequence_builder import create_single_prediction_sequence
                latest_sequence = create_single_prediction_sequence(data_clean, builder)
                future_pred, future_attention = attention_model.predict(latest_sequence, verbose=0)
                future_prices = builder.inverse_transform_predictions(future_pred)[0]

                # Test predictions for metrics
                test_pred, test_attention = attention_model.predict(X_test, verbose=0)
                test_pred_unscaled = builder.inverse_transform_predictions(test_pred)
                test_actual_unscaled = builder.inverse_transform_predictions(y_test)

                progress.progress(100)
                status.empty()

                # Display results
                last_price = data_clean['Close'].iloc[-1]
                avg_pred = np.mean(future_prices)
                final_pred = future_prices[-1]
                total_change = (final_pred - last_price) / last_price * 100

                # Trend determination
                if total_change > 3:
                    trend = "bullish"
                    trend_text = "Bullish"
                    trend_icon = "🟢"
                elif total_change < -3:
                    trend = "bearish"
                    trend_text = "Bearish"
                    trend_icon = "🔴"
                else:
                    trend = "neutral"
                    trend_text = "Neutral"
                    trend_icon = "🟡"

                # Prediction summary
                st.markdown(f"""
                <div class="prediction-card {trend} fade-in">
                    <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">
                        {trend_icon} {forecast_horizon}-Day Forecast: {trend_text}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 0.875rem; opacity: 0.8;">Expected Price</div>
                            <div style="font-size: 2rem; font-weight: 700;">${final_pred:.2f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.875rem; opacity: 0.8;">Expected Change</div>
                            <div style="font-size: 2rem; font-weight: 700;">{total_change:+.2f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Prediction chart
                st.markdown("##### Price Forecast")
                hist_prices = data_clean['Close'].iloc[-60:].values
                hist_dates = data_clean.index[-60:].tolist()
                forecast_dates = pd.date_range(
                    start=data_clean.index[-1] + timedelta(days=1),
                    periods=forecast_horizon,
                    freq='B'
                ).tolist()

                fig = create_prediction_chart(hist_prices, future_prices, hist_dates, forecast_dates)
                st.plotly_chart(fig, use_container_width=True)

                # Daily predictions table
                st.markdown("##### Daily Forecast Details")
                forecast_df = pd.DataFrame({
                    'Day': [f"Day {i+1}" for i in range(forecast_horizon)],
                    'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                    'Predicted Price': [f"${p:.2f}" for p in future_prices],
                    'Change': [f"{(p - last_price) / last_price * 100:+.2f}%" for p in future_prices]
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                # Model metrics
                rmse = np.sqrt(np.mean((test_pred_unscaled[:, 0] - test_actual_unscaled[:, 0])**2))
                mae = np.mean(np.abs(test_pred_unscaled[:, 0] - test_actual_unscaled[:, 0]))

                st.markdown("##### Model Performance")
                perf_cols = st.columns(3)
                with perf_cols[0]:
                    st.metric("RMSE", f"${rmse:.2f}", help="Root Mean Square Error - lower is better")
                with perf_cols[1]:
                    st.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error - lower is better")
                with perf_cols[2]:
                    accuracy = 100 - (mae / last_price * 100)
                    st.metric("Accuracy", f"{accuracy:.1f}%", help="Approximate prediction accuracy")

                # Attention weights
                st.markdown("##### Attention Analysis")
                st.markdown("*Which historical days influenced the prediction most?*")

                attention_fig = create_attention_chart(future_attention, sequence_length)
                st.plotly_chart(attention_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)
        else:
            st.info("👈 Configure settings in the sidebar and click **Run AI Prediction** to generate forecasts.")

            with st.expander("ℹ️ How does the AI prediction work?"):
                st.markdown("""
                Our prediction model uses:

                1. **LSTM (Long Short-Term Memory)**: A type of neural network designed for sequential data like stock prices
                2. **Attention Mechanism**: Helps the model focus on the most relevant historical patterns
                3. **Technical Indicators**: 15+ indicators including RSI, MACD, Bollinger Bands, etc.
                4. **Walk-Forward Validation**: Ensures the model is tested on unseen future data

                **Note**: Stock predictions are inherently uncertain. This tool is for educational purposes only.
                """)

    with tab4:
        st.markdown(f"### About {stock_info['name']}")

        if realtime_data and realtime_data.get('description'):
            st.markdown(f"""
            <div class="glass-card fade-in">
                <p style="line-height: 1.8; color: #475569;">{realtime_data['description']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Company details
        st.markdown("##### Company Details")
        detail_cols = st.columns(2)

        with detail_cols[0]:
            st.markdown(f"""
            | Attribute | Value |
            |-----------|-------|
            | **Sector** | {stock_info.get('sector', 'N/A')} |
            | **Industry** | {stock_info.get('industry', 'N/A')} |
            | **Headquarters** | {realtime_data.get('headquarters', 'N/A') if realtime_data else 'N/A'} |
            | **Employees** | {format_number(realtime_data.get('employees', 0)) if realtime_data else 'N/A'} |
            """)

        with detail_cols[1]:
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | **Market Cap** | {format_number(realtime_data.get('market_cap', 0), '$') if realtime_data else 'N/A'} |
            | **P/E Ratio** | {realtime_data.get('pe_ratio', 'N/A') if realtime_data else 'N/A'} |
            | **Beta** | {realtime_data.get('beta', 'N/A') if realtime_data else 'N/A'} |
            | **Dividend Yield** | {f"{realtime_data.get('dividend_yield', 0)*100:.2f}%" if realtime_data and realtime_data.get('dividend_yield') else 'N/A'} |
            """)

        if realtime_data and realtime_data.get('website'):
            st.markdown(f"🌐 **Website**: [{realtime_data['website']}]({realtime_data['website']})")

    with tab5:
        st.markdown("### 📚 Understanding Technical Indicators")

        st.markdown("Click on any indicator to learn more about what it means and how to interpret it.")

        for abbr, info in INDICATOR_EXPLANATIONS.items():
            with st.expander(f"**{abbr}** - {info['full_name']}"):
                st.markdown(f"""
                **Description:** {info['description']}

                **Range:** {info['range']}

                **How to use:**
                - Combine with other indicators for confirmation
                - Don't rely on a single indicator
                - Consider the overall market context
                """)

        st.markdown("---")
        st.markdown("### 📖 Glossary")

        glossary = {
            "Bull/Bullish": "Expecting prices to rise; optimistic market sentiment",
            "Bear/Bearish": "Expecting prices to fall; pessimistic market sentiment",
            "Support": "Price level where buying pressure prevents further decline",
            "Resistance": "Price level where selling pressure prevents further rise",
            "Volatility": "Measure of price fluctuation; higher volatility = higher risk",
            "Volume": "Number of shares traded; indicates strength of price movement",
            "Market Cap": "Total market value of a company's outstanding shares",
            "P/E Ratio": "Price-to-Earnings ratio; measures stock valuation",
            "EPS": "Earnings Per Share; company's profit divided by shares outstanding",
            "Beta": "Measure of stock's volatility relative to the overall market",
            "Dividend Yield": "Annual dividend payment as percentage of stock price",
        }

        for term, definition in glossary.items():
            st.markdown(f"**{term}**: {definition}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.875rem;">
        <p>Built with 💜 using TensorFlow, Streamlit & Plotly</p>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Not financial advice.</p>
        <p><a href="https://github.com/NK1425/stock-price-prediction-lstm" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
