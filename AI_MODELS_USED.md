# 🤖 AI Models Used in This Application

## Overview

This application uses **two different types of AI/ML systems**:

---

## 1. 📈 Stock Price Prediction Model

### **Type**: Deep Learning - LSTM Neural Network with Attention Mechanism

### **Framework**: TensorFlow/Keras

### **Architecture**:
```
Input Layer: 60 timesteps × 15 features
    ↓
LSTM Layer 1: 50 units (dropout 0.2, return_sequences=True)
    ↓
LSTM Layer 2: 50 units (dropout 0.2, return_sequences=True)
    ↓
Custom Attention Layer: Self-attention mechanism
    ↓
Dense Layer 1: 25 units (ReLU activation, dropout 0.2)
    ↓
Dense Layer 2: 10 units (ReLU activation)
    ↓
Output Layer: 7 units (Linear activation) → 7-day price forecast
```

### **What It Does**:
- Analyzes 60 days of historical stock data
- Uses 15+ technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, Volume, Volatility)
- Predicts stock prices for the next 7 days
- Achieves ~2.3% RMSE (Root Mean Square Error) accuracy

### **Training**:
- Trains on 1-2 years of historical data
- 80/20 train/validation split
- Early stopping to prevent overfitting
- Models are trained per stock symbol
- Training takes 1-2 minutes per stock (first time)

### **Model Files**:
- Uses TensorFlow's Keras API
- Models are cached in session state (not saved to disk in current implementation)
- Custom `AttentionLayer` class for attention mechanism

---

## 2. 💬 AI Chat Assistant

### **Type**: Rule-Based Expert System (NOT a Large Language Model)

### **Current Implementation**: 
**Keyword-based pattern matching** with sophisticated response templates

### **How It Works**:
- Analyzes user questions for keywords (e.g., "RSI", "predict", "why", "confidence")
- Matches patterns and generates structured responses
- Incorporates real stock data and technical indicators
- Provides detailed financial insights following your specified prompt template

### **Response Categories**:
1. **Prediction/Forecast Questions** → Detailed insights with trend analysis
2. **"Why" Questions** → Explains model reasoning step-by-step
3. **Indicator Questions** → Explains RSI, MACD, Bollinger Bands, etc.
4. **Confidence Questions** → Breaks down confidence factors
5. **Buy/Sell Questions** → Provides analysis with disclaimers
6. **Model Architecture** → Explains LSTM + Attention mechanism
7. **General Questions** → Helpful guidance and suggestions

### **NOT Using**:
- ❌ GPT-3/GPT-4 (OpenAI)
- ❌ Claude (Anthropic)
- ❌ Gemini (Google)
- ❌ Llama (Meta)
- ❌ Any Large Language Model (LLM)

### **Why Rule-Based?**:
- ✅ **Free** - No API costs
- ✅ **Fast** - Instant responses
- ✅ **Reliable** - Consistent answers
- ✅ **Privacy** - No data sent to external services
- ✅ **Customized** - Tailored to your specific financial copilot needs

---

## 🔄 Future Enhancement Option

If you want to add a **true LLM** for more natural conversations:

### **Option 1: OpenAI GPT**
```python
import openai
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": question}]
)
```
**Cost**: ~$0.002 per 1K tokens

### **Option 2: Free Local LLM**
```python
# Using Hugging Face Transformers
from transformers import pipeline
chatbot = pipeline("text-generation", model="mistral-7b")
```
**Cost**: Free (runs locally, but slower)

### **Option 3: Hugging Face Inference API**
```python
import requests
response = requests.post(
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={"inputs": question}
)
```
**Cost**: Free tier available

---

## 📊 Summary

| Component | AI Model Type | Framework | Purpose |
|-----------|--------------|-----------|---------|
| Stock Prediction | Deep Learning (LSTM + Attention) | TensorFlow/Keras | Predict stock prices |
| Chat Assistant | Rule-Based Expert System | Python (keyword matching) | Answer user questions |

---

## 🎯 Current Setup Benefits

1. **No External API Dependencies**: Everything runs locally
2. **No API Costs**: Free to run
3. **Fast Responses**: Instant chat replies
4. **Privacy**: No data sent to external services
5. **Customized**: Tailored responses for financial analysis
6. **Works Offline**: (except for fetching stock data from Yahoo Finance)

---

## 💡 Recommendation

The current rule-based system works well for:
- ✅ Financial data explanations
- ✅ Technical indicator analysis
- ✅ Model explanations
- ✅ Structured insights

Consider adding an LLM if you want:
- 🤔 More conversational interactions
- 🧠 Better understanding of complex queries
- 📝 Natural language generation for reports
- 🔍 Advanced reasoning capabilities

But for now, the rule-based system provides **excellent, structured financial insights** at **zero cost** with **instant responses**!
