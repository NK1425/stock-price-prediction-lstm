# 🚀 Deploy Stock Prediction App to Streamlit Cloud

## Quick Deploy to Streamlit Cloud (Recommended)

### Step 1: Update API URL in streamlit_app.py

**For Local Development:**
```python
API_BASE_URL = "http://localhost:5001/api"
```

**For Production (after deploying Flask API):**
```python
API_BASE_URL = "https://your-flask-api.herokuapp.com/api"  # Your deployed Flask API URL
```

### Step 2: Deploy to Streamlit Cloud

1. **Push to GitHub** (already done! ✅)
   - Your repo: `https://github.com/NK1425/stock-price-prediction-lstm`

2. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with GitHub

3. **Deploy App:**
   - Click "New app"
   - Select repository: `NK1425/stock-price-prediction-lstm`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

4. **Your app will be live at:**
   - `https://stock-price-prediction-lstm.streamlit.app`

---

## 🔧 Deploy Flask API (Required for AI Chat)

The Streamlit app needs the Flask API running. Deploy it separately:

### Option 1: Heroku (Recommended)

1. **Create Heroku App for API:**
   ```bash
   heroku create stock-prediction-api
   ```

2. **Update Procfile for Flask:**
   ```bash
   echo "web: python app.py" > Procfile
   ```

3. **Push to Heroku:**
   ```bash
   git push heroku main
   ```

4. **Your API will be at:**
   - `https://stock-prediction-api.herokuapp.com`

5. **Update streamlit_app.py:**
   ```python
   API_BASE_URL = "https://stock-prediction-api.herokuapp.com/api"
   ```

### Option 2: Railway

1. Go to: https://railway.app
2. Create new project
3. Deploy from GitHub: `NK1425/stock-price-prediction-lstm`
4. Set start command: `python app.py`
5. Set environment variable: `PORT=5001`
6. Get the URL and update `API_BASE_URL` in streamlit_app.py

---

## 📝 Setup Instructions

### Local Testing:

**Terminal 1 - Flask API:**
```bash
cd /Users/nk/NK1425.github.io-1
python3 app.py
```
API runs at: http://localhost:5001

**Terminal 2 - Streamlit App:**
```bash
cd /Users/nk/NK1425.github.io-1
streamlit run streamlit_app.py
```
App runs at: http://localhost:8501

---

## ✅ What's Included:

- ✅ Streamlit frontend (`streamlit_app.py`)
- ✅ Flask backend API (`app.py`)
- ✅ AI Chat integration via API
- ✅ Real-time stock data
- ✅ Interactive charts (Plotly)
- ✅ Technical indicators
- ✅ 7-day predictions

---

## 🔗 After Deployment:

**Streamlit App:**
- `https://stock-price-prediction-lstm.streamlit.app`

**Flask API (for AI Chat):**
- `https://stock-prediction-api.herokuapp.com/api`

Make sure to update the `API_BASE_URL` in `streamlit_app.py` after deploying the Flask API!

---

## 💡 Features Available:

- ✅ Stock search with autocomplete
- ✅ Real-time price charts
- ✅ AI predictions
- ✅ Technical indicators
- ✅ **AI Chat Assistant** (connects to Flask API)
- ✅ Modern, responsive UI

---

**Happy Deploying! 🎉**
