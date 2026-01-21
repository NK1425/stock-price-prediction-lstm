# 🎨 Deploy Flask API to Render (Alternative Method)

## ✅ Also Easy - No CLI Needed!

Render is another great option that connects directly to GitHub!

---

## 📋 Step-by-Step Guide

### Step 1: Go to Render
👉 **https://render.com**

### Step 2: Sign In
- Click "Get Started for Free"
- Choose "Sign in with GitHub"
- Authorize Render to access your GitHub

### Step 3: Create Web Service
1. Click **"New +"** button (top right)
2. Select **"Web Service"**
3. Click **"Connect"** next to your GitHub account (if not connected)
4. Choose repository: **`NK1425/stock-price-prediction-lstm`**

### Step 4: Configure Service
Fill in the settings:

**Basic Settings:**
- **Name:** `stock-prediction-api` (or any name you like)
- **Region:** Choose closest to you (e.g., US West)
- **Branch:** `main`

**Build & Deploy:**
- **Root Directory:** (Leave empty)
- **Environment:** `Python 3`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`

**Advanced Settings:**
- **Auto-Deploy:** `Yes` (auto-deploys on git push)
- **Health Check Path:** `/api/stock/AAPL` (optional)

### Step 5: Deploy!
1. Scroll down and click **"Create Web Service"**
2. Render will start building your app
3. Wait 3-5 minutes for deployment
4. You'll see the deployment URL!

### Step 6: Get Your API URL
After deployment:
- Your service URL will be shown at the top
- Example: `https://stock-prediction-api.onrender.com`
- Your API endpoint: `https://stock-prediction-api.onrender.com/api`

**Note:** Free tier apps may take 30 seconds to start (cold start)

---

## 🔧 Update Streamlit App

After getting your Render URL, update `streamlit_app.py`:

1. Open `streamlit_app.py`
2. Find line 9: `API_BASE_URL = "http://localhost:5001/api"`
3. Replace with your Render URL:

```python
API_BASE_URL = "https://stock-prediction-api.onrender.com/api"
```

4. Save and commit:

```bash
git add streamlit_app.py
git commit -m "Update API URL to Render deployment"
git push origin main
```

---

## ✅ Test Your API

Visit your Render URL in browser:
`https://stock-prediction-api.onrender.com/api/stock/AAPL`

You should see JSON data with stock information!

**Note:** First request might take 30 seconds (cold start on free tier)

---

## 🎉 That's It!

Your Flask API is now live on Render! 

**Next:** Deploy your Streamlit app to Streamlit Cloud:
👉 https://share.streamlit.io/

---

## 🔗 Quick Links

- **Render Dashboard:** https://dashboard.render.com
- **Your API:** `https://your-app-name.onrender.com`
- **Streamlit Cloud:** https://share.streamlit.io/

---

## 💡 Render Free Tier Notes

- ✅ Free tier available
- ⚠️ Apps sleep after 15 minutes of inactivity
- ⚠️ First request after sleep takes ~30 seconds (cold start)
- ✅ Auto-deploys on git push
- ✅ Custom domain support

---

**Need Help?** Render has great documentation at https://render.com/docs!
