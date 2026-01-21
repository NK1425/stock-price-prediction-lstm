# 🚂 Deploy Flask API to Railway (Easiest Method)

## ✅ No CLI Installation Needed!

Railway is the easiest way to deploy - it connects directly to GitHub!

---

## 📋 Step-by-Step Guide

### Step 1: Go to Railway
👉 **https://railway.app**

### Step 2: Sign In
- Click "Login" or "Start a New Project"
- Choose "Login with GitHub"
- Authorize Railway to access your GitHub

### Step 3: Create New Project
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose repository: **`NK1425/stock-price-prediction-lstm`**

### Step 4: Configure Deployment
Railway will auto-detect Python and Flask!

**Settings:**
- **Build Command:** (Leave empty, Railway auto-detects)
- **Start Command:** `python app.py`
- **Port:** Railway sets this automatically

**Environment Variables (if needed):**
- Add any if your app requires them
- Most likely not needed for this app

### Step 5: Deploy!
1. Click **"Deploy"**
2. Wait 2-5 minutes for deployment
3. Railway will show you the deployment URL!

### Step 6: Get Your API URL
After deployment:
1. Click on your service
2. Go to **"Settings"** tab
3. Find **"Domain"** or **"Public URL"**
4. Your API will be at: `https://your-app-name.up.railway.app`

**Example:** `https://stock-prediction-api.up.railway.app`

---

## 🔧 Update Streamlit App

After getting your Railway URL, update `streamlit_app.py`:

1. Open `streamlit_app.py`
2. Find line 9: `API_BASE_URL = "http://localhost:5001/api"`
3. Replace with your Railway URL:

```python
API_BASE_URL = "https://your-app-name.up.railway.app/api"
```

4. Save and commit:

```bash
git add streamlit_app.py
git commit -m "Update API URL to Railway deployment"
git push origin main
```

---

## ✅ Test Your API

Visit your Railway URL in browser:
`https://your-app-name.up.railway.app/api/stock/AAPL`

You should see JSON data with stock information!

---

## 🎉 That's It!

Your Flask API is now live on Railway! 

**Next:** Deploy your Streamlit app to Streamlit Cloud:
👉 https://share.streamlit.io/

---

## 🔗 Quick Links

- **Railway Dashboard:** https://railway.app/dashboard
- **Your API:** `https://your-app-name.up.railway.app`
- **Streamlit Cloud:** https://share.streamlit.io/

---

**Need Help?** Railway has excellent documentation and support!
