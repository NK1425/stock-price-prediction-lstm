# 🔍 Find Your Deployed App

## 🎉 Deployment Complete!

Your Flask API is now live! Here's how to find and access it.

---

## 📍 Where is Your App?

### **If You Deployed on Railway:**

1. Go to: https://railway.app/dashboard
2. Click on your project
3. Click on your service
4. Look for:
   - **"Public Domain"** section
   - OR **"Settings"** → **"Networking"** tab
5. Your URL will be something like:
   ```
   https://stock-price-prediction-lstm-production.up.railway.app
   ```

**Alternative:**
- Check the **"Deployments"** tab
- Look at the latest deployment
- URL will be shown there

---

### **If You Deployed on Render:**

1. Go to: https://dashboard.render.com
2. Click on your service
3. Look at the **top of the page** - your URL is displayed there
4. Your URL will be something like:
   ```
   https://stock-price-prediction-lstm.onrender.com
   ```

**Note:** Free tier apps may take 30 seconds to wake up (cold start)

---

### **If You Deployed on Heroku:**

1. Go to: https://dashboard.heroku.com/apps
2. Click on your app name
3. Look for:
   - **"Open app"** button (top right)
   - OR URL displayed at the top
4. Your URL will be something like:
   ```
   https://stock-prediction-api.herokuapp.com
   ```

---

## ✅ Test Your API

### **Method 1: Browser**
Open in your browser:
```
YOUR_API_URL/api/stock/AAPL
```

Example:
- Railway: `https://your-app.up.railway.app/api/stock/AAPL`
- Render: `https://your-app.onrender.com/api/stock/AAPL`
- Heroku: `https://your-app.herokuapp.com/api/stock/AAPL`

You should see JSON data with Apple stock information!

### **Method 2: Terminal (curl)**
```bash
curl https://YOUR_API_URL/api/stock/AAPL
```

### **Method 3: Python**
```python
import requests
response = requests.get('https://YOUR_API_URL/api/stock/AAPL')
print(response.json())
```

---

## 🔗 Available API Endpoints

Once you have your URL, you can access:

1. **Stock Data & Predictions:**
   ```
   GET /api/stock/{SYMBOL}
   Example: /api/stock/AAPL
   ```

2. **AI Chat:**
   ```
   POST /api/chat
   Body: {"question": "What is RSI?", "symbol": "AAPL"}
   ```

3. **Stock Search:**
   ```
   GET /api/search/{QUERY}
   Example: /api/search/AAPL
   ```

4. **Model Summary:**
   ```
   GET /api/model/summary/{SYMBOL}
   Example: /api/model/summary/AAPL
   ```

---

## 📱 Next Steps: Connect Streamlit App

### **Step 1: Update API URL in Streamlit App**

1. Open `streamlit_app.py`
2. Find line 9:
   ```python
   API_BASE_URL = "http://localhost:5001/api"
   ```
3. Replace with your deployed API URL:
   ```python
   API_BASE_URL = "https://YOUR_API_URL/api"
   ```
   (Don't forget to add `/api` at the end!)

### **Step 2: Commit and Push**

```bash
git add streamlit_app.py
git commit -m "Update API URL to deployed endpoint"
git push origin main
```

### **Step 3: Deploy Streamlit App**

1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Click **"New app"**
4. Select repository: `NK1425/stock-price-prediction-lstm`
5. Main file: `streamlit_app.py`
6. Click **"Deploy!"**

Your Streamlit app will be live at:
```
https://stock-price-prediction-lstm.streamlit.app
```

---

## 🎯 Quick Reference

| Platform | Dashboard URL | Your API URL Format |
|----------|--------------|---------------------|
| Railway | https://railway.app/dashboard | `https://*.up.railway.app` |
| Render | https://dashboard.render.com | `https://*.onrender.com` |
| Heroku | https://dashboard.heroku.com/apps | `https://*.herokuapp.com` |

---

## 💡 Tips

- **First request may be slow** (cold start on free tiers)
- **Check build logs** if API doesn't respond
- **Test all endpoints** to ensure everything works
- **Save your API URL** - you'll need it for Streamlit app

---

## 🆘 Troubleshooting

**Can't find the URL?**
- Check the deployment logs
- Look in Settings → Networking
- Check your email for deployment notifications

**API not responding?**
- Wait 30 seconds and try again (cold start)
- Check deployment status
- Verify the URL is correct

**Getting errors?**
- Check build logs in your platform dashboard
- Verify all environment variables are set
- Test locally first: `python app.py`

---

**Your app is live! Share the URL with anyone! 🎉**
