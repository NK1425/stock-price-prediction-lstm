# Deploying to Streamlit Cloud (Free Lifetime)

## Quick Start Guide

### Step 1: Push Your Code to GitHub

1. Create a new repository on GitHub (if you haven't already)
2. Push your code to the repository:
   ```bash
   git add .
   git commit -m "Enhanced Streamlit stock prediction app"
   git push origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to: `streamlit_app.py`
6. Click "Deploy"

### Step 3: Wait for Deployment

- Streamlit Cloud will automatically install dependencies from `requirements.txt`
- First deployment may take 5-10 minutes
- The app will be available at: `https://your-app-name.streamlit.app`

## Important Notes

### Requirements
- Your app must be in a GitHub repository (public or private)
- The main file should be named `streamlit_app.py`
- `requirements.txt` must be in the root directory

### File Structure
```
your-repo/
├── streamlit_app.py    # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md          # Optional
```

### Environment Variables
Streamlit Cloud automatically handles:
- Python version (defaults to 3.11)
- Port configuration
- HTTPS/SSL certificates

### Free Tier Limits
- **Unlimited apps**
- **No time limits**
- **No credit card required**
- Apps sleep after 1 hour of inactivity (wake up automatically on access)

### Troubleshooting

**Deployment fails:**
- Check that all dependencies in `requirements.txt` are correct
- Verify `streamlit_app.py` is in the root directory
- Check deployment logs in Streamlit Cloud dashboard

**App runs slow:**
- First model training for each stock takes 1-2 minutes
- Subsequent loads are faster due to caching
- Consider reducing training epochs if needed

**Memory issues:**
- Streamlit Cloud free tier has memory limits
- Models are cached in session state to reduce memory usage
- If issues persist, reduce batch size in model training

## Tips for Best Performance

1. **Caching**: The app uses `@st.cache_data` to cache stock data for 5 minutes
2. **Session State**: Models are cached in session state to avoid retraining
3. **First Load**: First time loading a stock may take longer (model training)

## Support

For issues with Streamlit Cloud:
- Check [Streamlit Cloud docs](https://docs.streamlit.io/streamlit-community-cloud)
- Visit [Streamlit forums](https://discuss.streamlit.io)
