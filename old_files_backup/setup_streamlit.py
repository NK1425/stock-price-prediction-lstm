#!/usr/bin/env python3
"""
Quick setup script to update API URL in streamlit_app.py
"""
import os

def update_api_url(new_url):
    """Update the API_BASE_URL in streamlit_app.py"""
    file_path = 'streamlit_app.py'
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace the API_BASE_URL
    import re
    pattern = r'API_BASE_URL\s*=\s*["\'].*["\']'
    replacement = f'API_BASE_URL = "{new_url}"'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Updated API_BASE_URL to: {new_url}")
    return True

if __name__ == "__main__":
    print("Streamlit App API URL Updater")
    print("=" * 40)
    print()
    print("1. Local development: http://localhost:5001/api")
    print("2. Heroku: https://your-app.herokuapp.com/api")
    print("3. Railway: https://your-app.up.railway.app/api")
    print()
    
    url = input("Enter your Flask API URL (include /api): ").strip()
    
    if url:
        if not url.endswith('/api'):
            url = url.rstrip('/') + '/api'
        update_api_url(url)
    else:
        print("No URL provided. Exiting.")
