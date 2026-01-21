#!/bin/bash

echo "🚀 Starting Stock Price Prediction App..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "📥 Installing/updating pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models

# Start the server
echo ""
echo "✅ Setup complete!"
echo "🌐 Starting Flask server on http://localhost:5000"
echo "📊 Open http://localhost:5000 in your browser to use the app"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
