#!/bin/bash

# AlgoTrade Pro - Startup Script
# This script starts the Flask backend server for the trading app

echo "🚀 Starting AlgoTrade Pro Backend..."
echo "================================"
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required Python packages are installed
echo "Checking dependencies..."
python3 -c "import flask, flask_cors, yfinance, pandas, numpy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required Python packages are missing"
    echo "Please install them using:"
    echo "  pip3 install flask flask-cors yfinance pandas numpy scikit-learn joblib"
    exit 1
fi

echo "✅ Dependencies OK"
echo ""

# Start the Flask server
echo "Starting Flask backend on http://localhost:5001"
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

python3 trading-backend.py
