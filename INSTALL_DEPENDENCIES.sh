#!/bin/bash

# Install Dependencies Script
# This script installs all required Python packages for the trading app

echo "📦 Installing AlgoTrade Pro Dependencies"
echo "========================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3 first:"
    echo "  brew install python3"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Upgrade pip first
echo "Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install from requirements.txt
echo "Installing packages from requirements.txt..."
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All dependencies installed successfully!"
    echo ""
    echo "You can now:"
    echo "  - Run tests: ./RUN_TESTS.sh"
    echo "  - Start backend: ./start-trading-app.sh"
else
    echo ""
    echo "❌ Error: Failed to install some dependencies"
    exit 1
fi
