#!/bin/bash

# Test Runner Script for macOS
# This script installs dependencies and runs tests using python3

echo "🧪 AlgoTrade Pro - Test Runner"
echo "=============================="
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3 first"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Install dependencies using python3 -m pip
echo "Installing dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Run tests using python3 -m pytest
echo "Running tests..."
echo "=============================="
python3 -m pytest test_trading_backend.py -v

TEST_EXIT_CODE=$?

echo ""
echo "=============================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $TEST_EXIT_CODE)"
fi

exit $TEST_EXIT_CODE
