# Testing Guide

## Running Tests Locally

### Quick Start (Easiest Method)
```bash
cd ~/Downloads/files
./RUN_TESTS.sh
```

### Manual Installation (Step by Step)

**For macOS/Linux (using python3):**

1. **Install test dependencies**
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

**Alternative if pip3 is available:**
```bash
pip3 install -r requirements.txt
```

2. **Run all tests**
```bash
python3 -m pytest test_trading_backend.py -v
```

**Alternative:**
```bash
python3 -m pytest test_trading_backend.py -v --tb=short
```

### Run specific test classes
```bash
python3 -m pytest test_trading_backend.py::TestTechnicalIndicators -v
python3 -m pytest test_trading_backend.py::TestAPIEndpoints -v
```

### Run with coverage
```bash
python3 -m pytest test_trading_backend.py --cov=. --cov-report=html
```

## Test Structure

The test suite includes:

1. **TestTechnicalIndicators** - Tests for all technical indicator calculations
   - SMA, EMA, RSI, MACD
   - Bollinger Bands, ATR
   - Stochastic, OBV, Williams %R, CCI

2. **TestCalculateAllIndicators** - Tests the indicator calculation pipeline

3. **TestTradingMLModel** - Tests the ML model
   - Initialization
   - Feature preparation
   - Training
   - Prediction

4. **TestAPIEndpoints** - Tests all Flask API endpoints
   - Health check
   - Stock data endpoints
   - Prediction endpoints
   - Backtest endpoints

## GitHub Actions

Tests run automatically on:
- Every push to main/master/develop branches
- Every pull request
- Manual workflow dispatch

The workflow tests against Python 3.9, 3.10, and 3.11.

## Troubleshooting

### If tests fail with "no tests collected"

Make sure:
1. Test file is named `test_*.py` (starts with `test_`)
2. Test functions are named `test_*`
3. Test classes are named `Test*`
4. `pytest.ini` is in the root directory

### If import errors occur

The test file uses `importlib` to import `trading-backend.py` (which has a hyphen). This should work automatically.

If you get import errors:
```bash
# Make sure you're in the right directory
cd /path/to/files

# Set PYTHONPATH if needed
export PYTHONPATH=$PWD:$PYTHONPATH
pytest test_trading_backend.py -v
```

### If API tests fail

Some API tests make actual HTTP requests. If you're running tests without internet or if Yahoo Finance is unavailable:
- Tests will fail gracefully
- Consider mocking the yfinance calls for unit tests
- Integration tests can be marked with `@pytest.mark.integration`
