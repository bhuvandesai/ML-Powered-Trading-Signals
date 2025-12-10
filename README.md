# AlgoTrade Pro - ML Trading Signals App

A full-stack trading application that calculates technical indicators and uses ML models to predict buy/sell signals.

## Features

### Technical Indicators Calculated:
- **RSI (Relative Strength Index)** - Momentum oscillator measuring overbought/oversold
- **MACD (Moving Average Convergence Divergence)** - Trend-following momentum
- **Bollinger Bands** - Volatility bands around moving average
- **Stochastic Oscillator** - Momentum indicator comparing closing price to price range
- **Williams %R** - Momentum indicator measuring overbought/oversold levels
- **CCI (Commodity Channel Index)** - Identifies cyclical trends
- **ATR (Average True Range)** - Volatility measure
- **SMA/EMA** - Simple and Exponential Moving Averages

### ML Model Features:
- Uses 15 technical indicators as features
- Gradient Boosting Classifier (backend) / Rule-based scoring (frontend)
- Predicts: **BUY**, **SELL**, or **HOLD** signals
- Shows confidence level and model accuracy
- Backtesting with strategy vs buy-and-hold comparison

---

## Files Included

### 1. `trading-app.html` - Self-Contained Frontend
A complete, standalone trading dashboard that runs entirely in the browser.

**To Use:**
1. Simply open `trading-app.html` in any modern web browser
2. Click on different stock symbols to analyze them
3. View the ML predictions, indicators, and backtest results

**Features:**
- No server required - runs 100% client-side
- All indicators calculated in JavaScript
- Rule-based ML model for predictions
- Interactive charts with Chart.js
- Beautiful dark theme UI

---

### 2. `trading-backend.py` - Python Flask Backend
A more advanced backend with scikit-learn ML models.

**To Use:**
```bash
# Install dependencies
pip install flask flask-cors pandas numpy scikit-learn joblib

# Run the server
python trading-backend.py

# Server runs at http://localhost:5000
```

**API Endpoints:**
- `GET /api/stocks` - List available stocks
- `GET /api/stock/<symbol>` - Get stock data with indicators
- `GET /api/indicators/<symbol>` - Get current technical indicators
- `GET /api/predict/<symbol>` - Get ML prediction (BUY/SELL/HOLD)
- `GET /api/backtest/<symbol>` - Run backtest simulation

**Example API Call:**
```bash
curl http://localhost:5000/api/predict/AAPL
```

**Response:**
```json
{
  "symbol": "AAPL",
  "signal": "BUY",
  "confidence": 72.5,
  "model_accuracy": 42.3,
  "current_price": 185.32,
  "analysis": {
    "trend": "BULLISH",
    "momentum": "POSITIVE",
    "volatility": "NORMAL",
    "rsi_status": "NEUTRAL"
  },
  "price_targets": {
    "support": 178.50,
    "resistance": 192.14,
    "stop_loss": 180.20,
    "take_profit": 196.80
  }
}
```

---

## How the ML Model Works

### Feature Engineering
The model uses these features extracted from technical indicators:
1. RSI value
2. MACD line and signal
3. MACD histogram
4. Stochastic %K and %D
5. Williams %R
6. CCI
7. ATR (normalized)
8. Price momentum (5-day change)
9. Volatility measure
10. Price relative to moving averages

### Prediction Logic
1. Calculate all technical indicators
2. Extract feature vector for current day
3. Model predicts probability of:
   - **BUY** (price expected to rise >2%)
   - **SELL** (price expected to fall >2%)
   - **HOLD** (price expected to stay within ±2%)
4. Returns signal with confidence percentage

### Backtest Methodology
- Start with $10,000 virtual capital
- Execute trades based on model signals
- Compare final equity to simple buy-and-hold strategy
- Track number of trades and returns

---

## Disclaimer

⚠️ **This is for educational purposes only!**

- Uses simulated/generated data
- Past performance doesn't guarantee future results
- Do not use for actual trading decisions
- Always consult financial professionals for investment advice

---

## Tech Stack

**Frontend:**
- React 18
- Chart.js 4
- Pure CSS (no framework)

**Backend:**
- Python 3
- Flask
- pandas & numpy
- scikit-learn (Gradient Boosting)

---

## Customization

To add real stock data, you can:
1. Replace `generate_stock_data()` with calls to Yahoo Finance API
2. Use `yfinance` library: `pip install yfinance`
3. Example:
```python
import yfinance as yf
data = yf.download('AAPL', period='2y')
```
