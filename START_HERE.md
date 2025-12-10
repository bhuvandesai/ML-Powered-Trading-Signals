# 🚀 Quick Start Guide - AlgoTrade Pro

## Method 1: Using the Startup Script (Easiest)

### Step 1: Make the script executable (only needed once)
```bash
chmod +x start-trading-app.sh
```

### Step 2: Run the startup script
```bash
./start-trading-app.sh
```

The backend server will start automatically!

---

## Method 2: Manual Terminal Commands (Step-by-Step)

### Step 1: Open Terminal
- Press `Cmd + Space` and type "Terminal"
- Or find Terminal in Applications → Utilities

### Step 2: Navigate to the project folder
```bash
cd ~/Downloads/files
```

### Step 3: Start the Flask backend server
```bash
python3 trading-backend.py
```

You should see output like:
```
 * Serving Flask app 'trading-backend'
 * Debug mode: on
 * Running on http://0.0.0.0:5001
```

### Step 4: Open the frontend in your browser

**Option A: Double-click the HTML file**
- Go to `~/Downloads/files/` folder
- Double-click `trading-app.html`
- It will open in your default browser

**Option B: Open from Terminal**
```bash
open trading-app.html
```

**Option C: Open manually**
- Open your browser
- Press `Cmd + O` (File → Open)
- Navigate to `~/Downloads/files/trading-app.html`
- Click Open

---

## Stopping the Server

When you're done:
1. Go back to the Terminal window running the server
2. Press `Ctrl + C` to stop it

---

## Troubleshooting

### If you get "command not found" errors:

**Check Python is installed:**
```bash
python3 --version
```

**Install missing dependencies:**
```bash
pip3 install flask flask-cors yfinance pandas numpy scikit-learn joblib
```

### If the backend won't start:

**Check if port 5001 is already in use:**
```bash
lsof -i :5001
```

If something is using it, either:
- Stop that process, or
- Change the port in `trading-backend.py` (line 692) from `5001` to another number like `5002`

### If the frontend shows connection errors:

1. Make sure the backend is running (check Terminal)
2. Verify the backend URL in `trading-app.html` matches (line 924 should be `http://localhost:5001`)
3. Try refreshing the browser page

---

## Quick Reference

**Start Backend:**
```bash
cd ~/Downloads/files && python3 trading-backend.py
```

**Open Frontend:**
```bash
cd ~/Downloads/files && open trading-app.html
```

**Test Backend:**
```bash
curl http://localhost:5001/api/health
```

---

## Tips

- Keep the Terminal window open while using the app (it's running the backend)
- The app fetches live data from Yahoo Finance, so you need an internet connection
- First load might take a few seconds as it downloads stock data and trains the ML model
