# 🚀 Quick Start Commands for macOS/zsh

## The Problem
On macOS with zsh, `pip` and `pytest` commands might not work. Use `python3 -m pip` and `python3 -m pytest` instead.

---

## ✅ Correct Commands (Use These!)

### Install Dependencies
```bash
cd ~/Downloads/files

# Method 1: Use the install script (Easiest)
./INSTALL_DEPENDENCIES.sh

# Method 2: Manual installation
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Run Tests
```bash
cd ~/Downloads/files

# Method 1: Use the test script (Easiest)
./RUN_TESTS.sh

# Method 2: Manual command
python3 -m pytest test_trading_backend.py -v
```

### Start the Backend
```bash
cd ~/Downloads/files
./start-trading-app.sh
```

---

## ❌ Wrong Commands (Don't Use These!)

```bash
# ❌ These might not work:
pip install -r requirements.txt
pytest test_trading_backend.py
python trading-backend.py
```

---

## ✅ Correct Alternatives

```bash
# ✅ Use these instead:
python3 -m pip install -r requirements.txt
python3 -m pytest test_trading_backend.py
python3 trading-backend.py
```

---

## Understanding the Difference

**On macOS, use:**
- `python3` instead of `python`
- `python3 -m pip` instead of `pip`
- `python3 -m pytest` instead of `pytest`

**Why?**
- macOS might have multiple Python versions
- `python3` ensures you're using Python 3
- `python3 -m pip` uses pip for that specific Python version
- This avoids "command not found" errors

---

## Troubleshooting

### If you get "command not found: python3"
```bash
# Check if Python is installed
which python3

# If not installed, install it:
brew install python3
```

### If you get "Permission denied"
```bash
# Make scripts executable:
chmod +x INSTALL_DEPENDENCIES.sh
chmod +x RUN_TESTS.sh
chmod +x start-trading-app.sh
```

### If pip install fails
```bash
# Try with user flag:
python3 -m pip install --user -r requirements.txt
```

### Check what's installed
```bash
# See installed packages:
python3 -m pip list

# See Python version:
python3 --version

# Check if pytest is installed:
python3 -m pytest --version
```

---

## Complete Workflow

```bash
# 1. Navigate to project folder
cd ~/Downloads/files

# 2. Install dependencies (first time only)
./INSTALL_DEPENDENCIES.sh

# 3. Run tests
./RUN_TESTS.sh

# 4. Start the backend (in one terminal)
./start-trading-app.sh

# 5. Open frontend (in another terminal or just double-click)
open trading-app.html
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Install deps | `python3 -m pip install -r requirements.txt` |
| Run tests | `python3 -m pytest test_trading_backend.py -v` |
| Start backend | `python3 trading-backend.py` |
| Check Python | `python3 --version` |
| Check pip | `python3 -m pip --version` |
| List packages | `python3 -m pip list` |

---

## Need Help?

If you're still having issues:
1. Make sure you're in the correct directory: `cd ~/Downloads/files`
2. Check Python is installed: `python3 --version`
3. Try the scripts: `./INSTALL_DEPENDENCIES.sh` and `./RUN_TESTS.sh`
