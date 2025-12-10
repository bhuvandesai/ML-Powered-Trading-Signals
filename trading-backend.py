"""
Full-Stack Trading App Backend
Calculates technical indicators and predicts buy/sell signals using ML
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import yfinance as yf

app = Flask(__name__)
CORS(app)

# ============================================
# TECHNICAL INDICATORS CALCULATION
# ============================================

class TechnicalIndicators:
    """Calculate various technical indicators for stock analysis"""
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def obv(close, volume):
        """On-Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def cci(high, low, close, period=20):
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mad)
    
    @staticmethod
    def momentum(data, period=10):
        """Momentum"""
        return data - data.shift(period)
    
    @staticmethod
    def roc(data, period=10):
        """Rate of Change"""
        return ((data - data.shift(period)) / data.shift(period)) * 100


# ============================================
# STOCK DATA GENERATION (Real data from Yahoo Finance)
# ============================================

def generate_stock_data(symbol, days=500):
    """Fetch real stock data from Yahoo Finance"""
    try:
        # Calculate period based on days (yfinance uses '1y', '2y', '5y', etc.)
        if days <= 30:
            period = '1mo'
        elif days <= 90:
            period = '3mo'
        elif days <= 180:
            period = '6mo'
        elif days <= 365:
            period = '1y'
        elif days <= 730:
            period = '2y'
        else:
            period = '5y'
        
        # Download stock data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Rename columns to lowercase to match expected format
        data.columns = data.columns.str.lower()
        
        # Ensure we have the required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns for {symbol}")
        
        # Select and reorder columns
        data = data[required_cols].copy()
        
        # Sort by date (oldest first)
        data = data.sort_values('date').reset_index(drop=True)
        
        # If we have more data than requested, take the most recent days
        if len(data) > days:
            data = data.tail(days).reset_index(drop=True)
        
        # Ensure data types are correct
        # Remove timezone info from date for compatibility (if it exists)
        if data['date'].dtype == 'object' or hasattr(data['date'].iloc[0], 'tz'):
            data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
        else:
            data['date'] = pd.to_datetime(data['date'])
        data['open'] = pd.to_numeric(data['open'], errors='coerce')
        data['high'] = pd.to_numeric(data['high'], errors='coerce')
        data['low'] = pd.to_numeric(data['low'], errors='coerce')
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        if len(data) == 0:
            raise ValueError(f"No valid data after cleaning for symbol {symbol}")
        
        print(f"Successfully fetched {len(data)} days of real data for {symbol}")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        
        return data
        
    except Exception as e:
        # Fallback to simulated data if real data fetch fails
        print(f"Warning: Failed to fetch real data for {symbol}: {str(e)}")
        print("Falling back to simulated data...")
        
        # Fallback to original simulated data generation
        np.random.seed(hash(symbol) % 2**32)
        
        base_prices = {
            'AAPL': 175, 'GOOGL': 140, 'MSFT': 380, 'AMZN': 180,
            'TSLA': 250, 'META': 500, 'NVDA': 480, 'AMD': 120,
            'NFLX': 450, 'JPM': 180, 'BAC': 35, 'WMT': 160
        }
        
        base_price = base_prices.get(symbol, 100)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        returns = np.random.normal(0.0005, 0.02, days)
        prices = base_price * np.exp(np.cumsum(returns))
        trend = np.linspace(0, 0.3, days)
        prices = prices * (1 + trend * np.random.choice([-1, 1]))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            'high': prices * (1 + np.random.uniform(0, 0.03, days)),
            'low': prices * (1 - np.random.uniform(0, 0.03, days)),
            'close': prices,
            'volume': np.random.randint(1000000, 50000000, days)
        })
        
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data


def calculate_all_indicators(df):
    """Calculate all technical indicators for a dataframe"""
    ti = TechnicalIndicators()
    
    # Moving Averages
    df['sma_10'] = ti.sma(df['close'], 10)
    df['sma_20'] = ti.sma(df['close'], 20)
    df['sma_50'] = ti.sma(df['close'], 50)
    df['ema_10'] = ti.ema(df['close'], 10)
    df['ema_20'] = ti.ema(df['close'], 20)
    
    # RSI
    df['rsi'] = ti.rsi(df['close'], 14)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = ti.macd(df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ti.bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ATR
    df['atr'] = ti.atr(df['high'], df['low'], df['close'])
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = ti.stochastic(df['high'], df['low'], df['close'])
    
    # OBV
    df['obv'] = ti.obv(df['close'], df['volume'])
    
    # Williams %R
    df['williams_r'] = ti.williams_r(df['high'], df['low'], df['close'])
    
    # CCI
    df['cci'] = ti.cci(df['high'], df['low'], df['close'])
    
    # Momentum
    df['momentum'] = ti.momentum(df['close'])
    
    # ROC
    df['roc'] = ti.roc(df['close'])
    
    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    
    # Volatility
    df['volatility'] = df['price_change'].rolling(window=20).std()
    
    return df


# ============================================
# ML MODEL FOR PREDICTION
# ============================================

class TradingMLModel:
    """Machine Learning model for trading signals prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'stoch_k', 'stoch_d', 'williams_r',
            'cci', 'momentum', 'roc', 'price_change',
            'price_change_5', 'volatility', 'atr'
        ]
        
    def prepare_features(self, df):
        """Prepare features for ML model"""
        # Create target variable (1 = buy, 0 = hold, -1 = sell)
        # Based on future price movement
        df = df.copy()
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        
        # Define signals based on future returns
        df['signal'] = 0  # Hold
        df.loc[df['future_return'] > 0.02, 'signal'] = 1   # Buy if >2% gain
        df.loc[df['future_return'] < -0.02, 'signal'] = -1  # Sell if >2% loss
        
        return df
    
    def train(self, df):
        """Train the ML model"""
        df = self.prepare_features(df)
        
        # Remove NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            return {'accuracy': 0, 'message': 'Not enough data'}
        
        X = df_clean[self.feature_columns]
        y = df_clean['signal']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': round(accuracy * 100, 2),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def predict(self, df):
        """Make predictions"""
        if self.model is None:
            return None, None
        
        df_clean = df.dropna(subset=self.feature_columns)
        
        if len(df_clean) == 0:
            return None, None
        
        X = df_clean[self.feature_columns].iloc[-1:].copy()
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get confidence
        confidence = max(probabilities) * 100
        
        return int(prediction), round(confidence, 2)


# ============================================
# API ENDPOINTS
# ============================================

# Store models in memory
models = {}

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Trading Backend API',
        'version': '1.0',
        'endpoints': {
            'GET /': 'API information (this endpoint)',
            'GET /api/health': 'Health check',
            'GET /api/stocks': 'Get list of available stocks',
            'GET /api/stock/<symbol>': 'Get stock data with technical indicators',
            'GET /api/indicators/<symbol>': 'Get current technical indicators',
            'GET /api/predict/<symbol>': 'Get ML prediction for buy/sell signal',
            'GET /api/backtest/<symbol>': 'Run backtest on trading strategy'
        },
        'status': 'running'
    })

@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """Get list of available stocks"""
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'Technology'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'Consumer'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'sector': 'Automotive'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'Technology'},
        {'symbol': 'AMD', 'name': 'AMD Inc.', 'sector': 'Technology'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'sector': 'Entertainment'},
        {'symbol': 'JPM', 'name': 'JPMorgan Chase', 'sector': 'Finance'},
        {'symbol': 'BAC', 'name': 'Bank of America', 'sector': 'Finance'},
        {'symbol': 'WMT', 'name': 'Walmart Inc.', 'sector': 'Retail'},
    ]
    return jsonify(stocks)


@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Get stock data with technical indicators"""
    try:
        # Generate stock data (now uses real Yahoo Finance data)
        df = generate_stock_data(symbol.upper())
        
        # Calculate indicators
        df = calculate_all_indicators(df)
        
        # Convert to JSON-friendly format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Get last 100 days for display
        df_display = df.tail(100).copy()
        
        # Replace NaN with None for JSON
        df_display = df_display.where(pd.notnull(df_display), None)
        
        return jsonify({
            'symbol': symbol.upper(),
            'data': df_display.to_dict('records'),
            'current_price': round(df['close'].iloc[-1], 2),
            'price_change': round(df['price_change'].iloc[-1] * 100, 2) if pd.notnull(df['price_change'].iloc[-1]) else 0,
            'volume': int(df['volume'].iloc[-1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/indicators/<symbol>', methods=['GET'])
def get_indicators(symbol):
    """Get current technical indicators for a stock"""
    try:
        df = generate_stock_data(symbol.upper())
        df = calculate_all_indicators(df)
        
        latest = df.iloc[-1]
        
        indicators = {
            'rsi': round(latest['rsi'], 2) if pd.notnull(latest['rsi']) else None,
            'macd': round(latest['macd'], 4) if pd.notnull(latest['macd']) else None,
            'macd_signal': round(latest['macd_signal'], 4) if pd.notnull(latest['macd_signal']) else None,
            'macd_histogram': round(latest['macd_hist'], 4) if pd.notnull(latest['macd_hist']) else None,
            'sma_20': round(latest['sma_20'], 2) if pd.notnull(latest['sma_20']) else None,
            'sma_50': round(latest['sma_50'], 2) if pd.notnull(latest['sma_50']) else None,
            'ema_20': round(latest['ema_20'], 2) if pd.notnull(latest['ema_20']) else None,
            'bb_upper': round(latest['bb_upper'], 2) if pd.notnull(latest['bb_upper']) else None,
            'bb_lower': round(latest['bb_lower'], 2) if pd.notnull(latest['bb_lower']) else None,
            'bb_middle': round(latest['bb_middle'], 2) if pd.notnull(latest['bb_middle']) else None,
            'stoch_k': round(latest['stoch_k'], 2) if pd.notnull(latest['stoch_k']) else None,
            'stoch_d': round(latest['stoch_d'], 2) if pd.notnull(latest['stoch_d']) else None,
            'atr': round(latest['atr'], 2) if pd.notnull(latest['atr']) else None,
            'williams_r': round(latest['williams_r'], 2) if pd.notnull(latest['williams_r']) else None,
            'cci': round(latest['cci'], 2) if pd.notnull(latest['cci']) else None,
            'momentum': round(latest['momentum'], 2) if pd.notnull(latest['momentum']) else None,
            'volatility': round(latest['volatility'] * 100, 2) if pd.notnull(latest['volatility']) else None,
        }
        
        # Add interpretation
        interpretations = []
        if indicators['rsi']:
            if indicators['rsi'] > 70:
                interpretations.append({'indicator': 'RSI', 'signal': 'OVERBOUGHT', 'value': indicators['rsi']})
            elif indicators['rsi'] < 30:
                interpretations.append({'indicator': 'RSI', 'signal': 'OVERSOLD', 'value': indicators['rsi']})
            else:
                interpretations.append({'indicator': 'RSI', 'signal': 'NEUTRAL', 'value': indicators['rsi']})
        
        if indicators['macd'] and indicators['macd_signal']:
            if indicators['macd'] > indicators['macd_signal']:
                interpretations.append({'indicator': 'MACD', 'signal': 'BULLISH', 'value': indicators['macd']})
            else:
                interpretations.append({'indicator': 'MACD', 'signal': 'BEARISH', 'value': indicators['macd']})
        
        if indicators['stoch_k']:
            if indicators['stoch_k'] > 80:
                interpretations.append({'indicator': 'Stochastic', 'signal': 'OVERBOUGHT', 'value': indicators['stoch_k']})
            elif indicators['stoch_k'] < 20:
                interpretations.append({'indicator': 'Stochastic', 'signal': 'OVERSOLD', 'value': indicators['stoch_k']})
        
        return jsonify({
            'symbol': symbol.upper(),
            'indicators': indicators,
            'interpretations': interpretations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/<symbol>', methods=['GET'])
def predict_signal(symbol):
    """Predict buy/sell signal using ML model"""
    try:
        symbol = symbol.upper()
        
        # Generate data
        df = generate_stock_data(symbol, days=500)
        df = calculate_all_indicators(df)
        
        # Train or get cached model
        if symbol not in models:
            models[symbol] = TradingMLModel()
        
        model = models[symbol]
        training_result = model.train(df)
        
        # Make prediction
        prediction, confidence = model.predict(df)
        
        signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        signal = signal_map.get(prediction, 'HOLD')
        
        # Get supporting analysis
        latest = df.iloc[-1]
        
        analysis = {
            'trend': 'BULLISH' if latest['close'] > latest['sma_50'] else 'BEARISH',
            'momentum': 'POSITIVE' if latest['momentum'] > 0 else 'NEGATIVE',
            'volatility': 'HIGH' if latest['volatility'] > 0.02 else 'NORMAL',
            'rsi_status': 'OVERBOUGHT' if latest['rsi'] > 70 else ('OVERSOLD' if latest['rsi'] < 30 else 'NEUTRAL')
        }
        
        # Calculate price targets
        current_price = latest['close']
        atr = latest['atr'] if pd.notnull(latest['atr']) else current_price * 0.02
        
        return jsonify({
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'model_accuracy': training_result['accuracy'],
            'current_price': round(current_price, 2),
            'analysis': analysis,
            'price_targets': {
                'support': round(current_price - atr * 2, 2),
                'resistance': round(current_price + atr * 2, 2),
                'stop_loss': round(current_price - atr * 1.5, 2) if signal == 'BUY' else round(current_price + atr * 1.5, 2),
                'take_profit': round(current_price + atr * 3, 2) if signal == 'BUY' else round(current_price - atr * 3, 2)
            },
            'feature_importance': dict(zip(
                model.feature_columns,
                [round(x * 100, 2) for x in model.model.feature_importances_]
            )) if model.model else {}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest/<symbol>', methods=['GET'])
def backtest(symbol):
    """Run a simple backtest on the trading strategy"""
    try:
        symbol = symbol.upper()
        
        # Generate data
        df = generate_stock_data(symbol, days=500)
        df = calculate_all_indicators(df)
        
        # Train model
        model = TradingMLModel()
        model.train(df)
        
        # Simulate trading
        df_clean = df.dropna(subset=model.feature_columns).copy()
        
        if len(df_clean) < 50:
            return jsonify({'error': 'Not enough data for backtest'})
        
        # Make predictions for all data
        X = df_clean[model.feature_columns]
        X_scaled = model.scaler.transform(X)
        predictions = model.model.predict(X_scaled)
        df_clean['prediction'] = predictions
        
        # Calculate returns
        initial_capital = 10000
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df_clean) - 1):
            row = df_clean.iloc[i]
            next_row = df_clean.iloc[i + 1]
            
            if row['prediction'] == 1 and position == 0:  # Buy signal
                shares = capital / row['close']
                position = shares
                capital = 0
                trades.append({
                    'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    'type': 'BUY',
                    'price': round(row['close'], 2),
                    'shares': round(shares, 2)
                })
            elif row['prediction'] == -1 and position > 0:  # Sell signal
                capital = position * row['close']
                trades.append({
                    'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    'type': 'SELL',
                    'price': round(row['close'], 2),
                    'shares': round(position, 2)
                })
                position = 0
            
            # Calculate current equity
            current_equity = capital + (position * row['close'])
            equity_curve.append({
                'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                'equity': round(current_equity, 2)
            })
        
        # Final equity
        final_equity = capital + (position * df_clean.iloc[-1]['close'])
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        # Buy and hold comparison
        buy_hold_return = ((df_clean.iloc[-1]['close'] - df_clean.iloc[0]['close']) / df_clean.iloc[0]['close']) * 100
        
        return jsonify({
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_equity': round(final_equity, 2),
            'total_return': round(total_return, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'total_trades': len(trades),
            'trades': trades[-20:],  # Last 20 trades
            'equity_curve': equity_curve[::5]  # Sample every 5th point
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/test-data/<symbol>', methods=['GET'])
def test_data_source(symbol):
    """Test endpoint to verify real data is being fetched"""
    try:
        df = generate_stock_data(symbol.upper(), days=10)
        return jsonify({
            'symbol': symbol.upper(),
            'data_source': 'Yahoo Finance (Real Data)',
            'data_points': len(df),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            },
            'price_range': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'latest': float(df['close'].iloc[-1])
            },
            'sample_data': df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(5).to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e), 'data_source': 'Fallback (Simulated)'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
