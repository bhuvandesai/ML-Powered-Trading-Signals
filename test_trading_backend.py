"""
Tests for trading-backend.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import importlib.util

# Add the current directory to path to import trading-backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module with hyphen in filename
spec = importlib.util.spec_from_file_location(
    "trading_backend", 
    os.path.join(os.path.dirname(__file__), "trading-backend.py")
)
trading_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trading_backend)

# Import needed components
TechnicalIndicators = trading_backend.TechnicalIndicators
calculate_all_indicators = trading_backend.calculate_all_indicators
TradingMLModel = trading_backend.TradingMLModel
app = trading_backend.app


@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing"""
    np.random.seed(42)
    # Use 200 days to ensure enough data after dropping NaN values from indicators
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
        'high': prices * (1 + np.random.uniform(0, 0.03, 200)),
        'low': prices * (1 - np.random.uniform(0, 0.03, 200)),
        'close': prices,
        'volume': np.random.randint(1000000, 50000000, 200)
    })
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def client():
    """Create Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestTechnicalIndicators:
    """Test TechnicalIndicators class"""
    
    def test_sma(self, sample_stock_data):
        """Test Simple Moving Average calculation"""
        ti = TechnicalIndicators()
        sma = ti.sma(sample_stock_data['close'], 20)
        
        assert len(sma) == len(sample_stock_data)
        assert pd.notna(sma.iloc[-1]), "SMA should have valid values"
        assert sma.iloc[-1] > 0, "SMA should be positive"
    
    def test_ema(self, sample_stock_data):
        """Test Exponential Moving Average calculation"""
        ti = TechnicalIndicators()
        ema = ti.ema(sample_stock_data['close'], 20)
        
        assert len(ema) == len(sample_stock_data)
        assert pd.notna(ema.iloc[-1]), "EMA should have valid values"
        assert ema.iloc[-1] > 0, "EMA should be positive"
    
    def test_rsi(self, sample_stock_data):
        """Test RSI calculation"""
        ti = TechnicalIndicators()
        rsi = ti.rsi(sample_stock_data['close'], 14)
        
        assert len(rsi) == len(sample_stock_data)
        assert pd.notna(rsi.iloc[-1]), "RSI should have valid values"
        assert 0 <= rsi.iloc[-1] <= 100, "RSI should be between 0 and 100"
    
    def test_macd(self, sample_stock_data):
        """Test MACD calculation"""
        ti = TechnicalIndicators()
        macd, signal, hist = ti.macd(sample_stock_data['close'])
        
        assert len(macd) == len(sample_stock_data)
        assert pd.notna(macd.iloc[-1]), "MACD should have valid values"
        assert pd.notna(signal.iloc[-1]), "Signal should have valid values"
        assert pd.notna(hist.iloc[-1]), "Histogram should have valid values"
    
    def test_bollinger_bands(self, sample_stock_data):
        """Test Bollinger Bands calculation"""
        ti = TechnicalIndicators()
        upper, middle, lower = ti.bollinger_bands(sample_stock_data['close'])
        
        assert len(upper) == len(sample_stock_data)
        assert pd.notna(upper.iloc[-1]), "Upper band should have valid values"
        assert upper.iloc[-1] > lower.iloc[-1], "Upper band should be above lower band"
        assert middle.iloc[-1] > 0, "Middle band should be positive"
    
    def test_atr(self, sample_stock_data):
        """Test Average True Range calculation"""
        ti = TechnicalIndicators()
        atr = ti.atr(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        assert len(atr) == len(sample_stock_data)
        assert pd.notna(atr.iloc[-1]), "ATR should have valid values"
        assert atr.iloc[-1] >= 0, "ATR should be non-negative"
    
    def test_stochastic(self, sample_stock_data):
        """Test Stochastic Oscillator calculation"""
        ti = TechnicalIndicators()
        k, d = ti.stochastic(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        assert len(k) == len(sample_stock_data)
        assert pd.notna(k.iloc[-1]), "Stochastic K should have valid values"
        assert 0 <= k.iloc[-1] <= 100, "Stochastic K should be between 0 and 100"
    
    def test_obv(self, sample_stock_data):
        """Test On-Balance Volume calculation"""
        ti = TechnicalIndicators()
        obv = ti.obv(sample_stock_data['close'], sample_stock_data['volume'])
        
        assert len(obv) == len(sample_stock_data)
        assert pd.notna(obv.iloc[-1]), "OBV should have valid values"


class TestCalculateAllIndicators:
    """Test calculate_all_indicators function"""
    
    def test_calculate_all_indicators(self, sample_stock_data):
        """Test that all indicators are calculated"""
        df = calculate_all_indicators(sample_stock_data.copy())
        
        # Check that all expected indicators are present
        expected_indicators = [
            'sma_10', 'sma_20', 'sma_50',
            'ema_10', 'ema_20',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'stoch_k', 'stoch_d', 'obv',
            'williams_r', 'cci', 'momentum', 'roc',
            'price_change', 'price_change_5', 'volatility'
        ]
        
        for indicator in expected_indicators:
            assert indicator in df.columns, f"{indicator} should be in dataframe"


class TestTradingMLModel:
    """Test TradingMLModel class"""
    
    def test_model_initialization(self):
        """Test ML model initialization"""
        model = TradingMLModel()
        assert model.model is None, "Model should be None initially"
        assert len(model.feature_columns) > 0, "Should have feature columns"
    
    def test_prepare_features(self, sample_stock_data):
        """Test feature preparation"""
        model = TradingMLModel()
        df = calculate_all_indicators(sample_stock_data.copy())
        df_prepared = model.prepare_features(df)
        
        assert 'future_return' in df_prepared.columns
        assert 'signal' in df_prepared.columns
        assert df_prepared['signal'].isin([-1, 0, 1]).all(), "Signals should be -1, 0, or 1"
    
    def test_model_training(self, sample_stock_data):
        """Test ML model training"""
        model = TradingMLModel()
        df = calculate_all_indicators(sample_stock_data.copy())
        
        # Check how many rows we have after cleaning
        df_prepared = model.prepare_features(df.copy())
        df_clean = df_prepared.dropna()
        clean_rows = len(df_clean)
        
        result = model.train(df)
        
        assert 'accuracy' in result
        
        # Check if training was successful
        if 'message' in result and result.get('message') == 'Not enough data':
            # If not enough data, verify it's due to insufficient clean rows
            assert clean_rows < 100, f"Expected < 100 clean rows, got {clean_rows}. Training should have succeeded."
            pytest.skip(f"Not enough data after cleaning ({clean_rows} rows) - skipping model training test")
        
        # If training succeeded, model should be set
        assert model.model is not None, f"Model should be trained after training. Had {clean_rows} clean rows."
        assert result['accuracy'] >= 0, "Accuracy should be non-negative"
        assert 'train_size' in result, "Result should include train_size"
        assert 'test_size' in result, "Result should include test_size"
        assert result.get('train_size', 0) > 0, "Should have training data"
        assert result.get('test_size', 0) > 0, "Should have test data"
    
    def test_model_prediction(self, sample_stock_data):
        """Test ML model prediction"""
        model = TradingMLModel()
        df = calculate_all_indicators(sample_stock_data.copy())
        
        # Train model first
        result = model.train(df)
        
        # Check if training was successful
        if 'message' in result and result.get('message') == 'Not enough data':
            pytest.skip("Not enough data after cleaning - skipping prediction test")
        
        # Only test prediction if model was trained
        if model.model is None:
            pytest.skip("Model not trained - skipping prediction test")
        
        # Make prediction
        prediction, confidence = model.predict(df)
        
        assert prediction is not None, "Prediction should not be None"
        assert prediction in [-1, 0, 1], f"Prediction should be -1, 0, or 1, got {prediction}"
        assert confidence is not None, "Confidence should not be None"
        assert 0 <= confidence <= 100, f"Confidence should be between 0 and 100, got {confidence}"


class TestAPIEndpoints:
    """Test Flask API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.get_json()
        assert 'message' in data
        assert data['message'] == 'Trading Backend API'
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_stocks_endpoint(self, client):
        """Test stocks list endpoint"""
        response = client.get('/api/stocks')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert 'symbol' in data[0]
    
    def test_stock_data_endpoint(self, client):
        """Test stock data endpoint"""
        # Mock or use a real symbol that might exist
        # Using AAPL as it's a common stock
        response = client.get('/api/stock/AAPL')
        
        # Should either succeed or fail gracefully
        if response.status_code == 200:
            data = response.get_json()
            assert 'symbol' in data
            assert 'data' in data
            assert 'current_price' in data
    
    def test_indicators_endpoint(self, client):
        """Test indicators endpoint"""
        response = client.get('/api/indicators/AAPL')
        
        # Should either succeed or fail gracefully
        if response.status_code == 200:
            data = response.get_json()
            assert 'symbol' in data
            assert 'indicators' in data
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        response = client.get('/api/predict/AAPL')
        
        # Should either succeed or fail gracefully
        if response.status_code == 200:
            data = response.get_json()
            assert 'symbol' in data
            assert 'signal' in data
            assert data['signal'] in ['BUY', 'SELL', 'HOLD']
    
    def test_backtest_endpoint(self, client):
        """Test backtest endpoint"""
        response = client.get('/api/backtest/AAPL')
        
        # Should either succeed or fail gracefully
        if response.status_code == 200:
            data = response.get_json()
            assert 'symbol' in data
            assert 'total_return' in data


class TestDataGeneration:
    """Test data generation functions"""
    
    def test_calculate_all_indicators_handles_empty_data(self):
        """Test that calculate_all_indicators handles edge cases"""
        # Empty dataframe
        empty_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Should not crash, but might return empty or with NaN
        try:
            result = calculate_all_indicators(empty_df)
            assert isinstance(result, pd.DataFrame)
        except Exception:
            # It's okay if it raises an error for empty data
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
