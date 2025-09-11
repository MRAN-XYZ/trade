import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import warnings
import os
warnings.filterwarnings('ignore')

# Technical Analysis Libraries
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
import joblib
import pickle

class DerivTradingBot:
    def __init__(self, app_id, token, model_dir="models"):
        """
        Initialize the Deriv Trading Bot for JD10 (Jump 10 Index)
        
        Args:
            app_id: Deriv API app ID
            token: API token for authentication
            model_dir: Directory containing the trained models
        """
        self.app_id = app_id
        self.token = token
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        self.model_dir = model_dir
        
        # Data storage
        self.price_data = {
            '15min': deque(maxlen=500),  # Store last 500 candles
            '60min': deque(maxlen=200)
        }
        self.current_price = None
        
        # ML Models - will be loaded from files
        self.lstm_model_15min = None
        self.lstm_model_60min = None
        self.rf_model = None
        self.meta_model = None
        
        # Risk Management Parameters
        self.position_size = 0
        self.max_drawdown = 0.15  # 15% max drawdown
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.current_position = None  # 'BUY', 'SELL', or None
        self.entry_price = None
        self.account_balance = 10000  # Starting balance
        self.peak_balance = 10000
        
        # Kelly Criterion parameters
        self.win_rate = 0.5
        self.avg_win = 0.04
        self.avg_loss = 0.02
        
        # Scalers for ML - will be loaded from files
        self.scaler_15min = None
        self.scaler_60min = None
        
    def load_models(self):
        """Load all ML models and scalers from files"""
        try:
            print("Loading ML models and scalers...")
            
            # Use a helper dictionary to load models
            model_files = {
                'lstm_15min': ('lstm_model_15min.h5', load_model),
                'lstm_60min': ('lstm_model_60min.h5', load_model),
                'rf_model': ('random_forest_model.joblib', joblib.load),
                'meta_model': ('meta_model.joblib', joblib.load),
                'scaler_15min': ('scaler_15min.joblib', joblib.load),
                'scaler_60min': ('scaler_60min.joblib', joblib.load),
            }
            
            for model_name, (filename, loader_func) in model_files.items():
                path = os.path.join(self.model_dir, filename)
                if os.path.exists(path):
                    setattr(self, model_name, loader_func(path))
                    print(f"✓ {model_name} loaded")
                else:
                    print(f"⚠ {model_name} not found at {path}")
            
            # Load additional model parameters if available
            params_path = os.path.join(self.model_dir, "model_parameters.pkl")
            if os.path.exists(params_path):
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                    self.win_rate = params.get('win_rate', self.win_rate)
                    self.avg_win = params.get('avg_win', self.avg_win)
                    self.avg_loss = params.get('avg_loss', self.avg_loss)
                    print("✓ Model parameters loaded")
            
            print("Model loading completed!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def save_model_parameters(self):
        """Save current model parameters for future use"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            params = {
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'account_balance': self.account_balance,
                'peak_balance': self.peak_balance
            }
            
            params_path = os.path.join(self.model_dir, "model_parameters.pkl")
            with open(params_path, 'wb') as f:
                pickle.dump(params, f)
            print("Model parameters saved")
            
        except Exception as e:
            print(f"Error saving parameters: {e}")
    
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        
        # EMA
        df_copy['ema_12'] = ta.trend.ema_indicator(df_copy['close'], window=12)
        df_copy['ema_26'] = ta.trend.ema_indicator(df_copy['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df_copy['close'])
        df_copy['macd'] = macd.macd()
        df_copy['macd_signal'] = macd.macd_signal()
        df_copy['macd_diff'] = macd.macd_diff()
        
        # RSI
        df_copy['rsi'] = ta.momentum.RSIIndicator(df_copy['close']).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df_copy['close'])
        df_copy['bb_high'] = bb.bollinger_hband()
        df_copy['bb_mid'] = bb.bollinger_mavg()
        df_copy['bb_low'] = bb.bollinger_lband()
        
        return df_copy.drop(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
    
    def prepare_ml_features(self, indicators_df, prices_df):
        """Prepare features for ML models"""
        feature_matrix = indicators_df.copy()
        feature_matrix['price_change'] = prices_df['close'].pct_change()
        feature_matrix['high_close_ratio'] = prices_df['high'] / prices_df['close'] - 1
        feature_matrix['close_low_ratio'] = prices_df['close'] / prices_df['low'] - 1
    
        price_returns = prices_df['close'].pct_change()
        for window in [5, 10, 20]:
            feature_matrix[f'ret_{window}'] = price_returns.rolling(window).mean()
            feature_matrix[f'vol_{window}'] = price_returns.rolling(window).std()
    
        feature_matrix.columns = [str(col) for col in feature_matrix.columns]
        feature_matrix = feature_matrix.fillna(0).replace([np.inf, -np.inf], 0)
        return feature_matrix
    
    # *** FIXED ***: Corrected LSTM prediction logic to be more efficient.
    def generate_lstm_predictions(self, features_df, model, scaler, lookback):
        """Generate predictions using an LSTM model."""
        if model is None or scaler is None or len(features_df) < lookback:
            return 0.0 # Return a neutral prediction
        
        try:
            # Select the last 'lookback' steps
            last_sequence = features_df.iloc[-lookback:].values
            
            # Scale the sequence
            scaled_sequence = scaler.transform(last_sequence)
            
            # Reshape for LSTM: (1, lookback, num_features)
            X = np.reshape(scaled_sequence, (1, lookback, scaled_sequence.shape[1]))
            
            # Make prediction
            prediction = model.predict(X, verbose=0)
            
            # Assuming regression output, return the single value
            return prediction[0, 0]
            
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return 0.0 # Return neutral on error

    # *** FIXED ***: Simplified RF prediction logic.
    def generate_rf_prediction(self, features_df, model):
        """Generate predictions using the Random Forest model."""
        if model is None or features_df.empty:
            return 0.0 # Return neutral
        
        try:
            # Use the most recent set of features
            X_test = features_df.iloc[-1:]
            prediction = model.predict(X_test)
            return prediction[0]
                
        except Exception as e:
            print(f"RF prediction error: {e}")
            return 0.0 # Return neutral on error

    # *** NEW ***: The correct stacked prediction pipeline.
    def make_stacked_prediction(self, features_15min, features_60min):
        """
        Executes the full stacked ensemble prediction pipeline.
        1. Get predictions from all base models (LSTMs, RF).
        2. Combine these predictions into a feature vector.
        3. Feed the vector to the meta-model to get the final prediction.
        """
        if self.meta_model is None:
            print("Meta model not loaded. Cannot make a stacked prediction.")
            return np.array([0.33, 0.33, 0.34]) # Return neutral probabilities

        # --- Step 1: Get predictions from base models ---
        
        # Prediction from LSTM 15min model (assuming 15 step lookback)
        pred_lstm_15 = self.generate_lstm_predictions(features_15min, self.lstm_model_15min, self.scaler_15min, lookback=15)
        
        # Prediction from LSTM 60min model (assuming 60 step lookback)
        pred_lstm_60 = self.generate_lstm_predictions(features_60min, self.lstm_model_60min, self.scaler_60min, lookback=60)
        
        # Prediction from Random Forest model (using 15min features is standard)
        pred_rf = self.generate_rf_prediction(features_15min, self.rf_model)

        print(f"Base Model Predictions -> LSTM_15: {pred_lstm_15:.4f}, LSTM_60: {pred_lstm_60:.4f}, RF: {pred_rf:.4f}")

        # --- Step 2: Combine predictions for the meta-model ---
        # The order MUST match the training order of the meta-model.
        meta_features = np.array([[pred_lstm_15, pred_lstm_60, pred_rf]])
        
        # --- Step 3: Get final prediction from the meta-model ---
        try:
            final_prediction_proba = self.meta_model.predict_proba(meta_features)
            return final_prediction_proba[0] # Return probabilities [BUY, SELL, HOLD]
        except Exception as e:
            print(f"Meta-model prediction error: {e}")
            return np.array([0.33, 0.33, 0.34])

    def calculate_kelly_position_size(self, confidence):
        """Calculate position size using Kelly Criterion"""
        if self.avg_loss == 0: return 0.0
        
        b = self.avg_win / self.avg_loss
        p = self.win_rate * confidence
        q = 1 - p
        
        kelly_fraction = max(0, (b * p - q) / b)
        safe_kelly = kelly_fraction * 0.25 # Use a fractional Kelly for safety
        
        return min(safe_kelly, 0.1) # Cap at 10% of capital

    # *** FIXED ***: Corrected typo `en_price` to `entry_price`.
    def check_risk_limits(self):
        """Check if we're within risk limits"""
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                return False, "Maximum drawdown exceeded"
        
        if self.current_position and self.entry_price:
            pnl_direction = 1 if self.current_position == 'BUY' else -1
            current_pnl_pct = pnl_direction * (self.current_price - self.entry_price) / self.entry_price
            
            if current_pnl_pct <= -self.stop_loss_pct:
                return False, "Stop loss hit"
            elif current_pnl_pct >= self.take_profit_pct:
                return False, "Take profit hit"
        
        return True, "Within risk limits"
    
    # *** REFACTORED ***: This function now prepares data and calls the new prediction pipeline.
    def generate_trading_signal(self):
        """Generate trading signal by preparing data and calling the stacked prediction pipeline."""
        
        # Ensure we have enough data for the longest lookback (60 for lstm_60min)
        if len(self.price_data['15min']) < 60 or len(self.price_data['60min']) < 60:
            return None # Not enough data to make a prediction

        # Prepare 15-minute features
        df_15min = pd.DataFrame(list(self.price_data['15min']))
        indicators_15min = self.calculate_technical_indicators(df_15min)
        features_15min = self.prepare_ml_features(indicators_15min, df_15min)

        # Prepare 60-minute features
        df_60min = pd.DataFrame(list(self.price_data['60min']))
        indicators_60min = self.calculate_technical_indicators(df_60min)
        features_60min = self.prepare_ml_features(indicators_60min, df_60min)

        # Get the final combined prediction from the stacked ensemble
        final_prediction_proba = self.make_stacked_prediction(features_15min, features_60min)
        
        # Determine action and confidence
        # Assuming the meta-model classes are ordered: 0=BUY, 1=SELL, 2=HOLD
        actions = ['BUY', 'SELL', 'HOLD']
        action_idx = np.argmax(final_prediction_proba)
        action = actions[action_idx]
        confidence = final_prediction_proba[action_idx]
        
        # Calculate position size
        position_size = self.calculate_kelly_position_size(confidence)
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'predictions': final_prediction_proba
        }
    
    async def connect_and_trade(self):
        """Main trading loop with WebSocket connection"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Authorize
                await websocket.send(json.dumps({"authorize": self.token}))
                print(f"Authorization response: {await websocket.recv()}")
                
                # Subscribe to JD10 ticks
                await websocket.send(json.dumps({"ticks": "JD10", "subscribe": 1}))
                
                # Subscribe to candles for both timeframes
                for granularity in [900, 3600]:  # 15min and 60min
                    await websocket.send(json.dumps({
                        "ticks_history": "JD10", "adjust_start_time": 1, "count": 500,
                        "end": "latest", "granularity": granularity, "style": "candles", "subscribe": 1
                    }))
                
                if not self.load_models():
                    print("CRITICAL: Models could not be loaded. Exiting.")
                    return
                
                print("\nBot is live and listening for market data...\n")
                
                async for message in websocket:
                    data = json.loads(message)
                    
                    if 'tick' in data:
                        self.current_price = float(data['tick']['quote'])
                        await self.process_trading_decision()
                    
                    elif 'ohlc' in data: # Handle real-time candle updates
                        self.update_candle_data([data['ohlc']], data['ohlc']['granularity'])
                    
                    elif 'candles' in data: # Handle initial history
                        self.update_candle_data(data['candles'], data['echo_req']['granularity'])

                    elif 'error' in data:
                        print(f"API Error: {data['error']['message']}")
                        
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}. Reconnecting...")
            await asyncio.sleep(5)
            await self.connect_and_trade()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            self.save_model_parameters()

    def update_candle_data(self, candles, granularity):
        """Update candle data storage based on granularity."""
        timeframe = '15min' if granularity == 900 else '60min'
        
        for candle in candles:
            candle_data = {
                'timestamp': candle['epoch'], 'open': float(candle['open']),
                'high': float(candle['high']), 'low': float(candle['low']),
                'close': float(candle['close']), 'volume': 1000 # Synthetic volume
            }
            # Append if new or update if it's the latest candle
            if self.price_data[timeframe] and self.price_data[timeframe][-1]['timestamp'] == candle['epoch']:
                self.price_data[timeframe][-1] = candle_data
            else:
                self.price_data[timeframe].append(candle_data)
    
    async def process_trading_decision(self):
        """Process trading decision based on signals"""
        within_limits, message = self.check_risk_limits()
        if not within_limits:
            print(f"Risk limit reached: {message}")
            if self.current_position:
                await self.close_position(message)
            return
        
        signal = self.generate_trading_signal()
        if signal is None:
            return # Not enough data yet
        
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Price: ${self.current_price:.2f} | "
            f"Signal: {signal['action']} ({signal['confidence']:.1%}) | "
            f"Position: {self.current_position or 'None'} | "
            f"Balance: ${self.account_balance:.2f}"
        )
        
        # Execute trade based on signal
        if signal['confidence'] > 0.65 and signal['action'] != 'HOLD':
            await self.execute_trade(signal)
    
    async def execute_trade(self, signal):
        """Execute trade based on signal"""
        action = signal['action']
        
        if self.current_position == 'BUY' and action == 'SELL':
            await self.close_position("Signal reversal")
            self.open_position('SELL', signal['position_size'])
            
        elif self.current_position == 'SELL' and action == 'BUY':
            await self.close_position("Signal reversal")
            self.open_position('BUY', signal['position_size'])
            
        elif self.current_position is None and action in ['BUY', 'SELL']:
            self.open_position(action, signal['position_size'])

    def open_position(self, action, position_size):
        """Helper to open a new position."""
        self.current_position = action
        self.entry_price = self.current_price
        self.position_size = position_size
        print(f"--- Opened {action} position of size {position_size:.2%} at ${self.current_price:.2f} ---")
    
    async def close_position(self, reason):
        """Close current position"""
        if not self.current_position: return
        
        pnl_direction = 1 if self.current_position == 'BUY' else -1
        pnl_pct = pnl_direction * (self.current_price - self.entry_price) / self.entry_price
        pnl_amount = self.account_balance * self.position_size * pnl_pct
        self.account_balance += pnl_amount
        
        self.peak_balance = max(self.peak_balance, self.account_balance)
        
        # Update Kelly parameters with exponential moving average
        if pnl_pct > 0:
            self.win_rate = (self.win_rate * 0.95) + (1 * 0.05)
            self.avg_win = (self.avg_win * 0.95) + (pnl_pct * 0.05)
        else:
            self.win_rate = (self.win_rate * 0.95) + (0 * 0.05)
            self.avg_loss = (self.avg_loss * 0.95) + (abs(pnl_pct) * 0.05)
        
        print(f"--- Closed {self.current_position} position for reason: {reason} | P&L: ${pnl_amount:.2f} ({pnl_pct:.2%}) ---")
        
        self.current_position, self.entry_price, self.position_size = None, None, 0

async def main():
    APP_ID = "96329"
    TOKEN = "lWFYLtfTp2sbWl8"
    MODEL_DIR = "models"
    
    bot = DerivTradingBot(APP_ID, TOKEN, MODEL_DIR)
    
    print("Starting Deriv JD10 Trading Bot...")
    await bot.connect_and_trade()

if __name__ == "__main__":
    # Ensure you have your credentials before running
    if "YOUR_APP_ID" in DerivTradingBot.__init__.__defaults__[0]:
         print("Please replace 'YOUR_APP_ID' and 'YOUR_API_TOKEN' in the main function before running.")
    else:
        asyncio.run(main())