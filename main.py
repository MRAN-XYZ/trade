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
        self.lstm_model_15 = None
        self.lstm_model_60 = None
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
        self.scaler_15 = None
        self.scaler_60 = None
        
    def load_models(self):
        """Load all ML models and scalers from files"""
        try:
            print("Loading ML models and scalers...")
            
            # Load LSTM models
            lstm_15_path = os.path.join(self.model_dir, "lstm_model_15min.h5")
            lstm_60_path = os.path.join(self.model_dir, "lstm_model_60min.h5")
            
            if os.path.exists(lstm_15_path):
                self.lstm_model_15 = load_model(lstm_15_path)
                print("✓ LSTM 15-minute model loaded")
            else:
                print(f"⚠ LSTM 15-minute model not found at {lstm_15_path}")
                
            if os.path.exists(lstm_60_path):
                self.lstm_model_60 = load_model(lstm_60_path)
                print("✓ LSTM 60-minute model loaded")
            else:
                print(f"⚠ LSTM 60-minute model not found at {lstm_60_path}")
            
            # Load Random Forest model
            rf_path = os.path.join(self.model_dir, "random_forest_model.joblib")
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                print("✓ Random Forest model loaded")
            else:
                print(f"⚠ Random Forest model not found at {rf_path}")
                # Fallback to untrained model
                self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Load Meta-learner model
            meta_path = os.path.join(self.model_dir, "meta_model.joblib")
            if os.path.exists(meta_path):
                self.meta_model = joblib.load(meta_path)
                print("✓ Meta-learner model loaded")
            else:
                print(f"⚠ Meta-learner model not found at {meta_path}")
                # Fallback to untrained model
                self.meta_model = LogisticRegression()
            
            # Load scalers
            scaler_15_path = os.path.join(self.model_dir, "scaler_15min.joblib")
            scaler_60_path = os.path.join(self.model_dir, "scaler_60min.joblib")
            
            if os.path.exists(scaler_15_path):
                self.scaler_15 = joblib.load(scaler_15_path)
                print("✓ 15-minute scaler loaded")
            else:
                print(f"⚠ 15-minute scaler not found at {scaler_15_path}")
                self.scaler_15 = StandardScaler()
                
            if os.path.exists(scaler_60_path):
                self.scaler_60 = joblib.load(scaler_60_path)
                print("✓ 60-minute scaler loaded")
            else:
                print(f"⚠ 60-minute scaler not found at {scaler_60_path}")
                self.scaler_60 = StandardScaler()
            
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
            print("Falling back to default models...")
            # Initialize fallback models
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.meta_model = LogisticRegression()
            self.scaler_15 = StandardScaler()
            self.scaler_60 = StandardScaler()
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
        indicators = {}
        
        # EMA
        indicators['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        indicators['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        indicators['macd'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()
        indicators['macd_diff'] = macd.macd_diff()
        
        # RSI
        indicators['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        indicators['bb_high'] = bb.bollinger_hband()
        indicators['bb_mid'] = bb.bollinger_mavg()
        indicators['bb_low'] = bb.bollinger_lband()
        indicators['bb_width'] = bb.bollinger_wband()
        indicators['bb_pct'] = bb.bollinger_pband()
        
        # Additional features
        indicators['volume_sma'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        return pd.DataFrame(indicators)
    
    def prepare_ml_features(self, indicators_df, prices_df):
        """Prepare features for ML models"""
        # Start with indicators DataFrame (already has proper string column names)
        feature_matrix = indicators_df.copy()
    
        # Add price-based features with explicit string column names
        feature_matrix['price_change'] = prices_df['close'].pct_change().fillna(0)
        feature_matrix['high_close_ratio'] = prices_df['high'] / prices_df['close'] - 1
        feature_matrix['close_low_ratio'] = prices_df['close'] / prices_df['low'] - 1
    
        # Add rolling statistics with explicit string column names
        price_returns = prices_df['close'].pct_change()
        for window in [5, 10, 20]:
            feature_matrix[f'ret_{window}'] = price_returns.rolling(window).mean().fillna(0)
            feature_matrix[f'vol_{window}'] = price_returns.rolling(window).std().fillna(0)
    
        # Ensure all columns are strings
        feature_matrix.columns = [str(col) for col in feature_matrix.columns]
    
        # Fill any remaining NaN values
        feature_matrix = feature_matrix.fillna(0)
    
        return feature_matrix
    
    def generate_lstm_predictions(self, features, model, scaler, lookback=20):
        """Generate predictions using LSTM"""
        if model is None or len(features) < lookback:
            return np.array([0.33, 0.33, 0.34])  # Equal probabilities
        
        try:
            # Prepare sequence data
            scaled_features = scaler.transform(features)  # Use transform, not fit_transform
            X = []
            for i in range(lookback, len(scaled_features)):
                X.append(scaled_features[i-lookback:i])
            
            if len(X) == 0:
                return np.array([0.33, 0.33, 0.34])
            
            X = np.array(X)
            
            # Make prediction
            prediction = model.predict(X[-1:], verbose=0)
            return prediction[0]  # [BUY_prob, SELL_prob, HOLD_prob]
            
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return np.array([0.33, 0.33, 0.34])
    
    def generate_rf_predictions(self, features):
        """Generate predictions using Random Forest"""
        if self.rf_model is None or len(features) < 50:
            return np.array([0.33, 0.33, 0.34])
    
        try:
            # If model is already trained, just predict
            if hasattr(self.rf_model, 'classes_') and len(self.rf_model.classes_) > 0:
                X_test = features.iloc[-1:]
                proba = self.rf_model.predict_proba(X_test)[0]
                
                # Handle different numbers of classes
                if len(proba) == 2:
                    # If only 2 classes were learned, pad with zeros
                    if 0 in self.rf_model.classes_ and 1 in self.rf_model.classes_:
                        return np.array([proba[0], proba[1], 0.0])
                    elif 0 in self.rf_model.classes_ and 2 in self.rf_model.classes_:
                        return np.array([proba[0], 0.0, proba[1]])
                    elif 1 in self.rf_model.classes_ and 2 in self.rf_model.classes_:
                        return np.array([0.0, proba[0], proba[1]])
                    else:
                        return np.array([0.33, 0.33, 0.34])
                elif len(proba) == 3:
                    # All 3 classes present - map to correct order [BUY, SELL, HOLD]
                    result = np.zeros(3)
                    for i, class_label in enumerate(self.rf_model.classes_):
                        result[int(class_label)] = proba[i]
                    return result
                else:
                    return np.array([0.33, 0.33, 0.34])
            else:
                # Model not trained yet, return neutral
                return np.array([0.33, 0.33, 0.34])
                
        except Exception as e:
            print(f"RF prediction error: {e}")
            return np.array([0.33, 0.33, 0.34])
    
    def combine_predictions_meta_learner(self, predictions_dict):
        """Combine all predictions using meta-learner"""
        try:
            # Stack all predictions
            all_predictions = []
            for timeframe in ['15min', '60min']:
                all_predictions.extend([
                    predictions_dict[timeframe]['lstm'],
                    predictions_dict[timeframe]['rf'],
                    predictions_dict[timeframe]['technical']
                ])
            
            # Flatten to feature vector
            meta_features = np.concatenate(all_predictions).reshape(1, -1)
            
            # If we have trained meta-learner, use it
            if self.meta_model is not None and hasattr(self.meta_model, 'classes_'):
                meta_proba = self.meta_model.predict_proba(meta_features)[0]
                return meta_proba
            else:
                # Simple averaging if no meta-model trained yet
                avg_predictions = np.mean([pred for pred in all_predictions], axis=0)
                return avg_predictions
                
        except Exception as e:
            print(f"Meta-learner prediction error: {e}")
            # Fallback to simple averaging
            all_predictions = []
            for timeframe in ['15min', '60min']:
                all_predictions.extend([
                    predictions_dict[timeframe]['lstm'],
                    predictions_dict[timeframe]['rf'],
                    predictions_dict[timeframe]['technical']
                ])
            return np.mean([pred for pred in all_predictions], axis=0)
    
    def calculate_kelly_position_size(self, confidence):
        """Calculate position size using Kelly Criterion"""
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = probability of win, q = probability of loss
        b = self.avg_win / self.avg_loss
        p = self.win_rate * confidence  # Adjust by confidence
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety factor (use 25% of Kelly)
        safe_kelly = kelly_fraction * 0.25
        
        # Ensure between 0 and max position size (10% of capital)
        return max(0, min(safe_kelly, 0.1))
    
    def check_risk_limits(self):
        """Check if we're within risk limits"""
        # Calculate current drawdown
        current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
        
        if current_drawdown > self.max_drawdown:
            return False, "Maximum drawdown exceeded"
        
        # Check if current position hits stop loss or take profit
        if self.current_position and self.entry_price:
            current_pnl = (self.current_price - self.entry_price) / self.en_price
            
            if self.current_position == 'BUY':
                if current_pnl <= -self.stop_loss_pct:
                    return False, "Stop loss hit"
                elif current_pnl >= self.take_profit_pct:
                    return False, "Take profit hit"
            elif self.current_position == 'SELL':
                if -current_pnl <= -self.stop_loss_pct:
                    return False, "Stop loss hit"
                elif -current_pnl >= self.take_profit_pct:
                    return False, "Take profit hit"
        
        return True, "Within risk limits"
    
    def generate_trading_signal(self):
        """Generate trading signal combining all models and indicators"""
        signals = {'15min': {}, '60min': {}}
        
        for timeframe in ['15min', '60min']:
            if len(self.price_data[timeframe]) < 100:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(list(self.price_data[timeframe]))
            df.columns = [str(col) for col in df.columns]
            
            # Calculate indicators
            indicators = self.calculate_technical_indicators(df)
            
            # Technical analysis signals
            tech_signal = self.get_technical_signal(df, indicators)
            signals[timeframe]['technical'] = tech_signal
            
            # Prepare ML features
            features = self.prepare_ml_features(indicators, df)
            
            #Ensure all column names are strings
            features.columns = [str(col) for col in features.columns]
            
            # LSTM predictions
            if timeframe == '15min':
                lstm_pred = self.generate_lstm_predictions(
                    features, self.lstm_model_15, self.scaler_15
                )
                signals[timeframe]['lstm'] = lstm_pred
            elif timeframe == '60min':
                lstm_pred = self.generate_lstm_predictions(
                    features, self.lstm_model_60, self.scaler_60
                )
                signals[timeframe]['lstm'] = lstm_pred
            else:
                signals[timeframe]['lstm'] = np.array([0.33, 0.33, 0.34])
            
            # Random Forest predictions
            rf_pred = self.generate_rf_predictions(features)
            signals[timeframe]['rf'] = rf_pred
        
        # Combine all signals using meta-learner
        final_prediction = self.combine_predictions_meta_learner(signals)
        
        # Determine action and confidence
        action_idx = np.argmax(final_prediction)
        actions = ['BUY', 'SELL', 'HOLD']
        action = actions[action_idx]
        confidence = final_prediction[action_idx]
        
        # Calculate position size
        position_size = self.calculate_kelly_position_size(confidence)
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'predictions': final_prediction,
            'signals': signals
        }
    
    def get_technical_signal(self, df, indicators):
        """Generate signal from technical indicators"""
        last_row = indicators.iloc[-1]
        scores = []
        
        # EMA Signal
        if last_row['ema_12'] > last_row['ema_26']:
            scores.append(1)  # Bullish
        else:
            scores.append(-1)  # Bearish
        
        # MACD Signal
        if last_row['macd'] > last_row['macd_signal']:
            scores.append(1)
        else:
            scores.append(-1)
        
        # RSI Signal
        if last_row['rsi'] < 30:
            scores.append(1)  # Oversold - Buy
        elif last_row['rsi'] > 70:
            scores.append(-1)  # Overbought - Sell
        else:
            scores.append(0)  # Neutral
        
        # Bollinger Bands Signal
        last_price = df['close'].iloc[-1]
        if last_price < last_row['bb_low']:
            scores.append(1)  # Below lower band - Buy
        elif last_price > last_row['bb_high']:
            scores.append(-1)  # Above upper band - Sell
        else:
            scores.append(0)  # Within bands
        
        # Average score
        avg_score = np.mean(scores)
        
        # Convert to probabilities
        if avg_score > 0.3:
            return np.array([0.6, 0.2, 0.2])  # BUY
        elif avg_score < -0.3:
            return np.array([0.2, 0.6, 0.2])  # SELL
        else:
            return np.array([0.2, 0.2, 0.6])  # HOLD
    
    async def connect_and_trade(self):
        """Main trading loop with WebSocket connection"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Authorize
                auth_request = {
                    "authorize": self.token
                }
                await websocket.send(json.dumps(auth_request))
                auth_response = await websocket.recv()
                print(f"Authorization response: {auth_response}")
                
                # Subscribe to JD10 ticks
                subscribe_request = {
                    "ticks": "JD10",
                    "subscribe": 1
                }
                await websocket.send(json.dumps(subscribe_request))
                
                # Subscribe to candles for both timeframes
                for granularity in [900, 3600]:  # 15min and 60min in seconds
                    candle_request = {
                        "ticks_history": "JD10",
                        "adjust_start_time": 1,
                        "count": 200,
                        "end": "latest",
                        "granularity": granularity,
                        "style": "candles",
                        "subscribe": 1
                    }
                    await websocket.send(json.dumps(candle_request))
                
                # Load models from files
                if not self.load_models():
                    print("Warning: Some models could not be loaded, using fallback models")
                
                # Trading loop
                tick_count = 0
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    # Handle tick data
                    if 'tick' in data:
                        tick = data['tick']
                        self.current_price = float(tick['quote'])
                        
                        await self.process_trading_decision()
                        print("done processing, exiting now")
                        break
                    
                    # Handle candle data
                    elif 'candles' in data:
                        candles = data['candles']
                        self.update_candle_data(candles, data.get('req_id', ''))
                    
                    # Handle errors
                    elif 'error' in data:
                        print(f"Error: {data['error']['message']}")
                        
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            # Save parameters when exiting
            self.save_model_parameters()
    
    def update_candle_data(self, candles, req_id):
        """Update candle data storage"""
        # Determine timeframe based on request
        for candle in candles:
            candle_data = {
                'timestamp': candle['epoch'],
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': 1000  # Synthetic volume
            }
            
            # Simple logic - you'd need to track req_id properly
            if '900' in str(req_id) or len(self.price_data['15min']) < 100:
                self.price_data['15min'].append(candle_data)
            else:
                self.price_data['60min'].append(candle_data)
    
    async def process_trading_decision(self):
        """Process trading decision based on signals"""
        # Check risk limits first
        within_limits, message = self.check_risk_limits()
        
        if not within_limits:
            print(f"Risk limit reached: {message}")
            if self.current_position:
                await self.close_position(message)
            return
        
        # Generate trading signal
        signal = self.generate_trading_signal()
        
        # Display signal
        print(f"\n{'='*50}")
        print(f"Timestamp: {datetime.now()}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Signal: {signal['action']}")
        print(f"Confidence: {signal['confidence']:.2%}")
        print(f"Position Size: {signal['position_size']:.2%}")
        print(f"Current Position: {self.current_position}")
        print(f"Account Balance: ${self.account_balance:.2f}")
        
        # Execute trade based on signal
        if signal['confidence'] > 0.6:  # Only trade with >60% confidence
            await self.execute_trade(signal)
    
    async def execute_trade(self, signal):
        """Execute trade based on signal"""
        action = signal['action']
        
        # Close opposite positions
        if self.current_position == 'BUY' and action == 'SELL':
            await self.close_position("Signal reversal")
            self.current_position = 'SELL'
            self.entry_price = self.current_price
            self.position_size = signal['position_size']
            print(f"Opened SELL position at ${self.current_price:.2f}")
            
        elif self.current_position == 'SELL' and action == 'BUY':
            await self.close_position("Signal reversal")
            self.current_position = 'BUY'
            self.entry_price = self.current_price
            self.position_size = signal['position_size']
            print(f"Opened BUY position at ${self.current_price:.2f}")
            
        elif self.current_position is None and action in ['BUY', 'SELL']:
            self.current_position = action
            self.entry_price = self.current_price
            self.position_size = signal['position_size']
            print(f"Opened {action} position at ${self.current_price:.2f}")
    
    async def close_position(self, reason):
        """Close current position"""
        if not self.current_position:
            return
        
        # Calculate P&L
        if self.current_position == 'BUY':
            pnl_pct = (self.current_price - self.entry_price) / self.entry_price
        else:  # SELL
            pnl_pct = (self.entry_price - self.current_price) / self.entry_price
        
        pnl_amount = self.account_balance * self.position_size * pnl_pct
        self.account_balance += pnl_amount
        
        # Update peak balance
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        
        # Update Kelly parameters
        if pnl_pct > 0:
            self.win_rate = self.win_rate * 0.95 + 0.05  # Exponential moving average
            self.avg_win = self.avg_win * 0.95 + pnl_pct * 0.05
        else:
            self.win_rate = self.win_rate * 0.95
            self.avg_loss = self.avg_loss * 0.95 + abs(pnl_pct) * 0.05
        
        print(f"Closed {self.current_position} position: P&L = ${pnl_amount:.2f} ({pnl_pct:.2%})")
        print(f"Reason: {reason}")
        
        self.current_position = None
        self.entry_price = None
        self.position_size = 0


async def main():
    # Initialize bot with your Deriv credentials
    APP_ID = "96329"
    TOKEN = "lWFYLtfTp2sbWl8"
    MODEL_DIR = "models"  # Directory containing your trained models
    
    bot = DerivTradingBot(APP_ID, TOKEN, MODEL_DIR)
    
    print("Starting Deriv JD10 Trading Bot...")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Initial Balance: ${bot.account_balance}")
    print(f"Max Drawdown: {bot.max_drawdown:.1%}")
    print(f"Stop Loss: {bot.stop_loss_pct:.1%}")
    print(f"Take Profit: {bot.take_profit_pct:.1%}")
    
    # Start trading
    await bot.connect_and_trade()


if __name__ == "__main__":
    asyncio.run(main())