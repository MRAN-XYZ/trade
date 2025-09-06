import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DerivTradingBot:
    def __init__(self, app_id: str, api_token: str):
        self.app_id = app_id
        self.api_token = api_token
        self.ws = None
        self.symbol = "R_10"  # JD10 Jump Index
        self.timeframes = [15, 60]  # minutes
        
        # Data storage
        self.price_data = pd.DataFrame()
        self.features_data = pd.DataFrame()
        
        # Models
        self.models = {}
        self.scaler = StandardScaler()
        self.meta_learner = LogisticRegression()
        
        # Trading parameters
        self.position_size = 0.01  # Base position size
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_drawdown = 0.20  # 20% max drawdown
        self.account_balance = 1000  # Starting balance
        self.current_balance = 1000
        self.peak_balance = 1000
        
        # Risk management
        self.open_positions = {}
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Signal storage
        self.signals_15m = {}
        self.signals_60m = {}
        self.confidence_15m = 0.0
        self.confidence_60m = 0.0

    async def connect(self):
        """Connect to Deriv WebSocket API"""
        uri = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        try:
            self.ws = await websockets.connect(uri)
            logger.info("Connected to Deriv WebSocket API")
            
            # Authorize
            auth_request = {
                "authorize": self.api_token
            }
            await self.ws.send(json.dumps(auth_request))
            response = await self.ws.recv()
            auth_data = json.loads(response)
            
            if "authorize" in auth_data:
                logger.info("Authorization successful")
                return True
            else:
                logger.error("Authorization failed")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def subscribe_to_ticks(self):
        """Subscribe to tick data for JD10"""
        tick_request = {
            "ticks": self.symbol,
            "subscribe": 1
        }
        await self.ws.send(json.dumps(tick_request))
        logger.info(f"Subscribed to {self.symbol} ticks")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if len(df) < 100:
            return df
            
        # 1. Exponential Moving Average (EMA)
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # 2. MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # 3. RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # 4. Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 5. ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Additional features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        df['volume_sma'] = df['volume'].rolling(window=20).mean() if 'volume' in df.columns else 0
        
        return df.dropna()

    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Detect support and resistance levels"""
        if len(df) < window * 2:
            return 0.0, 0.0
            
        # Use local minima/maxima for support/resistance
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        resistance_levels = df[df['high'] == highs]['high'].tail(5).mean()
        support_levels = df[df['low'] == lows]['low'].tail(5).mean()
        
        return support_levels, resistance_levels

    def create_lstm_model(self, input_shape: Tuple) -> Sequential:
        """Create LSTM model for price prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model

    def create_autoencoder(self, input_dim: int) -> Sequential:
        """Create autoencoder for anomaly detection"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(input_dim, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features = df.copy()
        
        # Technical indicator signals
        features['ema_signal'] = np.where(features['ema_9'] > features['ema_21'], 1, 0)
        features['macd_signal_flag'] = np.where(features['macd'] > features['macd_signal'], 1, 0)
        features['rsi_signal'] = np.where(features['rsi'] < 30, 1, np.where(features['rsi'] > 70, -1, 0))
        features['bb_signal'] = np.where(features['close'] < features['bb_lower'], 1, 
                                       np.where(features['close'] > features['bb_upper'], -1, 0))
        
        # Price momentum features
        features['momentum_5'] = features['close'].pct_change(5)
        features['momentum_10'] = features['close'].pct_change(10)
        features['momentum_20'] = features['close'].pct_change(20)
        
        # Support/Resistance features
        support, resistance = self.detect_support_resistance(features)
        features['distance_to_support'] = (features['close'] - support) / features['close']
        features['distance_to_resistance'] = (resistance - features['close']) / features['close']
        
        # Create target variable (future price movement)
        features['future_return_15m'] = features['close'].shift(-15).pct_change()
        features['future_return_60m'] = features['close'].shift(-60).pct_change()
        
        # Binary target: 1 if price goes up, 0 if down
        features['target_15m'] = np.where(features['future_return_15m'] > 0, 1, 0)
        features['target_60m'] = np.where(features['future_return_60m'] > 0, 1, 0)
        
        return features.dropna()

    def train_models(self, features_df: pd.DataFrame):
        """Train all ML models"""
        if len(features_df) < 500:
            logger.warning("Insufficient data for model training")
            return
            
        # Prepare feature columns
        feature_cols = ['ema_9', 'ema_21', 'ema_50', 'macd', 'macd_signal', 'macd_histogram',
                       'rsi', 'bb_width', 'atr', 'volatility', 'ema_signal', 'macd_signal_flag',
                       'rsi_signal', 'bb_signal', 'momentum_5', 'momentum_10', 'momentum_20',
                       'distance_to_support', 'distance_to_resistance']
        
        X = features_df[feature_cols].values
        y_15m = features_df['target_15m'].values
        y_60m = features_df['target_60m'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train_15m, y_test_15m = train_test_split(X_scaled, y_15m, test_size=0.2, random_state=42)
        _, _, y_train_60m, y_test_60m = train_test_split(X_scaled, y_60m, test_size=0.2, random_state=42)
        
        # Train Random Forest
        self.models['rf_15m'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['rf_60m'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['rf_15m'].fit(X_train, y_train_15m)
        self.models['rf_60m'].fit(X_train, y_train_60m)
        
        # Train XGBoost
        self.models['xgb_15m'] = xgb.XGBClassifier(random_state=42)
        self.models['xgb_60m'] = xgb.XGBClassifier(random_state=42)
        self.models['xgb_15m'].fit(X_train, y_train_15m)
        self.models['xgb_60m'].fit(X_train, y_train_60m)
        
        # Train LSTM
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        self.models['lstm_15m'] = self.create_lstm_model((1, X_train.shape[1]))
        self.models['lstm_60m'] = self.create_lstm_model((1, X_train.shape[1]))
        
        self.models['lstm_15m'].fit(X_train_lstm, y_train_15m, epochs=50, batch_size=32, verbose=0)
        self.models['lstm_60m'].fit(X_train_lstm, y_train_60m, epochs=50, batch_size=32, verbose=0)
        
        # Train Autoencoder for anomaly detection
        self.models['autoencoder'] = self.create_autoencoder(X_train.shape[1])
        self.models['autoencoder'].fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)
        
        # Train Meta-Learner
        rf_pred_15m = self.models['rf_15m'].predict_proba(X_test)[:, 1]
        xgb_pred_15m = self.models['xgb_15m'].predict_proba(X_test)[:, 1]
        lstm_pred_15m = self.models['lstm_15m'].predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).flatten()
        
        meta_features = np.column_stack([rf_pred_15m, xgb_pred_15m, lstm_pred_15m])
        self.meta_learner.fit(meta_features, y_test_15m)
        
        logger.info("All models trained successfully")

    def generate_signals(self, current_features: np.ndarray) -> Dict:
        """Generate trading signals from all algorithms"""
        signals = {}
        
        if not self.models or len(current_features) == 0:
            return signals
            
        try:
            # Scale features
            features_scaled = self.scaler.transform(current_features.reshape(1, -1))
            features_lstm = features_scaled.reshape((1, 1, features_scaled.shape[1]))
            
            # Technical Analysis Signals
            signals['ema'] = self.get_ema_signal(current_features)
            signals['macd'] = self.get_macd_signal(current_features)
            signals['rsi'] = self.get_rsi_signal(current_features)
            signals['bb'] = self.get_bollinger_signal(current_features)
            signals['atr'] = self.get_atr_signal(current_features)
            signals['support_resistance'] = self.get_support_resistance_signal(current_features)
            
            # ML Model Predictions
            for timeframe in ['15m', '60m']:
                signals[f'rf_{timeframe}'] = self.models[f'rf_{timeframe}'].predict_proba(features_scaled)[0][1]
                signals[f'xgb_{timeframe}'] = self.models[f'xgb_{timeframe}'].predict_proba(features_scaled)[0][1]
                signals[f'lstm_{timeframe}'] = self.models[f'lstm_{timeframe}'].predict(features_lstm)[0][0]
            
            # Autoencoder anomaly score
            reconstruction = self.models['autoencoder'].predict(features_scaled, verbose=0)
            signals['anomaly_score'] = np.mean(np.square(features_scaled - reconstruction))
            
            # Meta-learner prediction
            meta_features = np.array([[signals['rf_15m'], signals['xgb_15m'], signals['lstm_15m']]])
            signals['meta_prediction'] = self.meta_learner.predict_proba(meta_features)[0][1]
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals

    def get_ema_signal(self, features: np.ndarray) -> float:
        """EMA crossover signal"""
        ema_9 = features[0]  # Assuming first feature is EMA 9
        ema_21 = features[1]  # Assuming second feature is EMA 21
        return 1.0 if ema_9 > ema_21 else 0.0

    def get_macd_signal(self, features: np.ndarray) -> float:
        """MACD signal"""
        macd = features[3]  # Assuming MACD is 4th feature
        macd_signal = features[4]  # Assuming MACD signal is 5th feature
        return 1.0 if macd > macd_signal else 0.0

    def get_rsi_signal(self, features: np.ndarray) -> float:
        """RSI signal"""
        rsi = features[6]  # Assuming RSI is 7th feature
        if rsi < 30:
            return 1.0  # Oversold - buy signal
        elif rsi > 70:
            return 0.0  # Overbought - sell signal
        else:
            return 0.5  # Neutral

    def get_bollinger_signal(self, features: np.ndarray) -> float:
        """Bollinger Bands signal"""
        bb_signal = features[13]  # Assuming BB signal is in features
        return 1.0 if bb_signal == 1 else 0.0

    def get_atr_signal(self, features: np.ndarray) -> float:
        """ATR volatility signal"""
        atr = features[8]  # Assuming ATR is 9th feature
        volatility = features[9]  # Assuming volatility is 10th feature
        
        # High volatility might indicate trend continuation
        return 1.0 if atr > volatility else 0.5

    def get_support_resistance_signal(self, features: np.ndarray) -> float:
        """Support/Resistance signal"""
        distance_to_support = features[17]  # Assuming these are in features
        distance_to_resistance = features[18]
        
        if abs(distance_to_support) < 0.01:  # Near support
            return 1.0  # Buy signal
        elif abs(distance_to_resistance) < 0.01:  # Near resistance
            return 0.0  # Sell signal
        else:
            return 0.5  # Neutral

    def calculate_confidence(self, signals: Dict, timeframe: str) -> float:
        """Calculate confidence score for predictions"""
        if not signals:
            return 0.0
            
        # Weight different signal types
        weights = {
            'ema': 0.1,
            'macd': 0.1,
            'rsi': 0.1,
            'bb': 0.1,
            'atr': 0.05,
            'support_resistance': 0.1,
            f'rf_{timeframe}': 0.15,
            f'xgb_{timeframe}': 0.15,
            f'lstm_{timeframe}': 0.15,
            'meta_prediction': 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for signal_name, weight in weights.items():
            if signal_name in signals:
                weighted_score += signals[signal_name] * weight
                total_weight += weight
        
        confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Adjust confidence based on anomaly score
        if 'anomaly_score' in signals:
            anomaly_penalty = min(signals['anomaly_score'] * 0.1, 0.3)
            confidence = max(0.0, confidence - anomaly_penalty)
        
        return confidence

    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0.0
            
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Cap Kelly fraction at 25% for safety
        return max(0.0, min(0.25, kelly_fraction))

    async def execute_trade(self, signal: str, confidence: float, timeframe: int):
        """Execute trade based on signal and confidence"""
        if confidence < 0.6:  # Minimum confidence threshold
            logger.info(f"Confidence too low ({confidence:.2f}) for {timeframe}m trade")
            return
            
        # Check drawdown limit
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            logger.warning("Maximum drawdown reached. Stopping trading.")
            return
            
        # Calculate position size using Kelly Criterion
        # For simplicity, using fixed values here - in practice, calculate from historical performance
        win_rate = 0.55  # Example win rate
        avg_win = 0.02   # Example average win
        avg_loss = 0.015 # Example average loss
        
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        position_size = self.current_balance * kelly_fraction * confidence
        position_size = min(position_size, self.current_balance * self.max_risk_per_trade)
        
        if position_size < 1:  # Minimum trade size
            logger.info("Position size too small to trade")
            return
            
        # Create trade request
        trade_request = {
            "buy": 1,
            "subscribe": 1,
            "price": position_size,
            "parameters": {
                "contract_type": "CALL" if signal == "buy" else "PUT",
                "symbol": self.symbol,
                "duration": timeframe,
                "duration_unit": "m",
                "basis": "stake",
                "amount": position_size
            }
        }
        
        try:
            await self.ws.send(json.dumps(trade_request))
            response = await self.ws.recv()
            trade_data = json.loads(response)
            
            if "buy" in trade_data:
                contract_id = trade_data["buy"]["contract_id"]
                entry_price = trade_data["buy"]["start_spot"]
                
                # Store position for tracking
                self.open_positions[contract_id] = {
                    "entry_price": entry_price,
                    "position_size": position_size,
                    "signal": signal,
                    "timeframe": timeframe,
                    "timestamp": datetime.now(),
                    "stop_loss": entry_price * (1 - self.stop_loss_pct) if signal == "buy" else entry_price * (1 + self.stop_loss_pct),
                    "take_profit": entry_price * (1 + self.take_profit_pct) if signal == "buy" else entry_price * (1 - self.take_profit_pct)
                }
                
                logger.info(f"Trade executed: {signal} {self.symbol} size:{position_size} confidence:{confidence:.2f}")
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    async def process_tick_data(self, tick_data: Dict):
        """Process incoming tick data and generate signals"""
        try:
            # Extract tick information
            price = tick_data.get("tick", {}).get("quote", 0)
            timestamp = datetime.fromtimestamp(tick_data.get("tick", {}).get("epoch", 0))
            
            # Add to price data
            new_row = pd.DataFrame({
                'timestamp': [timestamp],
                'close': [price],
                'high': [price],  # For ticks, high = low = close
                'low': [price],
                'volume': [1]  # Synthetic volume
            })
            
            self.price_data = pd.concat([self.price_data, new_row], ignore_index=True)
            
            # Keep only last 1000 ticks for memory management
            if len(self.price_data) > 1000:
                self.price_data = self.price_data.tail(1000).reset_index(drop=True)
            
            # Process if we have enough data
            if len(self.price_data) >= 100:
                # Calculate technical indicators
                df_with_indicators = self.calculate_technical_indicators(self.price_data.copy())
                
                if len(df_with_indicators) > 0:
                    # Prepare features
                    features_df = self.prepare_features(df_with_indicators)
                    
                    if len(features_df) >= 500 and len(features_df) % 100 == 0:
                        # Retrain models periodically
                        logger.info("Retraining models with new data")
                        self.train_models(features_df)
                    
                    if len(features_df) > 0:
                        # Get current features for prediction
                        current_features = features_df.iloc[-1][[
                            'ema_9', 'ema_21', 'ema_50', 'macd', 'macd_signal', 'macd_histogram',
                            'rsi', 'bb_width', 'atr', 'volatility', 'ema_signal', 'macd_signal_flag',
                            'rsi_signal', 'bb_signal', 'momentum_5', 'momentum_10', 'momentum_20',
                            'distance_to_support', 'distance_to_resistance'
                        ]].values
                        
                        # Generate signals
                        signals = self.generate_signals(current_features)
                        
                        if signals:
                            # Calculate confidence for both timeframes
                            self.confidence_15m = self.calculate_confidence(signals, '15m')
                            self.confidence_60m = self.calculate_confidence(signals, '60m')
                            
                            # Determine trading signals
                            signal_15m = "buy" if self.confidence_15m > 0.5 else "hold"
                            signal_60m = "buy" if self.confidence_60m > 0.5 else "hold"
                            
                            logger.info(f"Signals - 15m: {signal_15m} (conf: {self.confidence_15m:.2f}), "
                                      f"60m: {signal_60m} (conf: {self.confidence_60m:.2f})")
                            
                            # Execute trades if conditions are met
                            if signal_15m == "buy" and self.confidence_15m > 0.7:
                                await self.execute_trade(signal_15m, self.confidence_15m, 15)
                            
                            if signal_60m == "buy" and self.confidence_60m > 0.7:
                                await self.execute_trade(signal_60m, self.confidence_60m, 60)
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")

    async def monitor_positions(self):
        """Monitor open positions for stop-loss and take-profit"""
        for contract_id, position in list(self.open_positions.items()):
            try:
                # Get current contract status
                status_request = {
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                    "subscribe": 1
                }
                
                await self.ws.send(json.dumps(status_request))
                response = await self.ws.recv()
                contract_data = json.loads(response)
                
                if "proposal_open_contract" in contract_data:
                    current_spot = contract_data["proposal_open_contract"].get("current_spot")
                    is_sold = contract_data["proposal_open_contract"].get("is_sold")
                    
                    if is_sold:
                        # Position closed, remove from tracking
                        pnl = contract_data["proposal_open_contract"].get("profit", 0)
                        self.current_balance += pnl
                        self.peak_balance = max(self.peak_balance, self.current_balance)
                        
                        logger.info(f"Position closed: {contract_id}, PnL: {pnl}")
                        del self.open_positions[contract_id]
                        
                    elif current_spot:
                        # Check stop-loss and take-profit levels
                        if position["signal"] == "buy":
                            if current_spot <= position["stop_loss"] or current_spot >= position["take_profit"]:
                                await self.close_position(contract_id)
                        else:  # sell signal
                            if current_spot >= position["stop_loss"] or current_spot <= position["take_profit"]:
                                await self.close_position(contract_id)
                                
            except Exception as e:
                logger.error(f"Error monitoring position {contract_id}: {e}")

    async def close_position(self, contract_id: str):
        """Close a specific position"""
        try:
            sell_request = {
                "sell": contract_id,
                "price": 0  # Market price
            }
            
            await self.ws.send(json.dumps(sell_request))
            response = await self.ws.recv()
            sell_data = json.loads(response)
            
            if "sell" in sell_data:
                logger.info(f"Position {contract_id} closed successfully")
                
        except Exception as e:
            logger.error(f"Error closing position {contract_id}: {e}")

    async def run(self):
        """Main trading loop"""
        if not await self.connect():
            return
            
        await self.subscribe_to_ticks()
        
        logger.info("Trading bot started. Listening for tick data...")
        
        try:
            while True:
                response = await self.ws.recv()
                data = json.loads(response)
                
                if "tick" in data:
                    await self.process_tick_data(data)
                elif "buy" in data:
                    logger.info(f"Trade confirmation: {data}")
                elif "proposal_open_contract" in data:
                    # Handle position updates
                    pass
                    
                # Monitor positions periodically
                if len(self.open_positions) > 0:
                    await self.monitor_positions()
                    
        except websockets.exceptions.ConnectionClosed:
            logger.error("WebSocket connection closed")
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            if self.ws:
                await self.ws.close()

# Usage example
async def main():
    # Replace with your actual Deriv API credentials
    APP_ID = "96329"
    API_TOKEN = "yKga73O12NCnU6a"
    
    bot = DerivTradingBot(APP_ID, API_TOKEN)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())