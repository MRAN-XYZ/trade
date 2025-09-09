import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Technical Analysis Libraries
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import joblib

class DerivTradingBotWithTraining:
    def __init__(self, app_id, token):
        """Initialize the Deriv Trading Bot with training capabilities"""
        self.app_id = app_id
        self.token = token
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        
        # Data storage
        self.price_data = {
            '15min': deque(maxlen=500),
            '60min': deque(maxlen=200)
        }
        self.current_price = None
        
        # ML Models
        self.lstm_model_15 = None
        self.lstm_model_60 = None
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_model = LogisticRegression(random_state=42)
        
        # Scalers for ML
        self.scaler_15 = StandardScaler()
        self.scaler_60 = StandardScaler()
        
        # Training data storage
        self.training_data = {
            '15min': None,
            '60min': None
        }
        
        # Model training flags
        self.models_trained = False
        
        # Risk Management Parameters (same as original)
        self.position_size = 0
        self.max_drawdown = 0.15
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.current_position = None
        self.entry_price = None
        self.account_balance = 10000
        self.peak_balance = 10000
        
        # Kelly Criterion parameters
        self.win_rate = 0.5
        self.avg_win = 0.04
        self.avg_loss = 0.02

    async def fetch_historical_data(self, granularity, count=1000):
        """Fetch historical candle data for training"""
        print(f"Fetching {count} candles with {granularity}s granularity...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Authorize
                auth_request = {"authorize": self.token}
                await websocket.send(json.dumps(auth_request))
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                
                if 'error' in auth_data:
                    raise Exception(f"Authorization failed: {auth_data['error']['message']}")
                
                # Request historical data
                history_request = {
                    "ticks_history": "JD10",
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "granularity": granularity,
                    "style": "candles"
                }
                
                await websocket.send(json.dumps(history_request))
                response = await websocket.recv()
                data = json.loads(response)
                
                if 'error' in data:
                    raise Exception(f"Data fetch failed: {data['error']['message']}")
                
                if 'candles' not in data:
                    raise Exception("No candle data received")
                
                candles = data['candles']
                historical_data = []
                
                for candle in candles:
                    historical_data.append({
                        'timestamp': candle['epoch'],
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': 1000  # Synthetic volume
                    })
                
                print(f"Successfully fetched {len(historical_data)} candles")
                return historical_data
                
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return []

    def create_labels_from_price_data(self, df, future_periods=5):
        """Create labels for classification based on future price movements"""
        # Calculate future returns
        future_returns = df['close'].shift(-future_periods) / df['close'] - 1
        
        # Create labels based on return thresholds
        labels = []
        for ret in future_returns:
            if pd.isna(ret):
                labels.append(2)  # HOLD for NaN values
            elif ret > 0.005:  # 0.5% threshold
                labels.append(0)  # BUY
            elif ret < -0.005:
                labels.append(1)  # SELL
            else:
                labels.append(2)  # HOLD
        
        return np.array(labels)

    def prepare_lstm_training_data(self, features, labels, lookback=20):
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(lookback, len(features) - 5):  # -5 to avoid future leak
            X.append(features[i-lookback:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)

    def train_lstm_model(self, timeframe, historical_data):
        """Train LSTM model for specific timeframe"""
        print(f"Training LSTM model for {timeframe}...")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(df)
        
        # Prepare features
        features = self.prepare_ml_features(indicators, df)
        features = features.fillna(0)
        
        # Create labels
        labels = self.create_labels_from_price_data(df)
        
        # Scale features
        scaler = self.scaler_15 if timeframe == '15min' else self.scaler_60
        scaled_features = scaler.fit_transform(features)
        
        # Prepare LSTM sequences
        X, y = self.prepare_lstm_training_data(scaled_features, labels)
        
        if len(X) == 0:
            print(f"Not enough data for LSTM training ({timeframe})")
            return False
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=3)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Build and train model
        model = self.build_lstm_model((X.shape[1], X.shape[2]))
        
        print(f"Training LSTM on {len(X_train)} samples...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(
            np.argmax(y_test, axis=1),
            np.argmax(test_predictions, axis=1)
        )
        
        print(f"LSTM {timeframe} Test Accuracy: {test_accuracy:.4f}")
        
        # Store model
        if timeframe == '15min':
            self.lstm_model_15 = model
        else:
            self.lstm_model_60 = model
        
        return True

    def train_random_forest(self, timeframe, historical_data):
        """Train Random Forest model for specific timeframe"""
        print(f"Training Random Forest for {timeframe}...")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(df)
        
        # Prepare features
        features = self.prepare_ml_features(indicators, df)
        features = features.fillna(0)
        
        # Create labels
        labels = self.create_labels_from_price_data(df)
        
        # Remove NaN labels
        valid_indices = ~pd.isna(labels)
        features = features[valid_indices]
        labels = labels[valid_indices]
        
        if len(features) < 100:
            print(f"Not enough data for RF training ({timeframe})")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = self.rf_model.score(X_train, y_train)
        test_accuracy = self.rf_model.score(X_test, y_test)
        
        print(f"RF {timeframe} Train Accuracy: {train_accuracy:.4f}")
        print(f"RF {timeframe} Test Accuracy: {test_accuracy:.4f}")
        
        # Print classification report
        y_pred = self.rf_model.predict(X_test)
        print(f"\nRandom Forest {timeframe} Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['BUY', 'SELL', 'HOLD']))
        
        return True

    def train_meta_learner(self):
        """Train meta-learner to combine predictions from all models"""
        print("Training meta-learner...")
        
        # We need predictions from both timeframes to train meta-learner
        all_predictions = []
        all_labels = []
        
        for timeframe in ['15min', '60min']:
            if self.training_data[timeframe] is None:
                continue
                
            df = pd.DataFrame(self.training_data[timeframe])
            indicators = self.calculate_technical_indicators(df)
            features = self.prepare_ml_features(indicators, df)
            features = features.fillna(0)
            labels = self.create_labels_from_price_data(df)
            
            # Generate predictions for meta-learning
            # Technical predictions
            tech_predictions = []
            for i in range(len(df)):
                tech_signal = self.get_technical_signal(df.iloc[:i+1], indicators.iloc[:i+1])
                tech_predictions.append(tech_signal)
            
            # LSTM predictions (simplified for training)
            scaler = self.scaler_15 if timeframe == '15min' else self.scaler_60
            scaled_features = scaler.transform(features)
            
            # RF predictions
            rf_predictions = []
            for i in range(50, len(features)):  # Start after minimum data
                try:
                    pred_proba = self.rf_model.predict_proba([scaled_features[i]])[0]
                    if len(pred_proba) == 3:
                        rf_predictions.append(pred_proba)
                    else:
                        rf_predictions.append([0.33, 0.33, 0.34])
                except:
                    rf_predictions.append([0.33, 0.33, 0.34])
            
            # Combine predictions for this timeframe
            min_length = min(len(tech_predictions), len(rf_predictions), len(labels))
            
            for i in range(min_length - 5):  # Avoid future leak
                if not pd.isna(labels[i]):
                    # Flatten all predictions into feature vector
                    meta_features = np.concatenate([
                        tech_predictions[i],
                        rf_predictions[i] if i < len(rf_predictions) else [0.33, 0.33, 0.34]
                    ])
                    
                    all_predictions.append(meta_features)
                    all_labels.append(labels[i])
        
        if len(all_predictions) < 100:
            print("Not enough data for meta-learner training")
            return False
        
        # Train meta-learner
        X_meta = np.array(all_predictions)
        y_meta = np.array(all_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_meta, y_meta, test_size=0.2, random_state=42
        )
        
        self.meta_model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = self.meta_model.score(X_train, y_train)
        test_accuracy = self.meta_model.score(X_test, y_test)
        
        print(f"Meta-learner Train Accuracy: {train_accuracy:.4f}")
        print(f"Meta-learner Test Accuracy: {test_accuracy:.4f}")
        
        return True

    async def train_all_models(self):
        """Train all models using historical data"""
        print("Starting model training process...")
        
        try:
            # Fetch historical data for both timeframes
            historical_15min = await self.fetch_historical_data(900, count=1000)  # 15 minutes
            historical_60min = await self.fetch_historical_data(3600, count=500)  # 1 hour
            
            if not historical_15min or not historical_60min:
                raise Exception("Failed to fetch sufficient historical data")
            
            # Store training data
            self.training_data['15min'] = historical_15min
            self.training_data['60min'] = historical_60min
            
            # Train LSTM models
            lstm_15_success = self.train_lstm_model('15min', historical_15min)
            lstm_60_success = self.train_lstm_model('60min', historical_60min)
            
            # Train Random Forest (use 15min data as primary)
            rf_success = self.train_random_forest('15min', historical_15min)
            
            # Train meta-learner
            meta_success = self.train_meta_learner()
            
            # Check if training was successful
            if lstm_15_success and lstm_60_success and rf_success and meta_success:
                self.models_trained = True
                print("\n✅ All models trained successfully!")
                
                # Save models
                self.save_trained_models()
                
            else:
                print("\n❌ Some models failed to train")
                
        except Exception as e:
            print(f"Training failed: {e}")
            return False
        
        return self.models_trained

    def save_trained_models(self):
        """Save trained models to disk"""
        try:
            # Save sklearn models
            joblib.dump(self.rf_model, 'rf_model.pkl')
            joblib.dump(self.meta_model, 'meta_model.pkl')
            joblib.dump(self.scaler_15, 'scaler_15.pkl')
            joblib.dump(self.scaler_60, 'scaler_60.pkl')
            
            # Save Keras models
            if self.lstm_model_15:
                self.lstm_model_15.save('lstm_model_15.h5')
            if self.lstm_model_60:
                self.lstm_model_60.save('lstm_model_60.h5')
            
            print("Models saved successfully!")
            
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_trained_models(self):
        """Load previously trained models from disk"""
        try:
            import os
            from tensorflow.keras.models import load_model
            
            # Check if model files exist
            model_files = [
                'rf_model.pkl', 'meta_model.pkl', 
                'scaler_15.pkl', 'scaler_60.pkl',
                'lstm_model_15.h5', 'lstm_model_60.h5'
            ]
            
            if not all(os.path.exists(f) for f in model_files):
                print("Some model files missing. Need to train models first.")
                return False
            
            # Load models
            self.rf_model = joblib.load('rf_model.pkl')
            self.meta_model = joblib.load('meta_model.pkl')
            self.scaler_15 = joblib.load('scaler_15.pkl')
            self.scaler_60 = joblib.load('scaler_60.pkl')
            self.lstm_model_15 = load_model('lstm_model_15.h5')
            self.lstm_model_60 = load_model('lstm_model_60.h5')
            
            self.models_trained = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    # [Include all the original methods: calculate_technical_indicators, 
    # prepare_ml_features, build_lstm_model, generate_trading_signal, etc.]
    
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
        feature_matrix = indicators_df.copy()
    
        # Add price-based features with explicit string column names
        feature_matrix['price_change'] = prices_df['close'].pct_change().fillna(0)
        feature_matrix['high_close_ratio'] = prices_df['high'] / prices_df['close'] - 1
        feature_matrix['close_low_ratio'] = prices_df['close'] / prices_df['low'] - 1
    
        # Add rolling statistics
        price_returns = prices_df['close'].pct_change()
        for window in [5, 10, 20]:
            feature_matrix[f'ret_{window}'] = price_returns.rolling(window).mean().fillna(0)
            feature_matrix[f'vol_{window}'] = price_returns.rolling(window).std().fillna(0)
    
        # Ensure all columns are strings
        feature_matrix.columns = [str(col) for col in feature_matrix.columns]
        feature_matrix = feature_matrix.fillna(0)
    
        return feature_matrix

    def build_lstm_model(self, input_shape):
        """Build LSTM model for price prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: BUY, SELL, HOLD
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def get_technical_signal(self, df, indicators):
        """Generate signal from technical indicators"""
        if len(df) == 0 or len(indicators) == 0:
            return np.array([0.33, 0.33, 0.34])
            
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


async def main_with_training():
    """Main function that includes training"""
    # Initialize bot
    APP_ID = "96329"
    TOKEN = "lWFYLtfTp2sbWl8"  # Read-only token
    
    bot = DerivTradingBotWithTraining(APP_ID, TOKEN)
    
    print("Deriv Trading Bot with Model Training")
    print("=====================================")
    
    # Try to load existing models first
    if not bot.load_trained_models():
        print("No existing models found. Starting training process...")
        
        # Train all models
        training_success = await bot.train_all_models()
        
        if not training_success:
            print("❌ Training failed! Cannot proceed with trading.")
            return
        
        print("✅ Training completed successfully!")
    
    print(f"\nModels ready for trading!")
    print(f"Models trained: {bot.models_trained}")
    
    # Now you can proceed with live trading
    # bot.connect_and_trade() would go here


if __name__ == "__main__":
    asyncio.run(main_with_training())