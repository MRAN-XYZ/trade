import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from main import DerivTradingBot  # reuse feature/indicator functions

APP_ID = "96329"   # replace with your Deriv app_id
TOKEN = "lWFYLtfTp2sbWl8"  # replace with your API token
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===================================================
# 1. Fetch 24h candle data from Deriv
# ===================================================
async def fetch_candles(granularity=900, count=2000):
    """Fetch candles for JD10"""
    ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"ticks_history": "JD10",
                                      "adjust_start_time": 1,
                                      "count": count,
                                      "end": "latest",
                                      "granularity": granularity,
                                      "style": "candles"}))
            response = await ws.recv()
            data = json.loads(response)
            
            # Check if response contains candles data
            if "candles" not in data:
                raise ValueError(f"No candles data received: {data}")
            
            candles = pd.DataFrame(data["candles"])
            
            # Validate required columns exist
            required_cols = ["open", "high", "low", "close"]
            for col in required_cols:
                if col not in candles.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            candles["open"] = candles["open"].astype(float)
            candles["high"] = candles["high"].astype(float)
            candles["low"] = candles["low"].astype(float)
            candles["close"] = candles["close"].astype(float)
            candles["volume"] = 1000  # synthetic volume
            
            print(f"Fetched {len(candles)} candles")
            return candles
            
    except Exception as e:
        print(f"Error fetching candles: {e}")
        raise

# ===================================================
# 2. Prepare features/labels
# ===================================================
def prepare_data(df, bot: DerivTradingBot):
    """Prepare features and labels from dataframe"""
    try:
        indicators = bot.calculate_technical_indicators(df)
        features = bot.prepare_ml_features(indicators, df)
        
        # Label: future price movement
        future_return = df["close"].pct_change().shift(-1).fillna(0)
        y = np.where(future_return > 0.002, 0,  # BUY
            np.where(future_return < -0.002, 1, 2))  # SELL / HOLD
        
        # Remove rows with NaN features
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        y = y[mask]
        
        print(f"Prepared {len(features)} samples with {features.shape[1]} features")
        return features, y
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise

# ===================================================
# 3. Train Random Forest
# ===================================================
def train_rf(X, y):
    """Train Random Forest model"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test, y_test)
        print(f"RF Accuracy: {accuracy:.4f}")
        
        model_path = os.path.join(MODEL_DIR, "random_forest_model.joblib")
        joblib.dump(rf, model_path)
        print(f"Random Forest model saved to {model_path}")
        
        return rf
        
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        raise

# ===================================================
# 4. Train LSTM
# ===================================================
def train_lstm(X, y, model_name):
    """Train LSTM model"""
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        
        # Save scaler
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{model_name}.joblib")
        joblib.dump(scaler, scaler_path)

        lookback = 20
        if len(scaled) <= lookback:
            raise ValueError(f"Not enough data points for lookback window. Need > {lookback}, got {len(scaled)}")
            
        X_seq, y_seq = [], []
        for i in range(lookback, len(scaled)):
            X_seq.append(scaled[i-lookback:i])
            y_seq.append(y[i])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        print(f"LSTM input shape: {X_seq.shape}")

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, X_seq.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(3, activation="softmax")
        ])
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(X_seq, y_seq, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        
        model_path = os.path.join(MODEL_DIR, f"lstm_model_{model_name}.h5")
        model.save(model_path)
        print(f"LSTM model saved to {model_path}")
        
        return model, scaler
        
    except Exception as e:
        print(f"Error training LSTM: {e}")
        raise

# ===================================================
# 5. Train Meta-Model
# ===================================================
def train_meta(rf, lstm_preds, tech_preds, X, y):
    """Train meta-model using ensemble predictions"""
    try:
        # Get RF predictions
        rf_preds = rf.predict_proba(X)
        
        # Ensure all prediction arrays have same length
        min_len = min(len(rf_preds), len(lstm_preds), len(tech_preds), len(y))
        
        meta_X = np.hstack([
            rf_preds[:min_len], 
            lstm_preds[:min_len], 
            tech_preds[:min_len]
        ])
        
        meta_y = y[:min_len]
        
        # Train meta-model
        meta = LogisticRegression(random_state=42)
        meta.fit(meta_X, meta_y)
        
        # Save meta-model
        meta_path = os.path.join(MODEL_DIR, "meta_model.joblib")
        joblib.dump(meta, meta_path)
        print(f"Meta-model trained and saved to {meta_path}")
        
        return meta
        
    except Exception as e:
        print(f"Error training meta-model: {e}")
        raise

# ===================================================
# Main training pipeline
# ===================================================
async def main():
    """Main training pipeline"""
    try:
        bot = DerivTradingBot(APP_ID, TOKEN, MODEL_DIR)

        # Fetch 15min and 60min candles (~24h each)
        print("Fetching 15min candles...")
        df15 = await fetch_candles(granularity=900, count=2000)
        
        print("Fetching 60min candles...")
        df60 = await fetch_candles(granularity=3600, count=2000)

        # Train on 15min data
        print("\n=== Training on 15min data ===")
        X15, y15 = prepare_data(df15, bot)
        rf_model = train_rf(X15, y15)
        lstm15, scaler15 = train_lstm(X15, y15, "15min")

        # Train on 60min data
        print("\n=== Training on 60min data ===")
        X60, y60 = prepare_data(df60, bot)
        lstm60, scaler60 = train_lstm(X60, y60, "60min")

        # Train meta-model using outputs
        print("\n=== Training meta-model ===")
        
        # Generate predictions for meta-model training
        rf_preds = rf_model.predict_proba(X15)
        
        # For demonstration, create placeholder predictions
        # In practice, you'd want actual LSTM and technical indicator predictions
        lstm_preds = np.tile([0.33, 0.33, 0.34], (len(X15), 1))
        tech_preds = np.tile([0.33, 0.33, 0.34], (len(X15), 1))
        
        meta_model = train_meta(rf_model, lstm_preds, tech_preds, X15, y15)

        print("\n✅ All models trained and saved in /models")
        print("Models saved:")
        for file in os.listdir(MODEL_DIR):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())