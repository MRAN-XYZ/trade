import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Import the bot class to reuse its data preparation functions
from main import DerivTradingBot

APP_ID = "96329"
TOKEN = "lWFYLtfTp2sbWl8"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===================================================
# 1. Fetch Candle Data
# ===================================================
async def fetch_candles(granularity=900, count=2000):
    """Fetch candles for JD10"""
    ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    print(f"Fetching {count} candles with {granularity}s granularity...")
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "ticks_history": "JD10", "adjust_start_time": 1, "count": count,
                "end": "latest", "granularity": granularity, "style": "candles"
            }))
            response = await ws.recv()
            data = json.loads(response)
            
            if "candles" not in data:
                raise ValueError(f"No candles data received: {data.get('error', 'Unknown error')}")
            
            candles = pd.DataFrame(data["candles"])
            candles = candles[['epoch', 'open', 'high', 'low', 'close']].astype(float)
            
            # *** THE FIX IS HERE: Rename 'epoch' to 'timestamp' for consistency ***
            candles.rename(columns={'epoch': 'timestamp'}, inplace=True)
            
            candles['volume'] = 1000  # synthetic volume
            
            print(f"✓ Fetched {len(candles)} candles.")
            return candles
            
    except Exception as e:
        print(f"Error fetching candles: {e}")
        raise

# ===================================================
# 2. Prepare Features and Labels
# ===================================================
def prepare_data(df, bot: DerivTradingBot):
    """Prepare features and two types of labels from the dataframe."""
    try:
        # This call will now work because the DataFrame has a 'timestamp' column
        indicators = bot.calculate_technical_indicators(df)
        features = bot.prepare_ml_features(indicators, df)
        
        y_reg = df["close"].pct_change().shift(-1).fillna(0)
        
        threshold = 0.001
        y_class = np.where(y_reg > threshold, 0,  # BUY
                   np.where(y_reg < -threshold, 1, 2))  # SELL / HOLD
        
        if features.isnull().values.any():
            features = features.dropna()
        
        aligned_indices = features.index
        y_reg = y_reg.loc[aligned_indices]
        y_class = pd.Series(y_class, index=df.index).loc[aligned_indices]

        print(f"✓ Prepared {len(features)} samples with {features.shape[1]} features.")
        return features, y_reg.values, y_class.values
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise

# ===================================================
# 3. Train Random Forest REGRESSOR
# ===================================================
def train_rf_regressor(X, y_reg):
    """Train Random Forest Regressor model."""
    print("Training Random Forest Regressor...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        score = rf.score(X_test, y_test)
        print(f"✓ RF Regressor R-squared score: {score:.4f}")
        
        model_path = os.path.join(MODEL_DIR, "random_forest_model.joblib")
        joblib.dump(rf, model_path)
        print(f"✓ Random Forest model saved to {model_path}")
        return rf
        
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        raise

# ===================================================
# 4. Train LSTM REGRESSOR
# ===================================================
def train_lstm_regressor(X, y_reg, model_name, lookback=20):
    """Train LSTM Regressor model."""
    print(f"Training LSTM Regressor ({model_name})...")
    try:
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{model_name}.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler for {model_name} saved.")
            
        X_seq, y_seq = [], []
        for i in range(lookback, len(scaled_X)):
            X_seq.append(scaled_X[i-lookback:i])
            y_seq.append(y_reg[i])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        if X_seq.shape[0] == 0:
            raise ValueError("Not enough data to create sequences for LSTM.")

        print(f"LSTM input shape: {X_seq.shape}")

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, X_seq.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_seq, y_seq, epochs=15, batch_size=32, validation_split=0.1, verbose=1)
        
        model_path = os.path.join(MODEL_DIR, f"lstm_model_{model_name}.h5")
        model.save(model_path)
        print(f"✓ LSTM model saved to {model_path}")
        return model, scaler
        
    except Exception as e:
        print(f"Error training LSTM: {e}")
        raise

# ===================================================
# 5. Train Meta-Model CLASSIFIER
# ===================================================
def train_meta_model(rf_model, lstm15_model, lstm60_model, X15, X60, y15_class):
    """Train meta-model using the regression outputs of the base models."""
    print("Training Meta-Model Classifier...")
    try:
        def get_lstm_preds(model, data, lookback):
            preds = []
            for i in range(lookback, len(data)):
                seq = data[i-lookback:i]
                seq = np.reshape(seq, (1, lookback, data.shape[1]))
                preds.append(model.predict(seq, verbose=0)[0, 0])
            return np.array(preds)

        scaler15 = joblib.load(os.path.join(MODEL_DIR, 'scaler_15min.joblib'))
        scaler60 = joblib.load(os.path.join(MODEL_DIR, 'scaler_60min.joblib'))
        
        scaled_X15 = scaler15.transform(X15)
        scaled_X60 = scaler60.transform(X60)

        pred_lstm15 = get_lstm_preds(lstm15_model, scaled_X15, lookback=15)
        pred_lstm60 = get_lstm_preds(lstm60_model, scaled_X60, lookback=60)
        pred_rf = rf_model.predict(X15)
        
        min_len = min(len(pred_rf), len(pred_lstm15), len(pred_lstm60))
        
        pred_rf_aligned = pred_rf[-min_len:]
        pred_lstm15_aligned = pred_lstm15[-min_len:]
        pred_lstm60_aligned = pred_lstm60[-min_len:]
        meta_y = y15_class[-min_len:]

        meta_X = np.vstack([pred_lstm15_aligned, pred_lstm60_aligned, pred_rf_aligned]).T
        print(f"Meta-model feature shape: {meta_X.shape}")
        
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(meta_X, meta_y)
        
        meta_path = os.path.join(MODEL_DIR, "meta_model.joblib")
        joblib.dump(meta_model, meta_path)
        print(f"✓ Meta-model trained and saved to {meta_path}")
        return meta_model
        
    except Exception as e:
        print(f"Error training meta-model: {e}")
        raise

# ===================================================
# Main Training Pipeline
# ===================================================
async def main():
    """Main training pipeline"""
    try:
        bot = DerivTradingBot(APP_ID, TOKEN, MODEL_DIR)

        df15 = await fetch_candles(granularity=900, count=5000)
        df60 = await fetch_candles(granularity=3600, count=5000)

        X15, y15_reg, y15_class = prepare_data(df15, bot)
        X60, y60_reg, y60_class = prepare_data(df60, bot)

        print("\n--- Training Base Models ---")
        rf_model = train_rf_regressor(X15, y15_reg)
        lstm15_model, _ = train_lstm_regressor(X15, y15_reg, "15min", lookback=15)
        lstm60_model, _ = train_lstm_regressor(X60, y60_reg, "60min", lookback=60)

        print("\n--- Training Meta-Model ---")
        train_meta_model(rf_model, lstm15_model, lstm60_model, X15, X60, y15_class)

        print("\n✅ All models trained successfully and are compatible with the bot.")
        print("Models saved in /models directory:")
        for file in sorted(os.listdir(MODEL_DIR)):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"\n❌ An error occurred in the main training pipeline: {e}")
        # raise # Optional: re-raise the exception for more detailed debug trace

if __name__ == "__main__":
    # Note: The TensorFlow/CUDA warnings at the start of the log are harmless and not related to the error.
    asyncio.run(main())