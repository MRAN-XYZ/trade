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

from main2 import DerivTradingBot  # reuse feature/indicator functions

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
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"ticks_history": "JD10",
                                  "adjust_start_time": 1,
                                  "count": count,
                                  "end": "latest",
                                  "granularity": granularity,
                                  "style": "candles"}))
        response = await ws.recv()
        data = json.loads(response)
        candles = pd.DataFrame(data["candles"])
        candles["open"] = candles["open"].astype(float)
        candles["high"] = candles["high"].astype(float)
        candles["low"] = candles["low"].astype(float)
        candles["close"] = candles["close"].astype(float)
        candles["volume"] = 1000  # synthetic volume
        return candles

# ===================================================
# 2. Prepare features/labels
# ===================================================
def prepare_data(df, bot: DerivTradingBot):
    indicators = bot.calculate_technical_indicators(df)
    features = bot.prepare_ml_features(indicators, df)
    # Label: future price movement
    future_return = df["close"].pct_change().shift(-1).fillna(0)
    y = np.where(future_return > 0.002, 0,  # BUY
        np.where(future_return < -0.002, 1, 2))  # SELL / HOLD
    return features, y

# ===================================================
# 3. Train Random Forest
# ===================================================
def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    print("RF Accuracy:", rf.score(X_test, y_test))
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_model.joblib"))
    return rf

# ===================================================
# 4. Train LSTM
# ===================================================
def train_lstm(X, y, scaler, model_name):
    scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{model_name}.joblib"))

    lookback = 20
    X_seq, y_seq = [], []
    for i in range(lookback, len(scaled)):
        X_seq.append(scaled[i-lookback:i])
        y_seq.append(y[i])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, X_seq.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_seq, y_seq, epochs=10, batch_size=32, validation_split=0.2)

    model.save(os.path.join(MODEL_DIR, f"lstm_model_{model_name}.h5"))
    return model

# ===================================================
# 5. Train Meta-Model
# ===================================================
def train_meta(rf, lstm_preds, tech_preds, y):
    meta_X = np.hstack([rf.predict_proba(X), lstm_preds, tech_preds])
    meta = LogisticRegression()
    meta.fit(meta_X, y[:len(meta_X)])  # align sizes
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta_model.joblib"))
    print("Meta-model trained and saved.")
    return meta

# ===================================================
# Main training pipeline
# ===================================================
async def main():
    bot = DerivTradingBot(APP_ID, TOKEN, MODEL_DIR)

    # Fetch 15min and 60min candles (~24h each)
    print("Fetching 15min candles...")
    df15 = await fetch_candles(granularity=900, count=2000)
    print("Fetching 60min candles...")
    df60 = await fetch_candles(granularity=3600, count=2000)

    # Train on 15min data
    X15, y15 = prepare_data(df15, bot)
    rf_model = train_rf(X15, y15)
    lstm15 = train_lstm(X15, y15, StandardScaler(), "15min")

    # Train on 60min data
    X60, y60 = prepare_data(df60, bot)
    lstm60 = train_lstm(X60, y60, StandardScaler(), "60min")

    # Train meta-model using outputs (simplified: just use RF probs + dummy LSTM/tech here)
    rf_preds = rf_model.predict_proba(X15)
    lstm_preds = np.tile([0.33,0.33,0.34], (len(X15),1))  # placeholder (real preds optional)
    tech_preds = np.tile([0.33,0.33,0.34], (len(X15),1))
    train_meta(rf_model, lstm_preds, tech_preds, y15)

    print("✅ All models trained and saved in /models")

if __name__ == "__main__":
    asyncio.run(main())
