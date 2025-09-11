import os import json import requests import numpy as np import pandas as pd from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier from sklearn.linear_model import LogisticRegression from sklearn.preprocessing import StandardScaler import joblib from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense, Dropout

Import bot class for feature engineering

from main2 import DerivTradingBot

Config

APP_ID = "96329"  # replace with your app id TOKEN = "lWFYLtfTp2sbWl8"  # replace with your token MODEL_DIR = "models" os.makedirs(MODEL_DIR, exist_ok=True)

Fetch historical candles from Deriv REST API

def fetch_candles(granularity, count=2000): url = "https://ws.binaryws.com/json" payload = { "ticks_history": "JD10", "adjust_start_time": 1, "count": count, "end": "latest", "granularity": granularity, "style": "candles" } resp = requests.get(url, params={"req": json.dumps(payload)}) data = resp.json() candles = data.get("candles", []) df = pd.DataFrame(candles) df["open"] = df["open"].astype(float) df["high"] = df["high"].astype(float) df["low"] = df["low"].astype(float) df["close"] = df["close"].astype(float) df["volume"] = 1000  # synthetic volume return df

Train RandomForest

print("Fetching 15m candles for training RandomForest...") df = fetch_candles(900) bot = DerivTradingBot(APP_ID, TOKEN, MODEL_DIR) indicators = bot.calculate_technical_indicators(df) features = bot.prepare_ml_features(indicators, df)

Label creation (simple: price up = BUY, down = SELL, flat = HOLD)

future_returns = df["close"].pct_change().shift(-1) labels = np.where(future_returns > 0.001, 0, np.where(future_returns < -0.001, 1, 2))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

rf = RandomForestClassifier(n_estimators=200, random_state=42) rf.fit(X_train, y_train) print("RF Accuracy:", rf.score(X_test, y_test)) joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_model.joblib"))

Train LSTM models

def train_lstm(features, labels, scaler_path, model_path, lookback=20): scaler = StandardScaler() scaled = scaler.fit_transform(features) joblib.dump(scaler, scaler_path)

X_seq, y_seq = [], []
for i in range(lookback, len(scaled)):
    X_seq.append(scaled[i-lookback:i])
    y_seq.append(labels[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_seq, y_seq, epochs=10, batch_size=32, validation_split=0.2)

model.save(model_path)

print("Training LSTM 15min...") train_lstm(features, labels, os.path.join(MODEL_DIR, "scaler_15min.joblib"), os.path.join(MODEL_DIR, "lstm_model_15min.h5"))

print("Fetching 60m candles for LSTM 60min...") df60 = fetch_candles(3600) ind60 = bot.calculate_technical_indicators(df60) feat60 = bot.prepare_ml_features(ind60, df60) future_returns60 = df60["close"].pct_change().shift(-1) labels60 = np.where(future_returns60 > 0.001, 0, np.where(future_returns60 < -0.001, 1, 2))

print("Training LSTM 60min...") train_lstm(feat60, labels60, os.path.join(MODEL_DIR, "scaler_60min.joblib"), os.path.join(MODEL_DIR, "lstm_model_60min.h5"))

Train Meta-model

print("Training Meta-model...") rf_probs = rf.predict_proba(features)

NOTE: For simplicity we use RF + tech indicators as meta features

tech_signals = [] for i in range(len(df)): sig = bot.get_technical_signal(df.iloc[:i+1], indicators.iloc[:i+1]) tech_signals.append(sig) tech_signals = np.array(tech_signals)

meta_X = np.hstack([rf_probs, tech_signals]) meta_y = labels[:len(meta_X)]

meta = LogisticRegression(max_iter=500) meta.fit(meta_X, meta_y) print("Meta Accuracy:", meta.score(meta_X, meta_y)) joblib.dump(meta, os.path.join(MODEL_DIR, "meta_model.joblib"))

print("\n✅ Training complete. Models saved in ./models
")
