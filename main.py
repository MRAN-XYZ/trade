import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
from talib import EMA, MACD, RSI, BBANDS, ATR  # TA-Lib for indicators
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Deriv API WebSocket URL
WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=YOUR_APP_ID"  # Replace with your app_id

# Your Deriv API token (get from api.deriv.com)
API_TOKEN = "YOUR_API_TOKEN"  # Replace with your token

# Symbol for Jump 10 index
SYMBOL = "JUMP10"

# Account balance (for simulation; replace with real query via API)
ACCOUNT_BALANCE = 10000.0
MAX_DRAWDOWN = 0.1  # 10% max drawdown
RISK_PER_TRADE = 0.01  # 1% risk per trade

# Global variables
ws = None
historical_data = pd.DataFrame()  # To store candles
current_position = None  # 'BUY' or 'SELL' or None
current_balance = ACCOUNT_BALANCE

# WebSocket handlers
def on_open(ws):
    print("Connected to Deriv API")
    authorize(ws)
    fetch_historical_data(ws)
    subscribe_to_ticks(ws)

def on_message(ws, message):
    data = json.loads(message)
    if 'error' in data:
        print("Error:", data['error'])
        return
    if data.get('msg_type') == 'authorize':
        print("Authorized")
    elif data.get('msg_type') == 'candles':
        process_historical_data(data['candles'])
    elif data.get('msg_type') == 'tick':
        process_tick(data['tick'])
    elif data.get('msg_type') == 'proposal':
        execute_trade(data['proposal'])
    elif data.get('msg_type') == 'buy':
        print("Trade executed:", data)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

# Authorize
def authorize(ws):
    ws.send(json.dumps({"authorize": API_TOKEN}))

# Fetch historical candles (last 5000 for training)
def fetch_historical_data(ws):
    ws.send(json.dumps({
        "ticks_history": SYMBOL,
        "adjust_start_time": 1,
        "count": 5000,
        "end": "latest",
        "start": 1,
        "style": "candles",
        "granularity": 60  # 1-minute candles
    }))

# Process historical candles
def process_historical_data(candles):
    global historical_data
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['epoch'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close']]
    historical_data = df
    print("Fetched historical data:", historical_data.shape)
    train_models()  # Train ML models once data is fetched
    threading.Thread(target=trading_loop).start()  # Start trading loop

# Subscribe to real-time ticks
def subscribe_to_ticks(ws):
    ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

# Process real-time tick
def process_tick(tick):
    global historical_data
    new_row = pd.DataFrame({
        'time': pd.to_datetime(tick['epoch'], unit='s'),
        'close': tick['quote']
    }, index=[0])
    # Approximate OHLC for simplicity (in real-time, you'd aggregate)
    historical_data = pd.concat([historical_data, new_row]).tail(5000)  # Keep last 5000

# Train ML models
def train_models():
    global historical_data
    if historical_data.empty:
        return

    df = historical_data.copy()
    df['return'] = df['close'].pct_change()
    df['target_15'] = np.where(df['close'].shift(-15) > df['close'], 1, 0)  # 1 = up (BUY), 0 = down (SELL) for 15-min
    df['target_60'] = np.where(df['close'].shift(-60) > df['close'], 1, 0)  # For 60-min
    df.dropna(inplace=True)

    # Features: close, volume (dummy), etc.
    features = ['close', 'return']
    X = df[features]
    y15 = df['target_15']
    y60 = df['target_60']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y15_train, y15_test = y15[:split], y15[split:]
    y60_train, y60_test = y60[:split], y60[split:]

    # LSTM
    global lstm_model_15, lstm_model_60
    lstm_model_15 = build_lstm(X_train.shape[1])
    lstm_model_15.fit(np.expand_dims(X_train, axis=2), y15_train, epochs=10, batch_size=32, verbose=0)
    lstm_model_60 = build_lstm(X_train.shape[1])
    lstm_model_60.fit(np.expand_dims(X_train, axis=2), y60_train, epochs=10, batch_size=32, verbose=0)

    # Random Forest
    global rf_model_15, rf_model_60
    rf_model_15 = RandomForestClassifier(n_estimators=100)
    rf_model_15.fit(X_train, y15_train)
    rf_model_60 = RandomForestClassifier(n_estimators=100)
    rf_model_60.fit(X_train, y60_train)

    # XGBoost
    global xgb_model_15, xgb_model_60
    xgb_model_15 = xgb.XGBClassifier()
    xgb_model_15.fit(X_train, y15_train)
    xgb_model_60 = xgb.XGBClassifier()
    xgb_model_60.fit(X_train, y60_train)

    # Autoencoder (for anomaly detection)
    global autoencoder
    autoencoder = build_autoencoder(X_train.shape[1])
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)

    # Meta-Learner (Logistic Regression) - train on predictions
    preds_15 = np.column_stack([
        lstm_model_15.predict(np.expand_dims(X_train, axis=2)).flatten(),
        rf_model_15.predict_proba(X_train)[:,1],
        xgb_model_15.predict_proba(X_train)[:,1]
    ])
    global meta_model_15
    meta_model_15 = LogisticRegression()
    meta_model_15.fit(preds_15, y15_train)

    preds_60 = np.column_stack([
        lstm_model_60.predict(np.expand_dims(X_train, axis=2)).flatten(),
        rf_model_60.predict_proba(X_train)[:,1],
        xgb_model_60.predict_proba(X_train)[:,1]
    ])
    global meta_model_60
    meta_model_60 = LogisticRegression()
    meta_model_60.fit(preds_60, y60_train)

    print("Models trained")

def build_lstm(input_shape):
    model = Sequential()
    model.add(Input(shape=(None, input_shape)))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    model = Model(inputs=input_layer, outputs=decoder)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Compute indicators and predictions
def compute_signals():
    df = historical_data.tail(100).copy()  # Last 100 candles for indicators
    close = df['close'].values

    # 1. EMA
    ema = EMA(close, timeperiod=14)

    # 2. MACD
    macd, macdsignal, _ = MACD(close)

    # 3. RSI
    rsi = RSI(close)

    # 4. Bollinger Bands
    upper, middle, lower = BBANDS(close)

    # 5. ATR
    atr = ATR(df['high'], df['low'], close)

    # 6. Support & Resistance (simple pivot points)
    pivot = (df['high'] + df['low'] + close) / 3
    support = pivot - atr
    resistance = pivot + atr

    # Latest features for prediction
    latest_features = np.array([[close[-1], (close[-1] - close[-2]) / close[-2]]])
    latest_scaled = StandardScaler().fit_transform(latest_features)  # Use global scaler if needed

    # Anomaly check with Autoencoder
    recon = autoencoder.predict(latest_scaled)
    mse = np.mean(np.power(latest_scaled - recon, 2), axis=1)
    anomaly_threshold = 0.1  # Adjust based on training
    if mse > anomaly_threshold:
        return {"15": ("HOLD", 0.0), "60": ("HOLD", 0.0)}  # Skip if anomaly

    # ML Predictions
    lstm_pred_15 = lstm_model_15.predict(np.expand_dims(latest_scaled, axis=2))[0][0]
    rf_pred_15 = rf_model_15.predict_proba(latest_scaled)[0][1]
    xgb_pred_15 = xgb_model_15.predict_proba(latest_scaled)[0][1]

    meta_input_15 = np.array([[lstm_pred_15, rf_pred_15, xgb_pred_15]])
    meta_pred_15 = meta_model_15.predict_proba(meta_input_15)[0][1]
    signal_15 = "BUY" if meta_pred_15 > 0.6 else "SELL" if meta_pred_15 < 0.4 else "HOLD"
    conf_15 = abs(meta_pred_15 - 0.5) * 2  # Confidence 0-1

    lstm_pred_60 = lstm_model_60.predict(np.expand_dims(latest_scaled, axis=2))[0][0]
    rf_pred_60 = rf_model_60.predict_proba(latest_scaled)[0][1]
    xgb_pred_60 = xgb_model_60.predict_proba(latest_scaled)[0][1]

    meta_input_60 = np.array([[lstm_pred_60, rf_pred_60, xgb_pred_60]])
    meta_pred_60 = meta_model_60.predict_proba(meta_input_60)[0][1]
    signal_60 = "BUY" if meta_pred_60 > 0.6 else "SELL" if meta_pred_60 < 0.4 else "HOLD"
    conf_60 = abs(meta_pred_60 - 0.5) * 2

    # Combine with indicators (simple voting)
    ind_signals = []
    if close[-1] > ema[-1]: ind_signals.append("BUY")
    if macd[-1] > macdsignal[-1]: ind_signals.append("BUY")
    if rsi[-1] > 50: ind_signals.append("BUY")
    if close[-1] > upper[-1]: ind_signals.append("SELL")  # Overbought
    if close[-1] < support[-1]: ind_signals.append("BUY")  # Bounce from support

    combined_signal_15 = max(set(ind_signals), key=ind_signals.count) if ind_signals else signal_15
    combined_signal_60 = combined_signal_15  # Simplify for 60-min (or extend logic)

    return {
        "15": (combined_signal_15, conf_15),
        "60": (combined_signal_60, conf_60)
    }

# Kelly Criterion for position sizing
def kelly_criterion(win_prob, win_loss_ratio=2):  # Assume 2:1 reward:risk
    return (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio

# Drawdown control
def check_drawdown():
    global current_balance
    drawdown = (ACCOUNT_BALANCE - current_balance) / ACCOUNT_BALANCE
    return drawdown < MAX_DRAWDOWN

# Stop-Loss / Take-Profit (example: 1% SL, 2% TP)
def apply_sl_tp(entry_price, signal):
    if signal == "BUY":
        sl = entry_price * 0.99
        tp = entry_price * 1.02
    else:
        sl = entry_price * 1.01
        tp = entry_price * 0.98
    return sl, tp

# Smart Order Execution (send proposal to Deriv API)
def smart_execute(signal, conf, timeframe):
    global ws, current_position, current_balance
    if conf < 0.7 or not check_drawdown() or current_position:
        return  # Skip if low conf, drawdown exceeded, or position open

    # Get current price via latest tick
    current_price = historical_data['close'].iloc[-1]

    # Kelly sizing
    stake_fraction = kelly_criterion(conf)
    stake = current_balance * stake_fraction * RISK_PER_TRADE

    # SL/TP
    sl, tp = apply_sl_tp(current_price, signal)

    # Send proposal (example for binary call/put; adjust for your contract type)
    contract_type = "CALL" if signal == "BUY" else "PUT"
    ws.send(json.dumps({
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "duration": int(timeframe),  # Duration in minutes
        "duration_unit": "m",
        "symbol": SYMBOL
    }))

# Handle proposal response and buy
def execute_trade(proposal):
    global ws
    if 'proposal' in proposal and proposal['proposal']['has_error'] == 0:
        ws.send(json.dumps({
            "buy": proposal['proposal']['proposal_id'],
            "price": proposal['proposal']['ask_price']
        }))
        current_position = "BUY" if proposal['proposal']['contract_type'] == "CALL" else "SELL"
        print(f"Executed {current_position} trade for {proposal['proposal']['longcode']}")

# Trading loop (runs every 60 seconds)
def trading_loop():
    while True:
        signals = compute_signals()
        print("Signals:", signals)

        for timeframe, (signal, conf) in signals.items():
            if signal != "HOLD":
                smart_execute(signal, conf, timeframe if timeframe == "15" else 60)

        # Simulate balance update (in real, query via API)
        global current_balance
        current_balance *= (1 + np.random.uniform(-0.01, 0.01))  # Dummy for testing

        # Check if position should be closed (dummy logic)
        global current_position
        if current_position:
            # Assume close after duration; in real, monitor via API
            current_position = None

        time.sleep(60)

# Start WebSocket
if __name__ == "__main__":
    ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()
