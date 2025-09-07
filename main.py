#!/usr/bin/env python3
"""
JD10 (Jump 10) Deriv Bot – One-file, batteries-included
Connects to Deriv via WebSocket, fetches 15 m + 60 m candles,
runs 11 strategies, meta-learns with Logistic Regression,
sizes with Kelly, manages risk, and prints BUY / SELL / HOLD
together with confidence, entry, SL, TP.
Everything (config + code) lives in this single file.
"""

import asyncio
import json
import logging
import math
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import talib
import websocket
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# 0. CONFIG – EDIT HERE ONLY
# ------------------------------------------------------------------
CONFIG = {
    "deriv_token": os.getenv("DERIV_TOKEN", "lWFYLtfTp2sbWl8"),
    "symbol": "JD10",
    "websocket_url": "wss://ws.derivws.com/websockets/v3?app_id=96329",
    "candle_size_15m": 900,  # seconds
    "candle_size_60m": 3600,
    "lookback": 500,  # how many historical candles to fetch
    "risk_per_trade": 0.01,  # 1 % of balance
    "max_drawdown": 0.05,  # 5 % equity drawdown kills trading
    "kelly_fraction_cap": 0.25,  # never bet more than ¼ Kelly
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 3.0,
    "min_confidence": 0.55,  # skip if meta-confidence < this
    "trade_stake": 1.0,  # base stake in USD
}

# ------------------------------------------------------------------
# 1. LOGGING
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jd10bot")

# ------------------------------------------------------------------
# 2. WEBSOCKET CLIENT
# ------------------------------------------------------------------
# ------------------------------------------------------------------

class DerivWS:
    def __init__(self, url, token):
        self.url       = url
        self.token     = token
        self._ws       = None
        self._queue    = asyncio.Queue()
        self._lock     = asyncio.Lock()
        self._running  = True
        self._connected = asyncio.Event()

    # ---------- low-level handlers ----------
    def _on_open(self, ws):
        log.info("WebSocket connected")
        # This runs in the background thread, so we must use a thread-safe method
        # to set the event. Fortunately, asyncio.Event methods are thread-safe.
        self._connected.set()

    def _on_close(self, ws, status, msg):
        log.warning("WebSocket disconnected")
        self._connected.clear()

    def _on_error(self, ws, error):
        log.error(f"WS error: {error}")
        self._connected.clear()

    def _on_message(self, ws, msg):
        try:
            # This also runs in the background thread. put_nowait is safe.
            self._queue.put_nowait(json.loads(msg))
        except json.JSONDecodeError:
            log.error(f"Could not decode JSON: {msg}")

    def _on_ping(self, ws, data):
        ws.send(data, opcode=websocket.ABNF.OPCODE_PONG)

    # ---------- public helpers ----------
    async def connect(self):
        """Start the socket in a background thread and authorize."""
        while self._running:
            try:
                self._queue = asyncio.Queue()
                self._connected.clear()
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open    = self._on_open,
                    on_close   = self._on_close,
                    on_error   = self._on_error,
                    on_message = self._on_message,
                    on_ping    = self._on_ping,
                )
                loop = asyncio.get_event_loop()
                loop.run_in_executor(
                    None,
                    lambda: self._ws.run_forever(
                        ping_interval=25,
                        ping_timeout=10
                    )
                )
                log.info("Connecting to WebSocket...")
                await asyncio.wait_for(self._connected.wait(), timeout=20)
                await self.authorize()
                return # Success
            except asyncio.TimeoutError:
                log.error("Connection timeout. Retrying in 10s...")
                if self._ws:
                    self._ws.close()
                await asyncio.sleep(10)
            except Exception as e:
                log.error(f"Connect failed: {e} – retry in 10s")
                if self._ws:
                    self._ws.close()
                await asyncio.sleep(10)

    async def authorize(self):
        log.info("Authorizing...")
        await self.send({"authorize": self.token})
        response = await self.wait_for("authorize")
        if response.get("error"):
            # Use single quotes inside the f-string to avoid conflict
            log.error(f"Authorization failed: {response['error']['message']}")
            raise ConnectionRefusedError("Authorization failed")
        log.info("Authorization successful")

    async def send(self, payload):
        async with self._lock:
            if self._ws and self._ws.sock and self._ws.sock.connected:
                self._ws.send(json.dumps(payload))
            else:
                log.warning("Attempted to send but socket not ready.")
                # Depending on desired behavior, you could raise an error here
                # raise RuntimeError("Socket not ready")

    async def wait_for(self, msg_type, timeout=15):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1)
                if msg.get("msg_type") == msg_type:
                    return msg
                # Optional: handle other messages that arrive while waiting
                # else:
                #    log.debug(f"Ignoring message of type {msg.get('msg_type')} while waiting for {msg_type}")
            except asyncio.TimeoutError:
                continue
        raise TimeoutError(f"Waiting for '{msg_type}' timed out after {timeout} seconds")

    async def get_candles(self, symbol, interval, count, style="candles"):
        payload = {
            "ticks_history": symbol,
            "count": count,
            "end": "latest",
            "granularity": interval,
            "style": style,
            "subscribe": 0, # Important: Not subscribing here, just getting history
        }
        await self.send(payload)
        response = await self.wait_for("candles")
        # Add error checking for the response itself
        if response.get("error"):
            log.error(f"Failed to get candles: {response['error']['message']}")
            return [] # Return empty list on error
        return response.get("candles", [])
# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------------
def compute_features(df):
    """Classic TA + micro features for ML models."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["ret"] = close.pct_change()
    df["ema9"] = talib.EMA(close, 9)
    df["ema21"] = talib.EMA(close, 21)
    df["macd"], df["macd_sig"], df["macd_hist"] = talib.MACD(close, 12, 26, 9)
    df["rsi"] = talib.RSI(close, 14)
    up, mid, low_bb = talib.BBANDS(close, 20, 2, 2)
    df["bb_pos"] = (close - low_bb) / (up - low_bb)
    df["atr"] = talib.ATR(high, low, close, 14)
    df["hl2"] = (high + low) / 2
    df["s3"], df["s2"], df["s1"], df["r1"], df["r2"], df["r3"] = pivot_points(high, low, close)
    df["label"] = (close.shift(-5) > close).astype(int)  # 5-bar ahead
    return df.dropna()


def pivot_points(high, low, close):
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    r3 = r1 + (high - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)
    s3 = s1 - (high - low)
    return s3, s2, s1, r1, r2, r3


# ------------------------------------------------------------------
# 4. INDIVIDUAL STRATEGIES
# ------------------------------------------------------------------
def strategy_ema(df):
    bullish = df["ema9"].iloc[-1] > df["ema21"].iloc[-1]
    bearish = df["ema9"].iloc[-1] < df["ema21"].iloc[-1]
    if bullish and df["ema9"].iloc[-2] <= df["ema21"].iloc[-2]:
        return 1
    if bearish and df["ema9"].iloc[-2] >= df["ema21"].iloc[-2]:
        return -1
    return 0


def strategy_macd(df):
    macd = df["macd"].iloc[-1]
    sig = df["macd_sig"].iloc[-1]
    prev_macd = df["macd"].iloc[-2]
    prev_sig = df["macd_sig"].iloc[-2]
    if prev_macd <= prev_sig and macd > sig:
        return 1
    if prev_macd >= prev_sig and macd < sig:
        return -1
    return 0


def strategy_rsi(df):
    rsi = df["rsi"].iloc[-1]
    if rsi < 30:
        return 1
    if rsi > 70:
        return -1
    return 0


def strategy_bb_mean_revert(df):
    pos = df["bb_pos"].iloc[-1]
    if pos < 0.1:
        return 1
    if pos > 0.9:
        return -1
    return 0


def strategy_snr_break(df):
    close = df["close"].iloc[-1]
    r1 = df["r1"].iloc[-1]
    s1 = df["s1"].iloc[-1]
    if close > r1 and df["close"].iloc[-2] <= r1:
        return 1
    if close < s1 and df["close"].iloc[-2] >= s1:
        return -1
    return 0


# ------------------------------------------------------------------
# 5. ML MODELS
# ------------------------------------------------------------------
def build_lstm_model(X, y):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except ImportError:
        log.warning("TensorFlow not available – LSTM returns 0")
        return lambda Xnew: np.zeros(len(Xnew))

    X3d = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X3d, y, epochs=10, batch_size=32, verbose=0)
    return lambda Xnew: model.predict(Xnew.reshape((Xnew.shape[0], Xnew.shape[1], 1)), verbose=0).flatten()


def build_rf(X, y):
    m = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    m.fit(X, y)
    return m.predict_proba


def build_xgb(X, y):
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(X, y)
    return m.predict_proba


def build_autoencoder(X):
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
    except ImportError:
        return lambda Xnew: np.zeros(len(Xnew))

    inp = Input(shape=(X.shape[1],))
    encoded = Dense(16, activation="relu")(inp)
    decoded = Dense(X.shape[1], activation="linear")(encoded)
    auto = Model(inp, decoded)
    auto.compile(optimizer="adam", loss="mse")
    auto.fit(X, X, epochs=15, batch_size=32, verbose=0)

    def recon_error(Xnew):
        pred = auto.predict(Xnew, verbose=0)
        return np.mean((Xnew - pred) ** 1, axis=1)

    return recon_error


# ------------------------------------------------------------------
# 6. META-LEARNER
# ------------------------------------------------------------------
class MetaModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.logreg = LogisticRegression()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.logreg.fit(Xs, y)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.logreg.predict_proba(Xs)[:, 1]


# ------------------------------------------------------------------
# 7. KELLY + RISK
# ------------------------------------------------------------------
def kelly_size(win_prob, avg_win, avg_loss):
    if avg_loss == 0:
        return 0
    q = 1 - win_prob
    k = (win_prob * avg_win - q * avg_loss) / avg_loss
    k = max(0, min(k, CONFIG["kelly_fraction_cap"]))
    return k


# ------------------------------------------------------------------
# 8. MAIN BRAIN
# ------------------------------------------------------------------
class JD10Bot:
    def __init__(self, ws):
        self.ws = ws
        self.balance = 10000  # demo
        self.equity_peak = self.balance
        self.hist_15m = deque(maxlen=CONFIG["lookback"])
        self.hist_60m = deque(maxlen=CONFIG["lookback"])
        self.meta = MetaModel()
        self.models_ready = False

    async def warm_up(self):
        log.info("Warming up history…")
        df15 = await self.ws.get_candles(CONFIG["symbol"], CONFIG["candle_size_15m"], CONFIG["lookback"])
        df60 = await self.ws.get_candles(CONFIG["symbol"], CONFIG["candle_size_60m"], CONFIG["lookback"] // 4)
        self.hist_15m.extend(df15.to_dict("records"))
        self.hist_60m.extend(df60.to_dict("records"))
        self.train_models()
        self.models_ready = True
        log.info("Models trained – ready to trade")

    def train_models(self):
        df15 = pd.DataFrame(self.hist_15m)
        df15 = compute_features(df15)
        feature_cols = [c for c in df15.columns if c not in ["label", "datetime"]]
        X = df15[feature_cols].values
        y = df15["label"].values

        # LSTM
        self.lstm_pred = build_lstm_model(X, y)
        # RF
        self.rf_pred = build_rf(X, y)
        # XGB
        self.xgb_pred = build_xgb(X, y)
        # Autoencoder
        self.recon = build_autoencoder(X)
        # Meta
        # Build meta features: prob from each model + recon error
        rf_p = self.rf_pred(X)[:, 1]
        xgb_p = self.xgb_pred(X)[:, 1]
        recon = self.recon(X)
        meta_X = np.column_stack([rf_p, xgb_p, recon])
        self.meta.fit(meta_X, y)
        log.info("Meta-learner trained")

    async def loop(self):
        await self.warm_up()
        while True:
            try:
                await self.tick()
                await asyncio.sleep(60)  # check every minute
            except Exception as e:
                log.error(f"Loop error: {e}")
                await asyncio.sleep(30)

    async def tick(self):
        # refresh last candle
        df15 = await self.ws.get_candles(CONFIG["symbol"], CONFIG["candle_size_15m"], 5)
        df60 = await self.ws.get_candles(CONFIG["symbol"], CONFIG["candle_size_60m"], 5)
        self.hist_15m.extend(df15.to_dict("records"))
        self.hist_60m.extend(df60.to_dict("records"))

        df15 = pd.DataFrame(self.hist_15m)
        df15 = compute_features(df15)
        df60 = pd.DataFrame(self.hist_60m)
        df60 = compute_features(df60)

        signal_15, conf_15 = self.generate_signal(df15)
        signal_60, conf_60 = self.generate_signal(df60)

        # Combine horizons: simple vote
        if signal_15 == signal_60:
            final_signal = signal_15
            final_conf = (conf_15 + conf_60) / 2
        else:
            final_signal = 0
            final_conf = 0.5

        if final_conf < CONFIG["min_confidence"]:
            final_signal = 0

        # Risk check
        dd = 1 - self.balance / self.equity_peak
        if dd > CONFIG["max_drawdown"]:
            log.warning("Max drawdown reached – flatten")
            final_signal = 0

        # Size
        atr = df15["atr"].iloc[-1]
        sl_pips = atr * CONFIG["sl_atr_mult"]
        tp_pips = atr * CONFIG["tp_atr_mult"]
        # dummy win/loss for Kelly
        win_prob = final_conf
        avg_win = tp_pips
        avg_loss = sl_pips
        k = kelly_size(win_prob, avg_win, avg_loss)
        size = CONFIG["trade_stake"] * k
        size = max(0.35, min(size, self.balance * CONFIG["risk_per_trade"]))

        # Decision
        decision = {1: "BUY", -1: "SELL", 0: "HOLD"}[final_signal]
        last_close = df15["close"].iloc[-1]
        sl_price = last_close - sl_pips if final_signal == 1 else last_close + sl_pips
        tp_price = last_close + tp_pips if final_signal == 1 else last_close - tp_pips

        log.info(
            f"{decision} | conf={final_conf:.2f} | size=${size:.2f} | "
            f"entry={last_close:.3f} | SL={sl_price:.3f} | TP={tp_price:.3f}"
        )

    def generate_signal(self, df):
        """Return signal ∈ {-1,0,1} and confidence ∈ [0,1]"""
        # Classic
        s1 = strategy_ema(df)
        s2 = strategy_macd(df)
        s3 = strategy_rsi(df)
        s4 = strategy_bb_mean_revert(df)
        s5 = strategy_snr_break(df)
        classic_vote = np.sign(s1 + s2 + s3 + s4 + s5)

        # ML
        feature_cols = [c for c in df.columns if c not in ["label", "datetime"]]
        X_last = df[feature_cols].values[-1:]
        rf_p = self.rf_pred(X_last)[0, 1]
        xgb_p = self.xgb_pred(X_last)[0, 1]
        lstm_p = self.lstm_pred(X_last)[0]
        recon = self.recon(X_last)[0]

        meta_X = np.column_stack([rf_p, xgb_p, recon])
        meta_p = self.meta.predict_proba(meta_X)[0]

        # Combine
        ml_signal = 1 if meta_p > 0.55 else (-1 if meta_p < 0.45 else 0)
        consensus = np.sign(classic_vote + ml_signal)
        confidence = meta_p if consensus == 1 else (1 - meta_p if consensus == -1 else 0.5)
        return int(consensus), float(confidence)


# ------------------------------------------------------------------
# 9. ENTRY
# ------------------------------------------------------------------
async def main():
    ws = DerivWS(CONFIG["websocket_url"], CONFIG["deriv_token"])
    await ws.connect()
    bot = JD10Bot(ws)
    try:
        await bot.loop()
    except asyncio.CancelledError:
        ws._running = False
        log.info("bot stopped by user")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped by user")