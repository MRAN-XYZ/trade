import math
import numpy as np
import asyncio
import json
import websockets
from datetime import datetime, timedelta
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")

# Configuration
API_TOKEN = "yKga73O12NCnU6a"  # Replace with your Deriv API token
APP_ID = 96329  # Deriv API application ID
SYMBOL = "JD10"  # Japan 10 Index
WS_URL = "wss://ws.derivws.com/websockets/v3"
ACCOUNT_CURRENCY = "USD"
INITIAL_BALANCE = 10000.0  # Initial account balance
RISK_PER_TRADE = 0.02  # Risk 2% of account per trade

Signal = namedtuple("Signal", ["name", "score", "weight", "age_ticks", "veto", "metadata"])
TradeSignal = namedtuple("TradeSignal", ["action", "confidence", "entry_price", "stop_loss", "take_profit", "size"])


class AdvancedNet(nn.Module):
    """Enhanced neural network with LSTM and attention mechanism"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(AdvancedNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x):
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        out = self.fc(context_vector)
        return out


class TradingSystem:
    def __init__(self, api_token, app_id, max_training_points=1000, sequence_length=60):
        self.api_token = api_token
        self.app_id = app_id
        self.websocket = None
        self.req_id = 1
        self.max_training_points = max_training_points
        self.sequence_length = sequence_length
        self.account_balance = INITIAL_BALANCE
        self.equity_curve = []
        self.current_position = None
        self.trade_history = []
        self.price_history = deque(maxlen=1000)
        self.time_history = deque(maxlen=1000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    async def connect(self):
        """Connect to Deriv WebSocket API"""
        try:
            self.websocket = await websockets.connect(
                f"{WS_URL}?app_id={self.app_id}", 
                ping_interval=30,
                ping_timeout=10,
                max_size=2**20  # 1MB max message size
            )
            logger.info("WebSocket connection established")

            # Authorize the connection
            auth_request = {
                "authorize": self.api_token,
                "req_id": self.req_id
            }
            self.req_id += 1

            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            auth_response = json.loads(response)

            if "error" in auth_response:
                logger.error(f"Authorization failed: {auth_response['error']['message']}")
                return False

            logger.info("Successfully authorized with Deriv API")
            
            # Subscribe to balance updates
            balance_request = {
                "balance": 1,
                "subscribe": 1,
                "req_id": self.req_id
            }
            self.req_id += 1
            await self.websocket.send(json.dumps(balance_request))
            
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def send_request(self, request, timeout=10):
        """Send a request and wait for response with timeout"""
        try:
            request["req_id"] = self.req_id
            self.req_id += 1

            await self.websocket.send(json.dumps(request))

            # Wait for response with timeout
            response = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            data = json.loads(response)

            if data.get("req_id") == request["req_id"]:
                return data
            else:
                logger.warning(f"Request ID mismatch: expected {request['req_id']}, got {data.get('req_id')}")
                return None

        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {request}")
            return None
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return None

    async def get_historical_data(self, symbol, count=1000, granularity='1m'):
        """Fetch historical price data for a symbol with different granularity"""
        try:
            # Map granularity to API parameters
            if granularity == 'tick':
                request = {
                    "ticks_history": symbol,
                    "count": count,
                    "end": "latest",
                    "style": "ticks"
                }
            else:
                # Convert granularity to seconds
                granularity_map = {'1m': 60, '5m': 300, '1h': 3600, '1d': 86400}
                if granularity not in granularity_map:
                    logger.warning(f"Unsupported granularity: {granularity}, using 1m")
                    granularity = '1m'
                
                request = {
                    "ticks_history": symbol,
                    "count": count,
                    "end": "latest",
                    "granularity": granularity_map[granularity],
                    "style": "candles"
                }

            response = await self.send_request(request)

            if response and "history" in response:
                if granularity == 'tick':
                    prices = [float(price) for price in response["history"]["prices"]]
                    times = response["history"]["times"]
                else:
                    # For candles, we get OHLC data
                    candles = response["history"]["candles"]
                    prices = [float(candle["close"]) for candle in candles]
                    times = [candle["epoch"] for candle in candles]
                
                logger.info(f"Fetched {len(prices)} historical prices for {symbol} ({granularity})")
                return prices, times
            elif response and "error" in response:
                logger.error(f"API Error: {response['error']['message']}")
                return None, None
            else:
                logger.warning("No historical data received")
                return None, None

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None, None

    async def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            request = {"ticks": symbol, "subscribe": 1}
            response = await self.send_request(request)

            if response and "tick" in response:
                current_price = float(response["tick"]["quote"])
                current_time = response["tick"]["epoch"]
                logger.info(f"Current price for {symbol}: {current_price}")
                return current_price, current_time
            elif response and "error" in response:
                logger.error(f"API Error: {response['error']['message']}")
                return None, None
            else:
                logger.warning("No current price received")
                return None, None

        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None, None

    async def place_trade(self, symbol, action, amount, stop_loss=None, take_profit=None):
        """Place a trade with risk management"""
        try:
            trade_request = {
                "buy": symbol,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "currency": ACCOUNT_CURRENCY,
                    "duration": 60,  # 1 hour
                    "duration_unit": "m",
                    "symbol": symbol
                }
            }
            
            if action.upper() == "SELL":
                trade_request["buy"] = symbol
                trade_request["price"] = -amount  # Negative amount for sell
            
            if stop_loss:
                trade_request["parameters"]["stop_loss"] = stop_loss
            if take_profit:
                trade_request["parameters"]["take_profit"] = take_profit
                
            response = await self.send_request(trade_request)
            
            if response and "error" not in response:
                logger.info(f"Trade executed: {action} {amount} {symbol}")
                return response
            else:
                error_msg = response["error"]["message"] if response and "error" in response else "Unknown error"
                logger.error(f"Trade failed: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return None

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

    def calculate_features(self, prices):
        """Calculate comprehensive feature set for prediction"""
        if len(prices) < 50:
            return None, None
            
        features = []
        price_array = np.array(prices)
        
        # Price changes
        for period in [1, 5, 10, 20, 50]:
            if len(prices) > period:
                change = (price_array[-1] / price_array[-period] - 1) * 100
                features.append(change)
            else:
                features.append(0)
                
        # Technical indicators
        rsi = self.calculate_rsi(prices)
        if rsi is not None:
            features.append(rsi[-1])
        else:
            features.append(50)  # Neutral RSI
            
        macd, signal, histogram = self.calculate_macd(prices)
        if macd is not None and signal is not None:
            features.append(macd[-1])
            features.append(signal[-1])
            features.append(histogram[-1])
        else:
            features.extend([0, 0, 0])
            
        # Moving averages
        for period in [5, 10, 20, 50]:
            ema = self.calculate_ema(prices, period)
            if ema is not None:
                features.append(ema[-1] / prices[-1] - 1)
            else:
                features.append(0)
                
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = self.calculate_bollinger(prices)
        if bb_middle is not None:
            bb_position = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            features.append(bb_position)
        else:
            features.append(0.5)  # Middle of band
            
        # Volatility
        if len(prices) > 20:
            returns = np.diff(np.log(prices))
            volatility = np.std(returns[-20:]) * np.sqrt(252) * 100  # Annualized percentage
            features.append(volatility)
        else:
            features.append(0)
            
        # Statistical features
        if len(prices) > 20:
            features.append(stats.skew(np.diff(prices[-20:])))
            features.append(stats.kurtosis(np.diff(prices[-20:])))
        else:
            features.extend([0, 0])
            
        return np.array(features), len(features)

    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM training"""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)

    def calculate_future_predictions(self, prices, periods=60):
        """Predict future prices using an advanced neural network with LSTM"""
        if len(prices) < 100:
            return None, None, None

        # Calculate features
        feature_vectors = []
        for i in range(50, len(prices) - periods):
            window_prices = prices[:i+1]
            features, n_features = self.calculate_features(window_prices)
            if features is not None:
                feature_vectors.append(features)
        
        if len(feature_vectors) < 50:
            return None, None, None
            
        feature_matrix = np.array(feature_vectors)
        
        # Create sequences
        X = self.create_sequences(feature_matrix, self.sequence_length)
        y = []
        
        for i in range(50 + self.sequence_length, len(prices) - periods):
            future_return = (prices[i + periods] / prices[i] - 1) * 100
            y.append(future_return)
            
        y = np.array(y)
        
        if len(X) != len(y) or len(X) == 0:
            return None, None, None
            
        # Standardize features
        X_means, X_stds = np.mean(X, axis=(0, 1)), np.std(X, axis=(0, 1))
        X_stds[X_stds == 0] = 1  # Prevent division by zero
        X_scaled = (X - X_means) / X_stds

        # Convert to torch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Define and train neural network
        model = AdvancedNet(input_size=X.shape[2]).to(self.device)
        criterion = nn.HuberLoss()  # More robust than MSE
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        # Train the model
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.debug(f"Early stopping at epoch {epoch}")
                break

        # Make prediction for current state
        current_features, n_features = self.calculate_features(prices)
        if current_features is None:
            return None, None, None
            
        # Create sequence for prediction
        recent_features = []
        for i in range(len(prices) - self.sequence_length, len(prices)):
            window_prices = prices[:i+1]
            features, _ = self.calculate_features(window_prices)
            if features is not None:
                recent_features.append(features)
                
        if len(recent_features) < self.sequence_length:
            return None, None, None
            
        recent_features = np.array(recent_features)
        recent_scaled = (recent_features - X_means) / X_stds
        recent_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        model.eval()
        with torch.no_grad():
            predicted_change = model(recent_tensor).item()

        predicted_price = prices[-1] * (1 + predicted_change / 100)

        # Calculate confidence based on model performance
        with torch.no_grad():
            predictions = model(X_tensor).detach().cpu().numpy().flatten()
        errors = np.abs(predictions - y)
        mae = np.mean(errors)
        y_std = np.std(y) if np.std(y) > 0 else 1
        confidence = 1 / (1 + mae / y_std)

        return predicted_price, predicted_change, confidence

    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None

        prices = np.array(prices)
        ema = np.zeros(len(prices))
        alpha = 2.0 / (period + 1)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0

        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            upval = delta if delta > 0 else 0
            downval = -delta if delta < 0 else 0

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return None, None, None

        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)

        if ema_fast is None or ema_slow is None:
            return None, None, None

        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)

        if signal_line is None:
            return macd_line, None, None

        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_bollinger(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None, None, None

        prices = np.array(prices)
        sma = np.convolve(prices, np.ones(period), 'valid') / period
        rolling_std = np.array([np.std(prices[i:i+period]) for i in range(len(prices) - period + 1)])
        upper = sma + std_dev * rolling_std
        lower = sma - std_dev * rolling_std
        
        # Pad with NaN to match input length
        pad_length = len(prices) - len(sma)
        sma = np.concatenate([np.full(pad_length, np.nan), sma])
        upper = np.concatenate([np.full(pad_length, np.nan), upper])
        lower = np.concatenate([np.full(pad_length, np.nan), lower])
        
        return sma, upper, lower

    def calculate_atr(self, high_prices, low_prices, close_prices, period=14):
        """Calculate Average True Range for volatility measurement"""
        if len(high_prices) < period or len(low_prices) < period or len(close_prices) < period:
            return None
            
        tr = np.zeros(len(high_prices))
        for i in range(1, len(high_prices)):
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i-1])
            tr3 = abs(low_prices[i] - close_prices[i-1])
            tr[i] = max(tr1, tr2, tr3)
            
        atr = np.zeros(len(high_prices))
        atr[period] = np.mean(tr[1:period+1])
        
        for i in range(period+1, len(high_prices)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        return atr

    def calculate_position_size(self, entry_price, stop_loss, risk_per_trade=RISK_PER_TRADE):
        """Calculate position size based on risk management"""
        risk_amount = self.account_balance * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
            
        position_size = risk_amount / risk_per_share
        return round(position_size, 2)

    def update_account_balance(self, new_balance):
        """Update account balance and equity curve"""
        self.account_balance = new_balance
        self.equity_curve.append((datetime.now(), new_balance))
        logger.info(f"Account balance updated: {new_balance:.2f} {ACCOUNT_CURRENCY}")

    def plot_equity_curve(self):
        """Plot equity curve over time"""
        if len(self.equity_curve) < 2:
            return
            
        times, balances = zip(*self.equity_curve)
        plt.figure(figsize=(10, 6))
        plt.plot(times, balances)
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel(f"Balance ({ACCOUNT_CURRENCY})")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("equity_curve.png")
        plt.close()


def normalize_prob(p):
    """Normalize probability to [-1, 1] range"""
    return 2 * (p - 0.5)


def exp_decay(age, tau=5.0):
    """Calculate exponential decay for signal aging"""
    return math.exp(-age / tau)


def generate_signals(prices, current_time, trading_system):
    """Generate trading signals from price data and technical indicators"""
    signals = []

    # RSI Signal
    rsi = trading_system.calculate_rsi(prices)
    if rsi is not None:
        rsi_value = rsi[-1]
        if rsi_value < 30:
            rsi_score = 1.0  # Oversold - buy signal
        elif rsi_value > 70:
            rsi_score = -1.0  # Overbought - sell signal
        else:
            # Normalize RSI to [-1, 1] range
            rsi_score = normalize_prob((rsi_value - 30) / 40)
        signals.append(Signal("rsi", rsi_score, 0.7, 0, False, {"value": rsi_value}))

    # MACD Signal
    macd_line, signal_line, histogram = trading_system.calculate_macd(prices)
    if macd_line is not None and signal_line is not None:
        if len(macd_line) > 1 and len(signal_line) > 1:
            # Check for crossovers
            if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
                macd_score = 1.0  # Bullish crossover
            elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
                macd_score = -1.0  # Bearish crossover
            else:
                # Use histogram relative to recent range
                max_hist = max(abs(histogram[-10:])) if len(histogram) >= 10 else 1.0
                macd_score = histogram[-1] / max_hist if max_hist != 0 else 0
            signals.append(Signal("macd", macd_score, 0.8, 0, False, {
                "macd": macd_line[-1],
                "signal": signal_line[-1],
                "histogram": histogram[-1]
            }))

    # EMA Trend Signal
    ema_short = trading_system.calculate_ema(prices, 10)
    ema_long = trading_system.calculate_ema(prices, 20)
    if ema_short is not None and ema_long is not None:
        ema_score = 1.0 if ema_short[-1] > ema_long[-1] else -1.0
        signals.append(Signal("ema_trend", ema_score, 0.6, 0, False, {
            "ema_short": ema_short[-1],
            "ema_long": ema_long[-1]
        }))

    # Momentum Signal
    if len(prices) >= 5:
        momentum = (prices[-1] - prices[-5]) / prices[-5] * 100
        momentum_score = np.tanh(momentum / 5)  # Normalize using tanh
        signals.append(Signal("momentum", momentum_score, 0.5, 0, False, {"value": momentum}))

    # Bollinger Bands Signal
    sma, upper, lower = trading_system.calculate_bollinger(prices)
    if sma is not None and not np.isnan(sma[-1]):
        current_price = prices[-1]
        position = (current_price - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) != 0 else 0.5
        if position < 0:
            bb_score = 1.0  # Below lower band - strong buy
        elif position > 1:
            bb_score = -1.0  # Above upper band - strong sell
        else:
            bb_score = normalize_prob(position)
        signals.append(Signal("bollinger", bb_score, 0.65, 0, False, {
            "position": position,
            "upper": upper[-1],
            "lower": lower[-1],
            "middle": sma[-1]
        }))

    # 15-minute Prediction Signal
    pred_15, change_15, confidence_15 = trading_system.calculate_future_predictions(prices, 15)
    if pred_15 is not None:
        pred_score_15 = np.tanh(change_15 / 10)  # Scale and normalize
        weight_15 = 0.6 * confidence_15
        signals.append(Signal("15min_pred", pred_score_15, weight_15, 0, False, {
            "predicted_price": pred_15,
            "predicted_change": change_15,
            "confidence": confidence_15
        }))

    # 60-minute Prediction Signal
    pred_60, change_60, confidence_60 = trading_system.calculate_future_predictions(prices, 60)
    if pred_60 is not None:
        pred_score_60 = np.tanh(change_60 / 10)  # Scale and normalize
        weight_60 = 0.7 * confidence_60
        signals.append(Signal("60min_pred", pred_score_60, weight_60, 0, False, {
            "predicted_price": pred_60,
            "predicted_change": change_60,
            "confidence": confidence_60
        }))

    # Volume-based signals (if available)
    # Note: Deriv API doesn't provide volume for all symbols
    
    # Market Regime Detection
    if len(prices) > 100:
        returns = np.diff(np.log(prices))
        volatility = np.std(returns[-20:]) * np.sqrt(252) * 100  # Annualized percentage
        
        if volatility < 10:
            regime_score = 0.2  # Low volatility - cautious
        elif volatility > 30:
            regime_score = -0.3  # High volatility - risk off
        else:
            regime_score = 0  # Normal market
            
        signals.append(Signal("market_regime", regime_score, 0.4, 0, False, {"volatility": volatility}))

    return signals


def aggregate_signals(signals, trend_ok=True, min_conv=0.35):
    """Aggregate multiple signals into a single trading decision"""
    # Check for veto signals (override all other signals)
    for s in signals:
        if s.veto:
            return {"action": "HOLD", "score": 0.0, "reason": f"veto:{s.name}"}

    # Calculate weighted ensemble score
    scores, weights = [], []
    for s in signals:
        decay = exp_decay(s.age_ticks, tau=5.0)
        scores.append(s.score * s.weight * decay)
        weights.append(s.weight * decay)

    if sum(weights) == 0:
        return {"action": "HOLD", "score": 0.0, "reason": "no_weight"}

    ensemble = sum(scores) / sum(weights)

    # Apply trend and conviction filters
    if not trend_ok and abs(ensemble) > 0:
        return {"action": "HOLD", "score": ensemble, "reason": "trend_block"}

    if abs(ensemble) < min_conv:
        return {"action": "HOLD", "score": ensemble, "reason": "low_conv"}

    action = "BUY" if ensemble > 0 else "SELL"
    return {"action": action, "score": ensemble, "reason": "ok"}


def calculate_stop_loss_take_profit(entry_price, action, atr=None, risk_reward_ratio=1.5):
    """Calculate stop loss and take profit levels"""
    if atr is None:
        # Default to 2% stop loss if ATR is not available
        if action == "BUY":
            stop_loss = entry_price * 0.98
            take_profit = entry_price * 1.03  # 1:1.5 risk-reward ratio
        else:
            stop_loss = entry_price * 1.02
            take_profit = entry_price * 0.97
    else:
        # Use ATR for volatility-based stops
        if action == "BUY":
            stop_loss = entry_price - 2 * atr
            take_profit = entry_price + 3 * atr  # 1:1.5 risk-reward ratio
        else:
            stop_loss = entry_price + 2 * atr
            take_profit = entry_price - 3 * atr
            
    return stop_loss, take_profit


async def trading_loop(trading_system, symbol, update_interval=60):
    """Main trading loop that runs continuously"""
    logger.info("Starting trading loop")
    
    try:
        if not await trading_system.connect():
            raise Exception("API connection failed")
            
        # Initial data collection
        prices, times = await trading_system.get_historical_data(symbol, count=500)
        
        if not prices or len(prices) == 0:
            raise Exception("No historical data received")
            
        # Main trading loop
        while True:
            try:
                # Get current price
                current_price, current_time = await trading_system.get_current_price(symbol)
                if current_price:
                    prices.append(current_price)
                    times.append(current_time)
                    trading_system.price_history.append(current_price)
                    trading_system.time_history.append(current_time)
                
                # Generate signals
                signals = generate_signals(prices, current_time, trading_system)
                
                # Generate trading decision
                decision = aggregate_signals(signals, trend_ok=True, min_conv=0.35)
                
                # Display results
                logger.info("\n=== TRADING SIGNALS ===")
                for signal in signals:
                    logger.info(f"{signal.name}: {signal.score:.3f} (weight: {signal.weight:.2f}, age: {signal.age_ticks})")

                logger.info(f"\n=== TRADING DECISION ===")
                logger.info(f"Action: {decision['action']}")
                logger.info(f"Confidence: {abs(decision['score']):.3f}")
                logger.info(f"Reason: {decision['reason']}")
                
                # Execute trade if needed
                if decision['action'] != 'HOLD' and trading_system.current_position is None:
                    # Calculate position size
                    atr = None  # Would need high/low data for accurate ATR
                    stop_loss, take_profit = calculate_stop_loss_take_profit(
                        current_price, decision['action'], atr
                    )
                    
                    position_size = trading_system.calculate_position_size(
                        current_price, stop_loss
                    )
                    
                    if position_size > 0:
                        # Place trade
                        trade_result = await trading_system.place_trade(
                            symbol, 
                            decision['action'], 
                            position_size,
                            stop_loss,
                            take_profit
                        )
                        
                        if trade_result:
                            trading_system.current_position = {
                                "action": decision['action'],
                                "entry_price": current_price,
                                "size": position_size,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "timestamp": datetime.now()
                            }
                            logger.info(f"Trade opened: {trading_system.current_position}")
                
                # Check if we need to close position
                if trading_system.current_position:
                    position = trading_system.current_position
                    
                    # Check stop loss and take profit
                    if (position['action'] == 'BUY' and current_price <= position['stop_loss']) or \
                       (position['action'] == 'SELL' and current_price >= position['stop_loss']):
                        # Stop loss hit
                        logger.info(f"Stop loss hit at {current_price}")
                        trading_system.current_position = None
                    elif (position['action'] == 'BUY' and current_price >= position['take_profit']) or \
                         (position['action'] == 'SELL' and current_price <= position['take_profit']):
                        # Take profit hit
                        logger.info(f"Take profit hit at {current_price}")
                        trading_system.current_position = None
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(update_interval)
                
    except Exception as e:
        logger.error(f"Fatal error in trading system: {e}")
    finally:
        await trading_system.disconnect()


async def main():
    """Main trading system execution"""
    trading_system = TradingSystem(API_TOKEN, APP_ID)

    # Run the trading loop
    await trading_loop(trading_system, SYMBOL, update_interval=60)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())