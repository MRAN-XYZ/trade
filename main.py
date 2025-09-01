import math
import numpy as np
import asyncio
import json
import websockets
from datetime import datetime
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

# Configuration
API_TOKEN = "yKga73O12NCnU6a"  # Replace with your Deriv API token
APP_ID = 96329  # Deriv API application ID
SYMBOL = "JD10"  # Japan 10 Index
WS_URL = "wss://ws.derivws.com/websockets/v3"

Signal = namedtuple("Signal", ["name", "score", "weight", "age_ticks", "veto"])


class TradingSystem:
    def __init__(self, api_token, app_id, max_training_points=500):
        self.api_token = api_token
        self.app_id = app_id
        self.websocket = None
        self.req_id = 1
        self.max_training_points = max_training_points

    async def connect(self):
        """Connect to Deriv WebSocket API"""
        try:
            self.websocket = await websockets.connect(f"{WS_URL}?app_id={self.app_id}")
            print("WebSocket connection established")
            
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
                print(f"Authorization failed: {auth_response['error']['message']}")
                return False
            
            print("Successfully authorized with Deriv API")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def send_request(self, request):
        """Send a request and wait for response"""
        try:
            request["req_id"] = self.req_id
            self.req_id += 1

            await self.websocket.send(json.dumps(request))
            
            while True:
                response = await self.websocket.recv()
                data = json.loads(response)
                
                if data.get("req_id") == request["req_id"]:
                    return data
                    
        except Exception as e:
            print(f"Error sending request: {e}")
            return None

    async def get_historical_data(self, symbol, count=500):
        """Fetch historical price data for a symbol"""
        try:
            request = {
                "ticks_history": symbol,
                "count": count,
                "end": "latest",
                "style": "ticks"
            }

            response = await self.send_request(request)

            if response and "history" in response:
                prices = [float(price) for price in response["history"]["prices"]]
                times = response["history"]["times"]
                print(f"Fetched {len(prices)} historical prices for {symbol}")
                return prices, times
            elif response and "error" in response:
                print(f"API Error: {response['error']['message']}")
                return None, None
            else:
                print("No historical data received")
                return None, None
                
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None, None

    async def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            request = {"ticks": symbol}
            response = await self.send_request(request)
            
            if response and "tick" in response:
                current_price = float(response["tick"]["quote"])
                current_time = response["tick"]["epoch"]
                print(f"Current price for {symbol}: {current_price}")
                return current_price, current_time
            elif response and "error" in response:
                print(f"API Error: {response['error']['message']}")
                return None, None
            else:
                print("No current price received")
                return None, None
                
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None, None

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            print("WebSocket connection closed")

    def calculate_future_predictions(self, prices, periods=60):
        """Predict future prices using a neural network with rolling window"""
        if len(prices) < 50:
            return None, None, None
        
        # Use only the last N points for training
        window_prices = prices[-self.max_training_points:]
        
        X, y = [], []
        for i in range(20, len(window_prices) - periods):
            # Calculate recent price changes
            recent_changes = [
                window_prices[i] / window_prices[i - 1] - 1,      # 1-tick change
                window_prices[i] / window_prices[i - 5] - 1,      # 5-tick change
                window_prices[i] / window_prices[i - 10] - 1,     # 10-tick change
                window_prices[i] / window_prices[i - 20] - 1,     # 20-tick change
            ]

            # Calculate technical indicators
            rsi = self.calculate_rsi(window_prices[:i + 1])
            macd, _, _ = self.calculate_macd(window_prices[:i + 1])
            ema_short = self.calculate_ema(window_prices[:i + 1], 10)
            ema_long = self.calculate_ema(window_prices[:i + 1], 20)

            if all(indicator is not None for indicator in [rsi, macd, ema_short, ema_long]):
                features = recent_changes + [
                    rsi[-1] / 100,                                    # Normalized RSI
                    macd[-1] / window_prices[i],                      # MACD relative to price
                    ema_short[-1] / window_prices[i] - 1,             # EMA short deviation
                    ema_long[-1] / window_prices[i] - 1,              # EMA long deviation
                ]
                
                X.append(features)
                y.append(window_prices[i + periods] / window_prices[i] - 1)  # Future return
        
        if len(X) < 20:
            return None, None, None
        
        X, y = np.array(X), np.array(y)
        
        # Standardize features
        X_means, X_stds = np.mean(X, axis=0), np.std(X, axis=0)
        X_stds[X_stds == 0] = 1  # Prevent division by zero
        X_scaled = (X - X_means) / X_stds
        
        # Convert to torch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Define neural network
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(8, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Train the model
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Make prediction for current state
        if len(window_prices) < 21:
            return None, None, None
        
        current_features = [
            window_prices[-1] / window_prices[-2] - 1,
            window_prices[-1] / window_prices[-6] - 1,
            window_prices[-1] / window_prices[-11] - 1,
            window_prices[-1] / window_prices[-21] - 1,
        ]

        current_rsi = self.calculate_rsi(window_prices)
        current_macd, _, _ = self.calculate_macd(window_prices)
        current_ema_short = self.calculate_ema(window_prices, 10)
        current_ema_long = self.calculate_ema(window_prices, 20)

        if any(indicator is None for indicator in [current_rsi, current_macd, current_ema_short, current_ema_long]):
            return None, None, None

        current_features.extend([
            current_rsi[-1] / 100,
            current_macd[-1] / window_prices[-1],
            current_ema_short[-1] / window_prices[-1] - 1,
            current_ema_long[-1] / window_prices[-1] - 1,
        ])

        current_scaled = (np.array(current_features) - X_means) / X_stds
        current_tensor = torch.tensor(current_scaled, dtype=torch.float32).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            predicted_change = model(current_tensor).item()
        
        predicted_price = window_prices[-1] * (1 + predicted_change)

        # Calculate confidence based on model performance
        with torch.no_grad():
            predictions = model(X_tensor).detach().numpy().flatten()
        errors = np.abs(predictions - y)
        mae = np.mean(errors)
        y_std = np.std(y) if np.std(y) > 0 else 1
        confidence = 1 / (1 + mae / y_std)

        # Filter out very small movements (noise)
        if abs(predicted_change) < 0.0005:  # Less than 0.05%
            predicted_change = 0.0

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
        return sma, upper, lower


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
        signals.append(Signal("rsi", rsi_score, 0.7, 0, False))

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
            signals.append(Signal("macd", macd_score, 0.8, 0, False))

    # EMA Trend Signal
    ema_short = trading_system.calculate_ema(prices, 10)
    ema_long = trading_system.calculate_ema(prices, 20)
    if ema_short is not None and ema_long is not None:
        ema_score = 1.0 if ema_short[-1] > ema_long[-1] else -1.0
        signals.append(Signal("ema_trend", ema_score, 0.6, 0, False))

    # Momentum Signal
    if len(prices) >= 5:
        momentum = (prices[-1] - prices[-5]) / prices[-5] * 100
        momentum_score = np.tanh(momentum / 5)  # Normalize using tanh
        signals.append(Signal("momentum", momentum_score, 0.5, 0, False))

    # Bollinger Bands Signal (new)
    sma, upper, lower = trading_system.calculate_bollinger(prices)
    if sma is not None:
        current_price = prices[-1]
        position = (current_price - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) != 0 else 0.5
        if position < 0:
            bb_score = 1.0  # Below lower band - strong buy
        elif position > 1:
            bb_score = -1.0  # Above upper band - strong sell
        else:
            bb_score = normalize_prob(position)
        signals.append(Signal("bollinger", bb_score, 0.65, 0, False))

    # 15-minute Prediction Signal
    pred_15, change_15, confidence_15 = trading_system.calculate_future_predictions(prices, 15)
    if pred_15 is not None:
        pred_score_15 = np.tanh(change_15 * 10)  # Scale and normalize
        weight_15 = 0.6 * confidence_15
        signals.append(Signal("15min_pred", pred_score_15, weight_15, 0, False))

    # 60-minute Prediction Signal
    pred_60, change_60, confidence_60 = trading_system.calculate_future_predictions(prices, 60)
    if pred_60 is not None:
        pred_score_60 = np.tanh(change_60 * 10)  # Scale and normalize
        weight_60 = 0.7 * confidence_60
        signals.append(Signal("60min_pred", pred_score_60, weight_60, 0, False))

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


async def main():
    """Main trading system execution"""
    trading_system = TradingSystem(API_TOKEN, APP_ID)

    prices_for_pred = None
    signals = None

    try:
        if not await trading_system.connect():
            raise Exception("API connection failed")

        # Fetch real market data
        prices, times = await trading_system.get_historical_data(SYMBOL, count=500)
        
        if prices and len(prices) > 0:
            current_price, current_time = await trading_system.get_current_price(SYMBOL)
            if current_price:
                prices.append(current_price)
                times.append(current_time)
            
            print(f"Using {len(prices)} price points for {SYMBOL}")
            signals = generate_signals(prices, current_time if current_price else times[-1], trading_system)
            prices_for_pred = prices
        else:
            raise Exception("No data received from API")

    except Exception as e:
        print(f"Error during API operations: {e}. Using mock data for demonstration.")
        np.random.seed(42)  # For reproducibility
        mock_prices = np.cumsum(np.random.normal(0, 1, 500)) + 100
        mock_time = datetime.now().timestamp()
        signals = generate_signals(mock_prices, mock_time, trading_system)
        prices_for_pred = mock_prices

    finally:
        await trading_system.disconnect()

    if signals is None or prices_for_pred is None:
        print("No signals generated. Exiting.")
        return

    # Generate trading decision
    res = aggregate_signals(signals, trend_ok=True, min_conv=0.35)

    # Display results
    print("\n=== TRADING SIGNALS ===")
    for signal in signals:
        print(f"{signal.name}: {signal.score:.3f} (weight: {signal.weight:.2f}, age: {signal.age_ticks})")

    print(f"\n=== TRADING DECISION ===")
    print(f"Action: {res['action']}")
    print(f"Confidence: {abs(res['score']):.3f}")
    print(f"Reason: {res['reason']}")

    # Display predictions
    pred_15, change_15, confidence_15 = trading_system.calculate_future_predictions(prices_for_pred, 15)
    pred_60, change_60, confidence_60 = trading_system.calculate_future_predictions(prices_for_pred, 60)

    if pred_15 is not None:
        print(f"\n=== 15-MINUTE PREDICTION ===")
        print(f"Predicted price: {pred_15:.2f}")
        print(f"Expected change: {change_15*100:.2f}%")
        print(f"Confidence: {confidence_15:.2f}")

    if pred_60 is not None:
        print(f"\n=== 60-MINUTE PREDICTION ===")
        print(f"Predicted price: {pred_60:.2f}")
        print(f"Expected change: {change_60*100:.2f}%")
        print(f"Confidence: {confidence_60:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
