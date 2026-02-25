"""
Ryan Kang, Parth Halani, Hadrien Courbe, Youti Wan, Aadith Jerfy
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=10.0):
            self.lookback = lookback
            self.position_size = position_size

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['Close'] > df['sma'], 'signal'] = 1
            df.loc[df['Close'] < df['sma'], 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)
    """

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(self, short_window: int = 20, long_window: int = 60, position_size: float = 10.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = 0
        df.loc[df["MA_short"] > df["MA_long"], "position"] = 1
        df.loc[df["MA_short"] < df["MA_long"], "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateStrategy(Strategy):
    """
    Starter strategy template for students. Modify the indicator and signal
    logic to build your own ideas.
    """

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 10.0,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = df["momentum"] > self.buy_threshold
        sell = df["momentum"] < self.sell_threshold

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendStrategy(Strategy):
    """
    Crypto trend-following strategy using fast/slow EMAs (long-only).
    """

    def __init__(self, short_window: int = 7, long_window: int = 21, position_size: float = 100.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df

class DemoStrategy(Strategy):
    """
    Simple demo strategy - buys 1 share when price up, sells 1 share when price down.
    Uses tiny position size to avoid margin/locate issues.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1   # Price went up -> buy
        df.loc[df["change"] < 0, "signal"] = -1  # Price went down -> sell
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Strategy
##
## class RSIStrategy(Strategy):
##     """Buy when RSI is oversold, sell when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=10.0):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##
##     def add_indicators(self, df):
##         delta = df['Close'].diff()
##         gain = delta.where(delta > 0, 0).rolling(self.period).mean()
##         loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
##         rs = gain / loss
##         df['RSI'] = 100 - (100 / (1 + rs))
##         return df
##
##     def generate_signals(self, df):
##         df['signal'] = 0
##         df.loc[df['RSI'] < self.oversold, 'signal'] = 1   # Buy when oversold
##         df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df
##
## To use your strategy:
##   python run_live.py --symbol AAPL --strategy mystrategy --live
## Lets goooooo

#def skibidi(int): yessir Bro bro bro

class MyStrategy(Strategy):

    def __init__(
        self,
        ema_window: int            = 20,
        atr_window: int            = 14,
        atr_sma_window: int        = 8,
        breakout_window: int       = 8,
        rsi_period: int            = 14,
        rsi_max_long: float        = 75.0,
        trailing_sma_window: int   = 10,
        hard_stop_atr_mult: float  = 2.0,
        max_hold_bars: int         = 30,
        cooldown_bars: int         = 3,
        risk_per_trade_pct: float  = 0.02,   # risk 2% of capital per trade
        capital: float             = 100000.0,
    ):
        self.ema_window          = ema_window
        self.atr_window          = atr_window
        self.atr_sma_window      = atr_sma_window
        self.breakout_window     = breakout_window
        self.rsi_period          = rsi_period
        self.rsi_max_long        = rsi_max_long
        self.trailing_sma_window = trailing_sma_window
        self.hard_stop_atr_mult  = hard_stop_atr_mult
        self.max_hold_bars       = max_hold_bars
        self.cooldown_bars       = cooldown_bars
        self.risk_per_trade_pct  = risk_per_trade_pct
        self.capital             = capital
        self._position           = 0
        self._entry_price        = 0.0
        self._hard_stop          = 0.0
        self._bars_in_trade      = 0
        self._bars_since_exit    = 999

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        df["EMA"] = close.ewm(span=self.ema_window, adjust=False).mean()

        prev = close.shift(1)
        tr   = pd.concat([
            (high - low).abs(),
            (high - prev).abs(),
            (low  - prev).abs(),
        ], axis=1).max(axis=1)
        df["ATR"]     = tr.rolling(self.atr_window, min_periods=2).mean().fillna(0)
        df["ATR_SMA"] = df["ATR"].rolling(self.atr_sma_window, min_periods=2).mean().fillna(0)

        df["swing_high"] = high.shift(1).rolling(self.breakout_window, min_periods=2).max()
        df["swing_low"]  = low.shift(1).rolling(self.breakout_window, min_periods=2).min()
        df["trail_SMA"]  = close.rolling(self.trailing_sma_window, min_periods=2).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(self.rsi_period, min_periods=2).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.rsi_period, min_periods=2).mean()
        rs    = gain / loss.replace(0, 1e-10)
        df["RSI"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

        df["atr_expanding"] = (
            (df["ATR"] > df["ATR_SMA"]) &
            (df["ATR"].shift(1) <= df["ATR_SMA"].shift(1))
        )
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"]     = 0
        df["position"]   = 0
        df["target_qty"] = 0.0

        min_bars = max(self.ema_window, self.atr_sma_window,
                       self.breakout_window, self.rsi_period,
                       self.trailing_sma_window) + 5
        if len(df) < min_bars:
            return df

        sig_col = df.columns.get_loc("signal")
        pos_col = df.columns.get_loc("position")
        qty_col = df.columns.get_loc("target_qty")

        for i in range(min_bars, len(df)):
            row        = df.iloc[i]
            price      = float(row["Close"])
            atr        = float(row["ATR"])
            trail_sma  = float(row["trail_SMA"])
            rsi        = float(row["RSI"])
            above_ema  = price > float(row["EMA"])
            swing_high = float(row["swing_high"])
            swing_low  = float(row["swing_low"])
            atr_cross  = bool(row["atr_expanding"])

            if self._position != 0:
                self._bars_in_trade += 1
            self._bars_since_exit += 1

            desired = self._position

            if self._position != 0:
                if self._position == 1 and price <= self._hard_stop:
                    desired = 0
                elif self._position == -1 and price >= self._hard_stop:
                    desired = 0
                elif self._position == 1 and price <= trail_sma:
                    desired = 0
                elif self._position == -1 and price >= trail_sma:
                    desired = 0
                elif self._bars_in_trade >= self.max_hold_bars:
                    desired = 0

            if desired == 0 and self._position == 0:
                if self._bars_since_exit >= self.cooldown_bars and atr_cross:
                    if above_ema and price > swing_high and rsi < self.rsi_max_long:
                        desired = 1

            if desired != self._position:
                if desired == 0:
                    df.iloc[i, sig_col]   = -self._position
                    if self._entry_price > 0 and atr > 0:
                        stop_dist   = self.hard_stop_atr_mult * atr
                        dollar_risk = self.capital * self.risk_per_trade_pct
                        notional    = dollar_risk / (stop_dist / self._entry_price)
                        df.iloc[i, qty_col] = min(notional, self.capital * 0.20)
                    self._position        = 0
                    self._entry_price     = 0.0
                    self._hard_stop       = 0.0
                    self._bars_in_trade   = 0
                    self._bars_since_exit = 0
                else:
                    df.iloc[i, sig_col] = desired
                    self._position      = desired
                    self._entry_price   = price
                    self._bars_in_trade = 0
                    stop_dist           = self.hard_stop_atr_mult * atr
                    self._hard_stop     = price - stop_dist if desired == 1 else price + stop_dist

            df.iloc[i, pos_col] = self._position
            if self._position != 0:
                # Risk-based sizing: risk 2% of capital per trade
                # qty = (capital × risk_pct) / (atr × hard_stop_mult)
                # This gives us the number of coins where a 1-stop loss = 2% capital
                # target_qty = USD notional (Alpaca live trader converts to coins)
                # Risk 2% of capital per trade
                stop_dist = self.hard_stop_atr_mult * atr
                if stop_dist > 0:
                    dollar_risk = self.capital * self.risk_per_trade_pct
                    notional = dollar_risk / (stop_dist / price)
                    df.iloc[i, qty_col] = min(notional, self.capital * 0.20)

        return df
