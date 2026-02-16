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
        sma_window: int = 20,          # SMA lookback for mean
        z_window: int = 20,            # Z-score lookback
        atr_window: int = 14,          # ATR lookback
        entry_z: float = 1.5,          # enter when |Z| > this
        exit_z: float = 0.3,           # exit when |Z| < this (reverted)
        atr_filter_mult: float = 1.5,  # max ATR vs median (reject breakouts)
        position_size: float = 5000.0,
        max_position: float = 15000.0,
    ):
        if z_window < 3:
            raise ValueError("z_window must be at least 3.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be greater than exit_z.")
        self.sma_window = sma_window
        self.z_window = z_window
        self.atr_window = atr_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.atr_filter_mult = atr_filter_mult
        self.position_size = position_size
        self.max_position = max_position
        self._prev_signal = 0

    def add_indicators(self, df):
        # --- Simple Moving Average (the "mean" we revert to) ---
        df["SMA"] = df["Close"].rolling(self.sma_window, min_periods=2).mean()

        # --- Z-score: how many std devs price is from SMA ---
        rolling_mean = df["Close"].rolling(self.z_window, min_periods=2).mean()
        rolling_std = df["Close"].rolling(self.z_window, min_periods=2).std()
        df["Z_score"] = (df["Close"] - rolling_mean) / rolling_std.replace(0, 1e-10)

        # --- ATR: Average True Range (volatility measure) ---
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        
        #Calculation of ATR --> measure of max volatility essentially
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        #Creates column of rolling ATR
        df["ATR"] = tr.rolling(self.atr_window, min_periods=2).mean().fillna(0)

        # ATR relative to its rolling median — tells us if vol is normal or extreme
        atr_median = df["ATR"].rolling(self.atr_window * 3, min_periods=2).median()
        df["ATR_ratio"] = df["ATR"] / atr_median.replace(0, 1e-10)

        # --- SMA slope: is the trend flat enough for mean-reversion? ---
        # Returns boolean --> If SMA flat, mean reversion else not mean reversion
        sma_pct_change = df["SMA"].pct_change(5).abs().fillna(0)
        df["SMA_flat"] = sma_pct_change < 0.005  # less than 0.5% move over 5 bars

        return df

    def generate_signals(self, df):
        """
        Incremental signal generation for the backtester.
        Only computes signal for the last row.
        """
        df["signal"] = 0
        df["position"] = 0
        df["target_qty"] = 0.0

        # Need enough data for indicators to warm up
        min_bars = max(self.sma_window, self.z_window, self.atr_window) + 5
        if len(df) < min_bars:
            return df

        #Capturing data of last row --> point we are trading at
        last = df.iloc[-1]
        z = last["Z_score"]
        atr_ratio = last["ATR_ratio"]
        sma_flat = last["SMA_flat"]

        # Volatility filter: ATR must be above 0.5x median (enough movement)
        # but below our multiplier (not a breakout)
        vol_ok = (atr_ratio > 0.5) and (atr_ratio < self.atr_filter_mult)

        #Desired tells us what we want to do based off our current position

        desired = 0

        if vol_ok:
            # --- ENTRY LOGIC ---
            if z < -self.entry_z:
                # Price far below mean -> BUY (expect reversion up)
                desired = 1
            elif z > self.entry_z:
                # Price far above mean -> SELL (expect reversion down)
                desired = -1

            # --- EXIT LOGIC ---
            # If we're long and Z crossed back above exit threshold -> close
            elif self._prev_signal == 1 and z > -self.exit_z:
                desired = 0  # will flatten
            # If we're short and Z crossed back below exit threshold -> close
            elif self._prev_signal == -1 and z < self.exit_z:
                desired = 0  # will flatten
            else:
                # Hold current position
                desired = self._prev_signal
        else:
            # Volatility outside goldilocks zone — flatten or stay flat
            if self._prev_signal != 0:
                desired = 0  # exit if vol becomes extreme
            else:
                desired = 0

        # Emit signal only on changes
        if desired != self._prev_signal:
            if desired == 0:
                # Exit signal: emit opposite of current position
                df.iloc[-1, df.columns.get_loc("signal")] = -self._prev_signal
            else:
                df.iloc[-1, df.columns.get_loc("signal")] = desired
            self._prev_signal = desired

        # Position and sizing
        df.iloc[-1, df.columns.get_loc("position")] = self._prev_signal

        # Scale size by Z-score magnitude — bigger deviation = more conviction
        z_abs = min(abs(z), 3.0)
        size_mult = 0.5 + (z_abs / 3.0) * 1.5  # ranges from 0.5x to 2.0x
        qty = abs(self._prev_signal) * self.position_size * size_mult
        df.iloc[-1, df.columns.get_loc("target_qty")] = min(qty, self.max_position)

        return df

