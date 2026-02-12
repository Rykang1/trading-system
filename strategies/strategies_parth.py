"""
Strategy base classes and built-in strategies updated for directional options trading.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy call, -1 for buy put, 0 for hold
    - target_qty: number of contracts (1 contract = 100 shares)
    - position: current position state (1=long call, -1=long put, 0=flat)

Options-specific columns used:
    - delta: how much option moves per $1 move in underlying (calls: 0 to 1, puts: -1 to 0)
    - iv: implied volatility of the option
    - theta: daily time decay (always negative for buyers)
    - option_type: "call" or "put"
    - strike: strike price
    - expiry: expiration date

Fetching options data from Alpaca:
    import alpaca_trade_api as tradeapi

    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

    options = api.get_option_contracts(
        underlying_symbol="AAPL",
        expiration_date_gte="2026-03-01",
        expiration_date_lte="2026-03-21",
        option_type="call"  # or "put"
    )

Note: target_qty is in contracts, not shares. Each contract controls 100 shares.
Note: Always close positions before expiration to avoid assignment risk.
Note: Theta decay works against buyers -- avoid holding options too long.

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=1):
            self.lookback = lookback
            self.position_size = position_size  # number of contracts

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            # Buy call when bullish, buy put when bearish
            df.loc[(df['Close'] > df['sma']) & (df['option_type'] == 'call'), 'signal'] = 1
            df.loc[(df['Close'] < df['sma']) & (df['option_type'] == 'put'), 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for directional options trading.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        Input:  Datetime, Open, High, Low, Close, Volume,
                strike, expiry, option_type, delta, iv, theta
        Output: signal, target_qty, position (from generate_signals)
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
        # Add default options columns if not present (e.g. when running on stock data)
        if "option_type" not in df.columns:
            df["option_type"] = "call"
        if "delta" not in df.columns:
            df["delta"] = 0.5
        if "iv" not in df.columns:
            df["iv"] = 0.25
        if "theta" not in df.columns:
            df["theta"] = -0.05
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class MovingAverageOptionsStrategy(Strategy):
    """
    Moving average crossover strategy adapted for directional options.

    Buys calls on bullish MA crossover, buys puts on bearish MA crossover.
    Filters by delta range to target near-ATM options with good leverage.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 60,
        position_size: float = 1.0,   # number of contracts
        delta_min: float = 0.4,        # minimum absolute delta (filter for near-ATM)
        delta_max: float = 0.6,        # maximum absolute delta
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.delta_min = delta_min
        self.delta_max = delta_max

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Shorter and longer rolling means on the underlying close price
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        # Daily returns and volatility of the underlying
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        # Crossover conditions on the underlying
        bullish_cross = (
            (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) &
            (df["MA_short"] > df["MA_long"])
        )
        bearish_cross = (
            (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) &
            (df["MA_short"] < df["MA_long"])
        )

        # Near-ATM delta filter (calls have positive delta, puts have negative delta)
        near_atm_call = (
            (df["option_type"] == "call") &
            (df["delta"] >= self.delta_min) &
            (df["delta"] <= self.delta_max)
        )
        near_atm_put = (
            (df["option_type"] == "put") &
            (df["delta"] >= -self.delta_max) &
            (df["delta"] <= -self.delta_min)
        )

        # Buy call on bullish crossover, buy put on bearish crossover
        df.loc[bullish_cross & near_atm_call, "signal"] = 1
        df.loc[bearish_cross & near_atm_put, "signal"] = -1

        # Position: 1 = long call, -1 = long put, 0 = flat
        df["position"] = 0
        df.loc[(df["MA_short"] > df["MA_long"]) & near_atm_call, "position"] = 1
        df.loc[(df["MA_short"] < df["MA_long"]) & near_atm_put, "position"] = -1

        # target_qty is number of contracts (each contract = 100 shares)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateOptionsStrategy(Strategy):
    """
    Starter options strategy template. Modify indicator and signal
    logic to build your own ideas.

    Buys calls on positive momentum, buys puts on negative momentum.
    """

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 1.0,    # number of contracts
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
        delta_min: float = 0.4,
        delta_max: float = 0.6,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.delta_min = delta_min
        self.delta_max = delta_max

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        near_atm_call = (
            (df["option_type"] == "call") &
            (df["delta"] >= self.delta_min) &
            (df["delta"] <= self.delta_max)
        )
        near_atm_put = (
            (df["option_type"] == "put") &
            (df["delta"] >= -self.delta_max) &
            (df["delta"] <= -self.delta_min)
        )

        # Buy call on positive momentum, buy put on negative momentum
        buy_call = (df["momentum"] > self.buy_threshold) & near_atm_call
        buy_put = (df["momentum"] < self.sell_threshold) & near_atm_put

        df.loc[buy_call, "signal"] = 1
        df.loc[buy_put, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendOptionsStrategy(Strategy):
    """
    Crypto trend-following options strategy using fast/slow EMAs.

    Uses EMA crossovers on the underlying crypto price to determine direction,
    then buys calls (bullish) or puts (bearish) accordingly.
    """

    def __init__(
        self,
        short_window: int = 7,
        long_window: int = 21,
        position_size: float = 1.0,   # number of contracts
        delta_min: float = 0.4,
        delta_max: float = 0.6,
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.delta_min = delta_min
        self.delta_max = delta_max

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Exponential moving averages give more weight to recent prices
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        bullish_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = bullish_regime.astype(int).diff().fillna(0)

        near_atm_call = (
            (df["option_type"] == "call") &
            (df["delta"] >= self.delta_min) &
            (df["delta"] <= self.delta_max)
        )
        near_atm_put = (
            (df["option_type"] == "put") &
            (df["delta"] >= -self.delta_max) &
            (df["delta"] <= -self.delta_min)
        )

        # Buy call when flipping bullish, buy put when flipping bearish
        df.loc[(flips > 0) & near_atm_call, "signal"] = 1
        df.loc[(flips < 0) & near_atm_put, "signal"] = -1

        # Position reflects current regime
        df["position"] = 0
        df.loc[bullish_regime & near_atm_call, "position"] = 1
        df.loc[~bullish_regime & near_atm_put, "position"] = -1

        df["target_qty"] = self.position_size
        return df


class DemoOptionsStrategy(Strategy):
    """
    Simple demo options strategy.

    Buys a call when price goes up, buys a put when price goes down.
    Uses a tiny position size to avoid large exposure.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size  # number of contracts

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        # Buy call when price went up, buy put when price went down
        df.loc[(df["change"] > 0) & (df["option_type"] == "call"), "signal"] = 1
        df.loc[(df["change"] < 0) & (df["option_type"] == "put"), "signal"] = -1

        df["position"] = df["signal"]
        df["target_qty"] = self.position_size  # contracts
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Options Strategy
##
## class RSIOptionsStrategy(Strategy):
##     """Buy calls when RSI is oversold, buy puts when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=1.0,
##                  delta_min=0.4, delta_max=0.6):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##         self.delta_min = delta_min
##         self.delta_max = delta_max
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
##         near_atm_call = (df['option_type'] == 'call') & df['delta'].between(self.delta_min, self.delta_max)
##         near_atm_put  = (df['option_type'] == 'put')  & df['delta'].between(-self.delta_max, -self.delta_min)
##         df.loc[(df['RSI'] < self.oversold)  & near_atm_call, 'signal'] = 1   # Buy call when oversold
##         df.loc[(df['RSI'] > self.overbought) & near_atm_put,  'signal'] = -1  # Buy put when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df


class MyStrategy(Strategy):
    """
    Custom directional options strategy using EMA crossovers.
    Same logic as CryptoTrendOptionsStrategy -- modify to your needs.
    """

    def __init__(
        self,
        short_window: int = 7,
        long_window: int = 21,
        position_size: float = 1.0,
        delta_min: float = 0.4,
        delta_max: float = 0.6,
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.delta_min = delta_min
        self.delta_max = delta_max

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        bullish_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = bullish_regime.astype(int).diff().fillna(0)

        near_atm_call = (
            (df["option_type"] == "call") &
            (df["delta"] >= self.delta_min) &
            (df["delta"] <= self.delta_max)
        )
        near_atm_put = (
            (df["option_type"] == "put") &
            (df["delta"] >= -self.delta_max) &
            (df["delta"] <= -self.delta_min)
        )

        df.loc[(flips > 0) & near_atm_call, "signal"] = 1
        df.loc[(flips < 0) & near_atm_put, "signal"] = -1

        df["position"] = 0
        df.loc[bullish_regime & near_atm_call, "position"] = 1
        df.loc[~bullish_regime & near_atm_put, "position"] = -1

        df["target_qty"] = self.position_size
        return df