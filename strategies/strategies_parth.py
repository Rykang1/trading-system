import numpy as np
import pandas as pd


class Strategy:
    def add_indicators(self, df):
        raise NotImplementedError

    def generate_signals(self, df):
        raise NotImplementedError

    def run(self, df):
        df = df.copy()
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


# =============================================================================
#  EXISTING STRATEGIES (unchanged)
# =============================================================================

class MovingAverageOptionsStrategy(Strategy):
    def __init__(self, short_window=20, long_window=60, position_size=1.0, delta_min=0.4, delta_max=0.6):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.delta_min = delta_min
        self.delta_max = delta_max

    def add_indicators(self, df):
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        bullish_cross = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        bearish_cross = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])
        near_atm_call = (df["option_type"] == "call") & (df["delta"] >= self.delta_min) & (df["delta"] <= self.delta_max)
        near_atm_put = (df["option_type"] == "put") & (df["delta"] >= -self.delta_max) & (df["delta"] <= -self.delta_min)
        df.loc[bullish_cross & near_atm_call, "signal"] = 1
        df.loc[bearish_cross & near_atm_put, "signal"] = -1
        df["position"] = 0
        df.loc[(df["MA_short"] > df["MA_long"]) & near_atm_call, "position"] = 1
        df.loc[(df["MA_short"] < df["MA_long"]) & near_atm_put, "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateOptionsStrategy(Strategy):
    def __init__(self, lookback=14, position_size=1.0, buy_threshold=0.01, sell_threshold=-0.01, delta_min=0.4, delta_max=0.6):
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

    def add_indicators(self, df):
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        near_atm_call = (df["option_type"] == "call") & (df["delta"] >= self.delta_min) & (df["delta"] <= self.delta_max)
        near_atm_put = (df["option_type"] == "put") & (df["delta"] >= -self.delta_max) & (df["delta"] <= -self.delta_min)
        df.loc[(df["momentum"] > self.buy_threshold) & near_atm_call, "signal"] = 1
        df.loc[(df["momentum"] < self.sell_threshold) & near_atm_put, "signal"] = -1
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendOptionsStrategy(Strategy):
    def __init__(self, short_window=7, long_window=21, position_size=1.0, delta_min=0.4, delta_max=0.6):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.delta_min = delta_min
        self.delta_max = delta_max

    def add_indicators(self, df):
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        bullish_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = bullish_regime.astype(int).diff().fillna(0)
        near_atm_call = (df["option_type"] == "call") & (df["delta"] >= self.delta_min) & (df["delta"] <= self.delta_max)
        near_atm_put = (df["option_type"] == "put") & (df["delta"] >= -self.delta_max) & (df["delta"] <= -self.delta_min)
        df.loc[(flips > 0) & near_atm_call, "signal"] = 1
        df.loc[(flips < 0) & near_atm_put, "signal"] = -1
        df["position"] = 0
        df.loc[bullish_regime & near_atm_call, "position"] = 1
        df.loc[~bullish_regime & near_atm_put, "position"] = -1
        df["target_qty"] = self.position_size
        return df


class DemoOptionsStrategy(Strategy):
    def __init__(self, position_size=1.0):
        self.position_size = position_size

    def add_indicators(self, df):
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        df.loc[(df["change"] > 0) & (df["option_type"] == "call"), "signal"] = 1
        df.loc[(df["change"] < 0) & (df["option_type"] == "put"), "signal"] = -1
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


# =============================================================================
#  MEAN REVERSION + VOLATILITY STRATEGY FOR STOCKS/ETFS
# =============================================================================

class DeltaHedgedVolStrategy(Strategy):
    """
    Mean-Reversion Strategy with SMA + Volatility Filter for Stocks/ETFs.

    Core idea:
      Price tends to revert to its moving average. When price deviates
      significantly (measured by Z-score), it's likely to snap back.
      But only trade mean-reversion when volatility is RIGHT — not too
      low (no movement) and not too high (trending/breakout).

    Signals:
      BUY  when Z-score < -entry_z  (price way below SMA = oversold)
      SELL when Z-score > +entry_z  (price way above SMA = overbought)
      EXIT when Z-score crosses back toward 0 (price reverted to mean)

    Filters:
      - ATR volatility filter: only trade when ATR is in a "goldilocks"
        zone (above median = enough movement to profit, but below
        extreme = not a breakout that will keep running)
      - Volume filter: confirm moves with above-average volume
      - Trend filter: avoid mean reversion during strong trends

    Designed for:
      - Stocks/ETFs (SPY, QQQ, AAPL, MSFT, etc.) on 1Hour to Daily timeframes
      - Works best with liquid large-cap equities
      - Mean reversion is cleaner in equity markets than crypto
    """
    
    def __init__(
        self,
        sma_window: int = 50,          # 50-period SMA (classic for daily/hourly)
        z_window: int = 20,            # Z-score over 20 periods
        atr_window: int = 14,          # Standard 14-period ATR
        entry_z: float = 1.5,          # More conservative entry (stocks less volatile)
        exit_z: float = 0.3,           # Exit when back toward mean
        atr_filter_min: float = 0.5,  # Min ATR ratio (avoid dead zones)
        atr_filter_max: float = 2.0,  # Max ATR ratio (avoid breakouts) - tighter for stocks
        volume_filter: bool = True,    # Require above-avg volume
        trend_filter: bool = True,     # Check for strong trends
        position_size: float = 10000.0, # $10K notional per trade
        max_position: float = 30000.0,  # Max $30K exposure
    ):
        if z_window < 3:
            raise ValueError("z_window must be at least 3.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be greater than exit_z.")
        if atr_filter_min >= atr_filter_max:
            raise ValueError("atr_filter_min must be less than atr_filter_max.")
        
        self.sma_window = sma_window
        self.z_window = z_window
        self.atr_window = atr_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.atr_filter_min = atr_filter_min
        self.atr_filter_max = atr_filter_max
        self.volume_filter = volume_filter
        self.trend_filter = trend_filter
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
        high = df["High"] if "High" in df.columns else df["Close"]
        low = df["Low"] if "Low" in df.columns else df["Close"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_window, min_periods=2).mean().fillna(0)

        # ATR relative to its rolling median — tells us if vol is normal or extreme
        atr_median = df["ATR"].rolling(self.atr_window * 3, min_periods=2).median()
        df["ATR_ratio"] = df["ATR"] / atr_median.replace(0, 1e-10)

        # --- Volume Filter (if available) ---
        if "Volume" in df.columns and self.volume_filter:
            vol_ma = df["Volume"].rolling(20, min_periods=2).mean()
            df["volume_ok"] = df["Volume"] > vol_ma
        else:
            df["volume_ok"] = True

        # --- Trend Filter: ADX-like logic or simple slope ---
        if self.trend_filter:
            # Calculate trend strength via price dispersion
            close_std = df["Close"].rolling(20, min_periods=2).std()
            close_mean = df["Close"].rolling(20, min_periods=2).mean()
            df["trend_strength"] = close_std / close_mean.replace(0, 1e-10)
            # Mean reversion works best when trend is weak (low dispersion)
            df["trend_ok"] = df["trend_strength"] < 0.02  # < 2% normalized std
        else:
            df["trend_ok"] = True

        # --- Distance from SMA (for additional filtering) ---
        df["distance_from_sma"] = ((df["Close"] - df["SMA"]) / df["SMA"]).fillna(0)

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

        last = df.iloc[-1]
        z = last["Z_score"]
        atr_ratio = last["ATR_ratio"]
        volume_ok = last["volume_ok"]
        trend_ok = last["trend_ok"]

        # Volatility filter: ATR must be in goldilocks zone
        # For stocks: tight range because breakouts are more persistent
        vol_ok = (atr_ratio > self.atr_filter_min) and (atr_ratio < self.atr_filter_max)

        # Combine all filters
        conditions_met = vol_ok and volume_ok and trend_ok

        desired = 0

        if conditions_met:
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
            # Conditions not met — flatten or stay flat
            if self._prev_signal != 0:
                desired = 0  # exit if conditions deteriorate
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


# Alias for auto-discovery
MyStrategy = DeltaHedgedVolStrategy