"""
Mean-Reversion Strategy v2 — Tuned from real backtest results.

Problems identified from v1 backtests:
───────────────────────────────────────
1. OVERTRADING: 25-30 trades in ~1600 bars on 1-min data = whipsaw city.
   Fix: Longer cooldown (10 bars), stricter entry filters, wider Z threshold.

2. LOW WIN RATE (7-24% on NVDA/MSFT/TSLA): Entering during micro-trends.
   Fix: Require RSI + Z-score + VWAP alignment (triple confirmation).
   Add minimum profit target — don't exit until we've made at least 0.5×ATR.

3. MSFT LOSS (-$122): Got chopped in a downtrend on 2/10.
   Fix: Stronger ADX filter (20 instead of 25), plus check if price is
   on the right side of VWAP before entering.

4. TSLA BARELY PROFITABLE: Too volatile for tight mean-reversion.
   Fix: Per-trade max loss (hard stop at 3×ATR), daily loss limit to
   stop trading after accumulating losses.

5. AAPL HOURLY (-$217 on 5 trades): Not enough bars for indicators.
   Fix: Already handled by min_bars check, but increased warmup period.
"""

import numpy as np
import pandas as pd


class Strategy:
    """Base Strategy interface."""

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


# =============================================================================
#  EXISTING STRATEGIES (unchanged)
# =============================================================================

class MovingAverageStrategy(Strategy):
    def __init__(self, short_window=20, long_window=60, position_size=10.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df):
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df):
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
    def __init__(self, lookback=14, position_size=10.0, buy_threshold=0.01, sell_threshold=-0.01):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df):
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        df.loc[df["momentum"] > self.buy_threshold, "signal"] = 1
        df.loc[df["momentum"] < self.sell_threshold, "signal"] = -1
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendStrategy(Strategy):
    def __init__(self, short_window=7, long_window=21, position_size=100.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df):
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df


class DemoStrategy(Strategy):
    def __init__(self, position_size=1.0):
        self.position_size = position_size

    def add_indicators(self, df):
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1
        df.loc[df["change"] < 0, "signal"] = -1
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


# =============================================================================
#  MEAN REVERSION v2 — TUNED FROM REAL BACKTEST RESULTS
# =============================================================================

class MyStrategy(Strategy):
    """
    Mean-Reversion v2: Fewer trades, higher win rate, hard risk limits.

    Key differences from v1:
      - Triple confirmation: Z-score + RSI + VWAP alignment
      - Minimum profit target: don't exit until 0.5×ATR profit (or hard stop)
      - Longer cooldown: 10 bars between trades (was 3)
      - Wider entry: Z > 2.0 (was 1.5) — only take the best setups
      - Hard stop at 3×ATR per trade
      - Daily loss limit: stop trading after -$200 cumulative in a session
      - Stronger ADX filter: threshold 20 (was 25)
    """

    def __init__(
        self,
<<<<<<< HEAD
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
=======
        # --- Core ---
        sma_window: int = 20,
        z_window: int = 20,
        entry_z: float = 2.0,           # WIDER: only enter on strong deviations
        exit_z: float = 0.3,
        # --- RSI ---
        rsi_period: int = 14,
        rsi_oversold: float = 28.0,      # TIGHTER: more extreme RSI required
        rsi_overbought: float = 72.0,
        # --- ATR / Vol ---
        atr_window: int = 14,
        atr_filter_mult: float = 1.6,    # TIGHTER: reject more volatile regimes
        hard_stop_atr_mult: float = 3.0, # NEW: max loss per trade
        min_profit_atr_mult: float = 0.5,# NEW: don't exit until this much profit
        trailing_stop_atr_mult: float = 2.5,
        # --- Regime ---
        adx_window: int = 14,
        adx_threshold: float = 20.0,     # TIGHTER: avoid even mild trends
        # --- Position sizing ---
        position_size: float = 5000.0,
        max_position: float = 15000.0,
        # --- Risk management ---
        cooldown_bars: int = 10,          # LONGER: was 3, prevents overtrading
        daily_loss_limit: float = 200.0,  # NEW: stop trading after this much loss
>>>>>>> 18fdf0e3bcef735f4cda3faebb93a84c3a00b695
    ):
        if z_window < 3:
            raise ValueError("z_window must be at least 3.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be greater than exit_z.")

        self.sma_window = sma_window
        self.z_window = z_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_window = atr_window
        self.atr_filter_mult = atr_filter_mult
        self.hard_stop_atr_mult = hard_stop_atr_mult
        self.min_profit_atr_mult = min_profit_atr_mult
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold
        self.position_size = position_size
        self.max_position = max_position
<<<<<<< HEAD
=======
        self.cooldown_bars = cooldown_bars
        self.daily_loss_limit = daily_loss_limit

        # ── Internal state ──
>>>>>>> 18fdf0e3bcef735f4cda3faebb93a84c3a00b695
        self._prev_signal = 0
        self._entry_price = 0.0
        self._entry_atr = 0.0
        self._trailing_stop = 0.0
        self._hard_stop = 0.0
        self._bars_since_exit = 999
        self._session_pnl = 0.0
        self._current_date = None
        self._stopped_for_day = False

    # ─────────────────────────────────────────────────────────────
    #  INDICATORS
    # ─────────────────────────────────────────────────────────────
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ── 1. SMA ──
        df["SMA"] = close.rolling(self.sma_window, min_periods=2).mean()

        # ── 2. Z-score ──
        rolling_mean = close.rolling(self.z_window, min_periods=2).mean()
        rolling_std = close.rolling(self.z_window, min_periods=2).std()
        df["Z_score"] = (close - rolling_mean) / rolling_std.replace(0, 1e-10)

        # ── 3. RSI ──
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_period, min_periods=2).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_period, min_periods=2).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        # ── 4. ATR ──
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_window, min_periods=2).mean().fillna(0)

        atr_median = df["ATR"].rolling(self.atr_window * 3, min_periods=2).median()
        df["ATR_ratio"] = df["ATR"] / atr_median.replace(0, 1e-10)

        # ── 5. VWAP (intraday anchor — better than SMA for 1-min bars) ──
        if "Volume" in df.columns:
            cum_vol = df["Volume"].cumsum()
            cum_vp = (close * df["Volume"]).cumsum()
            df["VWAP"] = cum_vp / cum_vol.replace(0, 1e-10)
        else:
            df["VWAP"] = df["SMA"]  # fallback

        # ── 6. ADX (trend strength) ──
        df["ADX"] = self._compute_adx(high, low, close, self.adx_window)

        # ── 7. Adaptive Z threshold ──
        vol_scale = df["ATR_ratio"].clip(0.6, 1.8)
        df["adaptive_entry_z"] = self.entry_z * vol_scale

        # ── 8. Price position relative to VWAP ──
        df["above_vwap"] = close > df["VWAP"]

        return df

    @staticmethod
    def _compute_adx(high, low, close, window):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=window, adjust=False).mean() / atr.replace(0, 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=window, adjust=False).mean() / atr.replace(0, 1e-10))
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10) * 100
        adx = dx.ewm(span=window, adjust=False).mean()
        return adx.fillna(0)

    # ─────────────────────────────────────────────────────────────
    #  SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df["position"] = 0
        df["target_qty"] = 0.0

        min_bars = max(self.sma_window, self.z_window, self.atr_window,
                       self.rsi_period, self.adx_window) + 10
        if len(df) < min_bars:
            return df

        last = df.iloc[-1]
        price = last["Close"]
        z = last["Z_score"]
        rsi = last["RSI"]
        atr = last["ATR"]
        atr_ratio = last["ATR_ratio"]
        adx = last["ADX"]
        adaptive_z = last["adaptive_entry_z"]
        above_vwap = last["above_vwap"]

        # ── Daily loss limit reset ──
        current_date = None
        if "Datetime" in df.columns:
            ts = last["Datetime"]
            if hasattr(ts, "date"):
                current_date = ts.date()
        if current_date and current_date != self._current_date:
            self._current_date = current_date
            self._session_pnl = 0.0
            self._stopped_for_day = False

        # ── Check if stopped for the day ──
        if self._stopped_for_day:
            # Still need to manage existing position
            if self._prev_signal != 0:
                self._force_exit(df, price)
            return df

        # ── Filter checks ──
        vol_ok = (atr_ratio > 0.5) and (atr_ratio < self.atr_filter_mult)
        regime_ok = adx < self.adx_threshold
        cooldown_ok = self._bars_since_exit >= self.cooldown_bars

        self._bars_since_exit += 1

        desired = self._prev_signal  # default: hold

        # ── HARD STOP CHECK (non-negotiable max loss per trade) ──
        if self._prev_signal != 0 and self._entry_price > 0:
            if self._prev_signal == 1 and price <= self._hard_stop:
                desired = 0  # hard stopped
            elif self._prev_signal == -1 and price >= self._hard_stop:
                desired = 0  # hard stopped

        # ── TRAILING STOP CHECK ──
        if desired == self._prev_signal and self._prev_signal != 0 and atr > 0:
            if self._prev_signal == 1:
                new_stop = price - self.trailing_stop_atr_mult * atr
                self._trailing_stop = max(self._trailing_stop, new_stop)
                if price < self._trailing_stop:
                    desired = 0
            elif self._prev_signal == -1:
                new_stop = price + self.trailing_stop_atr_mult * atr
                self._trailing_stop = min(self._trailing_stop, new_stop)
                if price > self._trailing_stop:
                    desired = 0

        # ── EXIT on Z-score reversion (only if min profit met) ──
        if desired == self._prev_signal and self._prev_signal != 0:
            unrealized = 0.0
            if self._prev_signal == 1:
                unrealized = price - self._entry_price
            else:
                unrealized = self._entry_price - price

            min_profit = self.min_profit_atr_mult * self._entry_atr
            has_min_profit = unrealized >= min_profit

            if self._prev_signal == 1 and z > -self.exit_z and has_min_profit:
                desired = 0
            elif self._prev_signal == -1 and z < self.exit_z and has_min_profit:
                desired = 0

            # Also exit if filters break down, regardless of profit
            if not vol_ok or not regime_ok:
                desired = 0

        # ── ENTRY LOGIC ──
        if desired == self._prev_signal and self._prev_signal == 0:
            if vol_ok and regime_ok and cooldown_ok:
                # TRIPLE CONFIRMATION: Z-score + RSI + VWAP alignment
                # Buy: price below VWAP (cheap), Z very negative, RSI oversold
                if (z < -adaptive_z
                        and rsi < self.rsi_oversold
                        and not above_vwap):
                    desired = 1

                # Sell: price above VWAP (expensive), Z very positive, RSI overbought
                elif (z > adaptive_z
                      and rsi > self.rsi_overbought
                      and above_vwap):
                    desired = -1

        # ── EMIT SIGNAL ──
        if desired != self._prev_signal:
            if desired == 0:
                # Exiting — calculate realized P&L for daily limit
                if self._prev_signal == 1:
                    trade_pnl = price - self._entry_price
                else:
                    trade_pnl = self._entry_price - price
                self._session_pnl += trade_pnl * 10  # rough: 10 shares

                df.iloc[-1, df.columns.get_loc("signal")] = -self._prev_signal
                self._bars_since_exit = 0
                self._entry_price = 0.0
                self._entry_atr = 0.0

                # Check daily loss limit
                if self._session_pnl <= -self.daily_loss_limit:
                    self._stopped_for_day = True

            else:
                # Entering
                df.iloc[-1, df.columns.get_loc("signal")] = desired
                self._entry_price = price
                self._entry_atr = atr

                # Set hard stop
                if desired == 1:
                    self._hard_stop = price - self.hard_stop_atr_mult * atr
                    self._trailing_stop = price - self.trailing_stop_atr_mult * atr
                else:
                    self._hard_stop = price + self.hard_stop_atr_mult * atr
                    self._trailing_stop = price + self.trailing_stop_atr_mult * atr

            self._prev_signal = desired

        # ── POSITION & SIZING ──
        df.iloc[-1, df.columns.get_loc("position")] = self._prev_signal

        z_abs = min(abs(z), 3.0)
        z_mult = 0.5 + (z_abs / 3.0) * 1.5
        qty = abs(self._prev_signal) * self.position_size * z_mult
        df.iloc[-1, df.columns.get_loc("target_qty")] = min(qty, self.max_position)

        return df

    def _force_exit(self, df, price):
        """Force close position (used when daily limit hit)."""
        if self._prev_signal == 1:
            trade_pnl = price - self._entry_price
        else:
            trade_pnl = self._entry_price - price
        self._session_pnl += trade_pnl * 10

        df.iloc[-1, df.columns.get_loc("signal")] = -self._prev_signal
        self._prev_signal = 0
        self._entry_price = 0.0
        self._bars_since_exit = 0