"""
ATR Breakout Strategy — Crypto
================================
Enters on ATR vol expansion + price breakout above swing high/low.
Exits on trailing SMA touch or hard stop.
Sizes by dollar value, not coin count, to prevent over-leverage.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy


class MyStrategy(Strategy):

    def __init__(
        self,
        ema_window: int            = 20,
        atr_window: int            = 14,
        atr_sma_window: int        = 8,
        breakout_window: int       = 8,
        rsi_period: int            = 14,
        rsi_max_long: float        = 75.0,
        rsi_min_short: float       = 20.0,
        trailing_sma_window: int   = 10,
        hard_stop_atr_mult: float  = 2.0,
        max_hold_bars: int         = 30,
        cooldown_bars: int         = 3,
        risk_per_trade_pct: float  = 0.02,   # risk 2% of capital per trade
        capital: float             = 50000.0,
    ):
        self.ema_window          = ema_window
        self.atr_window          = atr_window
        self.atr_sma_window      = atr_sma_window
        self.breakout_window     = breakout_window
        self.rsi_period          = rsi_period
        self.rsi_max_long        = rsi_max_long
        self.rsi_min_short       = rsi_min_short
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
