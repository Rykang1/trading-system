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


import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class MyStrategy(Strategy):
    """
    Custom directional options strategy using EMA crossovers.
    Same logic as CryptoTrendOptionsStrategy -- modify to your needs.
    """

    def __init__(
        self,
<<<<<<< HEAD
        short_window: int = 7,
        long_window: int = 21,
        position_size: float = 1.0,
        delta_min: float = 0.4,
        delta_max: float = 0.6,
=======
        rv_lookback: int = 20,
        iv_threshold: float = 0.05,
        position_size: float = 1000.0,
        delta_rebalance_threshold: float = 0.1,
        relative_vol_threshold: float = 0.03,
        risk_free_rate: float = 0.05
    ):
        self.rv_lookback = rv_lookback
        self.iv_threshold = iv_threshold
        self.position_size = position_size
        self.delta_rebalance_threshold = delta_rebalance_threshold
        self.relative_vol_threshold = relative_vol_threshold
        self.risk_free_rate = risk_free_rate
        
        # Track positions
        self.current_positions = {
            'BTC': {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'futures_hedge': 0.0},
            'ETH': {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'futures_hedge': 0.0}
        }
    
    def calculate_realized_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate realized volatility from returns.
        
        Uses standard deviation of log returns, annualized to match IV conventions.
        """
        if len(returns) < 2:
            return 0.0
        
        rv = returns.std()
        
        if annualize:
            # Annualize assuming 365 trading days for crypto
            rv = rv * np.sqrt(365)
        
        return rv
    
    def calculate_implied_volatility_proxy(self, df: pd.DataFrame, window: int = 10) -> float:
        """
        Proxy for implied volatility using ATM option prices.
        
        In practice, you would fetch this from options data APIs like Deribit.
        Here we use recent price volatility as a proxy + premium adjustment.
        """
        if len(df) < window:
            return 0.0
        
        recent_rv = self.calculate_realized_volatility(
            df['returns'].tail(window), 
            annualize=True
        )
        
        # Add a typical IV premium over RV (options usually trade at premium)
        # This is a simplified proxy - real implementation needs actual IV data
        iv_premium = 0.10  # 10% typical premium
        implied_vol = recent_rv * (1 + iv_premium)
        
        return implied_vol
    
    def black_scholes_delta(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float, 
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Black-Scholes delta for option pricing.
        
        Delta measures the sensitivity of option price to underlying price changes.
        """
        from scipy.stats import norm
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        return delta
    
    def black_scholes_gamma(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
    ) -> float:
        """
        Calculate Black-Scholes gamma.
        
        Gamma measures the rate of change of delta.
        """
        from scipy.stats import norm
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return gamma
    
    def black_scholes_vega(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
    ) -> float:
        """
        Calculate Black-Scholes vega.
        
        Vega measures the sensitivity to volatility changes.
        """
        from scipy.stats import norm
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return vega / 100  # Vega per 1% change in volatility
    
    def calculate_straddle_greeks(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        implied_vol: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks for an ATM straddle (long call + long put).
        """
        # Call delta
        call_delta = self.black_scholes_delta(
            spot_price, strike, time_to_expiry, 
            self.risk_free_rate, implied_vol, 'call'
        )
        
        # Put delta
        put_delta = self.black_scholes_delta(
            spot_price, strike, time_to_expiry,
            self.risk_free_rate, implied_vol, 'put'
        )
        
        # Straddle delta (should be close to zero for ATM)
        straddle_delta = call_delta + put_delta
        
        # Gamma is same for call and put
        gamma = self.black_scholes_gamma(
            spot_price, strike, time_to_expiry,
            self.risk_free_rate, implied_vol
        )
        straddle_gamma = 2 * gamma  # Long both call and put
        
        # Vega is same for call and put
        vega = self.black_scholes_vega(
            spot_price, strike, time_to_expiry,
            self.risk_free_rate, implied_vol
        )
        straddle_vega = 2 * vega
        
        return {
            'delta': straddle_delta,
            'gamma': straddle_gamma,
            'vega': straddle_vega
        }
    
    def generate_signal_iv_vs_rv(
        self,
        implied_vol: float,
        realized_vol: float
    ) -> int:
        """
        Signal 1: Compare IV to RV.
        
        Returns:
            1: Long volatility (IV < RV - threshold)
            0: No trade
            -1: Short volatility (IV > RV + threshold)
        """
        iv_rv_spread = (implied_vol - realized_vol) / realized_vol if realized_vol > 0 else 0
        
        if iv_rv_spread < -self.iv_threshold:
            # IV is cheap relative to RV -> Long volatility
            return 1
        elif iv_rv_spread > self.iv_threshold:
            # IV is expensive relative to RV -> Short volatility
            return -1
        else:
            return 0
    
    def generate_signal_relative_value(
        self,
        btc_iv: float,
        btc_rv: float,
        eth_iv: float,
        eth_rv: float
    ) -> Tuple[int, int]:
        """
        Signal 2: Relative value between BTC and ETH.
        
        Returns:
            (btc_signal, eth_signal): Each is 1 (long), 0 (neutral), or -1 (short)
        """
        # Calculate IV/RV ratios
        btc_ratio = btc_iv / btc_rv if btc_rv > 0 else 1.0
        eth_ratio = eth_iv / eth_rv if eth_rv > 0 else 1.0
        
        # Relative cheapness
        ratio_spread = (btc_ratio - eth_ratio) / eth_ratio if eth_ratio > 0 else 0
        
        btc_signal = 0
        eth_signal = 0
        
        if ratio_spread < -self.relative_vol_threshold:
            # BTC vol is relatively cheap -> Long BTC vol, Short ETH vol
            btc_signal = 1
            eth_signal = -1
        elif ratio_spread > self.relative_vol_threshold:
            # ETH vol is relatively cheap -> Long ETH vol, Short BTC vol
            btc_signal = -1
            eth_signal = 1
        
        return btc_signal, eth_signal
    
    def calculate_futures_hedge(
        self,
        portfolio_delta: float,
        spot_price: float
    ) -> float:
        """
        Calculate the futures position needed to hedge delta.
        
        Returns the number of futures contracts (negative for short hedge).
        """
        # Each futures contract has delta of 1.0
        # To hedge, we need opposite sign
        futures_position = -portfolio_delta
        
        return futures_position
    
    def add_indicators(self, df: pd.DataFrame, asset: str = 'BTC') -> pd.DataFrame:
        """
        Add volatility indicators to the DataFrame.
        """
        # Calculate returns
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        # Calculate realized volatility (rolling)
        df[f'{asset}_RV'] = df['returns'].rolling(
            window=self.rv_lookback, 
            min_periods=self.rv_lookback//2
        ).apply(lambda x: self.calculate_realized_volatility(x, annualize=True))
        
        # Calculate implied volatility proxy
        df[f'{asset}_IV'] = df.apply(
            lambda row: self.calculate_implied_volatility_proxy(
                df.loc[:row.name], 
                window=10
            ), 
            axis=1
        )
        
        # Calculate IV-RV spread
        df[f'{asset}_IV_RV_spread'] = df[f'{asset}_IV'] - df[f'{asset}_RV']
        
        # Calculate IV/RV ratio
        df[f'{asset}_IV_RV_ratio'] = np.where(
            df[f'{asset}_RV'] > 0,
            df[f'{asset}_IV'] / df[f'{asset}_RV'],
            1.0
        )
        
        return df
    
    def generate_signals(
        self, 
        btc_df: pd.DataFrame, 
        eth_df: pd.DataFrame,
        trade_size: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate trading signals for both BTC and ETH.
        
        Returns updated DataFrames with signals and position information.
        """
        if trade_size is None:
            trade_size = self.position_size
        
        # Ensure both dataframes have the same index
        common_index = btc_df.index.intersection(eth_df.index)
        btc_df = btc_df.loc[common_index].copy()
        eth_df = eth_df.loc[common_index].copy()
        
        # Initialize signal columns
        for asset, df in [('BTC', btc_df), ('ETH', eth_df)]:
            df[f'{asset}_signal_iv_rv'] = 0
            df[f'{asset}_signal_relative'] = 0
            df[f'{asset}_combined_signal'] = 0
            df[f'{asset}_position'] = 0
            df[f'{asset}_target_qty'] = 0.0
            df[f'{asset}_delta'] = 0.0
            df[f'{asset}_gamma'] = 0.0
            df[f'{asset}_vega'] = 0.0
            df[f'{asset}_futures_hedge'] = 0.0
        
        # Generate signals row by row
        for idx in common_index:
            # Get current volatility data
            btc_iv = btc_df.loc[idx, 'BTC_IV']
            btc_rv = btc_df.loc[idx, 'BTC_RV']
            eth_iv = eth_df.loc[idx, 'ETH_IV']
            eth_rv = eth_df.loc[idx, 'ETH_RV']
            
            # Skip if data is insufficient
            if pd.isna([btc_iv, btc_rv, eth_iv, eth_rv]).any():
                continue
            
            # Signal 1: IV vs RV for each asset
            btc_signal_iv_rv = self.generate_signal_iv_vs_rv(btc_iv, btc_rv)
            eth_signal_iv_rv = self.generate_signal_iv_vs_rv(eth_iv, eth_rv)
            
            # Signal 2: Relative value
            btc_signal_rel, eth_signal_rel = self.generate_signal_relative_value(
                btc_iv, btc_rv, eth_iv, eth_rv
            )
            
            # Combine signals (average of the two signals)
            btc_combined = np.sign((btc_signal_iv_rv + btc_signal_rel) / 2)
            eth_combined = np.sign((eth_signal_iv_rv + eth_signal_rel) / 2)
            
            # Store signals
            btc_df.loc[idx, 'BTC_signal_iv_rv'] = btc_signal_iv_rv
            btc_df.loc[idx, 'BTC_signal_relative'] = btc_signal_rel
            btc_df.loc[idx, 'BTC_combined_signal'] = btc_combined
            
            eth_df.loc[idx, 'ETH_signal_iv_rv'] = eth_signal_iv_rv
            eth_df.loc[idx, 'ETH_signal_relative'] = eth_signal_rel
            eth_df.loc[idx, 'ETH_combined_signal'] = eth_combined
            
            # Calculate position and Greeks for BTC
            if btc_combined != 0:
                btc_spot = btc_df.loc[idx, 'Close']
                btc_strike = btc_spot  # ATM straddle
                time_to_expiry = 7 / 365  # 1 week in years
                
                greeks_btc = self.calculate_straddle_greeks(
                    btc_spot, btc_strike, time_to_expiry, btc_iv
                )
                
                btc_df.loc[idx, 'BTC_position'] = btc_combined
                btc_df.loc[idx, 'BTC_target_qty'] = trade_size
                btc_df.loc[idx, 'BTC_delta'] = greeks_btc['delta'] * btc_combined
                btc_df.loc[idx, 'BTC_gamma'] = greeks_btc['gamma']
                btc_df.loc[idx, 'BTC_vega'] = greeks_btc['vega']
                
                # Calculate futures hedge
                btc_df.loc[idx, 'BTC_futures_hedge'] = self.calculate_futures_hedge(
                    btc_df.loc[idx, 'BTC_delta'], btc_spot
                )
            
            # Calculate position and Greeks for ETH
            if eth_combined != 0:
                eth_spot = eth_df.loc[idx, 'Close']
                eth_strike = eth_spot  # ATM straddle
                time_to_expiry = 7 / 365  # 1 week in years
                
                greeks_eth = self.calculate_straddle_greeks(
                    eth_spot, eth_strike, time_to_expiry, eth_iv
                )
                
                eth_df.loc[idx, 'ETH_position'] = eth_combined
                eth_df.loc[idx, 'ETH_target_qty'] = trade_size
                eth_df.loc[idx, 'ETH_delta'] = greeks_eth['delta'] * eth_combined
                eth_df.loc[idx, 'ETH_gamma'] = greeks_eth['gamma']
                eth_df.loc[idx, 'ETH_vega'] = greeks_eth['vega']
                
                # Calculate futures hedge
                eth_df.loc[idx, 'ETH_futures_hedge'] = self.calculate_futures_hedge(
                    eth_df.loc[idx, 'ETH_delta'], eth_spot
                )
        
        return btc_df, eth_df
    
    def run(
        self, 
        btc_df: pd.DataFrame, 
        eth_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full strategy pipeline for both assets.
        """
        # Add indicators
        btc_df = self.add_indicators(btc_df.copy(), 'BTC')
        eth_df = self.add_indicators(eth_df.copy(), 'ETH')
        
        # Generate signals
        btc_df, eth_df = self.generate_signals(btc_df, eth_df)
        
        return btc_df, eth_df


# Example usage for integration with existing framework
class _OldDeltaHedgedVolStrategy(Strategy):  # renamed to avoid conflict
    """
    Volatility Trading + Trend Filter Strategy
    
    Core Idea from Original Paper:
    - Trade when implied volatility (IV) misprices realized volatility (RV)
    - But ONLY trade with the trend (SMA filter)
    
    Key Innovation:
    - Use 50-period SMA to identify trend direction
    - Only buy volatility when price is above SMA (uptrend)
    - Exit when trend breaks or volatility normalizes
    
    This avoids the original problem: trading volatility in downtrends = losses
    """
    
    def __init__(
        self,
        rv_lookback: int = 20,
        sma_period: int = 50,  # Trend filter
        iv_threshold: float = 0.08,  # 8% IV/RV spread to trigger
        position_size: float = 0.005,  # 0.005 BTC = ~$500
        max_hold_periods: int = 100  # Force exit after 100 bars
>>>>>>> 173d1d72039f63110dc933aa8fdccccf1f6b5f8b
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