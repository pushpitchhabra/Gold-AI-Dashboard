from backtesting import Strategy
import pandas as pd
import numpy as np
import ta

class MovingAverageCrossover(Strategy):
    """Simple Moving Average Crossover Strategy"""
    
    # Strategy parameters
    fast_ma = 10
    slow_ma = 30
    
    def init(self):
        # Calculate moving averages
        self.ma_fast = self.I(ta.trend.sma_indicator, pd.Series(self.data.Close), window=self.fast_ma)
        self.ma_slow = self.I(ta.trend.sma_indicator, pd.Series(self.data.Close), window=self.slow_ma)
    
    def next(self):
        # Buy signal: fast MA crosses above slow MA
        if self.ma_fast[-1] > self.ma_slow[-1] and self.ma_fast[-2] <= self.ma_slow[-2]:
            if not self.position:
                self.buy()
        
        # Sell signal: fast MA crosses below slow MA
        elif self.ma_fast[-1] < self.ma_slow[-1] and self.ma_fast[-2] >= self.ma_slow[-2]:
            if self.position:
                self.sell()

class RSIStrategy(Strategy):
    """RSI Overbought/Oversold Strategy"""
    
    # Strategy parameters
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    
    def init(self):
        self.rsi = self.I(ta.momentum.rsi, pd.Series(self.data.Close), window=self.rsi_period)
    
    def next(self):
        # Buy signal: RSI crosses above oversold level
        if self.rsi[-1] > self.rsi_oversold and self.rsi[-2] <= self.rsi_oversold:
            if not self.position:
                self.buy()
        
        # Sell signal: RSI crosses below overbought level
        elif self.rsi[-1] < self.rsi_overbought and self.rsi[-2] >= self.rsi_overbought:
            if self.position:
                self.sell()

class MACDStrategy(Strategy):
    """MACD Signal Line Crossover Strategy"""
    
    # Strategy parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    def init(self):
        close_series = pd.Series(self.data.Close)
        self.macd_line = self.I(ta.trend.macd, close_series, 
                               window_fast=self.fast_period, 
                               window_slow=self.slow_period)
        self.signal_line = self.I(ta.trend.macd_signal, close_series,
                                 window_fast=self.fast_period,
                                 window_slow=self.slow_period,
                                 window_sign=self.signal_period)
    
    def next(self):
        # Buy signal: MACD crosses above signal line
        if (self.macd_line[-1] > self.signal_line[-1] and 
            self.macd_line[-2] <= self.signal_line[-2]):
            if not self.position:
                self.buy()
        
        # Sell signal: MACD crosses below signal line
        elif (self.macd_line[-1] < self.signal_line[-1] and 
              self.macd_line[-2] >= self.signal_line[-2]):
            if self.position:
                self.sell()

class BollingerBandsStrategy(Strategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    # Strategy parameters
    bb_period = 20
    bb_std = 2
    
    def init(self):
        close_series = pd.Series(self.data.Close)
        self.bb_upper = self.I(ta.volatility.bollinger_hband, close_series, 
                              window=self.bb_period, window_dev=self.bb_std)
        self.bb_lower = self.I(ta.volatility.bollinger_lband, close_series,
                              window=self.bb_period, window_dev=self.bb_std)
        self.bb_middle = self.I(ta.volatility.bollinger_mavg, close_series,
                               window=self.bb_period)
    
    def next(self):
        # Buy signal: Price touches lower band
        if self.data.Close[-1] <= self.bb_lower[-1]:
            if not self.position:
                self.buy()
        
        # Sell signal: Price touches upper band
        elif self.data.Close[-1] >= self.bb_upper[-1]:
            if self.position:
                self.sell()

class GoldenCrossStrategy(Strategy):
    """Golden Cross Strategy (50 MA vs 200 MA)"""
    
    # Strategy parameters
    short_ma = 50
    long_ma = 200
    
    def init(self):
        close_series = pd.Series(self.data.Close)
        self.ma_short = self.I(ta.trend.sma_indicator, close_series, window=self.short_ma)
        self.ma_long = self.I(ta.trend.sma_indicator, close_series, window=self.long_ma)
    
    def next(self):
        # Buy signal: Golden Cross (50 MA crosses above 200 MA)
        if (self.ma_short[-1] > self.ma_long[-1] and 
            self.ma_short[-2] <= self.ma_long[-2]):
            if not self.position:
                self.buy()
        
        # Sell signal: Death Cross (50 MA crosses below 200 MA)
        elif (self.ma_short[-1] < self.ma_long[-1] and 
              self.ma_short[-2] >= self.ma_long[-2]):
            if self.position:
                self.sell()

class StochasticStrategy(Strategy):
    """Stochastic Oscillator Strategy"""
    
    # Strategy parameters
    k_period = 14
    d_period = 3
    overbought = 80
    oversold = 20
    
    def init(self):
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)
        close_series = pd.Series(self.data.Close)
        
        self.stoch_k = self.I(ta.momentum.stoch, high_series, low_series, close_series,
                             window=self.k_period, smooth_window=self.d_period)
        self.stoch_d = self.I(ta.momentum.stoch_signal, high_series, low_series, close_series,
                             window=self.k_period, smooth_window=self.d_period)
    
    def next(self):
        # Buy signal: %K crosses above %D in oversold region
        if (self.stoch_k[-1] > self.stoch_d[-1] and 
            self.stoch_k[-2] <= self.stoch_d[-2] and
            self.stoch_k[-1] < self.oversold):
            if not self.position:
                self.buy()
        
        # Sell signal: %K crosses below %D in overbought region
        elif (self.stoch_k[-1] < self.stoch_d[-1] and 
              self.stoch_k[-2] >= self.stoch_d[-2] and
              self.stoch_k[-1] > self.overbought):
            if self.position:
                self.sell()

class EMAStrategy(Strategy):
    """Exponential Moving Average Strategy"""
    
    # Strategy parameters
    fast_ema = 12
    slow_ema = 26
    
    def init(self):
        close_series = pd.Series(self.data.Close)
        self.ema_fast = self.I(ta.trend.ema_indicator, close_series, window=self.fast_ema)
        self.ema_slow = self.I(ta.trend.ema_indicator, close_series, window=self.slow_ema)
    
    def next(self):
        # Buy signal: Fast EMA crosses above Slow EMA
        if (self.ema_fast[-1] > self.ema_slow[-1] and 
            self.ema_fast[-2] <= self.ema_slow[-2]):
            if not self.position:
                self.buy()
        
        # Sell signal: Fast EMA crosses below Slow EMA
        elif (self.ema_fast[-1] < self.ema_slow[-1] and 
              self.ema_fast[-2] >= self.ema_slow[-2]):
            if self.position:
                self.sell()
