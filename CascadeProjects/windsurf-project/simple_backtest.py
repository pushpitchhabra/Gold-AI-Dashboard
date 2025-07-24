#!/usr/bin/env python3
"""
Simplified Backtesting Engine for Gold Trading Strategies
========================================================

This module implements a custom backtesting framework that doesn't rely on
external backtesting libraries to avoid import issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleBacktester:
    def __init__(self, data, initial_capital=10000, commission=0.002):
        """
        Initialize the backtester
        
        Args:
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital
            commission: Commission rate per trade
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        
    def calculate_sma(self, period):
        """Calculate Simple Moving Average"""
        return self.data['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, period):
        """Calculate Exponential Moving Average"""
        return self.data['Close'].ewm(span=period).mean()
    
    def calculate_rsi(self, period=14):
        """Calculate RSI"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(period)
        std = self.data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    
    def backtest_strategy(self, strategy_name, signals):
        """
        Run backtest for a given strategy
        
        Args:
            strategy_name: Name of the strategy
            signals: DataFrame with 'buy' and 'sell' columns (1 for signal, 0 for no signal)
        """
        
        # Initialize tracking variables
        position = 0  # 0 = no position, 1 = long position
        cash = self.initial_capital
        shares = 0
        portfolio_value = []
        trades = []
        
        for i in range(len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_date = self.data.index[i]
            
            # Buy signal
            if signals['buy'].iloc[i] == 1 and position == 0:
                # Calculate shares to buy (use all available cash)
                shares = cash / current_price
                commission_cost = shares * current_price * self.commission
                shares = (cash - commission_cost) / current_price
                cash = 0
                position = 1
                
                trades.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'commission': commission_cost
                })
            
            # Sell signal
            elif signals['sell'].iloc[i] == 1 and position == 1:
                # Sell all shares
                cash = shares * current_price
                commission_cost = cash * self.commission
                cash -= commission_cost
                position = 0
                
                trades.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'commission': commission_cost
                })
                shares = 0
            
            # Calculate current portfolio value
            if position == 1:
                portfolio_value.append(shares * current_price)
            else:
                portfolio_value.append(cash)
        
        # If still holding position at the end, sell it
        if position == 1:
            final_price = self.data['Close'].iloc[-1]
            cash = shares * final_price * (1 - self.commission)
            portfolio_value[-1] = cash
        
        # Calculate performance metrics
        portfolio_series = pd.Series(portfolio_value, index=self.data.index)
        
        total_return = (portfolio_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        max_drawdown = self.calculate_max_drawdown(portfolio_series)
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_series)
        win_rate = self.calculate_win_rate(trades)
        
        # Buy and hold return
        buy_hold_return = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100
        
        result = {
            'strategy': strategy_name,
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'total_trades': len(trades),
            'win_rate': round(win_rate, 2),
            'final_value': round(portfolio_series.iloc[-1], 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'portfolio_series': portfolio_series,
            'trades': trades
        }
        
        return result
    
    def calculate_max_drawdown(self, portfolio_series):
        """Calculate maximum drawdown"""
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak * 100
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, portfolio_series):
        """Calculate Sharpe ratio (simplified)"""
        returns = portfolio_series.pct_change().dropna()
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252 * 48)  # Annualized for 30-min data
    
    def calculate_win_rate(self, trades):
        """Calculate win rate from trades"""
        if len(trades) < 2:
            return 0
        
        winning_trades = 0
        total_trades = 0
        
        for i in range(1, len(trades), 2):  # Every pair of buy-sell
            if i < len(trades):
                buy_price = trades[i-1]['price']
                sell_price = trades[i]['price']
                if sell_price > buy_price:
                    winning_trades += 1
                total_trades += 1
        
        return (winning_trades / total_trades * 100) if total_trades > 0 else 0

def run_all_strategies():
    """Run all trading strategies"""
    
    print("=" * 80)
    print("GOLD TRADING STRATEGY BACKTESTING")
    print("=" * 80)
    
    # Load data
    data_path = "/Users/pushpitchhabra/Desktop/Projects/Windsurf/Gold Windsurf/CascadeProjects/30 Min - Gold Histdata.com summary.csv"
    
    print("Loading data...")
    data = pd.read_csv(data_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%y %H:%M')
    data.set_index('Datetime', inplace=True)
    data.sort_index(inplace=True)
    data['Volume'] = 0
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"✓ Data loaded: {len(data)} records from {data.index[0]} to {data.index[-1]}")
    
    # Initialize backtester
    backtester = SimpleBacktester(data, initial_capital=10000, commission=0.002)
    
    results = []
    
    # Strategy 1: Moving Average Crossover (10/30)
    print("\n1. Running MA Crossover Strategy...")
    ma_fast = backtester.calculate_sma(10)
    ma_slow = backtester.calculate_sma(30)
    
    signals = pd.DataFrame(index=data.index)
    signals['buy'] = ((ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))).astype(int)
    signals['sell'] = ((ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))).astype(int)
    
    result = backtester.backtest_strategy("MA Crossover (10/30)", signals)
    results.append(result)
    
    # Strategy 2: RSI Strategy
    print("2. Running RSI Strategy...")
    rsi = backtester.calculate_rsi(14)
    
    signals = pd.DataFrame(index=data.index)
    signals['buy'] = ((rsi > 30) & (rsi.shift(1) <= 30)).astype(int)
    signals['sell'] = ((rsi < 70) & (rsi.shift(1) >= 70)).astype(int)
    
    result = backtester.backtest_strategy("RSI Strategy", signals)
    results.append(result)
    
    # Strategy 3: MACD Strategy
    print("3. Running MACD Strategy...")
    macd_line, signal_line = backtester.calculate_macd()
    
    signals = pd.DataFrame(index=data.index)
    signals['buy'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
    signals['sell'] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    
    result = backtester.backtest_strategy("MACD Strategy", signals)
    results.append(result)
    
    # Strategy 4: Bollinger Bands
    print("4. Running Bollinger Bands Strategy...")
    bb_upper, bb_lower, bb_middle = backtester.calculate_bollinger_bands()
    
    signals = pd.DataFrame(index=data.index)
    signals['buy'] = (data['Close'] <= bb_lower).astype(int)
    signals['sell'] = (data['Close'] >= bb_upper).astype(int)
    
    result = backtester.backtest_strategy("Bollinger Bands", signals)
    results.append(result)
    
    # Strategy 5: Golden Cross (50/200)
    print("5. Running Golden Cross Strategy...")
    ma_50 = backtester.calculate_sma(50)
    ma_200 = backtester.calculate_sma(200)
    
    signals = pd.DataFrame(index=data.index)
    signals['buy'] = ((ma_50 > ma_200) & (ma_50.shift(1) <= ma_200.shift(1))).astype(int)
    signals['sell'] = ((ma_50 < ma_200) & (ma_50.shift(1) >= ma_200.shift(1))).astype(int)
    
    result = backtester.backtest_strategy("Golden Cross (50/200)", signals)
    results.append(result)
    
    # Strategy 6: EMA Crossover (12/26)
    print("6. Running EMA Strategy...")
    ema_fast = backtester.calculate_ema(12)
    ema_slow = backtester.calculate_ema(26)
    
    signals = pd.DataFrame(index=data.index)
    signals['buy'] = ((ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))).astype(int)
    signals['sell'] = ((ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))).astype(int)
    
    result = backtester.backtest_strategy("EMA Crossover (12/26)", signals)
    results.append(result)
    
    # Display results
    print("\n" + "=" * 100)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 100)
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'Strategy': r['strategy'],
            'Total Return (%)': r['total_return'],
            'Max Drawdown (%)': r['max_drawdown'],
            'Sharpe Ratio': r['sharpe_ratio'],
            'Total Trades': r['total_trades'],
            'Win Rate (%)': r['win_rate'],
            'Final Value ($)': r['final_value'],
            'Buy & Hold (%)': r['buy_hold_return']
        }
        for r in results
    ])
    
    # Sort by total return
    results_df = results_df.sort_values('Total Return (%)', ascending=False)
    
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('backtest_results.csv', index=False)
    print(f"\n✓ Results saved to 'backtest_results.csv'")
    
    # Generate TradingView recommendations
    print("\n" + "=" * 80)
    print("TRADINGVIEW RECOMMENDATIONS")
    print("=" * 80)
    
    top_3 = results_df.head(3)
    
    print("\n🎯 TOP 3 STRATEGIES FOR TRADINGVIEW:")
    for idx, row in top_3.iterrows():
        print(f"\n{idx + 1}. {row['Strategy']}")
        print(f"   • Total Return: {row['Total Return (%)']}%")
        print(f"   • Max Drawdown: {row['Max Drawdown (%)']}%")
        print(f"   • Win Rate: {row['Win Rate (%)']}%")
        print(f"   • Total Trades: {row['Total Trades']}")
    
    print(f"\n💡 SETUP INSTRUCTIONS:")
    print("• Use 30-minute timeframe")
    print("• Set up alerts for crossover signals")
    print("• Consider risk management with stop-losses")
    print("• Paper trade before going live")
    
    return results_df

if __name__ == "__main__":
    results = run_all_strategies()
