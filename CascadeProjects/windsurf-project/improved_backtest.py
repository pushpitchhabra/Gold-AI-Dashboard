#!/usr/bin/env python3
"""
Improved Backtesting Engine with Risk Management
===============================================

This version includes:
- Better strategy logic
- Risk management (stop-loss, take-profit)
- Position sizing
- Reduced trading frequency
- Lower commission rates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedBacktester:
    def __init__(self, data, initial_capital=10000, commission=0.0005, stop_loss=0.02, take_profit=0.04):
        """
        Initialize the improved backtester
        
        Args:
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital
            commission: Commission rate per trade (reduced to 0.05%)
            stop_loss: Stop loss percentage (2%)
            take_profit: Take profit percentage (4%)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.stop_loss = stop_loss
        self.take_profit = take_profit
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
    
    def backtest_strategy_with_risk_mgmt(self, strategy_name, signals):
        """
        Run backtest with risk management
        """
        
        # Initialize tracking variables
        position = 0  # 0 = no position, 1 = long position
        cash = self.initial_capital
        shares = 0
        entry_price = 0
        portfolio_value = []
        trades = []
        
        # Risk management parameters
        position_size = 0.95  # Use 95% of available capital per trade
        
        for i in range(len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_date = self.data.index[i]
            
            # Risk management: Check stop-loss and take-profit
            if position == 1:
                price_change = (current_price - entry_price) / entry_price
                
                # Stop-loss hit
                if price_change <= -self.stop_loss:
                    cash = shares * current_price * (1 - self.commission)
                    trades.append({
                        'date': current_date,
                        'type': 'STOP_LOSS',
                        'price': current_price,
                        'shares': shares,
                        'pnl': cash - (shares * entry_price)
                    })
                    position = 0
                    shares = 0
                    entry_price = 0
                
                # Take-profit hit
                elif price_change >= self.take_profit:
                    cash = shares * current_price * (1 - self.commission)
                    trades.append({
                        'date': current_date,
                        'type': 'TAKE_PROFIT',
                        'price': current_price,
                        'shares': shares,
                        'pnl': cash - (shares * entry_price)
                    })
                    position = 0
                    shares = 0
                    entry_price = 0
            
            # Buy signal
            if signals['buy'].iloc[i] == 1 and position == 0:
                # Use position sizing
                investment_amount = cash * position_size
                shares = investment_amount / current_price
                commission_cost = shares * current_price * self.commission
                shares = (investment_amount - commission_cost) / current_price
                cash -= (shares * current_price + commission_cost)
                position = 1
                entry_price = current_price
                
                trades.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'pnl': 0
                })
            
            # Sell signal (only if not already stopped out)
            elif signals['sell'].iloc[i] == 1 and position == 1:
                cash += shares * current_price * (1 - self.commission)
                trades.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'pnl': cash - (shares * entry_price)
                })
                position = 0
                shares = 0
                entry_price = 0
            
            # Calculate current portfolio value
            if position == 1:
                portfolio_value.append(cash + shares * current_price)
            else:
                portfolio_value.append(cash)
        
        # If still holding position at the end, sell it
        if position == 1:
            final_price = self.data['Close'].iloc[-1]
            cash += shares * final_price * (1 - self.commission)
            portfolio_value[-1] = cash
        
        # Calculate performance metrics
        portfolio_series = pd.Series(portfolio_value, index=self.data.index)
        
        total_return = (portfolio_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        max_drawdown = self.calculate_max_drawdown(portfolio_series)
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_series)
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        
        # Buy and hold return
        buy_hold_return = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100
        
        result = {
            'strategy': strategy_name,
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'total_trades': len([t for t in trades if t['type'] == 'BUY']),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
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
        """Calculate Sharpe ratio"""
        returns = portfolio_series.pct_change().dropna()
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252 * 48)  # Annualized for 30-min data
    
    def calculate_win_rate(self, trades):
        """Calculate win rate from trades"""
        if not trades:
            return 0
        
        profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_trades = len([t for t in trades if t['type'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']])
        
        return (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    def calculate_profit_factor(self, trades):
        """Calculate profit factor"""
        profits = sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0])
        losses = abs(sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]))
        
        return profits / losses if losses > 0 else 0

def run_improved_strategies():
    """Run improved trading strategies with risk management"""
    
    print("=" * 80)
    print("IMPROVED GOLD TRADING STRATEGY BACKTESTING")
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
    
    # Initialize improved backtester with better parameters
    backtester = ImprovedBacktester(
        data, 
        initial_capital=10000, 
        commission=0.0005,  # Reduced commission to 0.05%
        stop_loss=0.02,     # 2% stop loss
        take_profit=0.04    # 4% take profit
    )
    
    results = []
    
    # Strategy 1: Trend Following MA (with confirmation)
    print("\n1. Running Trend Following MA Strategy...")
    ma_fast = backtester.calculate_sma(20)
    ma_slow = backtester.calculate_sma(50)
    rsi = backtester.calculate_rsi(14)
    
    signals = pd.DataFrame(index=data.index)
    # Buy: Fast MA > Slow MA AND RSI > 50 (momentum confirmation)
    signals['buy'] = ((ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1)) & (rsi > 50)).astype(int)
    # Sell: Fast MA < Slow MA OR RSI < 40
    signals['sell'] = ((ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))).astype(int)
    
    result = backtester.backtest_strategy_with_risk_mgmt("Trend Following MA", signals)
    results.append(result)
    
    # Strategy 2: Mean Reversion (Bollinger Bands + RSI)
    print("2. Running Mean Reversion Strategy...")
    bb_upper, bb_lower, bb_middle = backtester.calculate_bollinger_bands(20, 2)
    rsi = backtester.calculate_rsi(14)
    
    signals = pd.DataFrame(index=data.index)
    # Buy: Price near lower band AND RSI oversold
    signals['buy'] = ((data['Close'] <= bb_lower * 1.01) & (rsi < 35)).astype(int)
    # Sell: Price near upper band OR RSI overbought
    signals['sell'] = ((data['Close'] >= bb_upper * 0.99) | (rsi > 65)).astype(int)
    
    result = backtester.backtest_strategy_with_risk_mgmt("Mean Reversion", signals)
    results.append(result)
    
    # Strategy 3: MACD with Volume Confirmation (simulated)
    print("3. Running MACD Momentum Strategy...")
    macd_line, signal_line = backtester.calculate_macd(12, 26, 9)
    price_sma = backtester.calculate_sma(10)
    
    signals = pd.DataFrame(index=data.index)
    # Buy: MACD crosses above signal AND price above SMA (trend confirmation)
    signals['buy'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)) & 
                     (data['Close'] > price_sma)).astype(int)
    # Sell: MACD crosses below signal
    signals['sell'] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    
    result = backtester.backtest_strategy_with_risk_mgmt("MACD Momentum", signals)
    results.append(result)
    
    # Strategy 4: Golden Cross with RSI Filter
    print("4. Running Filtered Golden Cross Strategy...")
    ma_50 = backtester.calculate_sma(50)
    ma_200 = backtester.calculate_sma(200)
    rsi = backtester.calculate_rsi(14)
    
    signals = pd.DataFrame(index=data.index)
    # Buy: Golden cross AND RSI not overbought
    signals['buy'] = ((ma_50 > ma_200) & (ma_50.shift(1) <= ma_200.shift(1)) & (rsi < 70)).astype(int)
    # Sell: Death cross
    signals['sell'] = ((ma_50 < ma_200) & (ma_50.shift(1) >= ma_200.shift(1))).astype(int)
    
    result = backtester.backtest_strategy_with_risk_mgmt("Filtered Golden Cross", signals)
    results.append(result)
    
    # Strategy 5: Buy and Hold (for comparison)
    print("5. Running Buy and Hold Strategy...")
    signals = pd.DataFrame(index=data.index)
    signals['buy'] = 0
    signals['sell'] = 0
    signals['buy'].iloc[0] = 1  # Buy at the beginning
    signals['sell'].iloc[-1] = 1  # Sell at the end
    
    # Temporarily disable risk management for buy and hold
    original_stop_loss = backtester.stop_loss
    original_take_profit = backtester.take_profit
    backtester.stop_loss = 1.0  # 100% stop loss (effectively disabled)
    backtester.take_profit = 10.0  # 1000% take profit (effectively disabled)
    
    result = backtester.backtest_strategy_with_risk_mgmt("Buy and Hold", signals)
    results.append(result)
    
    # Restore original risk management
    backtester.stop_loss = original_stop_loss
    backtester.take_profit = original_take_profit
    
    # Display results
    print("\n" + "=" * 120)
    print("IMPROVED BACKTEST RESULTS SUMMARY")
    print("=" * 120)
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'Strategy': r['strategy'],
            'Total Return (%)': r['total_return'],
            'Max Drawdown (%)': r['max_drawdown'],
            'Sharpe Ratio': r['sharpe_ratio'],
            'Total Trades': r['total_trades'],
            'Win Rate (%)': r['win_rate'],
            'Profit Factor': r['profit_factor'],
            'Final Value ($)': r['final_value'],
            'Buy & Hold (%)': r['buy_hold_return']
        }
        for r in results
    ])
    
    # Sort by total return
    results_df = results_df.sort_values('Total Return (%)', ascending=False)
    
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('improved_backtest_results.csv', index=False)
    print(f"\n✓ Results saved to 'improved_backtest_results.csv'")
    
    # Generate TradingView recommendations
    print("\n" + "=" * 80)
    print("TRADINGVIEW SETUP RECOMMENDATIONS")
    print("=" * 80)
    
    # Filter out strategies with positive returns
    profitable_strategies = results_df[results_df['Total Return (%)'] > 0]
    
    if len(profitable_strategies) > 0:
        print("\n🎯 PROFITABLE STRATEGIES FOR TRADINGVIEW:")
        for idx, row in profitable_strategies.iterrows():
            print(f"\n✅ {row['Strategy']}")
            print(f"   • Total Return: {row['Total Return (%)']}%")
            print(f"   • Max Drawdown: {row['Max Drawdown (%)']}%")
            print(f"   • Win Rate: {row['Win Rate (%)']}%")
            print(f"   • Profit Factor: {row['Profit Factor']}")
            print(f"   • Total Trades: {row['Total Trades']}")
    else:
        print("\n⚠️  No profitable strategies found. Consider:")
        print("• Adjusting strategy parameters")
        print("• Using different timeframes")
        print("• Adding more sophisticated filters")
        print("• Considering market regime changes")
    
    print(f"\n💡 RISK MANAGEMENT RECOMMENDATIONS:")
    print("• Stop Loss: 2% per trade")
    print("• Take Profit: 4% per trade")
    print("• Position Size: 95% of available capital")
    print("• Commission: 0.05% per trade")
    print("• Use 30-minute timeframe")
    print("• Always paper trade first")
    
    return results_df

if __name__ == "__main__":
    results = run_improved_strategies()
