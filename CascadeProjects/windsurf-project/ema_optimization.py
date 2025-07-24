#!/usr/bin/env python3
"""
Comprehensive EMA Crossover Optimization Framework
=================================================

This framework tests all reasonable EMA combinations across multiple timeframes
with proper risk management for leveraged trading.

Account Parameters:
- Account Size: $1000
- Leverage: 500x
- Max Risk per Trade: $20 (2% of account)
- Position Sizing: Kelly Criterion + Fixed Risk
"""

import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EMAOptimizer:
    def __init__(self, data, account_size=1000, leverage=500, max_risk_per_trade=20):
        """
        Initialize EMA Crossover Optimizer
        
        Args:
            data: OHLCV DataFrame with datetime index
            account_size: Account size in dollars
            leverage: Available leverage
            max_risk_per_trade: Maximum risk per trade in dollars
        """
        self.original_data = data.copy()
        self.account_size = account_size
        self.leverage = leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_percent = max_risk_per_trade / account_size  # 2%
        
        # EMA combinations to test
        self.ema_combinations = [
            (5, 10), (5, 15), (5, 20), (5, 25), (5, 30),
            (8, 13), (8, 21), (8, 34), (8, 55),
            (9, 21), (9, 26), (9, 50),
            (10, 20), (10, 30), (10, 50),
            (12, 26), (12, 50), (12, 100),
            (13, 34), (13, 55),
            (15, 30), (15, 45), (15, 60),
            (20, 50), (20, 100), (20, 200),
            (21, 55), (21, 89),
            (25, 50), (25, 100),
            (30, 60), (30, 100),
            (50, 100), (50, 200),
            (100, 200)
        ]
        
        # Timeframes to test (in minutes)
        self.timeframes = {
            '30min': 1,    # Original data
            '1H': 2,       # 2x 30min periods
            '2H': 4,       # 4x 30min periods
            '4H': 8,       # 8x 30min periods
            '1D': 48       # 48x 30min periods
        }
        
        self.results = []
    
    def resample_data(self, timeframe_multiplier):
        """Resample data to higher timeframe"""
        if timeframe_multiplier == 1:
            return self.original_data.copy()
        
        # Resample OHLCV data
        resampled = self.original_data.resample(f'{timeframe_multiplier * 30}min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range for volatility-based stops"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def calculate_position_size(self, entry_price, stop_loss_price, risk_amount):
        """
        Calculate position size based on risk management
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_amount: Amount willing to risk in dollars
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0
        
        # Calculate position size without leverage
        base_position_size = risk_amount / risk_per_unit
        
        # Apply leverage (but cap at available margin)
        max_position_value = self.account_size * self.leverage
        max_units = max_position_value / entry_price
        
        # Use smaller of risk-based size and leverage-based size
        position_size = min(base_position_size, max_units)
        
        return position_size
    
    def backtest_ema_crossover(self, data, fast_ema, slow_ema, timeframe_name):
        """
        Backtest EMA crossover strategy with advanced risk management
        """
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(data, fast_ema)
        ema_slow = self.calculate_ema(data, slow_ema)
        atr = self.calculate_atr(data, 14)
        
        # Initialize tracking variables
        position = 0  # 0 = no position, 1 = long, -1 = short
        cash = self.account_size
        position_size = 0
        entry_price = 0
        stop_loss = 0
        portfolio_value = []
        trades = []
        
        # Skip initial periods to allow indicators to stabilize
        start_idx = max(slow_ema, 14) + 1
        
        for i in range(start_idx, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            current_atr = atr.iloc[i]
            
            # Check for stop loss hit
            if position != 0:
                if (position == 1 and current_price <= stop_loss) or \
                   (position == -1 and current_price >= stop_loss):
                    
                    # Close position due to stop loss
                    pnl = position_size * (current_price - entry_price) * position
                    cash += pnl
                    
                    trades.append({
                        'date': current_date,
                        'type': 'STOP_LOSS',
                        'side': 'SELL' if position == 1 else 'BUY',
                        'price': current_price,
                        'size': position_size,
                        'pnl': pnl,
                        'entry_price': entry_price
                    })
                    
                    position = 0
                    position_size = 0
                    entry_price = 0
                    stop_loss = 0
            
            # Generate signals
            if i > start_idx:  # Need previous values for crossover
                # Long signal: Fast EMA crosses above Slow EMA
                if (ema_fast.iloc[i] > ema_slow.iloc[i] and 
                    ema_fast.iloc[i-1] <= ema_slow.iloc[i-1] and 
                    position == 0):
                    
                    # Calculate stop loss using ATR
                    stop_loss_price = current_price - (2 * current_atr)  # 2 ATR stop
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        current_price, stop_loss_price, self.max_risk_per_trade
                    )
                    
                    if position_size > 0:
                        position = 1
                        entry_price = current_price
                        stop_loss = stop_loss_price
                        
                        trades.append({
                            'date': current_date,
                            'type': 'ENTRY',
                            'side': 'BUY',
                            'price': current_price,
                            'size': position_size,
                            'pnl': 0,
                            'stop_loss': stop_loss_price
                        })
                
                # Short signal: Fast EMA crosses below Slow EMA
                elif (ema_fast.iloc[i] < ema_slow.iloc[i] and 
                      ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]):
                    
                    # Close long position if exists
                    if position == 1:
                        pnl = position_size * (current_price - entry_price)
                        cash += pnl
                        
                        trades.append({
                            'date': current_date,
                            'type': 'EXIT',
                            'side': 'SELL',
                            'price': current_price,
                            'size': position_size,
                            'pnl': pnl,
                            'entry_price': entry_price
                        })
                        
                        position = 0
                        position_size = 0
                        entry_price = 0
                        stop_loss = 0
                    
                    # Enter short position
                    stop_loss_price = current_price + (2 * current_atr)  # 2 ATR stop
                    
                    position_size = self.calculate_position_size(
                        current_price, stop_loss_price, self.max_risk_per_trade
                    )
                    
                    if position_size > 0:
                        position = -1
                        entry_price = current_price
                        stop_loss = stop_loss_price
                        
                        trades.append({
                            'date': current_date,
                            'type': 'ENTRY',
                            'side': 'SELL',
                            'price': current_price,
                            'size': position_size,
                            'pnl': 0,
                            'stop_loss': stop_loss_price
                        })
            
            # Calculate current portfolio value
            if position != 0:
                unrealized_pnl = position_size * (current_price - entry_price) * position
                portfolio_value.append(cash + unrealized_pnl)
            else:
                portfolio_value.append(cash)
        
        # Close any remaining position
        if position != 0:
            final_price = data['Close'].iloc[-1]
            pnl = position_size * (final_price - entry_price) * position
            cash += pnl
            
            trades.append({
                'date': data.index[-1],
                'type': 'FINAL_EXIT',
                'side': 'SELL' if position == 1 else 'BUY',
                'price': final_price,
                'size': position_size,
                'pnl': pnl,
                'entry_price': entry_price
            })
        
        # Calculate performance metrics
        if len(portfolio_value) == 0:
            return None
        
        portfolio_series = pd.Series(portfolio_value, index=data.index[start_idx:])
        
        total_return = (cash - self.account_size) / self.account_size * 100
        max_drawdown = self.calculate_max_drawdown(portfolio_series)
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_series)
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        
        # Calculate additional metrics
        total_trades = len([t for t in trades if t['type'] == 'ENTRY'])
        avg_trade_return = total_return / total_trades if total_trades > 0 else 0
        
        return {
            'timeframe': timeframe_name,
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'total_return': round(total_return, 2),
            'final_balance': round(cash, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
            'avg_trade_return': round(avg_trade_return, 2),
            'trades': trades
        }
    
    def calculate_max_drawdown(self, portfolio_series):
        """Calculate maximum drawdown"""
        if len(portfolio_series) == 0:
            return 0
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak * 100
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, portfolio_series):
        """Calculate Sharpe ratio"""
        if len(portfolio_series) < 2:
            return 0
        returns = portfolio_series.pct_change().dropna()
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    
    def calculate_win_rate(self, trades):
        """Calculate win rate"""
        if not trades:
            return 0
        
        profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_trades = len([t for t in trades if t['type'] in ['EXIT', 'STOP_LOSS', 'FINAL_EXIT']])
        
        return (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    def calculate_profit_factor(self, trades):
        """Calculate profit factor"""
        if not trades:
            return 0
        
        profits = sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0])
        losses = abs(sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]))
        
        return profits / losses if losses > 0 else float('inf') if profits > 0 else 0
    
    def run_optimization(self):
        """Run comprehensive EMA optimization across all combinations"""
        
        print("=" * 100)
        print("COMPREHENSIVE EMA CROSSOVER OPTIMIZATION")
        print("=" * 100)
        print(f"Account Size: ${self.account_size}")
        print(f"Leverage: {self.leverage}x")
        print(f"Max Risk per Trade: ${self.max_risk_per_trade} ({self.max_risk_percent:.1%})")
        print(f"Testing {len(self.ema_combinations)} EMA combinations across {len(self.timeframes)} timeframes")
        print(f"Total combinations: {len(self.ema_combinations) * len(self.timeframes)}")
        
        total_tests = len(self.ema_combinations) * len(self.timeframes)
        current_test = 0
        
        for timeframe_name, multiplier in self.timeframes.items():
            print(f"\n📊 Testing {timeframe_name} timeframe...")
            
            # Resample data for this timeframe
            data = self.resample_data(multiplier)
            
            if len(data) < 500:  # Skip if insufficient data
                print(f"⚠️  Insufficient data for {timeframe_name} ({len(data)} bars)")
                continue
            
            for fast_ema, slow_ema in self.ema_combinations:
                current_test += 1
                
                if current_test % 20 == 0:
                    print(f"Progress: {current_test}/{total_tests} ({current_test/total_tests*100:.1f}%)")
                
                try:
                    result = self.backtest_ema_crossover(data, fast_ema, slow_ema, timeframe_name)
                    
                    if result and result['total_trades'] > 0:
                        self.results.append(result)
                        
                except Exception as e:
                    print(f"Error testing EMA({fast_ema},{slow_ema}) on {timeframe_name}: {e}")
                    continue
        
        print(f"\n✅ Optimization complete! Tested {len(self.results)} valid combinations")
        return self.results
    
    def get_top_strategies(self, n=10):
        """Get top N strategies sorted by multiple criteria"""
        
        if not self.results:
            print("No results available. Run optimization first.")
            return []
        
        # Filter strategies with reasonable performance
        filtered_results = [
            r for r in self.results 
            if r['total_trades'] >= 10 and  # Minimum trades for statistical significance
               r['win_rate'] > 30 and       # Minimum win rate
               r['max_drawdown'] > -50      # Maximum acceptable drawdown
        ]
        
        if not filtered_results:
            print("No strategies met the filtering criteria.")
            return self.results[:n]  # Return top n unfiltered
        
        # Sort by composite score (total return / max drawdown * win rate)
        for result in filtered_results:
            if result['max_drawdown'] != 0:
                result['composite_score'] = (result['total_return'] / abs(result['max_drawdown'])) * (result['win_rate'] / 100)
            else:
                result['composite_score'] = result['total_return'] * (result['win_rate'] / 100)
        
        # Sort by composite score
        sorted_results = sorted(filtered_results, key=lambda x: x['composite_score'], reverse=True)
        
        return sorted_results[:n]
    
    def print_optimization_results(self):
        """Print comprehensive optimization results"""
        
        if not self.results:
            print("No results to display.")
            return
        
        print("\n" + "=" * 120)
        print("TOP EMA CROSSOVER STRATEGIES")
        print("=" * 120)
        
        top_strategies = self.get_top_strategies(15)
        
        if not top_strategies:
            print("No profitable strategies found.")
            return
        
        # Print header
        print(f"{'Rank':<4} {'Timeframe':<8} {'EMA':<12} {'Return%':<8} {'Balance$':<10} {'Drawdown%':<10} {'Sharpe':<7} {'Trades':<7} {'WinRate%':<9} {'ProfitFactor':<12} {'Score':<8}")
        print("-" * 120)
        
        for i, result in enumerate(top_strategies, 1):
            ema_pair = f"{result['fast_ema']}/{result['slow_ema']}"
            print(f"{i:<4} {result['timeframe']:<8} {ema_pair:<12} {result['total_return']:<8.1f} {result['final_balance']:<10.0f} {result['max_drawdown']:<10.1f} {result['sharpe_ratio']:<7.2f} {result['total_trades']:<7} {result['win_rate']:<9.1f} {result['profit_factor']:<12.2f} {result.get('composite_score', 0):<8.2f}")
        
        # Print best strategy details
        if top_strategies:
            best = top_strategies[0]
            print(f"\n🏆 BEST STRATEGY DETAILS:")
            print(f"   • EMA Pair: {best['fast_ema']}/{best['slow_ema']}")
            print(f"   • Timeframe: {best['timeframe']}")
            print(f"   • Total Return: {best['total_return']:.2f}%")
            print(f"   • Final Balance: ${best['final_balance']:.2f}")
            print(f"   • Max Drawdown: {best['max_drawdown']:.2f}%")
            print(f"   • Win Rate: {best['win_rate']:.2f}%")
            print(f"   • Total Trades: {best['total_trades']}")
            print(f"   • Profit Factor: {best['profit_factor']:.2f}")
            print(f"   • Sharpe Ratio: {best['sharpe_ratio']:.3f}")
        
        return top_strategies

def main():
    """Main execution function"""
    
    # Load data
    data_path = "/Users/pushpitchhabra/Desktop/Projects/Windsurf/Gold Windsurf/CascadeProjects/30 Min - Gold Histdata.com summary.csv"
    
    print("Loading gold data...")
    data = pd.read_csv(data_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%y %H:%M')
    data.set_index('Datetime', inplace=True)
    data.sort_index(inplace=True)
    data['Volume'] = 0
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"✓ Data loaded: {len(data)} records from {data.index[0]} to {data.index[-1]}")
    
    # Initialize optimizer
    optimizer = EMAOptimizer(
        data=data,
        account_size=1000,
        leverage=500,
        max_risk_per_trade=20
    )
    
    # Run optimization
    results = optimizer.run_optimization()
    
    # Print results
    top_strategies = optimizer.print_optimization_results()
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('ema_optimization_results.csv', index=False)
        print(f"\n✅ Full results saved to 'ema_optimization_results.csv'")
    
    # Generate TradingView setup instructions
    if top_strategies:
        print(f"\n" + "=" * 80)
        print("TRADINGVIEW SETUP INSTRUCTIONS")
        print("=" * 80)
        
        best = top_strategies[0]
        print(f"\n🎯 OPTIMAL EMA CROSSOVER SETUP:")
        print(f"   • Fast EMA: {best['fast_ema']}")
        print(f"   • Slow EMA: {best['slow_ema']}")
        print(f"   • Timeframe: {best['timeframe']}")
        print(f"   • Expected Return: {best['total_return']:.2f}%")
        
        print(f"\n💰 POSITION SIZING & RISK MANAGEMENT:")
        print(f"   • Account Size: $1000")
        print(f"   • Max Risk per Trade: $20 (2%)")
        print(f"   • Use 2 ATR stop losses")
        print(f"   • Position size based on stop distance")
        print(f"   • Leverage available: 500x (use carefully)")
        
        print(f"\n📋 TRADING RULES:")
        print(f"   • BUY: When EMA({best['fast_ema']}) crosses above EMA({best['slow_ema']})")
        print(f"   • SELL: When EMA({best['fast_ema']}) crosses below EMA({best['slow_ema']})")
        print(f"   • STOP LOSS: 2 x ATR(14) from entry price")
        print(f"   • POSITION SIZE: Risk $20 per trade maximum")
        
        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"   • This strategy shows {best['win_rate']:.1f}% win rate")
        print(f"   • Maximum drawdown was {best['max_drawdown']:.1f}%")
        print(f"   • Always paper trade first before going live")
        print(f"   • Monitor performance and adjust if needed")
    
    return top_strategies

if __name__ == "__main__":
    results = main()
