#!/usr/bin/env python3
"""
Alternative Gold Trading Strategies Framework
===========================================

Testing multiple strategy types beyond EMA crossovers:
1. Mean Reversion (Bollinger Bands + RSI)
2. Momentum Breakout (ATR + Volume)
3. Support/Resistance Trading
4. Volatility Breakout
5. Multi-timeframe Analysis
6. Price Action Patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AlternativeStrategyTester:
    def __init__(self, data, account_size=1000, leverage=500, max_risk_per_trade=20):
        self.data = data.copy()
        self.account_size = account_size
        self.leverage = leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.results = []
        
    def resample_data(self, timeframe_hours):
        """Resample data to different timeframes"""
        if timeframe_hours == 0.5:  # 30min original
            return self.data.copy()
        
        periods = int(timeframe_hours * 2)  # 30min periods
        resampled = self.data.resample(f'{periods * 30}min').agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def calculate_indicators(self, data):
        """Calculate all technical indicators"""
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = data['Close'].rolling(20).mean()
        indicators['sma_50'] = data['Close'].rolling(50).mean()
        indicators['ema_12'] = data['Close'].ewm(span=12).mean()
        indicators['ema_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = data['Close'].rolling(bb_period).mean()
        bb_std_dev = data['Close'].rolling(bb_period).std()
        indicators['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
        indicators['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
        indicators['bb_middle'] = bb_sma
        
        # ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        indicators['atr'] = true_range.rolling(14).mean()
        
        # MACD
        macd_line = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = macd_line.ewm(span=9).mean()
        indicators['macd_histogram'] = macd_line - indicators['macd_signal']
        indicators['macd_line'] = macd_line
        
        # Stochastic
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        indicators['stoch_k'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        indicators['stoch_d'] = indicators['stoch_k'].rolling(3).mean()
        
        return indicators
    
    def strategy_mean_reversion(self, data, timeframe_name):
        """Mean Reversion Strategy using Bollinger Bands + RSI"""
        indicators = self.calculate_indicators(data)
        
        position = 0
        cash = self.account_size
        position_size = 0
        entry_price = 0
        trades = []
        
        for i in range(50, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            
            # Entry conditions
            if position == 0:
                # Buy when price touches lower BB and RSI oversold
                if (current_price <= indicators['bb_lower'].iloc[i] and 
                    indicators['rsi'].iloc[i] < 30):
                    
                    stop_loss = current_price - (2 * indicators['atr'].iloc[i])
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        position = 1
                        entry_price = current_price
                        trades.append({
                            'date': current_date, 'type': 'BUY', 'price': current_price,
                            'size': position_size, 'stop': stop_loss
                        })
            
            # Exit conditions
            elif position == 1:
                # Exit when price reaches middle BB or RSI overbought
                if (current_price >= indicators['bb_middle'].iloc[i] or 
                    indicators['rsi'].iloc[i] > 70 or
                    current_price <= trades[-1]['stop']):
                    
                    pnl = position_size * (current_price - entry_price)
                    cash += pnl
                    trades.append({
                        'date': current_date, 'type': 'SELL', 'price': current_price,
                        'size': position_size, 'pnl': pnl
                    })
                    position = 0
                    position_size = 0
        
        return self.calculate_performance(trades, timeframe_name, "Mean_Reversion")
    
    def strategy_momentum_breakout(self, data, timeframe_name):
        """Momentum Breakout Strategy using ATR and price action"""
        indicators = self.calculate_indicators(data)
        
        position = 0
        cash = self.account_size
        position_size = 0
        entry_price = 0
        trades = []
        
        for i in range(50, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            
            # Calculate 20-period high/low
            period_high = data['High'].iloc[i-20:i].max()
            period_low = data['Low'].iloc[i-20:i].min()
            
            if position == 0:
                # Buy on breakout above 20-period high with momentum
                if (current_price > period_high and 
                    indicators['rsi'].iloc[i] > 50 and
                    indicators['macd_line'].iloc[i] > indicators['macd_signal'].iloc[i]):
                    
                    stop_loss = current_price - (1.5 * indicators['atr'].iloc[i])
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        position = 1
                        entry_price = current_price
                        trades.append({
                            'date': current_date, 'type': 'BUY', 'price': current_price,
                            'size': position_size, 'stop': stop_loss
                        })
            
            elif position == 1:
                # Exit on breakdown or stop loss
                if (current_price < period_low or 
                    current_price <= trades[-1]['stop'] or
                    indicators['rsi'].iloc[i] < 30):
                    
                    pnl = position_size * (current_price - entry_price)
                    cash += pnl
                    trades.append({
                        'date': current_date, 'type': 'SELL', 'price': current_price,
                        'size': position_size, 'pnl': pnl
                    })
                    position = 0
                    position_size = 0
        
        return self.calculate_performance(trades, timeframe_name, "Momentum_Breakout")
    
    def strategy_volatility_breakout(self, data, timeframe_name):
        """Volatility Breakout Strategy"""
        indicators = self.calculate_indicators(data)
        
        position = 0
        cash = self.account_size
        position_size = 0
        entry_price = 0
        trades = []
        
        for i in range(50, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            
            # Volatility squeeze detection
            bb_width = (indicators['bb_upper'].iloc[i] - indicators['bb_lower'].iloc[i]) / indicators['bb_middle'].iloc[i]
            avg_bb_width = pd.Series([bb_width]).rolling(20).mean().iloc[-1] if i >= 70 else bb_width
            
            if position == 0:
                # Enter on volatility expansion
                if (bb_width > avg_bb_width * 1.2 and
                    abs(indicators['rsi'].iloc[i] - 50) > 20):
                    
                    direction = 1 if indicators['rsi'].iloc[i] > 50 else -1
                    stop_loss = current_price - (direction * 2 * indicators['atr'].iloc[i])
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        position = direction
                        entry_price = current_price
                        trades.append({
                            'date': current_date, 'type': 'BUY' if direction == 1 else 'SELL',
                            'price': current_price, 'size': position_size, 'stop': stop_loss
                        })
            
            elif position != 0:
                # Exit on volatility contraction or stop
                if (bb_width < avg_bb_width * 0.8 or
                    (position == 1 and current_price <= trades[-1]['stop']) or
                    (position == -1 and current_price >= trades[-1]['stop'])):
                    
                    pnl = position_size * (current_price - entry_price) * position
                    cash += pnl
                    trades.append({
                        'date': current_date, 'type': 'EXIT', 'price': current_price,
                        'size': position_size, 'pnl': pnl
                    })
                    position = 0
                    position_size = 0
        
        return self.calculate_performance(trades, timeframe_name, "Volatility_Breakout")
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            return 0
        
        base_position_size = self.max_risk_per_trade / risk_per_unit
        max_position_value = self.account_size * min(self.leverage, 20)  # Cap leverage at 20x
        max_units = max_position_value / entry_price
        
        return min(base_position_size, max_units)
    
    def calculate_performance(self, trades, timeframe, strategy_name):
        """Calculate strategy performance metrics"""
        if not trades or len(trades) < 2:
            return None
        
        total_pnl = sum([t.get('pnl', 0) for t in trades])
        total_return = (total_pnl / self.account_size) * 100
        
        profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_completed_trades = len([t for t in trades if 'pnl' in t])
        win_rate = (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0
        
        profits = sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0])
        losses = abs(sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]))
        profit_factor = profits / losses if losses > 0 else float('inf') if profits > 0 else 0
        
        return {
            'strategy': strategy_name,
            'timeframe': timeframe,
            'total_return': round(total_return, 2),
            'final_balance': round(self.account_size + total_pnl, 2),
            'total_trades': total_completed_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
            'trades': trades
        }
    
    def run_all_strategies(self):
        """Run all alternative strategies across multiple timeframes"""
        
        print("=" * 100)
        print("ALTERNATIVE GOLD TRADING STRATEGIES OPTIMIZATION")
        print("=" * 100)
        print(f"Account Size: ${self.account_size}")
        print(f"Max Risk per Trade: ${self.max_risk_per_trade}")
        print(f"Leverage Cap: 20x (for safety)")
        
        timeframes = {
            '30min': 0.5,
            '1H': 1,
            '2H': 2,
            '4H': 4,
            '1D': 24
        }
        
        strategies = [
            self.strategy_mean_reversion,
            self.strategy_momentum_breakout,
            self.strategy_volatility_breakout
        ]
        
        for tf_name, tf_hours in timeframes.items():
            print(f"\n📊 Testing {tf_name} timeframe...")
            
            try:
                data = self.resample_data(tf_hours)
                if len(data) < 100:
                    print(f"⚠️  Insufficient data for {tf_name}")
                    continue
                
                for strategy_func in strategies:
                    try:
                        result = strategy_func(data, tf_name)
                        if result and result['total_trades'] > 0:
                            self.results.append(result)
                            print(f"   ✓ {result['strategy']}: {result['total_return']:.2f}% return, {result['total_trades']} trades")
                        else:
                            print(f"   ✗ {strategy_func.__name__}: No valid trades")
                    except Exception as e:
                        print(f"   ✗ {strategy_func.__name__}: Error - {e}")
                        
            except Exception as e:
                print(f"⚠️  Error with {tf_name}: {e}")
        
        return self.results
    
    def print_results(self):
        """Print comprehensive results"""
        if not self.results:
            print("\n❌ No profitable strategies found.")
            return
        
        # Sort by total return
        sorted_results = sorted(self.results, key=lambda x: x['total_return'], reverse=True)
        
        print(f"\n" + "=" * 120)
        print("ALTERNATIVE STRATEGY RESULTS")
        print("=" * 120)
        
        print(f"{'Rank':<4} {'Strategy':<18} {'Timeframe':<10} {'Return%':<10} {'Balance$':<12} {'Trades':<8} {'WinRate%':<10} {'ProfitFactor':<12}")
        print("-" * 120)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:<4} {result['strategy']:<18} {result['timeframe']:<10} {result['total_return']:<10.2f} {result['final_balance']:<12.2f} {result['total_trades']:<8} {result['win_rate']:<10.2f} {result['profit_factor']:<12.3f}")
        
        # Best strategy details
        if sorted_results:
            best = sorted_results[0]
            print(f"\n🏆 BEST STRATEGY:")
            print(f"   • Strategy: {best['strategy']}")
            print(f"   • Timeframe: {best['timeframe']}")
            print(f"   • Total Return: {best['total_return']:.2f}%")
            print(f"   • Final Balance: ${best['final_balance']:.2f}")
            print(f"   • Total Trades: {best['total_trades']}")
            print(f"   • Win Rate: {best['win_rate']:.2f}%")
            print(f"   • Profit Factor: {best['profit_factor']:.3f}")
            
            # TradingView setup
            print(f"\n🎯 TRADINGVIEW SETUP:")
            if best['strategy'] == 'Mean_Reversion':
                print(f"   • Add Bollinger Bands(20,2)")
                print(f"   • Add RSI(14)")
                print(f"   • BUY: Price touches lower BB + RSI < 30")
                print(f"   • SELL: Price reaches middle BB or RSI > 70")
            elif best['strategy'] == 'Momentum_Breakout':
                print(f"   • Add 20-period high/low levels")
                print(f"   • Add RSI(14) and MACD(12,26,9)")
                print(f"   • BUY: Breakout above 20-period high + RSI > 50 + MACD bullish")
                print(f"   • SELL: Breakdown below 20-period low or RSI < 30")
            elif best['strategy'] == 'Volatility_Breakout':
                print(f"   • Add Bollinger Bands(20,2)")
                print(f"   • Add RSI(14)")
                print(f"   • ENTER: Volatility expansion + RSI extreme")
                print(f"   • EXIT: Volatility contraction")
            
            print(f"   • Timeframe: {best['timeframe']}")
            print(f"   • Risk: $20 per trade maximum")
            print(f"   • Use 2 ATR stop losses")
        
        return sorted_results

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
    
    # Initialize tester
    tester = AlternativeStrategyTester(
        data=data,
        account_size=1000,
        leverage=500,
        max_risk_per_trade=20
    )
    
    # Run all strategies
    results = tester.run_all_strategies()
    
    # Print results
    best_strategies = tester.print_results()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('alternative_strategies_results.csv', index=False)
        print(f"\n✅ Results saved to 'alternative_strategies_results.csv'")
    
    return best_strategies

if __name__ == "__main__":
    results = main()
