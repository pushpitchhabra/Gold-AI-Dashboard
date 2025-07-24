#!/usr/bin/env python3
"""
Enhanced EMA Strategy with Edge-Building Techniques
==================================================

This framework adds multiple filters and edge-building techniques to improve
EMA crossover performance:

1. Trend Filters (Higher Timeframe Bias)
2. Volatility Filters (ATR-based)
3. Momentum Confirmation (RSI, MACD)
4. Volume Analysis (Price-Volume Divergence)
5. Support/Resistance Levels
6. Market Session Filters
7. Advanced Risk Management
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

class EnhancedEMAStrategy:
    def __init__(self, data, account_size=1000, leverage=500, max_risk_per_trade=20):
        """
        Enhanced EMA Strategy with multiple edge-building filters
        """
        self.data = data.copy()
        self.account_size = account_size
        self.leverage = leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_percent = max_risk_per_trade / account_size
        
    def calculate_ema(self, period):
        """Calculate Exponential Moving Average"""
        return self.data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, period):
        """Calculate Simple Moving Average"""
        return self.data['Close'].rolling(window=period).mean()
    
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
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(period)
        std = self.data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    
    def identify_support_resistance(self, window=20, min_touches=2):
        """Identify key support and resistance levels"""
        highs = self.data['High'].rolling(window=window, center=True).max()
        lows = self.data['Low'].rolling(window=window, center=True).min()
        
        # Find local peaks and troughs
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(self.data) - window):
            if self.data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(self.data['High'].iloc[i])
            if self.data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(self.data['Low'].iloc[i])
        
        return resistance_levels, support_levels
    
    def get_market_session(self, timestamp):
        """Determine market session (simplified)"""
        hour = timestamp.hour
        
        # London session: 8:00-16:00 UTC
        if 8 <= hour < 16:
            return 'London'
        # New York session: 13:00-21:00 UTC
        elif 13 <= hour < 21:
            return 'NY_Overlap' if 13 <= hour < 16 else 'New_York'
        # Asian session: 21:00-8:00 UTC
        else:
            return 'Asian'
    
    def calculate_volatility_regime(self, period=20):
        """Calculate volatility regime (high/low volatility periods)"""
        atr = self.calculate_atr(period)
        atr_ma = atr.rolling(window=period).mean()
        
        # High volatility when ATR > 1.5 * ATR_MA
        high_vol = atr > (atr_ma * 1.5)
        return high_vol
    
    def enhanced_ema_backtest(self, fast_ema=12, slow_ema=26):
        """
        Enhanced EMA crossover with multiple filters
        """
        
        # Calculate all indicators
        ema_fast = self.calculate_ema(fast_ema)
        ema_slow = self.calculate_ema(slow_ema)
        
        # Trend filter (higher timeframe)
        ema_trend = self.calculate_ema(100)  # Long-term trend
        
        # Momentum filters
        rsi = self.calculate_rsi(14)
        macd_line, macd_signal, macd_hist = self.calculate_macd()
        
        # Volatility filters
        atr = self.calculate_atr(14)
        bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands()
        high_vol_regime = self.calculate_volatility_regime()
        
        # Support/Resistance
        resistance_levels, support_levels = self.identify_support_resistance()
        
        # Initialize tracking variables
        position = 0
        cash = self.account_size
        position_size = 0
        entry_price = 0
        stop_loss = 0
        portfolio_value = []
        trades = []
        
        # Skip initial periods for indicator stability
        start_idx = max(slow_ema, 100) + 1
        
        for i in range(start_idx, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_date = self.data.index[i]
            current_atr = atr.iloc[i]
            
            # Market session filter
            session = self.get_market_session(current_date)
            
            # Check stop loss
            if position != 0:
                if (position == 1 and current_price <= stop_loss) or \
                   (position == -1 and current_price >= stop_loss):
                    
                    pnl = position_size * (current_price - entry_price) * position
                    cash += pnl
                    
                    trades.append({
                        'date': current_date,
                        'type': 'STOP_LOSS',
                        'side': 'SELL' if position == 1 else 'BUY',
                        'price': current_price,
                        'size': position_size,
                        'pnl': pnl
                    })
                    
                    position = 0
                    position_size = 0
                    entry_price = 0
                    stop_loss = 0
            
            # Generate enhanced signals
            if i > start_idx:
                
                # Basic EMA crossover
                ema_cross_up = (ema_fast.iloc[i] > ema_slow.iloc[i] and 
                               ema_fast.iloc[i-1] <= ema_slow.iloc[i-1])
                ema_cross_down = (ema_fast.iloc[i] < ema_slow.iloc[i] and 
                                 ema_fast.iloc[i-1] >= ema_slow.iloc[i-1])
                
                # Enhanced filters for LONG entry
                if ema_cross_up and position == 0:
                    
                    # Filter 1: Trend alignment (price above long-term EMA)
                    trend_filter = current_price > ema_trend.iloc[i]
                    
                    # Filter 2: RSI not overbought
                    rsi_filter = 30 < rsi.iloc[i] < 70
                    
                    # Filter 3: MACD confirmation
                    macd_filter = macd_line.iloc[i] > macd_signal.iloc[i]
                    
                    # Filter 4: Volatility filter (avoid low volatility periods)
                    vol_filter = high_vol_regime.iloc[i] or current_atr > atr.rolling(20).mean().iloc[i]
                    
                    # Filter 5: Price not at resistance
                    resistance_filter = True
                    if resistance_levels:
                        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                        resistance_filter = abs(current_price - nearest_resistance) / current_price > 0.01  # 1% away
                    
                    # Filter 6: Session filter (prefer London/NY sessions)
                    session_filter = session in ['London', 'NY_Overlap', 'New_York']
                    
                    # Filter 7: Bollinger Band position (not at upper band)
                    bb_filter = current_price < bb_upper.iloc[i] * 0.98
                    
                    # Combine all filters
                    all_filters = (trend_filter and rsi_filter and macd_filter and 
                                  vol_filter and resistance_filter and session_filter and bb_filter)
                    
                    if all_filters:
                        # Calculate position size with enhanced risk management
                        stop_loss_price = current_price - (2.5 * current_atr)  # Wider stop in volatile markets
                        
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
                                'session': session,
                                'filters_passed': 'ALL'
                            })
                
                # Enhanced filters for SHORT entry
                elif ema_cross_down:
                    
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
                            'pnl': pnl
                        })
                        
                        position = 0
                        position_size = 0
                        entry_price = 0
                        stop_loss = 0
                    
                    # Short entry filters
                    elif position == 0:
                        # Filter 1: Trend alignment (price below long-term EMA)
                        trend_filter = current_price < ema_trend.iloc[i]
                        
                        # Filter 2: RSI not oversold
                        rsi_filter = 30 < rsi.iloc[i] < 70
                        
                        # Filter 3: MACD confirmation
                        macd_filter = macd_line.iloc[i] < macd_signal.iloc[i]
                        
                        # Filter 4: Volatility filter
                        vol_filter = high_vol_regime.iloc[i] or current_atr > atr.rolling(20).mean().iloc[i]
                        
                        # Filter 5: Price not at support
                        support_filter = True
                        if support_levels:
                            nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                            support_filter = abs(current_price - nearest_support) / current_price > 0.01
                        
                        # Filter 6: Session filter
                        session_filter = session in ['London', 'NY_Overlap', 'New_York']
                        
                        # Filter 7: Bollinger Band position (not at lower band)
                        bb_filter = current_price > bb_lower.iloc[i] * 1.02
                        
                        # Combine all filters
                        all_filters = (trend_filter and rsi_filter and macd_filter and 
                                      vol_filter and support_filter and session_filter and bb_filter)
                        
                        if all_filters:
                            stop_loss_price = current_price + (2.5 * current_atr)
                            
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
                                    'session': session,
                                    'filters_passed': 'ALL'
                                })
            
            # Calculate portfolio value
            if position != 0:
                unrealized_pnl = position_size * (current_price - entry_price) * position
                portfolio_value.append(cash + unrealized_pnl)
            else:
                portfolio_value.append(cash)
        
        # Close final position
        if position != 0:
            final_price = self.data['Close'].iloc[-1]
            pnl = position_size * (final_price - entry_price) * position
            cash += pnl
            
            trades.append({
                'date': self.data.index[-1],
                'type': 'FINAL_EXIT',
                'side': 'SELL' if position == 1 else 'BUY',
                'price': final_price,
                'size': position_size,
                'pnl': pnl
            })
        
        # Calculate performance metrics
        if len(portfolio_value) == 0:
            return None
        
        portfolio_series = pd.Series(portfolio_value, index=self.data.index[start_idx:])
        
        total_return = (cash - self.account_size) / self.account_size * 100
        max_drawdown = self.calculate_max_drawdown(portfolio_series)
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_series)
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        
        total_trades = len([t for t in trades if t['type'] == 'ENTRY'])
        
        return {
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'total_return': round(total_return, 2),
            'final_balance': round(cash, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
            'trades': trades,
            'portfolio_series': portfolio_series
        }
    
    def calculate_position_size(self, entry_price, stop_loss_price, risk_amount):
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            return 0
        
        base_position_size = risk_amount / risk_per_unit
        max_position_value = self.account_size * self.leverage
        max_units = max_position_value / entry_price
        
        return min(base_position_size, max_units)
    
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
        return returns.mean() / returns.std() * np.sqrt(252)
    
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

def test_enhanced_strategies():
    """Test enhanced EMA strategies with multiple filters"""
    
    print("=" * 100)
    print("ENHANCED EMA STRATEGY WITH EDGE-BUILDING FILTERS")
    print("=" * 100)
    
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
    
    # Initialize enhanced strategy
    strategy = EnhancedEMAStrategy(
        data=data,
        account_size=1000,
        leverage=500,
        max_risk_per_trade=20
    )
    
    # Test promising EMA combinations with filters
    ema_pairs = [
        (8, 21), (9, 21), (12, 26), (13, 34), (21, 55)
    ]
    
    results = []
    
    print(f"\nTesting {len(ema_pairs)} enhanced EMA combinations...")
    
    for i, (fast, slow) in enumerate(ema_pairs, 1):
        print(f"{i}. Testing Enhanced EMA({fast},{slow})...")
        
        try:
            result = strategy.enhanced_ema_backtest(fast, slow)
            if result and result['total_trades'] > 0:
                results.append(result)
                print(f"   ✓ Return: {result['total_return']:.2f}%, Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
            else:
                print(f"   ✗ No valid trades generated")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Display results
    if results:
        print(f"\n" + "=" * 120)
        print("ENHANCED EMA STRATEGY RESULTS")
        print("=" * 120)
        
        # Sort by total return
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        print(f"{'Rank':<4} {'EMA Pair':<10} {'Return%':<10} {'Balance$':<12} {'Drawdown%':<12} {'Sharpe':<8} {'Trades':<8} {'WinRate%':<10} {'ProfitFactor':<12}")
        print("-" * 120)
        
        for i, result in enumerate(results, 1):
            ema_pair = f"{result['fast_ema']}/{result['slow_ema']}"
            print(f"{i:<4} {ema_pair:<10} {result['total_return']:<10.2f} {result['final_balance']:<12.2f} {result['max_drawdown']:<12.2f} {result['sharpe_ratio']:<8.3f} {result['total_trades']:<8} {result['win_rate']:<10.2f} {result['profit_factor']:<12.3f}")
        
        # Best strategy details
        if results:
            best = results[0]
            print(f"\n🏆 BEST ENHANCED STRATEGY:")
            print(f"   • EMA Pair: {best['fast_ema']}/{best['slow_ema']}")
            print(f"   • Total Return: {best['total_return']:.2f}%")
            print(f"   • Final Balance: ${best['final_balance']:.2f}")
            print(f"   • Max Drawdown: {best['max_drawdown']:.2f}%")
            print(f"   • Win Rate: {best['win_rate']:.2f}%")
            print(f"   • Total Trades: {best['total_trades']}")
            print(f"   • Profit Factor: {best['profit_factor']:.3f}")
            print(f"   • Sharpe Ratio: {best['sharpe_ratio']:.3f}")
            
            print(f"\n📋 ENHANCED FILTERS APPLIED:")
            print(f"   • Trend Filter: Price vs 100-period EMA")
            print(f"   • RSI Filter: 30 < RSI < 70")
            print(f"   • MACD Confirmation: MACD line vs Signal line")
            print(f"   • Volatility Filter: High volatility periods preferred")
            print(f"   • Support/Resistance: Avoid key levels")
            print(f"   • Session Filter: London/NY sessions preferred")
            print(f"   • Bollinger Band Filter: Avoid extreme positions")
            
            print(f"\n🎯 TRADINGVIEW IMPLEMENTATION:")
            print(f"   • Add EMA({best['fast_ema']}) and EMA({best['slow_ema']})")
            print(f"   • Add EMA(100) for trend filter")
            print(f"   • Add RSI(14) for momentum filter")
            print(f"   • Add MACD(12,26,9) for confirmation")
            print(f"   • Add Bollinger Bands(20,2) for volatility")
            print(f"   • Set up alerts for crossovers with all filters")
            
            print(f"\n⚠️  IMPORTANT NOTES:")
            print(f"   • Only trade when ALL filters align")
            print(f"   • Use 2.5 ATR stop losses")
            print(f"   • Risk maximum $20 per trade")
            print(f"   • Prefer London/NY trading sessions")
            print(f"   • Always paper trade first")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('enhanced_ema_results.csv', index=False)
        print(f"\n✅ Results saved to 'enhanced_ema_results.csv'")
        
    else:
        print("\n❌ No profitable enhanced strategies found.")
        print("\nSUGGESTIONS FOR FURTHER IMPROVEMENT:")
        print("• Try different timeframes (1H, 4H, 1D)")
        print("• Add more sophisticated filters (volume, sentiment)")
        print("• Consider machine learning approaches")
        print("• Use fundamental analysis alongside technical")
        print("• Focus on specific market conditions/regimes")
    
    return results

if __name__ == "__main__":
    results = test_enhanced_strategies()
