#!/usr/bin/env python3
"""
Hybrid Adaptive Trading Strategy for Gold
Automatically switches between mean reversion and momentum breakout strategies
based on real-time market regime detection.

Account: $1,000
Max Risk per Trade: $20 (2%)
Leverage: 20x (Conservative)
Data: 30-minute Gold OHLC (2009-2025)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess gold data"""
    try:
        # Load the data
        data_path = "/Users/pushpitchhabra/Desktop/Projects/Windsurf/Gold Windsurf/CascadeProjects/30 Min - Gold Histdata.com summary.csv"
        df = pd.read_csv(data_path)
        
        # Parse datetime - the file has 'Datetime' column, not separate Date and Time
        df['DateTime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%y %H:%M')
        df = df.set_index('DateTime')
        
        # Keep only OHLC columns
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        
        # Add volume as zeros (not used but some indicators might expect it)
        df['Volume'] = 0
        
        print(f"✅ Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def calculate_indicators(df):
    """Calculate all technical indicators needed for regime detection and trading"""
    
    print("📊 Calculating technical indicators...")
    
    # ATR for volatility and stop losses
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                         np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                   abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # ADX for trend strength
    df['HL'] = df['High'] - df['Low']
    df['HC'] = abs(df['High'] - df['Close'].shift(1))
    df['LC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = np.maximum(df['HL'], np.maximum(df['HC'], df['LC']))
    
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                           np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                            np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    
    df['DIplus'] = 100 * (df['DMplus'].rolling(14).mean() / df['TR'].rolling(14).mean())
    df['DIminus'] = 100 * (df['DMminus'].rolling(14).mean() / df['TR'].rolling(14).mean())
    df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
    df['ADX'] = df['DX'].rolling(14).mean()
    
    # Bollinger Bands for volatility regime
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Width_MA'] = df['BB_Width'].rolling(20).mean()
    
    # RSI for momentum
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD for trend confirmation
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Rolling highs and lows for breakouts
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    
    # Price position within Bollinger Bands
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    print("✅ Technical indicators calculated successfully")
    return df

def detect_market_regime(df, i):
    """
    Detect current market regime: TRENDING, RANGING, or VOLATILE
    
    TRENDING: Strong trend with expanding volatility (ADX > 25 + BB expansion)
    RANGING: Weak trend with contracting volatility (ADX < 20 + BB contraction)
    VOLATILE: Mixed conditions (everything else)
    """
    
    adx = df['ADX'].iloc[i]
    bb_width = df['BB_Width'].iloc[i]
    bb_width_ma = df['BB_Width_MA'].iloc[i]
    
    # Handle NaN values
    if pd.isna(adx) or pd.isna(bb_width) or pd.isna(bb_width_ma):
        return 'RANGING'
    
    # Regime detection logic
    if adx > 25 and bb_width > bb_width_ma * 1.2:
        return 'TRENDING'
    elif adx < 20 and bb_width < bb_width_ma * 0.8:
        return 'RANGING'
    else:
        return 'VOLATILE'

def mean_reversion_signal(df, i):
    """
    Generate mean reversion signals for ranging markets
    
    LONG: Price near lower BB + RSI oversold
    SHORT: Price near upper BB + RSI overbought
    Target: Middle of Bollinger Bands
    """
    
    rsi = df['RSI'].iloc[i]
    bb_position = df['BB_Position'].iloc[i]
    close = df['Close'].iloc[i]
    bb_lower = df['BB_Lower'].iloc[i]
    bb_upper = df['BB_Upper'].iloc[i]
    bb_middle = df['BB_Middle'].iloc[i]
    
    # Handle NaN values
    if any(pd.isna(x) for x in [rsi, bb_position, close, bb_lower, bb_upper, bb_middle]):
        return None, None
    
    # Long signal: Price near lower BB and RSI oversold
    if bb_position < 0.2 and rsi < 35:
        return 'LONG', bb_middle  # Target middle of BB
    
    # Short signal: Price near upper BB and RSI overbought
    elif bb_position > 0.8 and rsi > 65:
        return 'SHORT', bb_middle  # Target middle of BB
    
    return None, None

def momentum_breakout_signal(df, i):
    """
    Generate momentum breakout signals for trending markets
    
    LONG: Price breaks above 20-period high + RSI > 50 + MACD bullish
    SHORT: Price breaks below 20-period low + RSI < 50 + MACD bearish
    Target: 2% move in direction of breakout
    """
    
    close = df['Close'].iloc[i]
    high_20 = df['High_20'].iloc[i-1] if i > 0 else df['High_20'].iloc[i]
    low_20 = df['Low_20'].iloc[i-1] if i > 0 else df['Low_20'].iloc[i]
    rsi = df['RSI'].iloc[i]
    macd = df['MACD'].iloc[i]
    macd_signal = df['MACD_Signal'].iloc[i]
    
    # Handle NaN values
    if any(pd.isna(x) for x in [close, high_20, low_20, rsi, macd, macd_signal]):
        return None, None
    
    # Long breakout: Price breaks above 20-period high with momentum confirmation
    if close > high_20 and rsi > 50 and macd > macd_signal:
        return 'LONG', close * 1.02  # Target 2% above entry
    
    # Short breakout: Price breaks below 20-period low with momentum confirmation
    elif close < low_20 and rsi < 50 and macd < macd_signal:
        return 'SHORT', close * 0.98  # Target 2% below entry
    
    return None, None

def calculate_position_size(account_balance, risk_per_trade, entry_price, stop_price, leverage=20):
    """
    Calculate position size based on risk management
    
    Position Size = (Risk per Trade / Risk per Share) * Leverage / Entry Price
    Max Position Value = Account Balance * Leverage
    """
    
    if pd.isna(entry_price) or pd.isna(stop_price) or entry_price <= 0 or stop_price <= 0:
        return 0
    
    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_price)
    
    if risk_per_share == 0:
        return 0
    
    # Calculate position size
    position_value = risk_per_trade / risk_per_share
    
    # Apply leverage
    position_size = position_value * leverage / entry_price
    
    # Ensure we don't exceed account balance with leverage
    max_position_value = account_balance * leverage
    if position_value > max_position_value:
        position_size = max_position_value / entry_price
    
    return position_size

def run_hybrid_backtest(df):
    """Run the hybrid adaptive strategy backtest"""
    
    # Initialize tracking variables
    account_balance = 1000  # Starting balance
    max_risk_per_trade = 20  # Maximum risk per trade
    leverage = 20  # Conservative leverage
    
    trades = []
    current_position = None
    regime_switches = []
    
    print("\n🚀 STARTING HYBRID ADAPTIVE STRATEGY BACKTEST")
    print("=" * 60)
    print(f"💰 Account Balance: ${account_balance:,}")
    print(f"⚠️  Max Risk per Trade: ${max_risk_per_trade}")
    print(f"📈 Leverage: {leverage}x")
    print(f"📊 Data Period: {df.index[0]} to {df.index[-1]}")
    print("=" * 60)
    
    regime_count = {'TRENDING': 0, 'RANGING': 0, 'VOLATILE': 0}
    
    for i in range(50, len(df)):  # Start after indicators are calculated
        current_time = df.index[i]
        current_price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        
        if pd.isna(current_price) or pd.isna(atr):
            continue
        
        # Detect market regime
        regime = detect_market_regime(df, i)
        regime_count[regime] += 1
        
        # Track regime switches
        if len(regime_switches) == 0 or regime_switches[-1][1] != regime:
            regime_switches.append((current_time, regime))
        
        # Close existing position if conditions are met
        if current_position:
            should_close = False
            close_reason = ""
            
            if current_position['type'] == 'LONG':
                # Close long position
                if current_price >= current_position['target']:
                    should_close = True
                    close_reason = "Target Hit"
                elif current_price <= current_position['stop']:
                    should_close = True
                    close_reason = "Stop Loss"
                elif regime != current_position['regime'] and i - current_position['entry_index'] > 10:
                    should_close = True
                    close_reason = "Regime Change"
            
            else:  # SHORT position
                if current_price <= current_position['target']:
                    should_close = True
                    close_reason = "Target Hit"
                elif current_price >= current_position['stop']:
                    should_close = True
                    close_reason = "Stop Loss"
                elif regime != current_position['regime'] and i - current_position['entry_index'] > 10:
                    should_close = True
                    close_reason = "Regime Change"
            
            if should_close:
                # Calculate P&L
                if current_position['type'] == 'LONG':
                    pnl = (current_price - current_position['entry_price']) * current_position['size']
                else:
                    pnl = (current_position['entry_price'] - current_price) * current_position['size']
                
                account_balance += pnl
                
                trade_record = {
                    'entry_time': current_position['entry_time'],
                    'exit_time': current_time,
                    'type': current_position['type'],
                    'regime': current_position['regime'],
                    'strategy': current_position['strategy'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': current_price,
                    'size': current_position['size'],
                    'pnl': pnl,
                    'close_reason': close_reason,
                    'account_balance': account_balance
                }
                trades.append(trade_record)
                
                current_position = None
        
        # Look for new entry signals if no current position
        if current_position is None:
            signal = None
            target = None
            strategy_used = ""
            
            if regime == 'RANGING':
                signal, target = mean_reversion_signal(df, i)
                strategy_used = "Mean Reversion"
            elif regime == 'TRENDING':
                signal, target = momentum_breakout_signal(df, i)
                strategy_used = "Momentum Breakout"
            elif regime == 'VOLATILE':
                # Use cautious mean reversion in volatile markets
                signal, target = mean_reversion_signal(df, i)
                strategy_used = "Cautious Mean Reversion"
                # Reduce position size in volatile conditions
                max_risk_per_trade = 15
            else:
                max_risk_per_trade = 20  # Reset risk
            
            if signal and target:
                # Calculate stop loss
                if signal == 'LONG':
                    stop_loss = current_price - (2 * atr if regime == 'RANGING' else 1.5 * atr)
                else:
                    stop_loss = current_price + (2 * atr if regime == 'RANGING' else 1.5 * atr)
                
                # Calculate position size
                position_size = calculate_position_size(
                    account_balance, max_risk_per_trade, current_price, stop_loss, leverage
                )
                
                if position_size > 0:
                    current_position = {
                        'type': signal,
                        'regime': regime,
                        'strategy': strategy_used,
                        'entry_time': current_time,
                        'entry_index': i,
                        'entry_price': current_price,
                        'target': target,
                        'stop': stop_loss,
                        'size': position_size
                    }
    
    # Close any remaining position
    if current_position:
        final_price = df['Close'].iloc[-1]
        if current_position['type'] == 'LONG':
            pnl = (final_price - current_position['entry_price']) * current_position['size']
        else:
            pnl = (current_position['entry_price'] - final_price) * current_position['size']
        
        account_balance += pnl
        
        trade_record = {
            'entry_time': current_position['entry_time'],
            'exit_time': df.index[-1],
            'type': current_position['type'],
            'regime': current_position['regime'],
            'strategy': current_position['strategy'],
            'entry_price': current_position['entry_price'],
            'exit_price': final_price,
            'size': current_position['size'],
            'pnl': pnl,
            'close_reason': 'End of Data',
            'account_balance': account_balance
        }
        trades.append(trade_record)
    
    print(f"\n📊 Regime Distribution:")
    total_periods = sum(regime_count.values())
    for regime, count in regime_count.items():
        percentage = (count / total_periods) * 100 if total_periods > 0 else 0
        print(f"   {regime}: {count:,} periods ({percentage:.1f}%)")
    
    return trades, regime_switches, account_balance

def analyze_results(trades, regime_switches, final_balance):
    """Analyze and display backtest results"""
    
    if not trades:
        print("\n❌ No trades were executed!")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # Basic performance metrics
    total_trades = len(trades)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    total_return = ((final_balance - 1000) / 1000) * 100
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
    
    # Calculate max drawdown
    running_balance = 1000
    peak_balance = 1000
    max_drawdown = 0
    
    for trade in trades:
        running_balance += trade['pnl']
        if running_balance > peak_balance:
            peak_balance = running_balance
        drawdown = (peak_balance - running_balance) / peak_balance * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    print("\n" + "="*80)
    print("🏆 HYBRID ADAPTIVE STRATEGY - BACKTEST RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"Initial Balance:     ${1000:,.2f}")
    print(f"Final Balance:       ${final_balance:,.2f}")
    print(f"Total Return:        {total_return:+.2f}%")
    print(f"Total P&L:           ${total_pnl:+,.2f}")
    print(f"Max Drawdown:        {max_drawdown:.2f}%")
    
    print(f"\n📈 TRADE STATISTICS:")
    print(f"Total Trades:        {total_trades}")
    print(f"Winning Trades:      {winning_trades}")
    print(f"Losing Trades:       {losing_trades}")
    print(f"Win Rate:            {win_rate:.1f}%")
    print(f"Average Win:         ${avg_win:+.2f}")
    print(f"Average Loss:        ${avg_loss:+.2f}")
    print(f"Profit Factor:       {profit_factor:.2f}")
    
    print(f"\n🎯 STRATEGY BREAKDOWN:")
    for strategy in trades_df['strategy'].unique():
        strategy_trades = trades_df[trades_df['strategy'] == strategy]
        strategy_pnl = strategy_trades['pnl'].sum()
        strategy_count = len(strategy_trades)
        strategy_winrate = (len(strategy_trades[strategy_trades['pnl'] > 0]) / strategy_count * 100) if strategy_count > 0 else 0
        print(f"{strategy:25} | Trades: {strategy_count:3d} | P&L: ${strategy_pnl:+8.2f} | Win Rate: {strategy_winrate:5.1f}%")
    
    print(f"\n🌊 REGIME BREAKDOWN:")
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        regime_pnl = regime_trades['pnl'].sum()
        regime_count = len(regime_trades)
        regime_winrate = (len(regime_trades[regime_trades['pnl'] > 0]) / regime_count * 100) if regime_count > 0 else 0
        print(f"{regime:15} | Trades: {regime_count:3d} | P&L: ${regime_pnl:+8.2f} | Win Rate: {regime_winrate:5.1f}%")
    
    print(f"\n🔄 REGIME ANALYSIS:")
    print(f"Total Regime Switches: {len(regime_switches)}")
    
    # Save detailed results
    trades_df.to_csv('hybrid_adaptive_results.csv', index=False)
    print(f"\n💾 Detailed results saved to: hybrid_adaptive_results.csv")
    
    # TradingView recommendations
    print(f"\n🎯 TRADINGVIEW SETUP RECOMMENDATIONS:")
    best_strategy = trades_df.groupby('strategy')['pnl'].sum().idxmax()
    print(f"Best Performing Strategy: {best_strategy}")
    
    if 'Mean Reversion' in best_strategy:
        print("📋 TradingView Indicators:")
        print("   • Bollinger Bands (20, 2)")
        print("   • RSI (14)")
        print("   • ADX (14)")
        print("📋 Entry Conditions:")
        print("   • LONG: Price < BB Lower + RSI < 35 + ADX < 20")
        print("   • SHORT: Price > BB Upper + RSI > 65 + ADX < 20")
    else:
        print("📋 TradingView Indicators:")
        print("   • 20-period High/Low")
        print("   • RSI (14)")
        print("   • MACD (12, 26, 9)")
        print("   • ADX (14)")
        print("📋 Entry Conditions:")
        print("   • LONG: Close > High[20] + RSI > 50 + MACD > Signal + ADX > 25")
        print("   • SHORT: Close < Low[20] + RSI < 50 + MACD < Signal + ADX > 25")
    
    print("\n" + "="*80)
    
    return trades_df

def main():
    """Main execution function"""
    print("🚀 HYBRID ADAPTIVE TRADING STRATEGY FOR GOLD")
    print("Automatically switches between Mean Reversion and Momentum Breakout")
    print("Based on real-time market regime detection using ADX and Bollinger Bands\n")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Run backtest
    trades, regime_switches, final_balance = run_hybrid_backtest(df)
    
    # Analyze results
    analyze_results(trades, regime_switches, final_balance)

if __name__ == "__main__":
    main()
