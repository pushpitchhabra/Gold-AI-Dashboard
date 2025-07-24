#!/usr/bin/env python3
"""
Gold Trading Strategy Backtesting Framework
===========================================

This script runs comprehensive backtests on historical gold data
to identify the best trading strategies for TradingView implementation.

Usage: python main.py
"""

import sys
import os
from data_loader import GoldDataLoader
from backtest_engine import BacktestEngine

def main():
    print("=" * 80)
    print("GOLD TRADING STRATEGY BACKTESTING FRAMEWORK")
    print("=" * 80)
    
    # Path to the historical data
    data_path = "/Users/pushpitchhabra/Desktop/Projects/Windsurf/Gold Windsurf/CascadeProjects/30 Min - Gold Histdata.com summary.csv"
    
    # Step 1: Load and preprocess data
    print("\n📊 STEP 1: LOADING DATA")
    print("-" * 40)
    
    loader = GoldDataLoader(data_path)
    data = loader.load_data()
    
    if data is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Display data information
    loader.get_data_info()
    loader.check_data_quality()
    
    # Step 2: Initialize backtesting engine
    print("\n🔧 STEP 2: INITIALIZING BACKTEST ENGINE")
    print("-" * 40)
    
    # Configuration
    initial_cash = 10000  # Starting with $10,000
    commission = 0.002    # 0.2% commission per trade
    
    engine = BacktestEngine(data, initial_cash=initial_cash, commission=commission)
    
    # Step 3: Run all backtests
    print("\n🚀 STEP 3: RUNNING BACKTESTS")
    print("-" * 40)
    
    results = engine.run_all_backtests()
    
    if not results:
        print("❌ No successful backtests. Exiting.")
        return
    
    # Step 4: Analyze results
    print("\n📈 STEP 4: ANALYZING RESULTS")
    print("-" * 40)
    
    summary_df = engine.print_performance_summary()
    
    # Step 5: Generate visualizations
    print("\n📊 STEP 5: GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    try:
        engine.create_equity_curves_plot()
        print("✓ Equity curves plot saved as 'equity_curves.png'")
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")
    
    # Step 6: TradingView recommendations
    print("\n🎯 STEP 6: TRADINGVIEW RECOMMENDATIONS")
    print("-" * 40)
    
    top_strategies = engine.generate_tradingview_recommendations()
    
    # Step 7: Save results to CSV
    print("\n💾 STEP 7: SAVING RESULTS")
    print("-" * 40)
    
    try:
        summary_df.to_csv('backtest_results.csv', index=False)
        print("✓ Results saved to 'backtest_results.csv'")
    except Exception as e:
        print(f"⚠️  Could not save CSV: {e}")
    
    print("\n" + "=" * 80)
    print("BACKTESTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📋 SUMMARY:")
    print(f"• Tested {len(results)} trading strategies")
    print(f"• Data period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"• Total data points: {len(data):,}")
    print(f"• Best performing strategy: {summary_df.iloc[0]['Strategy']}")
    print(f"• Best total return: {summary_df.iloc[0]['Total Return (%)']}%")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review the performance summary above")
    print("2. Check the equity curves plot (equity_curves.png)")
    print("3. Implement the top strategies in TradingView")
    print("4. Set up alerts for signal generation")
    print("5. Paper trade before going live")

if __name__ == "__main__":
    main()
