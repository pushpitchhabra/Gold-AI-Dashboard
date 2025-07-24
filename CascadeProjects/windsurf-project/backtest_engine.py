from backtesting import Backtest
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from strategies import (
    MovingAverageCrossover, RSIStrategy, MACDStrategy, 
    BollingerBandsStrategy, GoldenCrossStrategy, 
    StochasticStrategy, EMAStrategy
)

class BacktestEngine:
    def __init__(self, data, initial_cash=10000, commission=0.002):
        """
        Initialize the backtesting engine
        
        Args:
            data: OHLCV DataFrame with datetime index
            initial_cash: Starting capital
            commission: Commission rate (0.002 = 0.2%)
        """
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = {}
        
        # Define all strategies to test
        self.strategies = {
            'MA_Crossover': MovingAverageCrossover,
            'RSI_Strategy': RSIStrategy,
            'MACD_Strategy': MACDStrategy,
            'Bollinger_Bands': BollingerBandsStrategy,
            'Golden_Cross': GoldenCrossStrategy,
            'Stochastic': StochasticStrategy,
            'EMA_Strategy': EMAStrategy
        }
    
    def run_single_backtest(self, strategy_name, strategy_class):
        """Run backtest for a single strategy"""
        try:
            print(f"\nRunning backtest for {strategy_name}...")
            
            bt = Backtest(self.data, strategy_class, 
                         cash=self.initial_cash, 
                         commission=self.commission)
            
            result = bt.run()
            
            print(f"✓ {strategy_name} completed successfully")
            return result
            
        except Exception as e:
            print(f"✗ Error running {strategy_name}: {e}")
            return None
    
    def run_all_backtests(self):
        """Run backtests for all strategies"""
        print("=" * 60)
        print("STARTING COMPREHENSIVE BACKTEST")
        print("=" * 60)
        print(f"Data period: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"Total records: {len(self.data)}")
        print(f"Initial capital: ${self.initial_cash:,.2f}")
        print(f"Commission: {self.commission*100:.2f}%")
        
        for strategy_name, strategy_class in self.strategies.items():
            result = self.run_single_backtest(strategy_name, strategy_class)
            if result is not None:
                self.results[strategy_name] = result
        
        print(f"\n✓ Completed {len(self.results)} backtests successfully")
        return self.results
    
    def create_performance_summary(self):
        """Create a comprehensive performance summary"""
        if not self.results:
            print("No backtest results available. Run backtests first.")
            return None
        
        summary_data = []
        
        for strategy_name, result in self.results.items():
            summary_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': round(result['Return [%]'], 2),
                'Annual Return (%)': round(result['Return [%]'] * 365 / len(self.data) * 48, 2),  # Approx annualized
                'Max Drawdown (%)': round(result['Max. Drawdown [%]'], 2),
                'Sharpe Ratio': round(result['Sharpe Ratio'], 3),
                'Total Trades': result['# Trades'],
                'Win Rate (%)': round(result['Win Rate [%]'], 2),
                'Profit Factor': round(result['Profit Factor'], 3),
                'Final Value ($)': round(result['Equity Final [$]'], 2),
                'Buy & Hold Return (%)': round(result['Buy & Hold Return [%]'], 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Return (%)', ascending=False)
        
        return summary_df
    
    def print_performance_summary(self):
        """Print formatted performance summary"""
        summary_df = self.create_performance_summary()
        if summary_df is None:
            return
        
        print("\n" + "=" * 100)
        print("PERFORMANCE SUMMARY - RANKED BY TOTAL RETURN")
        print("=" * 100)
        
        # Display the summary table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(summary_df.to_string(index=False))
        
        # Highlight best performers
        print("\n" + "=" * 50)
        print("TOP PERFORMERS")
        print("=" * 50)
        
        best_return = summary_df.iloc[0]
        best_sharpe = summary_df.loc[summary_df['Sharpe Ratio'].idxmax()]
        best_winrate = summary_df.loc[summary_df['Win Rate (%)'].idxmax()]
        
        print(f"🏆 Best Total Return: {best_return['Strategy']} ({best_return['Total Return (%)']}%)")
        print(f"📈 Best Sharpe Ratio: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']})")
        print(f"🎯 Best Win Rate: {best_winrate['Strategy']} ({best_winrate['Win Rate (%)']}%)")
        
        return summary_df
    
    def create_equity_curves_plot(self):
        """Create equity curves comparison plot"""
        if not self.results:
            print("No backtest results available.")
            return
        
        plt.figure(figsize=(15, 10))
        
        for strategy_name, result in self.results.items():
            equity_curve = result._equity_curve['Equity']
            plt.plot(equity_curve.index, equity_curve.values, 
                    label=f"{strategy_name} ({result['Return [%]']:.1f}%)", 
                    linewidth=2)
        
        plt.title('Strategy Performance Comparison - Equity Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('/Users/pushpitchhabra/Desktop/Projects/Windsurf/Gold Windsurf/CascadeProjects/windsurf-project/equity_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_tradingview_recommendations(self):
        """Generate recommendations for TradingView setup"""
        if not self.results:
            print("No backtest results available.")
            return
        
        summary_df = self.create_performance_summary()
        
        print("\n" + "=" * 80)
        print("TRADINGVIEW SETUP RECOMMENDATIONS")
        print("=" * 80)
        
        # Get top 3 strategies
        top_strategies = summary_df.head(3)
        
        print("\n🎯 TOP 3 RECOMMENDED STRATEGIES FOR TRADINGVIEW:")
        print("-" * 60)
        
        for idx, row in top_strategies.iterrows():
            strategy_name = row['Strategy']
            print(f"\n{idx + 1}. {strategy_name}")
            print(f"   • Total Return: {row['Total Return (%)']}%")
            print(f"   • Max Drawdown: {row['Max Drawdown (%)']}%")
            print(f"   • Win Rate: {row['Win Rate (%)']}%")
            print(f"   • Sharpe Ratio: {row['Sharpe Ratio']}")
            
            # Strategy-specific TradingView setup
            if strategy_name == 'MA_Crossover':
                print("   • TradingView Setup: Add 10-period and 30-period Simple Moving Averages")
                print("   • Signal: Buy when 10 MA crosses above 30 MA, Sell when 10 MA crosses below 30 MA")
            
            elif strategy_name == 'RSI_Strategy':
                print("   • TradingView Setup: Add RSI(14) indicator")
                print("   • Signal: Buy when RSI crosses above 30, Sell when RSI crosses below 70")
            
            elif strategy_name == 'MACD_Strategy':
                print("   • TradingView Setup: Add MACD(12,26,9) indicator")
                print("   • Signal: Buy when MACD line crosses above Signal line, Sell when MACD crosses below Signal")
            
            elif strategy_name == 'Bollinger_Bands':
                print("   • TradingView Setup: Add Bollinger Bands(20,2) indicator")
                print("   • Signal: Buy when price touches lower band, Sell when price touches upper band")
            
            elif strategy_name == 'Golden_Cross':
                print("   • TradingView Setup: Add 50-period and 200-period Simple Moving Averages")
                print("   • Signal: Buy when 50 MA crosses above 200 MA, Sell when 50 MA crosses below 200 MA")
            
            elif strategy_name == 'EMA_Strategy':
                print("   • TradingView Setup: Add 12-period and 26-period Exponential Moving Averages")
                print("   • Signal: Buy when 12 EMA crosses above 26 EMA, Sell when 12 EMA crosses below 26 EMA")
        
        print(f"\n💡 GENERAL RECOMMENDATIONS:")
        print("• Use 30-minute timeframe to match your historical data")
        print("• Set up alerts for signal crossovers")
        print("• Consider risk management with stop-loss orders")
        print("• Monitor performance and adjust parameters as needed")
        print("• Paper trade first before going live")
        
        return top_strategies
