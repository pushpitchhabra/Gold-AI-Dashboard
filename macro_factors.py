"""
Macroeconomic Factors Module for Gold Trading Dashboard
Tracks and analyzes quantitative factors that influence gold prices
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroFactorsAnalyzer:
    """
    Analyzes macroeconomic and market factors that influence gold prices
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.fred_api_key = self.config.get('api_keys', {}).get('fred_key', '')
        
        # Define key factors that influence gold
        self.market_factors = {
            # Currency & Dollar Strength
            'DX-Y.NYB': {'name': 'US Dollar Index (DXY)', 'category': 'Currency', 'correlation': 'negative'},
            'EURUSD=X': {'name': 'EUR/USD', 'category': 'Currency', 'correlation': 'positive'},
            'GBPUSD=X': {'name': 'GBP/USD', 'category': 'Currency', 'correlation': 'positive'},
            'USDJPY=X': {'name': 'USD/JPY', 'category': 'Currency', 'correlation': 'negative'},
            
            # Interest Rates & Bonds
            '^TNX': {'name': '10-Year Treasury Yield', 'category': 'Interest Rates', 'correlation': 'negative'},
            '^FVX': {'name': '5-Year Treasury Yield', 'category': 'Interest Rates', 'correlation': 'negative'},
            '^IRX': {'name': '3-Month Treasury Bill', 'category': 'Interest Rates', 'correlation': 'negative'},
            'TLT': {'name': '20+ Year Treasury Bond ETF', 'category': 'Bonds', 'correlation': 'positive'},
            
            # Market Sentiment & Risk
            '^VIX': {'name': 'CBOE Volatility Index', 'category': 'Risk Sentiment', 'correlation': 'positive'},
            '^GSPC': {'name': 'S&P 500 Index', 'category': 'Equities', 'correlation': 'negative'},
            
            # Commodities & Inflation
            'CL=F': {'name': 'Crude Oil Futures', 'category': 'Commodities', 'correlation': 'positive'},
            'SI=F': {'name': 'Silver Futures', 'category': 'Precious Metals', 'correlation': 'positive'},
            'HG=F': {'name': 'Copper Futures', 'category': 'Industrial Metals', 'correlation': 'positive'},
            
            # Crypto (modern alternative store of value)
            'BTC-USD': {'name': 'Bitcoin', 'category': 'Digital Assets', 'correlation': 'mixed'}
        }
        
        # FRED economic indicators
        self.fred_indicators = {
            'CPIAUCSL': {'name': 'Consumer Price Index', 'category': 'Inflation', 'correlation': 'positive'},
            'CPILFESL': {'name': 'Core CPI', 'category': 'Inflation', 'correlation': 'positive'},
            'UNRATE': {'name': 'Unemployment Rate', 'category': 'Employment', 'correlation': 'positive'},
            'FEDFUNDS': {'name': 'Federal Funds Rate', 'category': 'Monetary Policy', 'correlation': 'negative'},
            'GDP': {'name': 'Gross Domestic Product', 'category': 'Economic Growth', 'correlation': 'negative'},
            'PAYEMS': {'name': 'Nonfarm Payrolls', 'category': 'Employment', 'correlation': 'negative'},
            'HOUST': {'name': 'Housing Starts', 'category': 'Real Estate', 'correlation': 'negative'},
            'UMCSENT': {'name': 'Consumer Sentiment', 'category': 'Sentiment', 'correlation': 'negative'}
        }
        
        self.factor_data = {}
        self.correlations = {}
        self.influence_scores = {}
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def fetch_market_factors(self, period='1mo') -> Dict:
        """
        Fetch market-based factors using yfinance
        """
        logger.info("Fetching market factors...")
        market_data = {}
        
        for symbol, info in self.market_factors.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    # Calculate volatility (20-day rolling std)
                    volatility = hist['Close'].pct_change().rolling(20).std().iloc[-1] * 100
                    
                    # Get last 30 data points for correlation analysis
                    close_data = hist['Close'].tail(30)
                    timestamps = close_data.index
                    
                    market_data[symbol] = {
                        'name': info['name'],
                        'category': info['category'],
                        'current_value': current_price,
                        'change_pct': change_pct,
                        'volatility': volatility if not pd.isna(volatility) else 0,
                        'correlation_type': info['correlation'],
                        'data': close_data.tolist(),  # Last 30 days for correlation
                        'timestamps': timestamps.strftime('%Y-%m-%d').tolist()
                    }
                    
                logger.info(f"✓ Fetched {info['name']}")
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                market_data[symbol] = {
                    'name': info['name'],
                    'category': info['category'],
                    'current_value': None,
                    'change_pct': 0,
                    'volatility': 0,
                    'correlation_type': info['correlation'],
                    'error': str(e)
                }
        
        return market_data
    
    def fetch_fred_indicators(self) -> Dict:
        """
        Fetch economic indicators from FRED API
        """
        if not self.fred_api_key or self.fred_api_key == 'your_fred_api_key_here':
            logger.warning("FRED API key not configured, skipping economic indicators")
            return {}
        
        logger.info("Fetching FRED economic indicators...")
        fred_data = {}
        
        for indicator, info in self.fred_indicators.items():
            try:
                # FRED API endpoint
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': indicator,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 100,
                    'sort_order': 'desc'
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get('observations', [])
                    
                    if observations:
                        # Get latest valid observation
                        latest_obs = None
                        prev_obs = None
                        
                        for obs in observations:
                            if obs['value'] != '.':
                                if latest_obs is None:
                                    latest_obs = obs
                                elif prev_obs is None:
                                    prev_obs = obs
                                    break
                        
                        if latest_obs:
                            current_value = float(latest_obs['value'])
                            prev_value = float(prev_obs['value']) if prev_obs else current_value
                            change_pct = ((current_value - prev_value) / prev_value) * 100 if prev_value != 0 else 0
                            
                            fred_data[indicator] = {
                                'name': info['name'],
                                'category': info['category'],
                                'current_value': current_value,
                                'change_pct': change_pct,
                                'correlation_type': info['correlation'],
                                'last_updated': latest_obs['date'],
                                'unit': data.get('units', 'N/A')
                            }
                            
                logger.info(f"✓ Fetched {info['name']}")
                
            except Exception as e:
                logger.error(f"Error fetching FRED indicator {indicator}: {e}")
                fred_data[indicator] = {
                    'name': info['name'],
                    'category': info['category'],
                    'current_value': None,
                    'change_pct': 0,
                    'correlation_type': info['correlation'],
                    'error': str(e)
                }
        
        return fred_data
    
    def calculate_correlations_with_gold(self, gold_data: pd.Series) -> Dict:
        """
        Calculate robust correlations using multiple models for highest accuracy
        """
        correlations = {}
        
        logger.info(f"Calculating robust correlations with gold data length: {len(gold_data)}")
        
        # Calculate correlations with both market and economic factors
        all_factors = {}
        all_factors.update(self.factor_data.get('market', {}))
        all_factors.update(self.factor_data.get('economic', {}))
        
        for symbol, data in all_factors.items():
            if 'data' in data and len(data['data']) > 3:  # More lenient minimum
                try:
                    factor_series = pd.Series(data['data'])
                    
                    # Multiple correlation approaches for robustness
                    correlation_results = self._calculate_multiple_correlations(
                        gold_data, factor_series, symbol, data['name']
                    )
                    
                    if correlation_results:
                        correlations[symbol] = correlation_results
                        logger.info(f"✓ {data['name']}: {correlation_results['correlation']:.3f} "
                                  f"({correlation_results['method']}, {correlation_results['data_points']} points)")
                        
                except Exception as e:
                    logger.error(f"Error calculating correlation for {symbol}: {e}")
        
        logger.info(f"Calculated {len(correlations)} robust correlations")
        return correlations
    
    def _calculate_multiple_correlations(self, gold_data: pd.Series, factor_data: pd.Series, 
                                       symbol: str, name: str) -> Dict:
        """
        Calculate correlation using multiple methods and return the most reliable one
        """
        from scipy.stats import pearsonr, spearmanr
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Data alignment and cleaning
            min_length = min(len(gold_data), len(factor_data))
            if min_length < 3:
                return None
            
            # Take most recent overlapping data
            gold_aligned = gold_data.iloc[-min_length:].reset_index(drop=True)
            factor_aligned = factor_data.iloc[-min_length:].reset_index(drop=True)
            
            # Create combined dataframe and remove NaN
            combined_df = pd.DataFrame({
                'gold': gold_aligned,
                'factor': factor_aligned
            }).dropna()
            
            if len(combined_df) < 3:
                return None
            
            gold_clean = combined_df['gold'].values
            factor_clean = combined_df['factor'].values
            
            # Method 1: Pearson correlation (linear relationships)
            try:
                pearson_corr, pearson_p = pearsonr(gold_clean, factor_clean)
                pearson_valid = not np.isnan(pearson_corr) and pearson_p < 0.1  # 90% confidence
            except:
                pearson_corr, pearson_valid = 0, False
            
            # Method 2: Spearman correlation (monotonic relationships)
            try:
                spearman_corr, spearman_p = spearmanr(gold_clean, factor_clean)
                spearman_valid = not np.isnan(spearman_corr) and spearman_p < 0.1
            except:
                spearman_corr, spearman_valid = 0, False
            
            # Method 3: Rolling correlation (recent trend)
            try:
                if len(combined_df) >= 5:
                    rolling_corr = combined_df['gold'].rolling(window=min(5, len(combined_df))).corr(
                        combined_df['factor']
                    ).iloc[-1]
                    rolling_valid = not np.isnan(rolling_corr)
                else:
                    rolling_corr, rolling_valid = 0, False
            except:
                rolling_corr, rolling_valid = 0, False
            
            # Method 4: Standardized correlation (normalized data)
            try:
                scaler_gold = StandardScaler()
                scaler_factor = StandardScaler()
                gold_scaled = scaler_gold.fit_transform(gold_clean.reshape(-1, 1)).flatten()
                factor_scaled = scaler_factor.fit_transform(factor_clean.reshape(-1, 1)).flatten()
                
                standardized_corr = np.corrcoef(gold_scaled, factor_scaled)[0, 1]
                standardized_valid = not np.isnan(standardized_corr)
            except:
                standardized_corr, standardized_valid = 0, False
            
            # Select the best correlation method
            correlations_methods = [
                ('Pearson', pearson_corr, pearson_valid, abs(pearson_corr)),
                ('Spearman', spearman_corr, spearman_valid, abs(spearman_corr)),
                ('Rolling', rolling_corr, rolling_valid, abs(rolling_corr)),
                ('Standardized', standardized_corr, standardized_valid, abs(standardized_corr))
            ]
            
            # Filter valid correlations and sort by absolute strength
            valid_correlations = [(method, corr, valid, strength) for method, corr, valid, strength 
                                in correlations_methods if valid and abs(corr) > 0.01]
            
            if not valid_correlations:
                # If no statistically significant correlations, use the strongest absolute correlation
                valid_correlations = [(method, corr, valid, strength) for method, corr, valid, strength 
                                    in correlations_methods if not np.isnan(corr)]
            
            if not valid_correlations:
                return None
            
            # Select the method with highest absolute correlation
            best_method, best_corr, _, _ = max(valid_correlations, key=lambda x: x[3])
            
            # Get expected direction from config
            factor_info = self.factor_data.get('market', {}).get(symbol) or \
                         self.factor_data.get('economic', {}).get(symbol, {})
            expected_direction = factor_info.get('correlation_type', 'unknown')
            
            return {
                'correlation': float(best_corr),
                'strength': self._classify_correlation_strength(abs(best_corr)),
                'direction': 'positive' if best_corr > 0 else 'negative',
                'expected_direction': expected_direction,
                'data_points': len(combined_df),
                'method': best_method,
                'confidence': 'high' if abs(best_corr) > 0.3 else 'medium' if abs(best_corr) > 0.1 else 'low',
                'all_methods': {
                    'pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0,
                    'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0,
                    'rolling': float(rolling_corr) if not np.isnan(rolling_corr) else 0,
                    'standardized': float(standardized_corr) if not np.isnan(standardized_corr) else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in multiple correlation calculation for {symbol}: {e}")
            return None
    
    def _classify_correlation_strength(self, abs_correlation: float) -> str:
        """Classify correlation strength"""
        if abs_correlation >= 0.7:
            return 'Strong'
        elif abs_correlation >= 0.4:
            return 'Moderate'
        elif abs_correlation >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def calculate_influence_scores(self) -> Dict:
        """
        Calculate current influence scores for each factor
        """
        influence_scores = {}
        
        for category in ['market', 'economic']:
            factors = self.factor_data.get(category, {})
            
            for symbol, data in factors.items():
                try:
                    # Base influence on volatility and recent change
                    volatility_score = min(abs(data.get('volatility', 0)) / 5, 1.0)  # Normalize to 0-1
                    change_score = min(abs(data.get('change_pct', 0)) / 10, 1.0)  # Normalize to 0-1
                    
                    # Get correlation strength
                    correlation_data = self.correlations.get(symbol, {})
                    correlation_strength = correlation_data.get('correlation', 0)
                    correlation_score = abs(correlation_strength)
                    
                    # Combined influence score (0-100)
                    influence_score = (volatility_score * 0.3 + change_score * 0.4 + correlation_score * 0.3) * 100
                    
                    influence_scores[symbol] = {
                        'score': influence_score,
                        'components': {
                            'volatility': volatility_score * 100,
                            'recent_change': change_score * 100,
                            'correlation': correlation_score * 100
                        },
                        'category': data['category'],
                        'name': data['name']
                    }
                    
                except Exception as e:
                    logger.error(f"Error calculating influence for {symbol}: {e}")
        
        return influence_scores
    
    def get_top_influencing_factors(self, limit: int = 5) -> List[Dict]:
        """
        Get top factors currently influencing gold prices
        """
        # Ensure influence scores are calculated
        if not hasattr(self, 'influence_scores') or not self.influence_scores:
            self.influence_scores = self.calculate_influence_scores()
        
        # Sort by influence score
        sorted_factors = sorted(
            self.influence_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        top_factors = []
        for symbol, data in sorted_factors[:limit]:
            factor_info = self.factor_data.get('market', {}).get(symbol, 
                         self.factor_data.get('economic', {}).get(symbol, {}))
            
            if factor_info:
                top_factors.append({
                    'symbol': symbol,
                    'name': data['name'],
                    'category': data['category'],
                    'influence_score': data['score'],
                    'current_value': factor_info.get('current_value'),
                    'change_pct': factor_info.get('change_pct', 0),
                    'correlation': self.correlations.get(symbol, {}).get('correlation', 0),
                    'expected_impact': self._determine_expected_impact(symbol, factor_info)
                })
        
        return top_factors
    
    def _determine_expected_impact(self, symbol: str, factor_data: Dict) -> str:
        """
        Determine expected impact on gold based on factor movement
        """
        change_pct = factor_data.get('change_pct', 0)
        correlation_type = factor_data.get('correlation_type', 'mixed')
        
        if abs(change_pct) < 0.1:
            return 'Neutral'
        
        if correlation_type == 'positive':
            return 'Bullish' if change_pct > 0 else 'Bearish'
        elif correlation_type == 'negative':
            return 'Bearish' if change_pct > 0 else 'Bullish'
        else:
            return 'Mixed'
    
    def generate_macro_summary(self) -> Dict:
        """
        Generate comprehensive macroeconomic summary
        """
        top_factors = self.get_top_influencing_factors(10)
        
        # Categorize impacts
        bullish_factors = [f for f in top_factors if f['expected_impact'] == 'Bullish']
        bearish_factors = [f for f in top_factors if f['expected_impact'] == 'Bearish']
        
        # Calculate overall macro sentiment
        bullish_score = sum(f['influence_score'] for f in bullish_factors)
        bearish_score = sum(f['influence_score'] for f in bearish_factors)
        
        if bullish_score > bearish_score * 1.2:
            overall_sentiment = 'Bullish'
        elif bearish_score > bullish_score * 1.2:
            overall_sentiment = 'Bearish'
        else:
            overall_sentiment = 'Mixed'
        
        return {
            'overall_sentiment': overall_sentiment,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'top_factors': top_factors,
            'bullish_factors': bullish_factors,
            'bearish_factors': bearish_factors,
            'summary': self._generate_narrative_summary(overall_sentiment, top_factors)
        }
    
    def _generate_narrative_summary(self, sentiment: str, top_factors: List[Dict]) -> str:
        """
        Generate narrative summary of macro conditions
        """
        if not top_factors:
            return "Insufficient data to generate macro analysis."
        
        top_factor = top_factors[0]
        
        summary = f"Current macro environment is **{sentiment.lower()}** for gold. "
        summary += f"The most influential factor is **{top_factor['name']}** "
        summary += f"(influence score: {top_factor['influence_score']:.1f}), "
        summary += f"which is currently {top_factor['expected_impact'].lower()} for gold prices. "
        
        if len(top_factors) > 1:
            summary += f"Other key factors include {', '.join([f['name'] for f in top_factors[1:3]])}. "
        
        return summary
    
    def update_all_factors(self, gold_data: pd.Series = None):
        """
        Update all macroeconomic factors and calculate relationships
        """
        logger.info("Updating all macroeconomic factors...")
        
        # Fetch all data
        self.factor_data['market'] = self.fetch_market_factors()
        self.factor_data['economic'] = self.fetch_fred_indicators()
        
        # Calculate correlations if gold data provided
        if gold_data is not None and not gold_data.empty:
            self.correlations = self.calculate_correlations_with_gold(gold_data)
        
        # Calculate influence scores
        self.influence_scores = self.calculate_influence_scores()
        
        logger.info("✓ All macroeconomic factors updated successfully")
        
        return self.generate_macro_summary()

if __name__ == "__main__":
    # Test the macro factors analyzer
    analyzer = MacroFactorsAnalyzer()
    
    # Create sample gold data for testing
    gold_data = pd.Series([2000, 2010, 2005, 2020, 2015] * 10)
    
    summary = analyzer.update_all_factors(gold_data)
    print("Macro Summary:", summary)
