"""
Streamlit Dashboard for AI-Powered Gold Trading
Main application interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import os
from datetime import datetime, timedelta
import time
import logging

# Configure page
st.set_page_config(
    page_title="AI Gold Trading Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from data_loader import GoldDataLoader
    from feature_engineer import GoldFeatureEngineer
    from model_trainer import GoldModelTrainer
    from predictor import GoldPredictor
    from updater import GoldModelUpdater
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Load configuration
@st.cache_data
def load_config():
    """Load configuration file"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components"""
    try:
        data_loader = GoldDataLoader()
        feature_engineer = GoldFeatureEngineer()
        model_trainer = GoldModelTrainer()
        predictor = GoldPredictor()
        updater = GoldModelUpdater()
        
        return data_loader, feature_engineer, model_trainer, predictor, updater
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None, None, None

def create_candlestick_chart(df, title="Gold Price Chart"):
    """Create candlestick chart with technical indicators"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Indicators', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Gold Price"
        ),
        row=1, col=1
    )
    
    # Add technical indicators if available
    if 'EMA_Fast' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_Fast'],
                name="EMA Fast",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_Slow'],
                name="EMA Slow",
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name="BB Upper",
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name="BB Lower",
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name="RSI",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name="MACD",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name="MACD Signal",
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        if 'MACD_Histogram' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name="MACD Histogram",
                    marker_color='gray',
                    opacity=0.6
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    return fig

def generate_enhanced_strategy_explanation(comprehensive_rec, price_action, market_context):
    """
    Generate an enhanced, LLM-style strategy explanation
    """
    direction = comprehensive_rec['direction']
    grade = comprehensive_rec['setup_grade']
    confidence = comprehensive_rec.get('trade_validity', {}).get('confidence_level', 'Medium')
    
    # Build comprehensive explanation
    explanation = f"""
### 🎯 **{direction} Setup Analysis - Grade {grade}**

**Market Assessment:**
The current gold market is showing **{market_context['market_sentiment'].lower()}** sentiment with **{market_context['volatility_regime'].lower()}** volatility conditions. 
Our AI model has identified a **{confidence.lower()} confidence** {direction.lower()} opportunity.

**Technical Foundation:**
{comprehensive_rec['technical_summary']}

**Price Action Insights:**
{comprehensive_rec['price_action_summary']}

**Risk Management Strategy:**
- **Account Protection:** Risking only {comprehensive_rec['risk_management']['account_risk_pct']:.1f}% of account (${comprehensive_rec['risk_management']['max_risk_usd']:.2f})
- **Position Sizing:** {comprehensive_rec['position_size']['quantity']} units with {comprehensive_rec['position_size'].get('leverage_used', 1)}x leverage
- **Risk:Reward:** Targeting 1:{comprehensive_rec['risk_management']['risk_reward_ratio']:.1f} ratio for optimal expectancy

**Setup Reasoning:**
{comprehensive_rec['setup_reasoning']}

**Execution Plan:**
1. **Entry:** ${comprehensive_rec['entry_price']:.2f} (current market price)
2. **Stop Loss:** ${comprehensive_rec['stop_loss']:.2f} (risk management level)
3. **Target:** ${comprehensive_rec['target_price']:.2f} (profit objective)
4. **Position Size:** {comprehensive_rec['position_size']['quantity']} units

**Key Considerations:**
- Monitor price action around key levels
- Adjust position size based on market volatility
- Use proper risk management at all times
- Consider market news and economic events

**Confidence Level:** {confidence} - This setup meets our criteria for a **Grade {grade}** trading opportunity.
    """
    
    return explanation

def display_prediction_card(prediction_result):
    """Display prediction result in a card format"""
    
    signal = prediction_result['signal']
    probability = prediction_result['probability']
    confidence = prediction_result['confidence']
    
    # Determine colors based on signal
    if 'BUY' in signal:
        color = 'green'
        icon = '📈'
    elif 'SELL' in signal:
        color = 'red'
        icon = '📉'
    else:
        color = 'gray'
        icon = '➡️'
    
    # Create columns for the prediction card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Trading Signal",
            value=signal,
            help=f"AI prediction based on technical analysis"
        )
    
    with col2:
        st.metric(
            label="📊 Probability",
            value=f"{probability:.1%}",
            help="Confidence level of the prediction"
        )
    
    with col3:
        st.metric(
            label="🔍 Confidence",
            value=confidence,
            help="Overall confidence in the signal"
        )
    
    # Additional info
    if prediction_result['current_price']:
        st.info(f"💰 Current Gold Price: ${prediction_result['current_price']:.2f}")
    
    # Data quality indicator
    quality_color = {
        'live': '🟢',
        'historical': '🟡',
        'unavailable': '🔴'
    }
    
    st.caption(f"Data Quality: {quality_color.get(prediction_result['data_quality'], '🔴')} {prediction_result['data_quality'].title()}")

def display_technical_indicators(indicators):
    """Display technical indicators in a formatted way"""
    
    if not indicators:
        st.warning("No technical indicators available")
        return
    
    st.subheader("📊 Technical Indicators")
    
    # Create columns for indicators
    col1, col2 = st.columns(2)
    
    with col1:
        if 'RSI' in indicators:
            rsi_value = indicators['RSI']
            rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
            st.metric("RSI (14)", f"{rsi_value:.1f}", help=f"Status: {rsi_status}")
        
        if 'MACD' in indicators:
            st.metric("MACD", f"{indicators['MACD']:.4f}")
        
        if 'BB_Position' in indicators:
            bb_pos = indicators['BB_Position']
            bb_status = "Near Upper Band" if bb_pos > 0.8 else "Near Lower Band" if bb_pos < 0.2 else "Middle Range"
            st.metric("Bollinger Band Position", f"{bb_pos:.2f}", help=f"Status: {bb_status}")
    
    with col2:
        if 'EMA_Signal' in indicators:
            ema_signal = indicators['EMA_Signal']
            ema_status = "Bullish" if ema_signal > 0 else "Bearish"
            st.metric("EMA Signal", f"{ema_signal:.2f}", help=f"Trend: {ema_status}")
        
        if 'ATR' in indicators:
            st.metric("ATR (14)", f"{indicators['ATR']:.2f}", help="Average True Range - Volatility measure")
        
        if 'Volatility' in indicators:
            st.metric("Volatility (20)", f"{indicators['Volatility']:.4f}", help="20-period price volatility")

def generate_strategy_explanation(indicators, signal):
    """Generate strategy explanation (placeholder for GPT integration)"""
    
    # This is a simplified rule-based explanation
    # In a full implementation, you would integrate with OpenAI GPT here
    
    explanations = []
    
    if 'RSI' in indicators:
        rsi = indicators['RSI']
        if rsi < 30:
            explanations.append("RSI indicates oversold conditions, suggesting potential mean reversion opportunity.")
        elif rsi > 70:
            explanations.append("RSI shows overbought levels, indicating possible reversal or consolidation.")
    
    if 'BB_Position' in indicators:
        bb_pos = indicators['BB_Position']
        if bb_pos < 0.2:
            explanations.append("Price near lower Bollinger Band suggests potential bounce (mean reversion).")
        elif bb_pos > 0.8:
            explanations.append("Price near upper Bollinger Band indicates potential resistance.")
    
    if 'MACD' in indicators and 'MACD_Signal' in indicators:
        macd = indicators['MACD']
        macd_signal = indicators['MACD_Signal']
        if macd > macd_signal:
            explanations.append("MACD above signal line suggests bullish momentum.")
        else:
            explanations.append("MACD below signal line indicates bearish momentum.")
    
    # Strategy recommendation
    if 'BUY' in signal:
        strategy = "Consider a **breakout strategy** if momentum indicators align, or **mean reversion** if price is oversold."
    elif 'SELL' in signal:
        strategy = "Consider **short-term selling** or **profit-taking** if holding long positions."
    else:
        strategy = "**Wait and watch** - mixed signals suggest staying on the sidelines."
    
    if explanations:
        return " ".join(explanations) + f" {strategy}"
    else:
        return f"Limited indicator data available. {strategy}"

def main():
    """Main Streamlit application"""
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader, feature_engineer, model_trainer, predictor, updater = initialize_components()
    
    if not all([data_loader, feature_engineer, model_trainer, predictor, updater]):
        st.error("Failed to initialize components. Please check your setup.")
        return
    
    # Sidebar
    st.sidebar.title("🥇 Gold Trading AI")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["📊 Dashboard", "🎯 Advanced Trading", "🌍 Macro Factors", "📈 Live Prediction", "🔧 Model Management", "📋 System Status"]
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 5 minutes")
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Main content based on selected page
    if page == "📊 Dashboard":
        st.title("🥇 AI-Powered Gold Trading Dashboard")
        st.markdown("Real-time gold price analysis with machine learning predictions")
        
        # Get live prediction
        with st.spinner("Getting live prediction..."):
            prediction_result = predictor.get_live_prediction()
        
        # Display prediction
        st.subheader("🎯 Current Prediction")
        display_prediction_card(prediction_result)
        
        # Display technical indicators
        if prediction_result['indicators']:
            display_technical_indicators(prediction_result['indicators'])
        
        # Strategy explanation
        st.subheader("💡 Strategy Insight")
        explanation = generate_strategy_explanation(
            prediction_result['indicators'], 
            prediction_result['signal']
        )
        st.markdown(explanation)
        
        # Load and display chart
        st.subheader("📈 Price Chart with Technical Indicators")
        
        with st.spinner("Loading chart data..."):
            try:
                # Get recent data for chart
                live_data = data_loader.fetch_live_data(days_back=7)
                
                if not live_data.empty:
                    # Add technical indicators
                    chart_data = feature_engineer.add_technical_indicators(live_data)
                    
                    # Create and display chart
                    fig = create_candlestick_chart(chart_data.tail(200))  # Last 200 candles
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Unable to load chart data")
                    
            except Exception as e:
                st.error(f"Error loading chart: {e}")
        
        # Market sentiment
        sentiment = predictor.get_market_sentiment()
        sentiment_colors = {
            'BULLISH': '🟢',
            'BEARISH': '🔴',
            'NEUTRAL': '🟡'
        }
        st.info(f"📊 Market Sentiment: {sentiment_colors.get(sentiment, '🟡')} {sentiment}")
    
    elif page == "🎯 Advanced Trading":
        st.title("🎯 Advanced Trading Analysis & Risk Management")
        st.markdown("**Professional-grade trading recommendations with risk management and position sizing**")
        
        # Get comprehensive trading recommendation
        with st.spinner("Analyzing market conditions and generating trading recommendation..."):
            try:
                comprehensive_rec = predictor.get_comprehensive_trading_recommendation()
            except Exception as e:
                st.error(f"Error getting comprehensive recommendation: {e}")
                comprehensive_rec = predictor._get_fallback_comprehensive_recommendation()
        
        # Display setup grade prominently
        grade_colors = {
            'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'
        }
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            grade = comprehensive_rec['setup_grade']
            st.markdown(f"### {grade_colors.get(grade, '⚪')} Setup Grade: **{grade}** ({comprehensive_rec['setup_quality']})", 
                       unsafe_allow_html=True)
        
        # Main trading recommendation
        st.subheader("📋 Trading Recommendation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            direction_color = "🟢" if comprehensive_rec['direction'] == 'LONG' else "🔴" if comprehensive_rec['direction'] == 'SHORT' else "⚪"
            st.metric(
                label="🎯 Direction",
                value=f"{direction_color} {comprehensive_rec['direction']}",
                help="Recommended trading direction"
            )
        
        with col2:
            if comprehensive_rec['current_price']:
                st.metric(
                    label="💰 Entry Price",
                    value=f"${comprehensive_rec['entry_price']:.2f}",
                    help="Recommended entry price"
                )
        
        with col3:
            if comprehensive_rec['stop_loss']:
                st.metric(
                    label="🛡️ Stop Loss",
                    value=f"${comprehensive_rec['stop_loss']:.2f}",
                    help="Risk management stop loss level"
                )
        
        with col4:
            if comprehensive_rec['target_price']:
                st.metric(
                    label="🎯 Target",
                    value=f"${comprehensive_rec['target_price']:.2f}",
                    help="Profit target level"
                )
        
        # Risk Management Section
        st.subheader("⚖️ Risk Management & Position Sizing")
        
        risk_mgmt = comprehensive_rec['risk_management']
        position_info = comprehensive_rec['position_size']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Risk Metrics**")
            st.info(f"**Max Risk:** ${risk_mgmt['max_risk_usd']:.2f} ({risk_mgmt['account_risk_pct']:.1f}% of account)")
            st.info(f"**Potential Profit:** ${risk_mgmt['potential_profit_usd']:.2f}")
            st.info(f"**Risk:Reward Ratio:** 1:{risk_mgmt['risk_reward_ratio']:.1f}")
            
            # Position sizing details
            st.markdown("**📈 Position Details**")
            if position_info['quantity'] > 0:
                st.success(f"**Recommended Quantity:** {position_info['quantity']} units")
                st.info(f"**Leveraged Position:** ${position_info['leveraged_position_usd']:,.2f}")
                st.info(f"**Leverage Used:** {position_info['leverage_used']}x")
            else:
                st.warning("No position recommended due to poor setup quality")
        
        with col2:
            st.markdown("**🎯 Setup Analysis**")
            st.info(f"**Setup Reasoning:** {comprehensive_rec['setup_reasoning']}")
            st.info(f"**Technical Summary:** {comprehensive_rec['technical_summary']}")
            st.info(f"**Price Action:** {comprehensive_rec['price_action_summary']}")
        
        # Trade Validity Check
        trade_validity = comprehensive_rec['trade_validity']
        
        if trade_validity['is_valid']:
            st.success(f"✅ **{trade_validity['recommendation']}** - {trade_validity['confidence_level']} Confidence")
        else:
            st.error(f"❌ **{trade_validity['recommendation']}** - Issues: {', '.join(trade_validity['issues'])}")
        
        # Price Action Analysis
        st.subheader("🕯️ Advanced Price Action Analysis")
        
        price_action = comprehensive_rec.get('price_action_analysis', {})
        
        if price_action:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Candlestick Patterns**")
                patterns = [
                    ("Bullish Candle", price_action.get('bullish_candle', False)),
                    ("Bearish Candle", price_action.get('bearish_candle', False)),
                    ("Doji", price_action.get('doji', False)),
                    ("Hammer", price_action.get('hammer', False)),
                    ("Shooting Star", price_action.get('shooting_star', False))
                ]
                
                for pattern, detected in patterns:
                    if detected:
                        st.success(f"✅ {pattern}")
                    else:
                        st.info(f"⚪ {pattern}")
            
            with col2:
                st.markdown("**Engulfing Patterns**")
                if price_action.get('bullish_engulfing'):
                    st.success("✅ Bullish Engulfing")
                else:
                    st.info("⚪ Bullish Engulfing")
                
                if price_action.get('bearish_engulfing'):
                    st.success("✅ Bearish Engulfing")
                else:
                    st.info("⚪ Bearish Engulfing")
            
            with col3:
                st.markdown("**Price Action Strength**")
                strength = price_action.get('price_action_strength', 0)
                if strength > 0.5:
                    st.success(f"🟢 Strong Bullish ({strength:.2f})")
                elif strength < -0.5:
                    st.error(f"🔴 Strong Bearish ({strength:.2f})")
                else:
                    st.info(f"🟡 Neutral ({strength:.2f})")
                
                candle_type = price_action.get('current_candle_type', 'Unknown')
                st.info(f"**Current Candle:** {candle_type}")
        
        # Market Context
        st.subheader("🌍 Market Context")
        
        market_context = comprehensive_rec['market_context']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment = market_context['market_sentiment']
            sentiment_colors = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡'}
            st.info(f"**Market Sentiment:** {sentiment_colors.get(sentiment, '🟡')} {sentiment}")
        
        with col2:
            volatility = market_context['volatility_regime']
            vol_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            st.info(f"**Volatility Regime:** {vol_colors.get(volatility, '⚪')} {volatility}")
        
        with col3:
            data_quality = market_context['data_quality']
            quality_colors = {'live': '🟢', 'historical': '🟡', 'unavailable': '🔴'}
            st.info(f"**Data Quality:** {quality_colors.get(data_quality, '🔴')} {data_quality.title()}")
        
        # Enhanced Strategy Explanation with LLM-style analysis
        st.subheader("🧠 AI Strategy Explanation")
        
        strategy_explanation = generate_enhanced_strategy_explanation(
            comprehensive_rec, price_action, market_context
        )
        
        st.markdown(strategy_explanation)
        
        # Trading Journal Entry Template
        st.subheader("📝 Trading Journal Template")
        
        with st.expander("Click to generate trading journal entry"):
            journal_entry = f"""
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Symbol:** Gold (GC=F)
**Setup Grade:** {comprehensive_rec['setup_grade']} ({comprehensive_rec['setup_quality']})
**Direction:** {comprehensive_rec['direction']}
**Entry:** ${comprehensive_rec['entry_price']:.2f}
**Stop Loss:** ${comprehensive_rec['stop_loss']:.2f}
**Target:** ${comprehensive_rec['target_price']:.2f}
**Risk:** ${risk_mgmt['max_risk_usd']:.2f}
**R:R Ratio:** 1:{risk_mgmt['risk_reward_ratio']:.1f}
**Position Size:** {position_info['quantity']} units

**Analysis:**
{comprehensive_rec['setup_reasoning']}

**Technical Conditions:**
{comprehensive_rec['technical_summary']}

**Price Action:**
{comprehensive_rec['price_action_summary']}

**Market Context:**
Sentiment: {market_context['market_sentiment']}
Volatility: {market_context['volatility_regime']}

**Trade Decision:** {trade_validity['recommendation']}
**Confidence:** {trade_validity['confidence_level']}
            """
            
            st.code(journal_entry, language='markdown')
    
    elif page == "🌍 Macro Factors":
        st.title("🌍 Macroeconomic Factors Analysis")
        st.markdown("**Real-time monitoring of quantitative factors influencing gold prices**")
        
        # Get macro analysis
        with st.spinner("Analyzing macroeconomic factors and their impact on gold..."):
            try:
                macro_analysis = predictor.get_macro_analysis()
            except Exception as e:
                st.error(f"Error getting macro analysis: {e}")
                macro_analysis = {
                    'overall_sentiment': 'Unknown',
                    'top_factors': [],
                    'summary': 'Macro analysis unavailable'
                }
        
        # Overall Macro Sentiment
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            sentiment = macro_analysis.get('overall_sentiment', 'Unknown')
            sentiment_colors = {
                'Bullish': '🟢',
                'Bearish': '🔴', 
                'Mixed': '🟡',
                'Unknown': '⚪'
            }
            st.markdown(f"### {sentiment_colors.get(sentiment, '⚪')} Overall Macro Sentiment: **{sentiment}**")
        
        # Sentiment Scores
        bullish_score = macro_analysis.get('bullish_score', 0)
        bearish_score = macro_analysis.get('bearish_score', 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="🟢 Bullish Factors Score",
                value=f"{bullish_score:.1f}",
                help="Combined influence score of bullish factors"
            )
        
        with col2:
            st.metric(
                label="🔴 Bearish Factors Score",
                value=f"{bearish_score:.1f}",
                help="Combined influence score of bearish factors"
            )
        
        with col3:
            net_score = bullish_score - bearish_score
            st.metric(
                label="⚖️ Net Sentiment Score",
                value=f"{net_score:+.1f}",
                delta=f"{'Bullish' if net_score > 0 else 'Bearish' if net_score < 0 else 'Neutral'} Bias",
                help="Net macro sentiment (Bullish - Bearish)"
            )
        
        # Top Influencing Factors
        st.subheader("📊 Top Influencing Factors")
        
        top_factors = macro_analysis.get('top_factors', [])
        
        if top_factors:
            # Create a comprehensive factors table
            factors_data = []
            for factor in top_factors[:10]:  # Show top 10
                impact_color = {
                    'Bullish': '🟢',
                    'Bearish': '🔴',
                    'Neutral': '🟡',
                    'Mixed': '🟠'
                }.get(factor.get('expected_impact', 'Neutral'), '⚪')
                
                factors_data.append({
                    'Factor': factor.get('name', 'Unknown'),
                    'Category': factor.get('category', 'N/A'),
                    'Current Value': f"{factor.get('current_value', 0):.2f}" if factor.get('current_value') else 'N/A',
                    'Change %': f"{factor.get('change_pct', 0):+.2f}%",
                    'Correlation': f"{factor.get('correlation', 0):.3f}",
                    'Influence Score': f"{factor.get('influence_score', 0):.1f}",
                    'Expected Impact': f"{impact_color} {factor.get('expected_impact', 'Neutral')}"
                })
            
            factors_df = pd.DataFrame(factors_data)
            st.dataframe(factors_df, use_container_width=True, hide_index=True)
            
            # Detailed factor analysis by category
            st.subheader("📈 Factor Analysis by Category")
            
            # Group factors by category
            categories = {}
            for factor in top_factors:
                category = factor.get('category', 'Other')
                if category not in categories:
                    categories[category] = []
                categories[category].append(factor)
            
            # Display each category
            for category, factors in categories.items():
                with st.expander(f"📊 {category} Factors ({len(factors)} factors)"):
                    
                    for factor in factors:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**{factor.get('name', 'Unknown')}**")
                            st.write(f"Symbol: {factor.get('symbol', 'N/A')}")
                        
                        with col2:
                            current_val = factor.get('current_value')
                            if current_val is not None:
                                st.metric(
                                    label="Current Value",
                                    value=f"{current_val:.2f}",
                                    delta=f"{factor.get('change_pct', 0):+.2f}%"
                                )
                            else:
                                st.write("Value: N/A")
                        
                        with col3:
                            correlation = factor.get('correlation', 0)
                            corr_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"
                            st.write(f"**Correlation:** {correlation:.3f}")
                            st.write(f"**Strength:** {corr_strength}")
                        
                        with col4:
                            influence = factor.get('influence_score', 0)
                            impact = factor.get('expected_impact', 'Neutral')
                            impact_color = {
                                'Bullish': '🟢',
                                'Bearish': '🔴',
                                'Neutral': '🟡',
                                'Mixed': '🟠'
                            }.get(impact, '⚪')
                            
                            st.write(f"**Influence:** {influence:.1f}")
                            st.write(f"**Impact:** {impact_color} {impact}")
                        
                        st.divider()
            
            # Macro Summary Narrative
            st.subheader("🧠 AI Macro Analysis")
            
            macro_summary = macro_analysis.get('summary', 'No macro summary available.')
            st.markdown(f"**Current Assessment:** {macro_summary}")
            
            # Enhanced macro insights
            bullish_factors = macro_analysis.get('bullish_factors', [])
            bearish_factors = macro_analysis.get('bearish_factors', [])
            
            if bullish_factors or bearish_factors:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Bullish Factors:**")
                    if bullish_factors:
                        for factor in bullish_factors[:5]:  # Top 5
                            st.success(f"✅ {factor.get('name', 'Unknown')} ({factor.get('influence_score', 0):.1f})")
                    else:
                        st.info("No significant bullish factors identified")
                
                with col2:
                    st.markdown("**🔴 Bearish Factors:**")
                    if bearish_factors:
                        for factor in bearish_factors[:5]:  # Top 5
                            st.error(f"❌ {factor.get('name', 'Unknown')} ({factor.get('influence_score', 0):.1f})")
                    else:
                        st.info("No significant bearish factors identified")
            
            # Correlation Heatmap (if we have enough data)
            if len(top_factors) >= 5:
                st.subheader("🔥 Factor Correlation Heatmap")
                
                # Create correlation matrix for visualization
                factor_names = [f.get('name', 'Unknown')[:15] + '...' if len(f.get('name', '')) > 15 else f.get('name', 'Unknown') for f in top_factors[:8]]
                correlations = [f.get('correlation', 0) for f in top_factors[:8]]
                
                # Create a simple correlation visualization
                fig = go.Figure(data=go.Bar(
                    x=factor_names,
                    y=correlations,
                    marker_color=['green' if c > 0 else 'red' for c in correlations],
                    text=[f"{c:.3f}" for c in correlations],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Gold Price Correlations with Key Factors",
                    xaxis_title="Factors",
                    yaxis_title="Correlation with Gold",
                    height=400,
                    xaxis_tickangle=-45
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            # Data Quality and Update Info
            st.subheader("ℹ️ Data Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                last_updated = macro_analysis.get('last_updated', 'Unknown')
                if last_updated != 'Unknown':
                    try:
                        update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        formatted_time = update_time.strftime('%Y-%m-%d %H:%M:%S')
                        st.info(f"**Last Updated:** {formatted_time}")
                    except:
                        st.info(f"**Last Updated:** {last_updated}")
                else:
                    st.info("**Last Updated:** Unknown")
            
            with col2:
                total_factors = len(top_factors)
                st.info(f"**Factors Analyzed:** {total_factors}")
            
            with col3:
                if 'error' in macro_analysis:
                    st.warning("⚠️ Some data unavailable")
                else:
                    st.success("✅ Data quality: Good")
            
            # Manual refresh button
            if st.button("🔄 Refresh Macro Analysis", key="refresh_macro"):
                st.rerun()
        
        else:
            st.warning("⚠️ No macro factors data available. Please check your API keys and internet connection.")
            
            # Show configuration help
            with st.expander("📋 Configuration Help"):
                st.markdown("""
                **To enable macro factors analysis:**
                
                1. **FRED API Key**: Get a free API key from [FRED Economic Data](https://fred.stlouisfed.org/docs/api/api_key.html)
                2. **Update config.yaml**: Add your FRED API key to the `api_keys.fred_key` field
                3. **Internet Connection**: Ensure you have internet access for real-time data
                
                **Supported Data Sources:**
                - **Market Data**: yfinance (DXY, Treasury yields, currencies, commodities)
                - **Economic Data**: FRED API (inflation, employment, GDP, Fed rates)
                
                **Key Factors Monitored:**
                - 🏛️ **Monetary Policy**: Fed Funds Rate, Treasury yields
                - 💱 **Currency**: USD Index (DXY), major forex pairs
                - 📊 **Economic**: Inflation (CPI), employment, GDP growth
                - 📈 **Market Sentiment**: VIX, S&P 500, risk-on/risk-off
                - 🛢️ **Commodities**: Oil, silver, copper prices
                """)
    
    elif page == "📈 Live Prediction":
        st.title("📈 Live Prediction Analysis")
        
        # Get detailed prediction
        prediction_result = predictor.get_live_prediction()
        
        # Display detailed prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Prediction")
            display_prediction_card(prediction_result)
        
        with col2:
            st.subheader("Prediction History")
            
            # Get prediction history
            history = predictor.get_prediction_history(days=3)
            
            if history:
                history_df = pd.DataFrame(history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                
                # Display recent predictions
                st.dataframe(
                    history_df[['timestamp', 'price', 'signal', 'probability']].tail(10),
                    use_container_width=True
                )
                
                # Plot probability over time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['probability'],
                    mode='lines+markers',
                    name='Prediction Probability'
                ))
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title="Prediction Probability Over Time",
                    yaxis_title="Probability",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No prediction history available")
        
        # Accuracy analysis
        st.subheader("📊 Recent Accuracy Analysis")
        
        accuracy_result = predictor.analyze_prediction_accuracy(days=30)
        
        if accuracy_result:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy_result['accuracy']:.1%}")
            
            with col2:
                st.metric("Precision", f"{accuracy_result['precision']:.1%}")
            
            with col3:
                st.metric("Recall", f"{accuracy_result['recall']:.1%}")
            
            with col4:
                st.metric("Total Predictions", accuracy_result['total_predictions'])
            
            st.info(f"Analysis based on last {accuracy_result['analysis_period_days']} days")
        else:
            st.warning("Unable to calculate accuracy metrics")
    
    elif page == "🔧 Model Management":
        st.title("🔧 Model Management")
        
        # Model information
        st.subheader("📋 Current Model Info")
        
        model_info = model_trainer.get_model_info()
        if model_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Training Date:** {model_info.get('training_date', 'Unknown')}")
                st.info(f"**Features:** {model_info.get('n_features', 'Unknown')}")
            
            with col2:
                st.info(f"**Training Months:** {model_info.get('training_months', 'Unknown')}")
                st.info(f"**Model Type:** XGBoost Classifier")
        else:
            st.warning("No model information available")
        
        # Manual operations
        st.subheader("🔧 Manual Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Update Data", help="Fetch latest market data"):
                with st.spinner("Updating data..."):
                    success = updater.update_historical_data()
                    if success:
                        st.success("Data updated successfully!")
                    else:
                        st.warning("No new data to update")
        
        with col2:
            if st.button("🔄 Retrain Model", help="Retrain model with latest data"):
                with st.spinner("Retraining model... This may take a few minutes."):
                    success = updater.force_retrain_now()
                    if success:
                        st.success("Model retrained successfully!")
                        st.cache_resource.clear()  # Clear cached components
                    else:
                        st.error("Model retraining failed")
        
        with col3:
            if st.button("🏃 Full Update", help="Update data and retrain model"):
                with st.spinner("Running full update..."):
                    result = updater.run_update_now()
                    
                    if result['data_updated']:
                        st.success("✅ Data updated")
                    
                    if result['model_retrained']:
                        st.success("✅ Model retrained")
                        st.cache_resource.clear()
                    
                    if result['errors']:
                        for error in result['errors']:
                            st.error(f"❌ {error}")
    
    elif page == "📋 System Status":
        st.title("📋 System Status")
        
        # Update status
        st.subheader("🔄 Update Status")
        
        status = updater.get_update_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Last Update:** {status['last_update'] or 'Never'}")
        
        with col2:
            st.info(f"**Last Retrain:** {status['last_retrain'] or 'Never'}")
        
        # Recent updates
        if status['recent_updates']:
            st.subheader("📝 Recent Update Log")
            
            updates_df = pd.DataFrame(status['recent_updates'])
            st.dataframe(updates_df, use_container_width=True)
        
        # File status
        st.subheader("📁 File Status")
        
        files_to_check = [
            ('Historical Data', config.get('data', {}).get('historical_file', 'data/gold_30min.csv')),
            ('Model File', config.get('model', {}).get('model_file', 'saved_models/gold_model_xgb.pkl')),
            ('Feature File', config.get('model', {}).get('feature_file', 'saved_models/feature_columns.pkl'))
        ]
        
        for file_name, file_path in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                st.success(f"✅ {file_name}: {file_size:,} bytes (Modified: {file_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                st.error(f"❌ {file_name}: File not found")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🥇 Gold Trading AI Dashboard**")
    st.sidebar.markdown("Powered by XGBoost & Technical Analysis")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
