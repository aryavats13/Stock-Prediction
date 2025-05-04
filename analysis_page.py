import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from utils import fetch_stock_data, calculate_returns, calculate_technical_indicators

def analysis_page():
    # Custom CSS for modern UI
    st.markdown("""
        <style>
        .analysis-header {
            background-color: var(--bg-secondary);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .chart-container {
            background-color: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            background-color: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        
        .indicator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .indicator-card {
            background-color: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 10px;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltip-text {
            visibility: hidden;
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            text-align: center;
            padding: 5px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Analysis Header
    st.markdown("""
        <div class="analysis-header">
            <h1>Stock Analysis</h1>
            <p>Analyze stocks with advanced technical indicators and AI predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Stock Selection
    col1, col2 = st.columns([2,1])
    with col1:
        stock_symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    with col2:
        timeframe = st.selectbox("Select Timeframe", 
                               ["1D", "5D", "1M", "3M", "6M", "1Y", "2Y", "5Y"],
                               index=3)
    
    if stock_symbol:
        try:
            # Fetch stock data
            data = fetch_stock_data(stock_symbol, period=timeframe)
            
            # Calculate technical indicators
            indicators = calculate_technical_indicators(data)
            
            # Ensure moving averages are calculated correctly
            if 'MA50' not in indicators.columns:
                st.error("Moving Average calculation error: MA50 not found.")
                return
            
            # Display stock metrics
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            st.markdown("""
                <div class="metric-container">
                    <div class="metric-card">
                        <h3>Current Price</h3>
                        <div class="stock-price">
                            ${:.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Price Change</h3>
                        <div class="stock-price {}">
                            {}{:.2f} ({:.2f}%)
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Volume</h3>
                        <div class="stock-price">
                            {:,.0f}
                        </div>
                    </div>
                </div>
            """.format(
                current_price,
                "price-up" if price_change >= 0 else "price-down",
                "+" if price_change >= 0 else "",
                price_change,
                price_change_pct,
                data['Volume'].iloc[-1]
            ), unsafe_allow_html=True)
            
            # Create interactive chart
            fig = make_subplots(rows=2, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.7, 0.3])
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="OHLC"
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in data.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name="Volume",
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=800,
                margin=dict(t=30, b=30),
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Indicators
            st.markdown("### Technical Indicators")
            st.markdown("""
                <div class="indicator-grid">
                    <div class="indicator-card">
                        <h4>RSI (14)</h4>
                        <div class="stock-price">
                            {:.2f}
                        </div>
                        <div class="tooltip">
                            <i class="fas fa-info-circle"></i>
                            <span class="tooltip-text">
                                RSI > 70: Overbought<br>
                                RSI < 30: Oversold
                            </span>
                        </div>
                    </div>
                    <div class="indicator-card">
                        <h4>MACD</h4>
                        <div class="stock-price">
                            {:.2f}
                        </div>
                    </div>
                    <div class="indicator-card">
                        <h4>50-Day MA</h4>
                        <div class="stock-price">
                            {:.2f}
                        </div>
                    </div>
                </div>
            """.format(
                indicators['RSI'].iloc[-1],
                indicators['MACD'].iloc[-1],
                indicators['MA50'].iloc[-1]
            ), unsafe_allow_html=True)
            
            # Educational Content
            st.markdown("""
                <h4 style=\"color: white; font-size: 18px; margin-top: 20px;\">Understanding Candlestick Charts</h4>
                <ul style=\"color: #e0e0e0; font-size: 16px;\">
                    <li>Green candles: Price closed higher than it opened (bullish)</li>
                    <li>Red candles: Price closed lower than it opened (bearish)</li>
                    <li>The body of the candle shows opening and closing prices</li>
                    <li>The wicks (thin lines) show the highest and lowest prices during that period</li>
                </ul>
                
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error fetching data for {stock_symbol}: {str(e)}")

if __name__ == "__main__":
    analysis_page()
