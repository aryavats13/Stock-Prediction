import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
import ta

def fetch_stock_data(symbol, period='2y'):
    try:
        # First try with yfinance
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        # If yfinance data is empty, try Alpha Vantage API
        if df.empty:
            # Silently try alternative data source without showing info message
            # Import Alpha Vantage API module
            import alpha_vantage_api as av
            
            # Map yfinance period to Alpha Vantage period
            av_period = "1y"
            if period == "1mo" or period == "1m":
                av_period = "1mo"
            elif period == "3mo" or period == "3m":
                av_period = "3mo"
            elif period == "6mo" or period == "6m":
                av_period = "6mo"
            elif period == "1y":
                av_period = "1y"
            elif period == "2y":
                av_period = "1y"  # Alpha Vantage doesn't have 2y directly
            
            # Try to get data from Alpha Vantage
            df = av.get_stock_data(symbol, period=av_period)
            
            if df.empty:
                # Instead of raising an error, return empty DataFrame
                # The calling function will handle this case
                return pd.DataFrame()
        return df
    except Exception as e:
        # Silently handle errors without showing error messages
        return pd.DataFrame()

def calculate_technical_indicators(df):
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'])
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    return df

def plot_stock_data(df, symbol):
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
    )
    
    # Add Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', 
                            line=dict(color='#AB47BC', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', 
                            line=dict(color='#7E57C2', width=1.5)))
    
    # Add volume bars
    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' 
              for index, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            marker_line_color='rgb(0,0,0)',
            marker_line_width=0.5,
            opacity=0.7,
            yaxis='y2'
        )
    )
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High', 
                            line=dict(color='rgba(236, 64, 122, 0.3)', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low',
                            line=dict(color='rgba(236, 64, 122, 0.3)', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Mid',
                            line=dict(color='rgba(236, 64, 122, 0.8)', dash='dash')))
    
    # Update layout with improved fonts and styling
    fig.update_layout(
        title=dict(
            text=f'{symbol} Stock Analysis - Last 2 Years',
            font=dict(size=24, color='white')
        ),
        yaxis_title=dict(
            text='Stock Price ($)',
            font=dict(size=16, color='white')
        ),
        yaxis2=dict(
            title='Volume',
            titlefont=dict(size=16, color='white'),
            tickfont=dict(size=12, color='white'),
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            title='Date',
            titlefont=dict(size=16, color='white'),
            tickfont=dict(size=12, color='white')
        ),
        yaxis=dict(
            tickfont=dict(size=12, color='white'),
            tickformat='$,.2f'
        ),
        height=800,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            font=dict(size=14, color='white'),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_technical_indicators(df):
    # RSI Plot
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                line=dict(color='#7E57C2', width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a")
    fig_rsi.update_layout(
        title='Relative Strength Index (RSI)',
        height=300,
        template='plotly_dark',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(color='white')
    )
    
    # MACD Plot
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                 line=dict(color='#AB47BC', width=2)))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal Line',
                                 line=dict(color='#26a69a', width=2)))
    fig_macd.update_layout(
        title='Moving Average Convergence Divergence (MACD)',
        height=300,
        template='plotly_dark',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Date',
        yaxis_title='MACD Value',
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(color='white')
    )
    
    return fig_rsi, fig_macd

def create_prophet_model(df):
    # Prepare data for Prophet
    prophet_df = df.reset_index()
    
    # The date column will be the former index, now as a column named 'index' or 'Date'
    # First, identify which column contains the date
    if 'Date' in prophet_df.columns:
        date_col = 'Date'
    else:
        date_col = 'index'  # When reset_index() is called, the index becomes a column named 'index'
    
    # Select only the date column and Close price
    prophet_df = prophet_df[[date_col, 'Close']]
    
    # Remove timezone from dates if present
    if hasattr(prophet_df[date_col].dtype, 'tz'):
        prophet_df[date_col] = prophet_df[date_col].dt.tz_localize(None)
    
    # Rename columns to Prophet's required format
    prophet_df.columns = ['ds', 'y']
    
    # Ensure the date column is datetime type
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Create and fit the model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    # Create future dates for 25 days
    future_dates = model.make_future_dataframe(periods=25)
    forecast = model.predict(future_dates)
    
    return forecast

def stock_page():
    # Set page title with larger font and better spacing
    st.markdown('<div style="padding: 20px; margin-bottom: 30px; text-align: center; background-color: rgba(0,0,0,0.3); border-radius: 10px;">', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size: 38px; color: white; margin-bottom: 10px;">Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="font-size: 22px; color: #9e9e9e; font-weight: 400;">Analyzing: {st.session_state.stock}</h3>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a loading animation while fetching data
    with st.spinner('Loading market data...'):
        # Fetch stock data
        df = fetch_stock_data(st.session_state.stock)
    
    if df.empty:
        # Show a more user-friendly message without error styling
        st.markdown('<div style="padding: 30px; text-align: center; background-color: rgba(0,0,0,0.3); border-radius: 10px; margin: 50px 0;">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #e0e0e0;">We couldn\'t find data for this symbol</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 18px; color: #9e9e9e; margin: 20px 0;">Please try a different stock symbol or check your internet connection.</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 16px; color: #9e9e9e;">Examples: <span style="color: #4fc3f7;">AAPL</span> (Apple), <span style="color: #4fc3f7;">MSFT</span> (Microsoft), <span style="color: #4fc3f7;">RELIANCE.NS</span> (Reliance Industries)</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a back button
        if st.button("← Go Back to Home", key="back_button"):
            st.session_state.page = 'home'
            st.rerun()
        return
    
    # Add a progress bar for data processing steps
    progress_bar = st.progress(0)
    
    # Calculate technical indicators
    progress_bar.progress(25)
    st.markdown('<div style="padding: 5px; background-color: rgba(0,0,0,0.2); border-radius: 5px; margin-bottom: 10px;"><p style="color: #9e9e9e; font-size: 14px;">Calculating technical indicators...</p></div>', unsafe_allow_html=True)
    df = calculate_technical_indicators(df)
    
    # Create Prophet forecast
    progress_bar.progress(50)
    st.markdown('<div style="padding: 5px; background-color: rgba(0,0,0,0.2); border-radius: 5px; margin-bottom: 10px;"><p style="color: #9e9e9e; font-size: 14px;">Building prediction model...</p></div>', unsafe_allow_html=True)
    forecast = create_prophet_model(df)
    
    # Display stock info with error handling
    progress_bar.progress(75)
    st.markdown('<div style="padding: 5px; background-color: rgba(0,0,0,0.2); border-radius: 5px; margin-bottom: 10px;"><p style="color: #9e9e9e; font-size: 14px;">Retrieving company information...</p></div>', unsafe_allow_html=True)
    stock = yf.Ticker(st.session_state.stock)
    
    # Try to get stock info with error handling
    try:
        info = stock.info
    except Exception as e:
        # Create a fallback info dictionary with default values without showing warning
        info = {
            'currentPrice': df['Close'].iloc[-1] if not df.empty else 'N/A',
            'regularMarketChangePercent': 0,
            'marketCap': 0,
            'fiftyTwoWeekHigh': df['High'].max() if not df.empty else 'N/A'
        }
    
    # Complete the progress bar
    progress_bar.progress(100)
    
    # Add a small delay to show the completed progress bar
    import time
    time.sleep(0.5)
    
    # Remove the progress bar
    progress_bar.empty()
    
    # Company Info Section with improved fonts and spacing
    st.markdown('<div style="padding: 20px; background-color: rgba(0,0,0,0.3); border-radius: 10px; margin: 20px 0;">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 26px; color: white; margin-bottom: 20px;">Company Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            current_price = info.get('currentPrice', df['Close'].iloc[-1] if not df.empty else 'N/A')
            price_display = f"${current_price}" if isinstance(current_price, (int, float)) else "N/A"
            percent_change = info.get('regularMarketChangePercent', 0)
            percent_display = f"{percent_change:.2f}%" if isinstance(percent_change, (int, float)) else "0.00%"
            st.metric("Current Price", price_display, percent_display)
            
            # Add tooltip explanation for beginners
            st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">The current trading price of the stock</div>', unsafe_allow_html=True)
        except:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}" if not df.empty else "N/A", "0.00%")
            st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">The current trading price of the stock</div>', unsafe_allow_html=True)
    with col2:
        try:
            market_cap = info.get('marketCap', 0)
            market_cap_display = f"${market_cap/1e9:.2f}B" if isinstance(market_cap, (int, float)) else "N/A"
            st.metric("Market Cap", market_cap_display)
            st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Total market value of the company</div>', unsafe_allow_html=True)
        except:
            st.metric("Market Cap", "N/A")
            st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Total market value of the company</div>', unsafe_allow_html=True)
    with col3:
        try:
            high_52week = info.get('fiftyTwoWeekHigh', df['High'].max() if not df.empty else 'N/A')
            high_display = f"${high_52week}" if isinstance(high_52week, (int, float)) else "N/A"
            st.metric("52 Week High", high_display)
            st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Highest price in the past year</div>', unsafe_allow_html=True)
        except:
            st.metric("52 Week High", f"${df['High'].max():.2f}" if not df.empty else "N/A")
            st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Highest price in the past year</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs with improved fonts and spacing
    st.markdown("""
    <style>
    .stTab {
        font-size: 20px;
        font-weight: bold;
        padding: 10px 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px 5px 0 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📈 Price Analysis", "📊 Technical Indicators", "🔮 Predictions"])
    
    with tabs[0]:  # Price Analysis Tab
        # Add educational header for beginners
        st.markdown('<div style="padding: 15px; background-color: rgba(0,0,0,0.3); border-radius: 10px; margin: 10px 0 25px 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size: 22px; color: white; margin-bottom: 10px;">Price Chart & Volume</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #9e9e9e; font-size: 16px;">This chart shows the stock\'s price movement over time. <strong>Candlesticks</strong> show daily price ranges - green means price went up, red means it went down. <strong>Volume bars</strong> at the bottom show how many shares were traded each day.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the chart
        fig = plot_stock_data(df, st.session_state.stock)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add chart reading tips for beginners
        with st.expander("📚 How to Read This Chart"):
            st.markdown("""
            <div style="padding: 15px; background-color: rgba(0,0,0,0.2); border-radius: 10px;">
                <h4 style="color: white; font-size: 18px;">Understanding Candlestick Charts</h4>
                <ul style="color: #e0e0e0; font-size: 16px;">
                    <li><strong style="color: #26a69a;">Green candles</strong>: Price closed higher than it opened (bullish)</li>
                    <li><strong style="color: #ef5350;">Red candles</strong>: Price closed lower than it opened (bearish)</li>
                    <li>The <strong>body</strong> of the candle shows opening and closing prices</li>
                    <li>The <strong>wicks</strong> (thin lines) show the highest and lowest prices during that period</li>
                </ul>
                
                <h4 style="color: white; font-size: 18px; margin-top: 20px;">Moving Averages</h4>
                <ul style="color: #e0e0e0; font-size: 16px;">
                    <li><strong>MA20</strong> (20-day moving average): Shows short-term trend</li>
                    <li><strong>MA50</strong> (50-day moving average): Shows medium-term trend</li>
                    <li>When shorter MA crosses above longer MA, it\'s often a bullish signal</li>
                    <li>When shorter MA crosses below longer MA, it\'s often a bearish signal</li>
                </ul>
                
                <h4 style="color: white; font-size: 18px; margin-top: 20px;">Volume</h4>
                <ul style="color: #e0e0e0; font-size: 16px;">
                    <li>Higher volume often confirms the strength of a price move</li>
                    <li>Low volume may indicate lack of conviction in the price movement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Statistics with improved fonts and layout
        st.markdown('<div style="padding: 15px; background-color: rgba(0,0,0,0.3); border-radius: 10px; margin: 25px 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size: 22px; color: white; margin-bottom: 15px;">Key Statistics</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #9e9e9e; font-size: 16px; margin-bottom: 20px;">These metrics help you understand the stock\'s valuation, volatility, and trading activity.</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Handle potential formatting errors with try-except
            try:
                pe_value = info.get('trailingPE', 'N/A')
                pe_display = f"{pe_value:.2f}" if isinstance(pe_value, (int, float)) else "N/A"
                st.metric("P/E Ratio", pe_display)
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Price relative to earnings (lower can mean better value)</div>', unsafe_allow_html=True)
            except:
                st.metric("P/E Ratio", "N/A")
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Price relative to earnings (lower can mean better value)</div>', unsafe_allow_html=True)
        with col2:
            try:
                beta_value = info.get('beta', 'N/A')
                beta_display = f"{beta_value:.2f}" if isinstance(beta_value, (int, float)) else "N/A"
                st.metric("Beta", beta_display)
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Volatility compared to market (>1 means more volatile)</div>', unsafe_allow_html=True)
            except:
                st.metric("Beta", "N/A")
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Volatility compared to market (>1 means more volatile)</div>', unsafe_allow_html=True)
        with col3:
            try:
                volume_value = info.get('volume', df['Volume'].iloc[-1] if not df.empty else 'N/A')
                volume_display = f"{volume_value:,}" if isinstance(volume_value, (int, float)) else "N/A"
                st.metric("Volume", volume_display)
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Number of shares traded today</div>', unsafe_allow_html=True)
            except:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,}" if not df.empty else "N/A")
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Number of shares traded today</div>', unsafe_allow_html=True)
        with col4:
            try:
                avg_volume = info.get('averageVolume', df['Volume'].mean() if not df.empty else 'N/A')
                avg_volume_display = f"{avg_volume:,}" if isinstance(avg_volume, (int, float)) else "N/A"
                st.metric("Avg Volume", avg_volume_display)
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Average daily trading volume</div>', unsafe_allow_html=True)
            except:
                st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}" if not df.empty else "N/A")
                st.markdown('<div style="font-size: 12px; color: #9e9e9e; margin-top: 5px;">Average daily trading volume</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:  # Technical Indicators Tab
        # Add educational header for technical analysis
        st.markdown('<div style="padding: 15px; background-color: rgba(0,0,0,0.3); border-radius: 10px; margin: 10px 0 25px 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size: 22px; color: white; margin-bottom: 10px;">Technical Analysis</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #9e9e9e; font-size: 16px;">Technical indicators help identify potential trading opportunities by analyzing price movements and patterns. These tools can help predict future price movements based on historical data.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display RSI with explanation
        st.markdown('<div style="padding: 15px; background-color: rgba(0,0,0,0.2); border-radius: 10px; margin-bottom: 25px;">', unsafe_allow_html=True)
        st.markdown('<h4 style="font-size: 18px; color: white;">Relative Strength Index (RSI)</h4>', unsafe_allow_html=True)
        st.markdown('<p style="color: #9e9e9e; font-size: 15px; margin-bottom: 15px;">RSI measures the speed and change of price movements on a scale of 0-100. Values above 70 suggest the stock may be <span style="color: #ef5350;">overbought</span> (potentially overvalued), while values below 30 suggest it may be <span style="color: #26a69a;">oversold</span> (potentially undervalued).</p>', unsafe_allow_html=True)
        
        # Display RSI chart
        fig_rsi, fig_macd = plot_technical_indicators(df)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Add RSI interpretation guide
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div style="text-align: center; padding: 10px; background-color: rgba(38, 166, 154, 0.2); border-radius: 5px;">', unsafe_allow_html=True)
            st.markdown('<h5 style="color: #26a69a;">RSI Below 30</h5>', unsafe_allow_html=True)
            st.markdown('<p style="color: #e0e0e0; font-size: 14px;">Potentially oversold<br>Consider buying opportunity</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="text-align: center; padding: 10px; background-color: rgba(255, 255, 255, 0.1); border-radius: 5px;">', unsafe_allow_html=True)
            st.markdown('<h5 style="color: #e0e0e0;">RSI Between 30-70</h5>', unsafe_allow_html=True)
            st.markdown('<p style="color: #e0e0e0; font-size: 14px;">Neutral territory<br>No strong signal</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div style="text-align: center; padding: 10px; background-color: rgba(239, 83, 80, 0.2); border-radius: 5px;">', unsafe_allow_html=True)
            st.markdown('<h5 style="color: #ef5350;">RSI Above 70</h5>', unsafe_allow_html=True)
            st.markdown('<p style="color: #e0e0e0; font-size: 14px;">Potentially overbought<br>Consider selling opportunity</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display MACD with explanation
        st.markdown('<div style="padding: 15px; background-color: rgba(0,0,0,0.2); border-radius: 10px; margin: 25px 0;">', unsafe_allow_html=True)
        st.markdown('<h4 style="font-size: 18px; color: white;">Moving Average Convergence Divergence (MACD)</h4>', unsafe_allow_html=True)
        st.markdown('<p style="color: #9e9e9e; font-size: 15px; margin-bottom: 15px;">MACD helps identify changes in the strength, direction, momentum, and duration of a trend. When the MACD line crosses above the signal line, it\'s often a <span style="color: #26a69a;">bullish signal</span>. When it crosses below, it\'s often a <span style="color: #ef5350;">bearish signal</span>.</p>', unsafe_allow_html=True)
        
        # Display MACD chart
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # Add MACD interpretation guide
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div style="text-align: center; padding: 10px; background-color: rgba(38, 166, 154, 0.2); border-radius: 5px;">', unsafe_allow_html=True)
            st.markdown('<h5 style="color: #26a69a;">Bullish Signal</h5>', unsafe_allow_html=True)
            st.markdown('<p style="color: #e0e0e0; font-size: 14px;">MACD line crosses above signal line<br>Potential upward momentum</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="text-align: center; padding: 10px; background-color: rgba(239, 83, 80, 0.2); border-radius: 5px;">', unsafe_allow_html=True)
            st.markdown('<h5 style="color: #ef5350;">Bearish Signal</h5>', unsafe_allow_html=True)
            st.markdown('<p style="color: #e0e0e0; font-size: 14px;">MACD line crosses below signal line<br>Potential downward momentum</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a simple moving average explanation
        with st.expander("📚 Learn More About Technical Indicators"):
            st.markdown('''
            <div style="padding: 15px; background-color: rgba(0,0,0,0.2); border-radius: 10px;">
                <h4 style="color: white; font-size: 18px;">Common Technical Indicators</h4>
                
                <h5 style="color: #4fc3f7; margin-top: 15px;">Moving Averages</h5>
                <p style="color: #e0e0e0; font-size: 15px;">Moving averages smooth out price data to create a single flowing line, making it easier to identify the direction of the trend. A rising moving average indicates an uptrend, while a falling moving average indicates a downtrend.</p>
                
                <h5 style="color: #4fc3f7; margin-top: 15px;">Bollinger Bands</h5>
                <p style="color: #e0e0e0; font-size: 15px;">Bollinger Bands consist of a middle band (simple moving average) with an upper and lower band. These bands widen when volatility increases and contract when volatility decreases. Price reaching the upper band may indicate overbought conditions, while price reaching the lower band may indicate oversold conditions.</p>
                
                <h5 style="color: #4fc3f7; margin-top: 15px;">Volume</h5>
                <p style="color: #e0e0e0; font-size: 15px;">Volume represents the total number of shares traded during a given time period. High volume during price increases suggests strong buying pressure, while high volume during price decreases suggests strong selling pressure.</p>
            </div>
            ''', unsafe_allow_html=True)
    
    with tabs[2]:  # Predictions Tab
        st.markdown('<h3 style="font-size: 24px; color: white;">25-Day Price Prediction</h3>', unsafe_allow_html=True)
        
        # Current price and stats
        last_price = df['Close'].iloc[-1]
        predicted_price = forecast['yhat'].iloc[-1]
        upper_price = forecast['yhat_upper'].iloc[-1]
        lower_price = forecast['yhat_lower'].iloc[-1]
        price_change = ((predicted_price - last_price) / last_price) * 100
        
        # Display predictions in a clean format
        st.markdown("""
        <style>
        .prediction-box {
            background-color: rgba(0,0,0,0.5);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .price-text {
            color: white;
            font-size: 18px;
            margin: 10px 0;
        }
        .highlight {
            color: #00ff00;
            font-weight: bold;
        }
        .warning {
            color: #ff9800;
            font-size: 16px;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Current Price Box
        st.markdown(f"""
        <div class="prediction-box">
            <h4 style="color: white; font-size: 20px;">Current Status</h4>
            <p class="price-text">Current Price: <span class="highlight">${last_price:.2f}</span></p>
            <p class="price-text">Trading Volume: <span class="highlight">{df['Volume'].iloc[-1]:,.0f}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction Box
        st.markdown(f"""
        <div class="prediction-box">
            <h4 style="color: white; font-size: 20px;">25-Day Forecast</h4>
            <p class="price-text">Predicted Price: <span class="highlight">${predicted_price:.2f}</span></p>
            <p class="price-text">Expected Change: <span class="highlight">{price_change:+.2f}%</span></p>
            <p class="price-text">Price Range:</p>
            <ul class="price-text">
                <li>Upper Estimate: ${upper_price:.2f}</li>
                <li>Lower Estimate: ${lower_price:.2f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis Box
        trend = "Upward" if price_change > 0 else "Downward"
        confidence_range = ((upper_price - lower_price) / predicted_price) * 100
        
        st.markdown(f"""
        <div class="prediction-box">
            <h4 style="color: white; font-size: 20px;">Analysis</h4>
            <p class="price-text">• Predicted Trend: <span class="highlight">{trend}</span></p>
            <p class="price-text">• Confidence Range: <span class="highlight">±{confidence_range:.1f}%</span></p>
            <p class="price-text">• Based on historical patterns and market indicators, the stock shows:</p>
            <ul class="price-text">
                <li>{'Strong' if abs(price_change) > 10 else 'Moderate'} {trend.lower()} momentum</li>
                <li>{'High' if confidence_range > 20 else 'Moderate'} price volatility expected</li>
            </ul>
            <p class="warning">Note: These predictions are based on historical data and should not be the sole basis for investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer with improved fonts
    st.markdown("""
    <div style='background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 5px; margin-top: 30px;'>
        <p style='color: white; font-size: 16px; font-family: Arial, sans-serif;'>
            This analysis is based on historical data and technical indicators. 
            Past performance is not indicative of future results. 
            Please conduct your own research before making investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)