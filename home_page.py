import streamlit as st

def home_page():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Stock Market Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your intelligent companion for stock market analysis and predictions</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Welcome section with project information
    st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
    st.markdown('<h2>Welcome to Your Stock Analysis Tool</h2>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">This platform uses advanced algorithms and machine learning to analyze stock market data and provide valuable insights for your investment decisions. Our AI-powered prediction model helps you understand market trends and make informed choices.</p>', unsafe_allow_html=True)
    
    # Key features in the welcome section
    st.markdown('<div class="welcome-features">', unsafe_allow_html=True)
    st.markdown('<h3>Key Features:</h3>', unsafe_allow_html=True)
    st.markdown('<ul class="feature-list-welcome">', unsafe_allow_html=True)
    st.markdown('<li><span class="feature-highlight">Real-time Analysis:</span> Get up-to-date information on stock performance</li>', unsafe_allow_html=True)
    st.markdown('<li><span class="feature-highlight">Technical Indicators:</span> Access RSI, MACD, and other key indicators</li>', unsafe_allow_html=True)
    st.markdown('<li><span class="feature-highlight">AI Predictions:</span> View 25-day forecasts based on machine learning models</li>', unsafe_allow_html=True)
    st.markdown('<li><span class="feature-highlight">Multiple Markets:</span> Analyze US stocks, Indian stocks, and cryptocurrencies</li>', unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Start analyzing button
    st.markdown('<div class="start-button-container">', unsafe_allow_html=True)
    if st.button("Start Analyzing Now", key="start_analyzing_button", help="Click to begin your stock analysis journey"):
        st.session_state.page = 'analysis'
        st.session_state.stock = "AAPL"  # Default to Apple as a starting point
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market Selection Tabs
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    tabs = st.tabs(["📈 Stocks", "🪙 Cryptocurrencies"])
    
    with tabs[0]:  # Stocks Tab
        st.markdown('<div class="market-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="market-card us-market">', unsafe_allow_html=True)
            st.markdown('<h2>US Market</h2>', unsafe_allow_html=True)
            
            us_stocks = [
                "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V",
                "PG", "UNH", "MA", "HD", "DIS", "BAC", "ADBE", "CRM", "NFLX", "CMCSA"
            ]
            
            selected_us_stock = st.selectbox("Select US Stock", us_stocks, key="us_stock_selector")
            if st.button("Analyze US Stock", key="us_analyze_button"):
                st.session_state.stock = selected_us_stock
                st.session_state.page = 'analysis'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="market-card indian-market">', unsafe_allow_html=True)
            st.markdown('<h2>Indian Market</h2>', unsafe_allow_html=True)
            
            indian_stocks = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
                "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "WIPRO.NS",
                "HCLTECH.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "TITAN.NS", "ULTRACEMCO.NS"
            ]
            
            selected_indian_stock = st.selectbox("Select Indian Stock", indian_stocks, key="indian_stock_selector")
            if st.button("Analyze Indian Stock", key="indian_analyze_button"):
                st.session_state.stock = selected_indian_stock
                st.session_state.page = 'analysis'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:  # Crypto Tab
        st.markdown('<div class="market-card crypto-market">', unsafe_allow_html=True)
        st.markdown('<h2>Cryptocurrency Market</h2>', unsafe_allow_html=True)
        
        cryptocurrencies = [
            "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "XRP-USD", "SOL-USD", 
            "ADA-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "DOT-USD", "LINK-USD",
            "SHIB-USD", "LTC-USD", "UNI-USD", "XLM-USD", "ATOM-USD", "BCH-USD",
            "FIL-USD", "ETC-USD"
        ]
        
        col1, col2 = st.columns([2,1])
        with col1:
            selected_crypto = st.selectbox("Select Cryptocurrency", cryptocurrencies, key="crypto_selector")
            if st.button("Analyze Cryptocurrency", key="crypto_analyze_button"):
                st.session_state.stock = selected_crypto
                st.session_state.page = 'analysis'
                st.rerun()
        
        with col2:
            st.markdown('''
            <div class="feature-list">
                <h3>Market Features</h3>
                <ul>
                    <li>24/7 Trading</li>
                    <li>High Volatility</li>
                    <li>Global Market</li>
                    <li>Emerging Tech</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Platform Features Section
    st.markdown('<div class="features-section">', unsafe_allow_html=True)
    st.markdown('<h2>Platform Features</h2>', unsafe_allow_html=True)
    
    feature_cols = st.columns(4)
    
    features = [
        {"icon": "📊", "title": "Real-Time Data"},
        {"icon": "📈", "title": "Technical Analysis"},
        {"icon": "🤖", "title": "AI Predictions"},
        {"icon": "📱", "title": "Mobile Friendly"}
    ]
    
    for col, feature in zip(feature_cols, features):
        with col:
            st.markdown(f'''
            <div class="feature-card">
                <div class="feature-icon">{feature['icon']}</div>
                <h3>{feature['title']}</h3>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market Insights Section
    st.markdown('<div class="insights-section">', unsafe_allow_html=True)
    st.markdown('<h2>Market Insights</h2>', unsafe_allow_html=True)
    
    insight_cols = st.columns(3)
    insights = [
        {
            "title": "Technical Analysis",
            "items": ["Moving Averages", "RSI Indicator", "MACD Analysis", "Volume Analysis"]
        },
        {
            "title": "AI Predictions",
            "items": ["25-Day Forecast", "Trend Analysis", "Price Targets", "Risk Assessment"]
        },
        {
            "title": "Market Data",
            "items": ["Real-time Prices", "Historical Data", "Volume Metrics", "Price Alerts"]
        }
    ]
    
    for col, insight in zip(insight_cols, insights):
        with col:
            st.markdown(f'''
            <div class="insight-card">
                <h3>{insight['title']}</h3>
                <ul>
                    {"".join(f'<li>{item}</li>' for item in insight['items'])}
                </ul>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown('''
    <div class="disclaimer">
        <p>This platform provides analysis based on historical data and market trends. 
           Investment decisions should be made with careful consideration of risks involved.</p>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container
