# app.py

import streamlit as st
import yfinance as yf
import re
import random

# =========================== Stock Context ===========================

def get_stock_context(ticker):
    """Gather comprehensive context about a stock to provide to Gemini."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        hist = stock.history(period="1mo")
        if not hist.empty:
            current_price = round(hist['Close'].iloc[-1], 2)
            start_price = hist['Close'].iloc[0]
            percent_change = ((current_price - start_price) / start_price) * 100

            hist_year = stock.history(period="1y")
            week_52_high = round(hist_year['High'].max(), 2) if not hist_year.empty else "N/A"
            week_52_low = round(hist_year['Low'].min(), 2) if not hist_year.empty else "N/A"
        else:
            current_price = "N/A"
            percent_change = "N/A"
            week_52_high = "N/A"
            week_52_low = "N/A"

        # News extraction (max 3)
        news = stock.news[:3] if hasattr(stock, 'news') and stock.news else []
        news_summaries = [{
            "title": item.get('title', 'No title'),
            "publisher": item.get('publisher', 'Unknown'),
            "link": item.get('link', '#')
        } for item in news]

        context = {
            "ticker": ticker.upper(),
            "company_name": info.get('shortName', ticker.upper()),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "current_price": f"${current_price}",
            "monthly_change": f"{percent_change:.2f}%" if percent_change != "N/A" else "N/A",
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "eps": info.get('trailingEps', 'N/A'),
            "dividend_yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A",
            "52_week_high": f"${week_52_high}",
            "52_week_low": f"${week_52_low}",
            "business_summary": info.get('longBusinessSummary', 'No summary available'),
            "recent_news": news_summaries
        }

        return context

    except Exception as e:
        return {"error": f"Error retrieving stock context: {str(e)}"}

# =========================== Utility ===========================

def extract_tickers(message):
    """Extract ticker symbols from a message."""
    potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', message)
    common_words = {"I", "A", "TO", "IN", "IS", "IT", "BE", "AS", "AT", "BY",
                    "OR", "ON", "DO", "IF", "ME", "MY", "WE", "GO", "NO", "SO",
                    "US", "AM"}
    return [ticker for ticker in potential_tickers if ticker not in common_words]

# =========================== Streamlit UI ===========================

def display_chatbot(default_ticker=None):
    """Display the chatbot interface with fallback to rule-based responses."""

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        # Add a welcome message
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": "👋 Hello! I'm your Stock Chat Assistant. Ask me anything about stocks, market trends, or financial terms. You can also ask about specific stocks by mentioning their ticker symbol."
        })

    # Sidebar: Stock Context
    st.sidebar.subheader("📊 Stock Context")
    ticker = st.sidebar.text_input("Enter stock ticker:", default_ticker or "AAPL")

    if ticker:
        with st.spinner(f"Loading context for {ticker}..."):
            context = get_stock_context(ticker)
        if "error" in context:
            st.sidebar.error(context["error"])
            context = None
        else:
            st.sidebar.success(f"Loaded {context['company_name']} ({ticker.upper()})")
            st.sidebar.metric("Current Price", context['current_price'], context['monthly_change'])
    else:
        context = None

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask me about stocks..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check for tickers in prompt
        mentioned_tickers = extract_tickers(prompt)
        if mentioned_tickers and (not ticker or ticker.upper() not in mentioned_tickers):
            with st.spinner(f"Updating context to {mentioned_tickers[0]}..."):
                new_context = get_stock_context(mentioned_tickers[0])
                if "error" not in new_context:
                    context = new_context
                    ticker = mentioned_tickers[0]
                    st.success(f"Context updated to {context['company_name']} ({ticker.upper()})")

        # Generate response using rule-based system
        with st.spinner("Thinking..."):
            response = get_rule_based_response(prompt, context)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

def get_rule_based_response(prompt, context=None):
    """Generate a rule-based response to user queries."""
    
    # Convert prompt to lowercase for easier matching
    prompt_lower = prompt.lower()
    
    # Check if the prompt contains ticker-specific questions
    if context and any(term in prompt_lower for term in ["price", "worth", "value", "cost", "trading at"]):
        return f"{context['company_name']} ({context['ticker']}) is currently trading at ${context['current_price']}. " + \
               f"The stock has {context['monthly_change_description']} in the past month."
    
    # Check for questions about stock recommendations
    if any(term in prompt_lower for term in ["buy", "sell", "invest", "recommendation", "good stock"]):
        disclaimer = "Please note that I cannot provide personalized investment advice. Always do your own research and consider consulting with a financial advisor."
        
        if context:
            analyst_sentiment = random.choice(["generally positive", "mixed", "cautiously optimistic", "neutral", "somewhat bearish"])
            return f"Regarding {context['company_name']} ({context['ticker']}), market analysts have {analyst_sentiment} sentiment. " + \
                   f"The stock is currently trading at ${context['current_price']}. {disclaimer}"
        else:
            return f"When considering stock investments, it's important to look at factors like company fundamentals, industry trends, and your own investment goals. {disclaimer}"
    
    # Check for questions about market trends
    if any(term in prompt_lower for term in ["market trend", "market outlook", "market forecast", "bull market", "bear market"]):
        trends = [
            "Market trends are influenced by economic indicators, geopolitical events, and investor sentiment.",
            "Recent market movements have been characterized by volatility due to inflation concerns and central bank policies.",
            "Many analysts are watching economic data closely to gauge the direction of the market.",
            "Technology and AI-related stocks have been particularly volatile in recent trading sessions."
        ]
        return random.choice(trends) + " Remember that past performance is not indicative of future results."
    
    # Check for questions about financial terms
    financial_terms = {
        "p/e ratio": "Price-to-Earnings (P/E) ratio is a valuation metric that compares a company's stock price to its earnings per share. A high P/E may indicate that investors expect high growth in the future.",
        "eps": "Earnings Per Share (EPS) represents a company's profit divided by its outstanding shares of common stock. It's a key indicator of a company's profitability.",
        "dividend": "A dividend is a payment made by a corporation to its shareholders, usually as a distribution of profits. Not all companies pay dividends.",
        "market cap": "Market Capitalization is the total value of a company's outstanding shares, calculated by multiplying the stock price by the number of shares outstanding.",
        "bear market": "A bear market refers to a market condition where prices are falling and widespread pessimism causes the negative sentiment to be self-sustaining.",
        "bull market": "A bull market is a financial market where prices are rising or expected to rise. It's characterized by optimism and investor confidence."
    }
    
    for term, definition in financial_terms.items():
        if term in prompt_lower:
            return definition
    
    # Default responses for general questions
    general_responses = [
        "I'm here to help with stock information and basic financial questions. For specific stock details, try mentioning a ticker symbol.",
        "Could you provide more details about what you'd like to know? I can help with stock information, market trends, and financial terms.",
        "I can assist with basic stock information and financial concepts. What specific aspect are you interested in learning more about?",
        "For the most accurate and up-to-date information, I recommend checking financial news sources and official company reports."
    ]
    
    return random.choice(general_responses)

# =========================== MAIN ===========================

if __name__ == "__main__":
    display_chatbot()
