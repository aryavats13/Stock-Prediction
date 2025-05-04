import streamlit as st
import home_page
import stock_page
from bot import display_chatbot
import config

# === MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="Stock Prediction & Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Load custom CSS ===
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.title("📊 Stock Analytics")
st.sidebar.markdown("Analyze stocks and predict future prices")

nav_options = {
    "Home": "🏠 Home",
    "Stock Analysis": "📈 Stock Analysis",
    "Chat Assistant": "💬 Chat Assistant",
    "About": "ℹ️ About"
}

nav = st.sidebar.radio("Navigation", list(nav_options.values()))

# === Routing ===
if nav == nav_options["Home"]:
    home_page.home_page()

elif nav == nav_options["Stock Analysis"]:
    stock_page.app()

elif nav == nav_options["Chat Assistant"]:
    st.title("💬 Stock Chat Assistant")
    st.markdown("""
        Welcome to the Stock Chat Assistant!  
        Ask me anything about stocks, market trends, financial terms,  
        or get help with stock analysis.
    """)
    display_chatbot()

elif nav == nav_options["About"]:
    st.title("ℹ️ About Stock Prediction App")
    st.markdown("""
    ## Stock Prediction & Analysis

    This application provides tools for analyzing stocks and predicting future stock prices.

    ### Features:
    - **Stock Analysis**: View detailed stock information, charts, key metrics, and more  
    - **Price Prediction**: Machine learning-based stock price predictions  
    - **Detailed Guides**: Comprehensive guides for individual stocks  
    - **Chat Assistant**: Ask questions about stocks and get instant answers  

    ### Technologies Used:
    - **Streamlit**: For the web interface  
    - **yfinance**: For retrieving stock data  
    - **TensorFlow/Keras**: For LSTM neural network prediction models  
    - **Pandas & NumPy**: For data manipulation  
    - **Matplotlib**: For data visualization  
    - **BeautifulSoup**: For web scraping stock information  

    ### Disclaimer:
    This application is for educational and informational purposes only.  
    The predictions and analyses provided should not be considered as financial advice.  
    Always conduct your own research before making investment decisions.
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 Stock Analytics App")
