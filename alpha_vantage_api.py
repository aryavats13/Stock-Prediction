import requests
import pandas as pd
from datetime import datetime
import os

ALPHA_VANTAGE_API_KEY = "RNZPXZ6Q9FEFMEHM"  # Updated API key

def get_stock_data(symbol, period="1mo"):
    """
    Fetch stock data from Alpha Vantage API
    
    Parameters:
    symbol (str): Stock symbol (e.g., TSLA, AAPL)
    period (str): Time period for data (1mo, 3mo, etc.)
    
    Returns:
    pandas.DataFrame: DataFrame with stock price data
    """
    # Map period to Alpha Vantage output size
    if period in ["1d", "5d", "1wk", "1mo"]:
        outputsize = "compact"  # Returns the latest 100 data points
    else:
        outputsize = "full"     # Returns up to 20 years of historical data
    
    # Determine function based on period
    if period in ["1d", "5d"]:
        function = "TIME_SERIES_INTRADAY"
        interval = "60min"  # For intraday data
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
    else:
        function = "TIME_SERIES_DAILY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            print(f"Alpha Vantage API Error: {data['Error Message']}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Extract time series data
        if function == "TIME_SERIES_INTRADAY":
            time_series_key = f"Time Series ({interval})"
        else:
            time_series_key = "Time Series (Daily)"
        
        if time_series_key not in data:
            print(f"No time series data found for {symbol}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Convert to DataFrame
        time_series = data[time_series_key]
        df = pd.DataFrame(time_series).T
        
        # Rename columns to match yfinance format
        df.columns = [col.split(". ")[1].capitalize() for col in df.columns]
        df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }, inplace=True)
        
        # Convert string values to float
        for col in df.columns:
            df[col] = df[col].astype(float)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date (most recent last, to match yfinance)
        df.sort_index(inplace=True)
        
        # Limit data points based on period
        if period == "1mo":
            df = df.last('30D')
        elif period == "3mo":
            df = df.last('90D')
        elif period == "6mo":
            df = df.last('180D')
        elif period == "1y":
            df = df.last('365D')
        
        return df
    
    except Exception as e:
        print(f"Error fetching data from Alpha Vantage: {e}")
        return pd.DataFrame()  # Return empty DataFrame

def get_stock_price(symbol):
    """
    Get the latest stock price for a given symbol
    
    Parameters:
    symbol (str): Stock symbol (e.g., TSLA, AAPL)
    
    Returns:
    float: Latest stock price
    """
    df = get_stock_data(symbol, period="1d")
    if not df.empty:
        return df["Close"].iloc[-1]
    return None