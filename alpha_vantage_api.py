import requests
import pandas as pd
from datetime import datetime
import os
import time
import streamlit as st

# Get the API key from Streamlit secrets
try:
    ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except Exception:
    # Fallback for local development or if secrets not available
    ALPHA_VANTAGE_API_KEY = "FWK038M3HOADVG4T"  # Updated API key

def get_stock_data(symbol, period="1mo", max_retries=3, retry_delay=15):
    """
    Fetch stock data from Alpha Vantage API with retry logic and improved error handling
    
    Parameters:
    symbol (str): Stock symbol (e.g., TSLA, AAPL)
    period (str): Time period for data (1mo, 3mo, etc.)
    max_retries (int): Maximum number of retry attempts
    retry_delay (int): Delay in seconds between retries
    
    Returns:
    pandas.DataFrame: DataFrame with stock price data
    """
    # Map period to Alpha Vantage output size
    if period in ["1d", "5d", "1wk", "1mo"]:
        outputsize = "compact"  # Returns the latest 100 data points
    else:
        outputsize = "full"     # Returns up to 20 years of historical data
    
    # Special handling for market indices which use ^ prefix
    if symbol.startswith('^'):
        print(f"Warning: Market indices like {symbol} may not be supported by the free Alpha Vantage API.")
        # For market indices, use a different endpoint
        function = "TIME_SERIES_DAILY"
        # Convert ^ to %5E for URL encoding
        encoded_symbol = symbol.replace('^', '%5E')
        url = f"https://www.alphavantage.co/query?function={function}&symbol={encoded_symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
    # Determine function based on period for regular stocks
    elif period in ["1d", "5d"]:
        function = "TIME_SERIES_INTRADAY"
        interval = "60min"  # For intraday data
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
    else:
        function = "TIME_SERIES_DAILY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    print(f"Making request to Alpha Vantage API: {url}")
    
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries}")
        
        try:
            response = requests.get(url)
            print(f"Response Status Code: {response.status_code}")
            
            # Check if response is successful
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()
            
            data = response.json()
            
            # Print response keys for debugging
            print(f"Response contains keys: {list(data.keys())}")
            
            # If we get a rate limit message, wait and retry
            if "Note" in data and "call frequency" in data["Note"]:
                print(f"Rate limit reached: {data['Note']}")
                if attempt < max_retries - 1:  # Don't wait on last attempt
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Max retries reached. Returning empty DataFrame.")
                    return pd.DataFrame()
            
            # Check for error messages
            if "Error Message" in data:
                print(f"Alpha Vantage API Error: {data['Error Message']}")
                return pd.DataFrame()
            
            # Check for Information message (usually indicates invalid symbol or API limit)
            if "Information" in data:
                print(f"Alpha Vantage Information: {data['Information']}")
                print("This typically means you've reached your API call limit or there's an issue with your API key.")
                return pd.DataFrame()
            
            # Check for empty response
            if not data:
                print(f"Empty response received for symbol: {symbol}")
                return pd.DataFrame()
            
            # Extract time series data
            if function == "TIME_SERIES_INTRADAY":
                time_series_key = f"Time Series ({interval})"
            else:
                time_series_key = "Time Series (Daily)"
            
            if time_series_key not in data:
                print(f"No time series data found. Available keys: {list(data.keys())}")
                # Check if there's metadata but no time series (sometimes happens)
                if "Meta Data" in data:
                    print("Metadata found but no time series. This might be an API limit issue.")
                return pd.DataFrame()
            
            # Convert to DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame(time_series).T
            
            # Rename columns for consistency
            if function == "TIME_SERIES_DAILY":
                df.columns = [col.split(". ")[1] for col in df.columns]
            
            df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            }, inplace=True)
            
            # Convert string values to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date
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
            
            print(f"Successfully retrieved {len(df)} data points")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Returning empty DataFrame.")
                return pd.DataFrame()

def test_api_key():
    """Test if the Alpha Vantage API key is valid and working"""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
    
    print(f"Testing API key: {ALPHA_VANTAGE_API_KEY[:4]}...{ALPHA_VANTAGE_API_KEY[-4:]}")
    print(f"Request URL: {url}")
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data and "Invalid API call" in data["Error Message"]:
            print("ERROR: Invalid API key")
            return False
        
        if "Information" in data:
            print(f"WARNING: Information message received: {data['Information']}")
            print("This may indicate you've reached your API call limit.")
            return False
        
        # Check if time series data exists
        if "Time Series (Daily)" in data:
            print("SUCCESS: API key is valid and working")
            # Print sample of data received
            first_date = list(data["Time Series (Daily)"].keys())[0]
            print(f"Sample data from {first_date}:")
            print(data["Time Series (Daily)"][first_date])
            return True
            
        print(f"UNKNOWN RESPONSE: Received keys {list(data.keys())}")
        return False
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_api_connection(ticker="IBM"):
    """
    Test the API connection with a specific ticker symbol
    
    Parameters:
    ticker (str): Stock ticker symbol to test
    
    Returns:
    dict: Dictionary with test results
    """
    # Special handling for market indices which use ^ prefix
    if ticker.startswith('^'):
        # For market indices, use a different endpoint
        function = "TIME_SERIES_DAILY"
        # Convert ^ to %5E for URL encoding
        encoded_ticker = ticker.replace('^', '%5E')
        url = f"https://www.alphavantage.co/query?function={function}&symbol={encoded_ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
    else:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
    
    result = {
        "success": False,
        "message": "",
        "api_key_masked": f"{ALPHA_VANTAGE_API_KEY[:4]}...{ALPHA_VANTAGE_API_KEY[-4:]}",
        "data_sample": None
    }
    
    try:
        response = requests.get(url)
        
        # Check if response is successful
        if response.status_code != 200:
            result["message"] = f"API request failed with status code {response.status_code}"
            return result
        
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            result["message"] = f"API Error: {data['Error Message']}"
            return result
        
        # Check for Information message (usually indicates invalid symbol or API limit)
        if "Information" in data:
            result["message"] = f"API Information: {data['Information']}"
            return result
        
        # Check for empty response
        if not data:
            result["message"] = f"Empty response received for ticker: {ticker}"
            return result
        
        # Check for invalid ticker symbol
        if "Meta Data" not in data:
            result["message"] = f"Invalid ticker symbol: {ticker}"
            return result
        
        # Check if time series data exists
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            result["message"] = f"No time series data found for {ticker}"
            return result
        
        # Success - return sample data
        result["success"] = True
        result["message"] = f"Successfully connected to Alpha Vantage API with ticker {ticker}"
        
        # Get first date as sample
        first_date = list(data[time_series_key].keys())[0]
        result["data_sample"] = {
            "date": first_date,
            "data": data[time_series_key][first_date]
        }
        
        return result
        
    except Exception as e:
        result["message"] = f"Exception: {str(e)}"
        return result

def list_popular_symbols():
    """Print a list of popular stock symbols to try"""
    popular_symbols = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Google
        "AMZN",  # Amazon
        "META",  # Meta (Facebook)
        "TSLA",  # Tesla
        "IBM",   # IBM
        "NFLX",  # Netflix
        "NVDA",  # NVIDIA
        "JPM"    # JPMorgan Chase
    ]
    
    print("\nPopular symbols to try:")
    for symbol in popular_symbols:
        print(f"- {symbol}")

if __name__ == "__main__":
    # Test the API key first
    print("=" * 50)
    print("STEP 1: TESTING API KEY")
    print("=" * 50)
    key_valid = test_api_key()
    
    if not key_valid:
        print("\nYour API key might be invalid or has reached its limit.")
        print("Free Alpha Vantage API keys have a limit of 25 calls per day.")
        print("Consider getting a new API key from: https://www.alphavantage.co/support/#api-key")
    
    # Try to get stock data for a few different symbols
    print("\n" + "=" * 50)
    print("STEP 2: TESTING WITH MULTIPLE SYMBOLS")
    print("=" * 50)
    
    test_symbols = ["IBM", "AAPL", "MSFT"]
    
    for symbol in test_symbols:
        print(f"\nTesting with symbol: {symbol}")
        print("-" * 30)
        df = get_stock_data(symbol, period="1mo")
        
        if not df.empty:
            print(f"✓ Successfully retrieved data for {symbol}")
        else:
            print(f"✗ Failed to retrieve data for {symbol}")
    
    # Show list of popular symbols
    list_popular_symbols()
    
    print("\n" + "=" * 50)
    print("TROUBLESHOOTING TIPS:")
    print("=" * 50)
    print("1. If all symbols fail, your API key might be invalid or you've reached your daily limit.")
    print("2. Free Alpha Vantage accounts have a limit of 25 API calls per day.")
    print("3. Some symbols might not be available or might have changed.")
    print("4. Try with different symbols from the popular symbols list.")
    print("5. Get a new API key from: https://www.alphavantage.co/support/#api-key")