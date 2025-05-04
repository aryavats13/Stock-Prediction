import requests
import pandas as pd
from datetime import datetime
import os
import time
import streamlit as st
import numpy as np
from datetime import timedelta

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
    try:
        # Map period to Alpha Vantage output size
        output_size = "full" if period in ["6mo", "1y"] else "compact"
        
        # Special handling for market indices which use ^ prefix
        if symbol.startswith('^'):
            # For market indices, use a different endpoint
            function = "TIME_SERIES_DAILY"
            # Convert ^ to %5E for URL encoding
            encoded_symbol = symbol.replace('^', '%5E')
            url = f"https://www.alphavantage.co/query?function={function}&symbol={encoded_symbol}&outputsize={output_size}&apikey={ALPHA_VANTAGE_API_KEY}"
        else:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        # Try to get the data with retries
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                data = response.json()
                
                # Check for error messages
                if "Error Message" in data:
                    raise Exception(f"API Error: {data['Error Message']}")
                
                # Check for Information message (usually indicates API limit)
                if "Information" in data and "API call frequency" in data["Information"]:
                    raise Exception(f"API Limit: {data['Information']}")
                
                # Check if time series data exists
                time_series_key = "Time Series (Daily)"
                if time_series_key not in data:
                    raise Exception(f"No time series data found for {symbol}")
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
                
                # Convert string values to float
                for col in df.columns:
                    df[col] = df[col].astype(float)
                
                # Rename columns
                df.columns = [col.split(". ")[1].capitalize() for col in df.columns]
                
                # Sort by date and filter based on period
                df.index = pd.to_datetime(df.index)
                df = df.sort_index(ascending=True)
                
                # Filter based on period
                if period == "1mo":
                    df = df.last('30D')
                elif period == "3mo":
                    df = df.last('90D')
                elif period == "6mo":
                    df = df.last('180D')
                elif period == "1y":
                    df = df.last('365D')
                
                return df, "alpha_vantage", None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # On final attempt, fall back to synthetic data
                    print(f"Failed to get data from Alpha Vantage after {max_retries} attempts: {str(e)}")
                    break
    except Exception as e:
        print(f"Error accessing Alpha Vantage API: {str(e)}")
    
    # Fallback to synthetic data
    return generate_synthetic_data(symbol, period), "synthetic", "Using synthetic data due to API limitations."

def generate_synthetic_data(symbol, period="1y"):
    """Generate synthetic stock data for demonstration"""
    # Use ticker string to generate a consistent seed
    seed_value = sum(ord(c) for c in symbol)
    np.random.seed(seed_value)
    
    # Generate dates
    end_date = datetime.now()
    if period == "1mo":
        days = 30
    elif period == "3mo":
        days = 90
    elif period == "6mo":
        days = 180
    else:  # Default to 1y
        days = 365
    
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate a random walk with drift
    returns = np.random.normal(0.0005, 0.015, size=len(dates)) 
    
    # Add some cyclicality and initial price based on ticker
    price = 50 + (seed_value % 200)  # Initial price between 50 and 250
    prices = [price]
    
    for ret in returns:
        price = price * (1 + ret)
        prices.append(price)
    
    prices = prices[:-1]  # Remove the extra price
    
    # Create synthetic data
    synthetic_data = pd.DataFrame({
        'Open': prices * np.random.uniform(0.98, 0.995, size=len(prices)),
        'High': prices * np.random.uniform(1.01, 1.03, size=len(prices)),
        'Low': prices * np.random.uniform(0.97, 0.99, size=len(prices)),
        'Close': prices,
        'Volume': np.random.randint(100000, 10000000, size=len(prices))
    }, index=dates)
    
    return synthetic_data

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
        df, source, message = get_stock_data(symbol, period="1mo")
        
        if not df.empty:
            print(f"✓ Successfully retrieved data for {symbol} from {source}")
            if message:
                print(message)
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