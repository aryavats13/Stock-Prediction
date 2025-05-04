import alpha_vantage_api as av

def main():
    print("Testing Alpha Vantage API connection...")
    result = av.test_api_connection()
    
    print(f"API Test Results:")
    print(f"API Key: {result['api_key_masked']}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print()
    
    if result['success']:
        print(f"Sample Data Date: {result['data_sample']['date']}")
        print(f"Sample Close Price: {result['data_sample']['data']['4. close']}")
    
    # Try getting data for a specific stock
    symbol = "AAPL"
    print(f"\nTesting data retrieval for {symbol}...")
    df = av.get_stock_data(symbol, period="1mo")
    
    if not df.empty:
        print(f"Successfully retrieved data for {symbol}")
        print(f"Data shape: {df.shape}")
        print(f"Latest close price: ${df['Close'].iloc[-1]:.2f}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    else:
        print(f"Failed to retrieve data for {symbol}")

if __name__ == "__main__":
    main()