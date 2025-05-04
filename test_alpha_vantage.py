import requests
import pandas as pd
import os
import json

# Use the API key from your code
ALPHA_VANTAGE_API_KEY = "FWK038M3HOADVG4T"  # Updated API key

def test_alpha_vantage_connection():
    """Test the Alpha Vantage API connection and print detailed information"""
    print("Testing Alpha Vantage API connection...")
    
    # Test URL - using a simple request to test the API key
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
    
    print(f"Making request to: {url}")
    
    try:
        # Make the request
        response = requests.get(url)
        print(f"Response Status Code: {response.status_code}")
        
        # Try to parse JSON
        try:
            data = response.json()
            print(f"Response Keys: {list(data.keys())}")
            
            # Print the full response for inspection
            print("\nFull Response:")
            print(json.dumps(data, indent=2))
            
            # Check for common error patterns
            if "Error Message" in data:
                print(f"\nERROR: {data['Error Message']}")
                print("This usually indicates an invalid API key or request format.")
            
            elif "Information" in data:
                print(f"\nINFORMATION: {data['Information']}")
                print("This usually indicates you've reached your API call limit or there's an issue with your API key.")
            
            elif "Note" in data and "call frequency" in data["Note"]:
                print(f"\nRATE LIMIT: {data['Note']}")
                print("You've reached the API call frequency limit.")
            
            elif "Time Series (Daily)" in data:
                print("\nSUCCESS: Received time series data.")
                # Print the first data point
                first_date = list(data["Time Series (Daily)"].keys())[0]
                print(f"First data point ({first_date}):")
                print(json.dumps(data["Time Series (Daily)"][first_date], indent=2))
            
            else:
                print("\nUNKNOWN RESPONSE FORMAT")
        
        except ValueError:
            print("Response is not valid JSON. Raw response:")
            print(response.text[:500])  # Print first 500 chars of the response
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    test_alpha_vantage_connection()