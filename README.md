# Stock Market Prediction Web App

## Overview
This repository contains a web application for predicting stock prices using machine learning models. The application allows users to select a stock, view historical data, analyze it with moving averages and candlestick charts, and predict future prices using trained models. The app is built using Streamlit and incorporates a combination of Keras and Scikit-learn models for price predictions.

## Features
- **Historical Data Visualization**: Display historical stock data in tabular form.
- **Candlestick Charts**: Visualize stock data using candlestick charts for the past 20 days.
- **Moving Averages**: Plot and compare different moving averages (MA50, MA100, MA200) with the actual closing prices.
- **Price Prediction**: Predict stock prices using a hybrid model combining Keras LSTM and Scikit-learn algorithms.
- **Comparison of Predictions**: Compare predicted prices with actual prices for the last 3 days.
- **Next Day Prediction**: Predict the stock price for the next trading day.
- **Analyst Insights**: View analyst recommendations and price targets.
- **Detailed Guides**: Access comprehensive stock guides.
- **Chat Assistant**: Ask questions about stocks and get instant answers.

## Installation & Setup

### Local Development
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Stock-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.streamlit/secrets.toml` file with your API keys:
   ```toml
   ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"
   GEMINI_API_KEY = "your_gemini_api_key"
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Deployment

#### Deploying to Streamlit Cloud
1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
3. Create a new app and connect it to your GitHub repository.
4. In the app settings, add your secrets (API keys) under the "Secrets" section.
5. Deploy the app.

#### Deploying to Heroku
1. Create a `Procfile` with the following content:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. Initialize a git repository (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. Create a Heroku app and deploy:
   ```bash
   heroku create your-app-name
   heroku config:set ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   heroku config:set GEMINI_API_KEY=your_gemini_api_key
   git push heroku main
   ```

## Troubleshooting Deployment Issues

### Analysis and Predictions Not Showing
If your deployed app isn't showing analysis and predictions, check the following:

1. **API Keys**: Ensure your API keys are properly set in the deployment environment's secrets or config vars.

2. **Fallback Mechanism**: The app is designed to use synthetic data if API calls fail. Check the console logs to see if there are API-related errors.

3. **Dependencies**: Make sure all dependencies in requirements.txt are properly installed in the deployment environment.

4. **Memory Limits**: Some deployment platforms have memory limits. If your app is crashing due to memory issues, try reducing the size of data being processed or using more efficient algorithms.

5. **Logs**: Check the deployment platform's logs for any errors or warnings.

### Common Solutions
- Restart the app on the deployment platform
- Verify that your API keys are valid and have not expired
- Check for rate limiting on the Alpha Vantage API (free tier has limitations)
- Ensure your deployment platform supports TensorFlow/Keras if you're using those for predictions
