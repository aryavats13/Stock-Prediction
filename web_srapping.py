import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import re
import time

def clean_text(text):
    """Clean scraped text by removing extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def get_stock_summary(ticker):
    """Scrape summary information about a stock from Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"Failed to retrieve data for {ticker}. Status code: {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the company description section
        description_div = soup.find('section', {'class': 'quote-sub-section Mt(30px)'})
        if description_div:
            description = description_div.find('p')
            if description:
                return clean_text(description.text)
        
        # If we couldn't find it in the expected place, try the fallback method with YFinance
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'longBusinessSummary' in info:
            return info['longBusinessSummary']
        
        return f"Could not find description for {ticker}"
    
    except Exception as e:
        return f"Error retrieving stock summary: {str(e)}"

def get_analyst_opinions(ticker):
    """Scrape analyst opinions about a stock from Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/analysis"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"Failed to retrieve analyst data for {ticker}. Status code: {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the analyst recommendation section
        recommendation_div = soup.find('div', {'class': 'Mb(10px) Pend(20px) Pstart(20px)'})
        if recommendation_div:
            recommendation_text = recommendation_div.find('span')
            if recommendation_text:
                return clean_text(recommendation_text.text)
        
        # Fallback to YFinance
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'recommendationMean' in info:
            mean = info['recommendationMean']
            key = "Buy" if mean < 2.5 else "Hold" if mean < 3.5 else "Sell"
            return f"Analyst consensus: {key} (Rating: {mean}/5)"
        
        return "Analyst opinions not available"
    
    except Exception as e:
        return f"Error retrieving analyst opinions: {str(e)}"

def get_financial_highlights(ticker):
    """Get key financial metrics for a stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        highlights = {}
        
        # Key financial metrics
        metrics = [
            'marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 
            'returnOnEquity', 'revenueGrowth', 'operatingMargins',
            'ebitdaMargins', 'profitMargins', 'grossMargins'
        ]
        
        for metric in metrics:
            if metric in info:
                value = info[metric]
                
                # Format percentages
                if "Yield" in metric or "Growth" in metric or "Margins" in metric or "OnEquity" in metric:
                    if value is not None:
                        highlights[metric] = f"{value:.2%}"
                # Format market cap
                elif metric == 'marketCap':
                    if value > 1_000_000_000_000:
                        highlights[metric] = f"${value/1_000_000_000_000:.2f}T"
                    elif value > 1_000_000_000:
                        highlights[metric] = f"${value/1_000_000_000:.2f}B"
                    elif value > 1_000_000:
                        highlights[metric] = f"${value/1_000_000:.2f}M"
                    else:
                        highlights[metric] = f"${value:,.0f}"
                # Format PE ratios
                elif "PE" in metric:
                    if value is not None and value > 0:
                        highlights[metric] = f"{value:.2f}"
                    else:
                        highlights[metric] = "N/A"
                else:
                    highlights[metric] = value
        
        # Format the data for display
        formatted_highlights = []
        readable_names = {
            'marketCap': 'Market Cap',
            'trailingPE': 'P/E (TTM)',
            'forwardPE': 'Forward P/E',
            'dividendYield': 'Dividend Yield',
            'returnOnEquity': 'Return on Equity',
            'revenueGrowth': 'Revenue Growth',
            'operatingMargins': 'Operating Margin',
            'ebitdaMargins': 'EBITDA Margin',
            'profitMargins': 'Profit Margin',
            'grossMargins': 'Gross Margin'
        }
        
        for metric, value in highlights.items():
            formatted_highlights.append(f"{readable_names.get(metric, metric)}: {value}")
        
        return "\n".join(formatted_highlights)
    
    except Exception as e:
        return f"Error retrieving financial highlights: {str(e)}"

def get_news_with_summary(ticker, max_articles=5):
    """Get recent news with summaries."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return "No recent news found."
        
        # Limit to max_articles
        news = news[:max_articles]
        
        formatted_news = []
        for i, article in enumerate(news):
            title = article.get('title', 'No title available')
            link = article.get('link', '#')
            publisher = article.get('publisher', 'Unknown')
            published_time = article.get('providerPublishTime', 0)
            
            # Convert timestamp to readable format
            if published_time:
                published_date = pd.to_datetime(published_time, unit='s').strftime('%Y-%m-%d')
            else:
                published_date = 'Unknown date'
            
            summary = article.get('summary', 'No summary available')
            
            article_text = f"### {i+1}. {title}\n"
            article_text += f"**Source**: {publisher} | **Date**: {published_date}\n\n"
            article_text += f"{summary}\n\n"
            
            formatted_news.append(article_text)
        
        return "\n".join(formatted_news)
    
    except Exception as e:
        return f"Error retrieving news: {str(e)}"

def get_comprehensive_stock_guide(ticker):
    """Generate a comprehensive guide for a specific stock."""
    try:
        # Sleep to avoid rate limiting
        time.sleep(1)
        
        # Get basic info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return f"Could not retrieve information for {ticker}"
        
        company_name = info.get('shortName', ticker)
        
        # Build the guide
        guide = [f"# Comprehensive Guide: {company_name} ({ticker})"]
        
        # Company Overview
        guide.append("## Company Overview")
        summary = get_stock_summary(ticker)
        guide.append(summary)
        
        # Basic Information
        guide.append("\n## Basic Information")
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        employees = info.get('fullTimeEmployees', 'N/A')
        website = info.get('website', 'N/A')
        
        guide.append(f"**Sector**: {sector}")
        guide.append(f"**Industry**: {industry}")
        guide.append(f"**Employees**: {employees}")
        guide.append(f"**Website**: {website}")
        
        # Financial Highlights
        guide.append("\n## Financial Highlights")
        highlights = get_financial_highlights(ticker)
        guide.append(highlights)
        
        # Analyst Opinions
        guide.append("\n## Analyst Opinions")
        opinions = get_analyst_opinions(ticker)
        guide.append(opinions)
        
        # Recent News
        guide.append("\n## Recent News")
        news = get_news_with_summary(ticker, 3)
        guide.append(news)
        
        return "\n\n".join(guide)
    
    except Exception as e:
        return f"Error generating stock guide: {str(e)}"