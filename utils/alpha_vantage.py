"""
Alpha Vantage API Client for Stock Data and News
"""
import requests
import os
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

class AlphaVantageClient:
    """Client for Alpha Vantage Stock Market API"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_stock_price(self, ticker: str) -> Optional[Dict]:
        """
        Get current stock price
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'RELIANCE.BSE')
        
        Returns:
            dict: Current price data or None
        """
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
                return {
                    "symbol": quote.get("01. symbol", ticker),
                    "price": float(quote.get("05. price", 0)),
                    "change": float(quote.get("09. change", 0)),
                    "change_percent": quote.get("10. change percent", "0%"),
                    "volume": int(quote.get("06. volume", 0)),
                    "latest_trading_day": quote.get("07. latest trading day", ""),
                    "previous_close": float(quote.get("08. previous close", 0))
                }
            
            return None
            
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            return None
    
    def get_historical_data(self, ticker: str, months: int = 6) -> Optional[pd.DataFrame]:
        """
        Get historical stock data (daily)
        
        Args:
            ticker: Stock symbol
            months: Number of months of historical data (default: 6)
        
        Returns:
            DataFrame with historical data or None
        """
        try:
            # Use TIME_SERIES_DAILY for detailed data
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "compact",  # Last 100 days
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Rename columns
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                # Filter to last N months
                cutoff_date = datetime.now() - timedelta(days=months * 30)
                df = df[df.index >= cutoff_date]
                
                return df
            
            return None
            
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return None
    
    def get_stock_overview(self, ticker: str) -> Optional[Dict]:
        """
        Get company overview and fundamentals
        
        Args:
            ticker: Stock symbol
        
        Returns:
            dict: Company information
        """
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and "Symbol" in data:
                return {
                    "symbol": data.get("Symbol", ""),
                    "name": data.get("Name", ""),
                    "sector": data.get("Sector", ""),
                    "industry": data.get("Industry", ""),
                    "market_cap": data.get("MarketCapitalization", ""),
                    "pe_ratio": data.get("PERatio", ""),
                    "dividend_yield": data.get("DividendYield", ""),
                    "52_week_high": data.get("52WeekHigh", ""),
                    "52_week_low": data.get("52WeekLow", ""),
                    "description": data.get("Description", "")[:300] + "..."
                }
            
            return None
            
        except Exception as e:
            print(f"Error fetching overview for {ticker}: {e}")
            return None
    
    def get_news_sentiment(self, tickers: List[str] = None, limit: int = 10) -> List[Dict]:
        """
        Get news and sentiment for stocks
        
        Args:
            tickers: List of stock symbols (optional)
            limit: Maximum number of news articles
        
        Returns:
            list: News articles with sentiment
        """
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": limit
            }
            
            if tickers:
                params["tickers"] = ",".join(tickers)
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "feed" in data:
                news_items = []
                
                for item in data["feed"]:
                    # Extract sentiment
                    sentiment_score = float(item.get("overall_sentiment_score", 0))
                    if sentiment_score > 0.15:
                        sentiment = "Bullish"
                        sentiment_color = "🟢"
                    elif sentiment_score < -0.15:
                        sentiment = "Bearish"
                        sentiment_color = "🔴"
                    else:
                        sentiment = "Neutral"
                        sentiment_color = "🟡"
                    
                    news_items.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "time_published": item.get("time_published", ""),
                        "summary": item.get("summary", "")[:200] + "...",
                        "source": item.get("source", ""),
                        "sentiment": sentiment,
                        "sentiment_color": sentiment_color,
                        "sentiment_score": sentiment_score,
                        "relevant_tickers": [t["ticker"] for t in item.get("ticker_sentiment", [])]
                    })
                
                return news_items[:limit]
            
            return []
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def calculate_returns(self, historical_df: pd.DataFrame, invested_amount: float) -> Dict:
        """
        Calculate returns based on historical data
        
        Args:
            historical_df: Historical price data
            invested_amount: Amount invested
        
        Returns:
            dict: Return metrics
        """
        if historical_df is None or len(historical_df) == 0:
            return {"error": "No data available"}
        
        try:
            first_price = historical_df['close'].iloc[0]
            current_price = historical_df['close'].iloc[-1]
            
            # Calculate shares bought (assuming bought at first recorded price)
            shares = invested_amount / first_price
            current_value = shares * current_price
            
            total_return = current_value - invested_amount
            return_percentage = (total_return / invested_amount) * 100
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = historical_df['close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            # Calculate max drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            return {
                "invested_amount": invested_amount,
                "shares": round(shares, 4),
                "first_price": round(first_price, 2),
                "current_price": round(current_price, 2),
                "current_value": round(current_value, 2),
                "total_return": round(total_return, 2),
                "return_percentage": round(return_percentage, 2),
                "volatility": round(volatility, 2),
                "max_drawdown": round(max_drawdown, 2)
            }
            
        except Exception as e:
            return {"error": str(e)}