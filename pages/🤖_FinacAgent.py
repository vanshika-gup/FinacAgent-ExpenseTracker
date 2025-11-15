"""
🤖 Advanced Conversational AI Financial Agent
Y-Finance for Stock Data, News & Market Insights (NSE/BSE Support)
"""
import streamlit as st
import sys
from pathlib import Path
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
import re
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================
class PortfolioManager:
    """Manage user's stock portfolio with JSON storage"""

    def __init__(self, filepath: str = "data/portfolio.json"):
        self.filepath = filepath
        self._ensure_file_exists()
        logger.info(f"PortfolioManager initialized with file: {filepath}")

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            self._save_portfolio([])

    def _save_portfolio(self, portfolio: List[Dict]):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(portfolio, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            return False

    def _load_portfolio(self) -> List[Dict]:
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return []

    def add_stock(self, ticker: str, invested_amount: float, purchase_date: str = None) -> Dict:
        try:
            if not ticker or len(ticker.strip()) == 0:
                return {"status": "error", "message": "Ticker cannot be empty"}
            if invested_amount <= 0:
                return {"status": "error", "message": "Invested amount must be positive"}

            ticker = ticker.strip().upper()
            portfolio = self._load_portfolio()

            existing = next((s for s in portfolio if s['ticker'] == ticker), None)
            if existing:
                return {"status": "error", "message": f"{ticker} already exists"}

            new_stock = {
                "ticker": ticker,
                "invested_amount": round(invested_amount, 2),
                "purchase_date": purchase_date or datetime.now().strftime("%Y-%m-%d"),
                "added_on": datetime.now().isoformat()
            }

            portfolio.append(new_stock)

            if self._save_portfolio(portfolio):
                logger.info(f"Added stock: {ticker}")
                return {"status": "success", "message": f"✅ Added {ticker}", "stock": new_stock}
            else:
                return {"status": "error", "message": "Failed to save portfolio"}
        except Exception as e:
            logger.error(f"Error adding stock: {e}")
            return {"status": "error", "message": f"Error: {str(e)}"}

    def get_all_stocks(self) -> List[Dict]:
        return self._load_portfolio()

    def get_stock(self, ticker: str) -> Optional[Dict]:
        portfolio = self._load_portfolio()
        ticker = ticker.strip().upper()
        return next((s for s in portfolio if s['ticker'] == ticker), None)

    def delete_stock(self, ticker: str) -> Dict:
        try:
            portfolio = self._load_portfolio()
            ticker = ticker.strip().upper()
            initial_length = len(portfolio)
            portfolio = [s for s in portfolio if s['ticker'] != ticker]

            if len(portfolio) == initial_length:
                return {"status": "error", "message": f"{ticker} not found"}

            if self._save_portfolio(portfolio):
                logger.info(f"Deleted stock: {ticker}")
                return {"status": "success", "message": f"✅ Deleted {ticker}"}
            else:
                return {"status": "error", "message": "Failed to save"}
        except Exception as e:
            logger.error(f"Error deleting stock: {e}")
            return {"status": "error", "message": str(e)}

    def get_portfolio_summary(self) -> Dict:
        portfolio = self._load_portfolio()
        if not portfolio:
            return {"total_stocks": 0, "total_invested": 0, "stocks": [], "tickers": []}

        return {
            "total_stocks": len(portfolio),
            "total_invested": sum(s['invested_amount'] for s in portfolio),
            "stocks": portfolio,
            "tickers": [s['ticker'] for s in portfolio]
        }


# ============================================================================
# Y-FINANCE CLIENT (For Stock Data - NSE/BSE Support)
# ============================================================================
class YFinanceClient:
    """Y-Finance client for stock data with NSE/BSE support"""

    INDIAN_STOCK_MAPPINGS = {
        'RELIANCE': 'RELIANCE.NS', 'RELIANCE INDUSTRIES': 'RELIANCE.NS',
        'TCS': 'TCS.NS', 'TATA CONSULTANCY': 'TCS.NS', 'TATA CONSULTANCY SERVICES': 'TCS.NS',
        'INFOSYS': 'INFY.NS', 'INFY': 'INFY.NS',
        'HDFC': 'HDFCBANK.NS', 'HDFC BANK': 'HDFCBANK.NS', 'HDFCBANK': 'HDFCBANK.NS',
        'ICICI': 'ICICIBANK.NS', 'ICICI BANK': 'ICICIBANK.NS', 'ICICIBANK': 'ICICIBANK.NS',
        'WIPRO': 'WIPRO.NS', 'BHARTI': 'BHARTIARTL.NS', 'BHARTI AIRTEL': 'BHARTIARTL.NS',
        'AIRTEL': 'BHARTIARTL.NS', 'ITC': 'ITC.NS', 'SBI': 'SBIN.NS',
        'STATE BANK': 'SBIN.NS', 'STATE BANK OF INDIA': 'SBIN.NS',
        'TATA MOTORS': 'TATAMOTORS.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
        'TATA STEEL': 'TATASTEEL.NS', 'TATASTEEL': 'TATASTEEL.NS',
        'MARUTI': 'MARUTI.NS', 'MARUTI SUZUKI': 'MARUTI.NS',
        'BAJAJ': 'BAJFINANCE.NS', 'BAJAJ FINANCE': 'BAJFINANCE.NS',
        'ASIAN PAINTS': 'ASIANPAINT.NS', 'ASIANPAINT': 'ASIANPAINT.NS',
        'HUL': 'HINDUNILVR.NS', 'HINDUSTAN UNILEVER': 'HINDUNILVR.NS',
    }

    def __init__(self):
        logger.info("YFinanceClient initialized for stock data")

    def normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol by handling Indian stock mappings and suffixes"""
        ticker_upper = ticker.strip().upper()

        if ticker_upper.endswith('.NS') or ticker_upper.endswith('.BO'):
            return ticker_upper

        if ticker_upper in self.INDIAN_STOCK_MAPPINGS:
            return self.INDIAN_STOCK_MAPPINGS[ticker_upper]

        return ticker_upper
    
    def get_currency_symbol(self, ticker: str) -> str:
        """Get appropriate currency symbol based on ticker"""
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            return '₹'
        return '$'

    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive stock information"""
        try:
            normalized_ticker = self.normalize_ticker(ticker)
            logger.info(f"Fetching stock info for {ticker} (normalized: {normalized_ticker})")
            stock = yf.Ticker(normalized_ticker)
            info = stock.info

            if not info or 'regularMarketPrice' not in info:
                logger.warning(f"No data found for {ticker}")
                return None

            result = {
                'ticker': normalized_ticker,
                'name': info.get('longName', normalized_ticker),
                'price': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('regularMarketOpen', info.get('open', 0)),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'currency': info.get('currency', 'USD')
            }

            if result['previous_close'] > 0:
                change = result['price'] - result['previous_close']
                change_percent = (change / result['previous_close']) * 100
                result['change'] = change
                result['change_percent'] = f"{change_percent:+.2f}%"
            else:
                result['change'] = 0
                result['change_percent'] = "0%"

            logger.info(f"✅ Successfully fetched info for {ticker}: {result['price']}")
            return result
        except Exception as e:
            logger.error(f"❌ Error fetching stock info for {ticker}: {e}")
            return None

    def get_historical_data(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Get historical stock data"""
        try:
            normalized_ticker = self.normalize_ticker(ticker)
            logger.info(f"Fetching historical data for {ticker} ({period})")
            stock = yf.Ticker(normalized_ticker)
            df = stock.history(period=period)

            if df.empty:
                logger.warning(f"No historical data for {ticker}")
                return None

            df.columns = [col.lower() for col in df.columns]
            logger.info(f"✅ Successfully fetched {len(df)} days of data for {ticker}")
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching historical data for {ticker}: {e}")
            return None

    def get_stock_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get news for a specific stock using yfinance"""
        try:
            normalized_ticker = self.normalize_ticker(ticker)
            logger.info(f"Fetching news for {ticker} (normalized: {normalized_ticker})")
            stock = yf.Ticker(normalized_ticker)
            news = stock.news
            
            if not news:
                logger.warning(f"No news found for {ticker}")
                return []
            
            news_list = []
            for article in news[:limit]:
                content = article.get('content', {})
                title = content.get('title')
                
                if not title:
                    continue

                provider = content.get('provider', {})
                canonical_url = content.get('canonicalUrl', {})
                
                news_list.append({
                    'title': title,
                    'publisher': provider.get('displayName', 'Unknown'),
                    'link': canonical_url.get('url', article.get('link', '')),
                    'published': content.get('pubDate', 'N/A')
                })

            logger.info(f"✅ Successfully fetched {len(news_list)} news articles for {ticker}")
            return news_list
        except Exception as e:
            logger.error(f"❌ Error fetching news for {ticker}: {e}")
            return []

    def get_market_news(self, limit: int = 10) -> List[Dict]:
        """Get general market news using major index tickers"""
        try:
            logger.info(f"Fetching general market news (limit: {limit})")
            indices = ['^GSPC', '^DJI', '^IXIC', '^NSEI'] # Added NSEI for Indian market context
            
            all_news = []
            seen_titles = set()
            
            for index in indices:
                try:
                    stock = yf.Ticker(index)
                    news = stock.news
                    
                    if news:
                        for article in news:
                            content = article.get('content', {})
                            title = content.get('title')
                            
                            if not title or title in seen_titles:
                                continue
                            
                            seen_titles.add(title)
                            provider = content.get('provider', {})
                            canonical_url = content.get('canonicalUrl', {})
                            
                            all_news.append({
                                'title': title,
                                'publisher': provider.get('displayName', 'Unknown'),
                                'link': canonical_url.get('url', article.get('link', '')),
                                'published': content.get('pubDate', 'N/A')
                            })
                            
                            if len(all_news) >= limit:
                                break
                except Exception as e:
                    logger.warning(f"Error fetching news from {index}: {e}")
                    continue
                
                if len(all_news) >= limit:
                    break
            
            logger.info(f"✅ Successfully fetched {len(all_news)} market news articles")
            return all_news[:limit]
        except Exception as e:
            logger.error(f"❌ Error fetching market news: {e}")
            return []

    def compare_stocks(self, tickers: List[str]) -> Dict:
        """Compare multiple stocks with parallel data fetching"""
        try:
            if not tickers or len(tickers) < 2:
                return {'error': 'Need at least 2 tickers for comparison'}
            
            logger.info(f"Comparing stocks: {', '.join(tickers)}")
            comparison_data = {}
            
            def fetch_stock_data(ticker: str) -> tuple:
                try:
                    stock_info = self.get_stock_info(ticker)
                    if stock_info:
                        historical_data = self.get_historical_data(ticker, '1y')
                        return (ticker, stock_info, historical_data)
                    return (ticker, None, None)
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
                    return (ticker, None, None)
            
            with ThreadPoolExecutor(max_workers=min(len(tickers), 5)) as executor:
                future_to_ticker = {
                    executor.submit(fetch_stock_data, ticker): ticker 
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker, stock_info, historical_data = future.result()
                    
                    if stock_info:
                        metrics = {
                            'ticker': ticker,
                            'name': stock_info.get('name', ticker),
                            'price': stock_info.get('price', 0),
                            'change': stock_info.get('change', 0),
                            'change_percent': stock_info.get('change_percent', '0%'),
                            'volume': stock_info.get('volume', 0),
                            'market_cap': stock_info.get('market_cap', 0),
                            'pe_ratio': stock_info.get('pe_ratio', 0),
                            'sector': stock_info.get('sector', 'N/A'),
                            'currency': stock_info.get('currency', 'USD')
                        }
                        
                        if historical_data is not None and not historical_data.empty:
                            try:
                                start_price = historical_data['close'].iloc[0]
                                current_price = historical_data['close'].iloc[-1]
                                year_return = ((current_price - start_price) / start_price) * 100
                                metrics['year_return'] = year_return
                                
                                daily_returns = historical_data['close'].pct_change().dropna()
                                if len(daily_returns) > 0:
                                    volatility = daily_returns.std() * (252 ** 0.5) * 100
                                    metrics['volatility'] = volatility
                            except Exception as e:
                                logger.warning(f"Error calculating performance metrics for {ticker}: {e}")
                                metrics['year_return'] = 0
                                metrics['volatility'] = 0
                        
                        comparison_data[ticker] = metrics
            
            if not comparison_data:
                return {'error': 'No data available for any of the provided tickers'}
            
            logger.info(f"✅ Successfully compared {len(comparison_data)} stocks")
            
            return {
                'stocks': comparison_data,
                'comparison_count': len(comparison_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in compare_stocks: {e}")
            return {'error': str(e)}


# ============================================================================
# CHART GENERATOR
# ============================================================================
class ChartGenerator:
    """Generate interactive visualizations for stock data"""
    
    @staticmethod
    def create_single_stock_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
        """Create historical price chart for a single stock."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.1, subplot_titles=(f'{ticker} Price', 'Volume'),
                          row_heights=[0.7, 0.3])

        # Price chart
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Price'),
                      row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(0, 0, 255, 0.3)'),
                      row=2, col=1)

        fig.update_layout(
            title_text=f"{ticker} Historical Data",
            xaxis_rangeslider_visible=False,
            height=500,
            showlegend=False
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig

    @staticmethod
    def create_comparison_chart(comparison_data: Dict) -> go.Figure:
        """Create side-by-side comparison visualization"""
        stocks_data = comparison_data.get('stocks', {})
        
        if not stocks_data or len(stocks_data) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for comparison", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        tickers = list(stocks_data.keys())
        names = [stocks_data[t].get('name', t)[:20] for t in tickers]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Current Price', 'Market Cap', '1-Year Return (%)', 'P/E Ratio'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Price
        prices = [stocks_data[t].get('price', 0) for t in tickers]
        fig.add_trace(
            go.Bar(x=names, y=prices, marker_color=[colors[i % len(colors)] for i in range(len(tickers))],
                   text=[f"₹{p:,.2f}" if stocks_data[t].get('currency') == 'INR' else f"${p:,.2f}" for t, p in zip(tickers, prices)], 
                   textposition='outside', showlegend=False),
            row=1, col=1
        )
        
        # Market Cap
        market_caps = [stocks_data[t].get('market_cap', 0) / 1e9 for t in tickers]
        fig.add_trace(
            go.Bar(x=names, y=market_caps, marker_color=[colors[i % len(colors)] for i in range(len(tickers))],
                   text=[f"{mc:.1f}B" for mc in market_caps], textposition='outside', showlegend=False),
            row=1, col=2
        )
        
        # Year Return
        year_returns = [stocks_data[t].get('year_return', 0) for t in tickers]
        return_colors = ['#28a745' if yr > 0 else '#dc3545' for yr in year_returns]
        fig.add_trace(
            go.Bar(x=names, y=year_returns, marker_color=return_colors,
                   text=[f"{yr:+.1f}%" for yr in year_returns], textposition='outside', showlegend=False),
            row=2, col=1
        )
        
        # P/E Ratio
        pe_ratios = [stocks_data[t].get('pe_ratio', 0) for t in tickers]
        fig.add_trace(
            go.Bar(x=names, y=pe_ratios, marker_color=[colors[i % len(colors)] for i in range(len(tickers))],
                   text=[f"{pe:.1f}" if pe > 0 else "N/A" for pe in pe_ratios], textposition='outside', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            title_text=f"Stock Comparison: {', '.join(tickers)}",
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Market Cap (Billions)", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="P/E Ratio", row=2, col=2)
        
        return fig


# ============================================================================
# ADVANCED CONVERSATIONAL AI AGENT
# ============================================================================
class AdvancedFinancialAgent:
    """Sophisticated conversational AI financial advisor with function calling"""
    
    def __init__(self, portfolio_manager: PortfolioManager, yfinance_client: YFinanceClient):
        api_key = os.getenv('GEMINI_API_KEY', '').strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=api_key)
        
        system_instruction = """You are "FinBot," a friendly, expert AI financial assistant.

        **Your Persona:**
        - You are conversational, insightful, and always eager to help.
        - You use emojis to make the conversation engaging and friendly. 🤖
        - You break down complex financial topics into easy-to-understand explanations.
        - You are proactive. If a user asks about a stock, you might also offer to fetch recent news or compare it to a competitor.

        **Your Core Capabilities (Tools):**
        1.  **Get Stock Info:** You can fetch real-time data for any stock (US, NSE, BSE) like price, market cap, P/E ratio, etc.
        2.  **Get Stock News:** You can find the latest news for one or multiple stocks.
        3.  **Compare Stocks:** You can compare two or more stocks side-by-side. When you do this, a visual chart is AUTOMATICALLY generated and displayed.
        4.  **Show Stock Charts:** You can generate and display a historical price chart for any single stock.
        5.  **Analyze Portfolio:** You can access the user's portfolio to provide a summary or analyze their holdings.

        **How to Interact:**
        - **Be Conversational:** Don't just dump data. Frame it within a helpful conversation. For example, instead of just stating the price, say "RELIANCE.NS is currently trading at ₹X, which is up Y% today! Looks like a positive day for them. 👍"
        - **Think Step-by-Step (Function Chaining):** For complex queries, you must chain your tools together.
          - *User:* "How are the stocks in my portfolio doing compared to INFY.NS?"
          - *Your Thought Process:*
            1. I need to know what's in the portfolio. I'll call `get_portfolio_summary`.
            2. Now I have the portfolio tickers. I'll add 'INFY.NS' to that list.
            3. I will call `compare_stocks` with the combined list of tickers. This will automatically create a chart.
            4. I'll then summarize the comparison in a friendly, easy-to-read message.
        - **Always Offer More:** Be proactive! After showing a stock's info, ask, "Would you like to see a historical price chart for this, or perhaps compare it with another stock?"
        - **Clarify When Needed:** If a user's request is ambiguous (e.g., "Tell me about Tata"), ask for clarification ("Sure! Do you mean Tata Motors (TATAMOTORS.NS) or Tata Steel (TATASTEEL.NS)?")
        
        You are empowered to use your tools to provide the best possible financial insights in a conversational manner. Let's get started! 🚀"""

        self.model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            system_instruction=system_instruction
        )
        
        self.portfolio_manager = portfolio_manager
        self.yfinance_client = yfinance_client
        
        logger.info("✅ AdvancedFinancialAgent initialized with function calling support")
    
    def get_function_definitions(self) -> List[Dict]:
        """Get function definitions for Gemini function calling with corrected UPPERCASE types"""
        return [
            {
                "name": "get_stock_info",
                "description": "Get current stock information including price, change, volume, market cap, and other key metrics.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "ticker": {"type": "STRING", "description": "Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')"}
                    }, 
                    "required": ["ticker"]
                }
            },
            {
                "name": "get_historical_chart",
                "description": "Generates and displays a historical price and volume chart for a single stock over a specified period.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "ticker": {"type": "STRING", "description": "Stock ticker symbol"}, 
                        "period": {"type": "STRING", "description": "Time period (e.g., '1mo', '6mo', '1y')"}
                    }, 
                    "required": ["ticker", "period"]
                }
            },
            {
                "name": "get_stock_news",
                "description": "Get recent news articles for a list of one or more stock tickers.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "tickers": {
                            "type": "ARRAY", 
                            "items": {"type": "STRING"},
                            "description": "List of stock ticker symbols"
                        }
                    }, 
                    "required": ["tickers"]
                }
            },
            {
                "name": "get_market_news",
                "description": "Get general market news from major indices.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "limit": {"type": "INTEGER", "description": "Max number of articles"}
                    },
                    "required": []
                }
            },
            {
                "name": "compare_stocks",
                "description": "Compare multiple stocks side-by-side. This automatically generates a comparison chart.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "tickers": {
                            "type": "ARRAY", 
                            "items": {"type": "STRING"},
                            "description": "List of stock ticker symbols to compare (minimum 2)"
                        }
                    }, 
                    "required": ["tickers"]
                }
            },
            {
                "name": "get_portfolio_summary",
                "description": "Get a summary of the user's current stock portfolio, including tickers and total investment.",
                "parameters": {"type": "OBJECT", "properties": {}},
            }
        ]
    
    def _get_function_map(self) -> Dict:
        """Create mapping of function names to their implementations"""
        return {
            "get_stock_info": self.yfinance_client.get_stock_info,
            "get_historical_chart": self._func_get_historical_chart,
            "get_stock_news": self._func_get_stock_news,
            "get_market_news": self.yfinance_client.get_market_news,
            "compare_stocks": self.yfinance_client.compare_stocks,
            "get_portfolio_summary": self.portfolio_manager.get_portfolio_summary,
        }
    
    def _func_get_historical_chart(self, ticker: str, period: str) -> Dict:
        df = self.yfinance_client.get_historical_data(ticker, period)
        if df is None:
            return {"error": f"Could not fetch historical data for {ticker}."}
        chart = ChartGenerator.create_single_stock_chart(ticker, df)
        return {"chart": chart, "summary": f"Generated historical chart for {ticker} for the period {period}."}

    def _func_get_stock_news(self, tickers: List[str]) -> Dict:
        all_news = {}
        with ThreadPoolExecutor(max_workers=len(tickers)) as executor:
            future_to_ticker = {executor.submit(self.yfinance_client.get_stock_news, ticker): ticker for ticker in tickers}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    news_data = future.result()
                    if news_data:
                        all_news[ticker] = news_data
                except Exception as exc:
                    logger.error(f'{ticker} generated an exception: {exc}')
        return {"news": all_news}


    def chat(self, user_message: str, chat_history: List[Dict]) -> Dict:
        """Multi-turn function calling chat method"""
        try:
            logger.info(f"Processing user message: {user_message}")
            
            gemini_history = []
            for msg in chat_history:
                role = 'user' if msg['role'] == 'user' else 'model'
                content = msg.get('content', '')
                if content:
                    gemini_history.append({'role': role, 'parts': [content]})
            
            chat_session = self.model.start_chat(history=gemini_history)
            tools = self.get_function_definitions()
            
            response = chat_session.send_message(user_message, tools=tools)
            
            function_calls_made = []
            metadata = {'charts': []}
            
            # This loop handles one or more function calls from the model
            while response.candidates[0].content.parts[0].function_call.name:
                function_call = response.candidates[0].content.parts[0].function_call
                function_name = function_call.name
                function_args = dict(function_call.args)
                
                logger.info(f"Function call requested: {function_name} with args: {function_args}")
                function_calls_made.append({'name': function_name, 'arguments': function_args})
                
                function_map = self._get_function_map()
                if function_name in function_map:
                    function_result = function_map[function_name](**function_args)
                else:
                    function_result = {'error': f'Unknown function: {function_name}'}
                
                # Special handling for functions that generate charts
                if function_name == 'compare_stocks' and 'error' not in function_result:
                    chart = ChartGenerator.create_comparison_chart(function_result)
                    metadata['charts'].append(chart)
                
                if function_name == 'get_historical_chart' and 'error' not in function_result:
                    metadata['charts'].append(function_result.get("chart"))
                    # We only send a summary back to the model, not the chart object itself
                    function_result = {"status": "success", "summary": function_result.get("summary")}

                # Send the function result back to the model
                response = chat_session.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_name,
                                response={'result': json.dumps(function_result, default=str)} # Use json.dumps for safety
                            )
                        )]
                    ),
                    tools=tools
                )

            final_response = ""
            try:
                # Safely extract text from the final response by iterating through its parts
                final_response = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            except IndexError:
                # This handles rare cases where the response might be empty
                final_response = "I'm sorry, I had trouble generating a final response."
                logger.error("Could not extract final response text due to empty candidates or parts.")
            
            return {
                'content': final_response,
                'function_calls': function_calls_made,
                'metadata': metadata,
            }
            
        except Exception as e:
            logger.error(f"Error in chat method: {e}", exc_info=True)
            return {
                'content': "I encountered an error processing your request. Please try again.",
                'function_calls': [],
                'metadata': {},
            }

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="AI Financial Assistant", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stock-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def init_services():
    """Initialize all services"""
    try:
        portfolio_mgr = PortfolioManager()
        yfinance_client = YFinanceClient()
        agent = AdvancedFinancialAgent(portfolio_mgr, yfinance_client)
        return {
            'portfolio_mgr': portfolio_mgr,
            'yfinance': yfinance_client,
            'agent': agent
        }
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        st.error(f"❌ Critical Error on Startup: {e}")
        return None

services = init_services()

if services is None:
    st.stop()

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Header
st.markdown('<p class="main-header">🤖 Your AI Financial Assistant</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><em>Y-Finance Data & News • Real-time Market Insights</em></p>", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.header("📊 Dashboard")
    summary = services['portfolio_mgr'].get_portfolio_summary()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stocks", summary['total_stocks'])
    with col2:
        st.metric("Invested", f"₹{summary['total_invested']:,.0f}")
    
    st.divider()
        
    if st.button("🔄 Clear Chat", width="stretch"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")
        st.rerun()
    


# Main content
tab1, tab2 = st.tabs(["💬 Chat with Assistant", "💼 Portfolio Manager"])

with tab1:
    st.subheader("💬 Chat with Assistant")
    
    # Initialize chat with a greeting if history is empty
    if not st.session_state.chat_history:
        if summary['total_stocks'] > 0:
            greeting = f"Hi! 👋 I'm FinBot. I see you have {summary['total_stocks']} stocks in your portfolio worth ₹{summary['total_invested']:,.0f}. How can I help you explore the markets today?"
        else:
            greeting = "Hi! 👋 I'm FinBot, your AI financial assistant. You can ask me to analyze stocks, get market news, or compare investments. What's on your mind?"
        st.session_state.chat_history.append({"role": "assistant", "content": greeting})

    # Display chat history
    for message in st.session_state.chat_history:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            if "function_calls" in message and message["function_calls"]:
                with st.expander("🔧 Agent Activity", expanded=False):
                    for func_call in message["function_calls"]:
                        st.caption(f"Tool Used: `{func_call['name']}` with arguments: `{func_call.get('arguments', {})}`")

            st.markdown(message["content"])

            if "charts" in message and message["charts"]:
                for chart in message["charts"]:
                    if chart:
                        st.plotly_chart(chart, width="stretch")

            if "news" in message and message["news"]:
                for ticker, articles in message["news"].items():
                    st.markdown(f"#### 📰 News for {ticker}")
                    for article in articles[:3]: # Show top 3 news per ticker
                         with st.container(border=True):
                            st.markdown(f"**{article['title']}**")
                            st.caption(f"Source: {article['publisher']} | Published: {article.get('published', 'N/A')}")
                            if article.get('link'):
                                st.markdown(f"[Read full article]({article['link']})")


    # Chat input
    if prompt := st.chat_input("Ask me anything about stocks..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("FinBot is thinking..."):
                response_data = services['agent'].chat(
                    user_message=prompt,
                    chat_history=st.session_state.chat_history[:-1]
                )
            
            # Prepare assistant message for history
            assistant_message = {
                "role": "assistant",
                "content": response_data.get('content', ''),
                "function_calls": response_data.get('function_calls', []),
                "charts": response_data.get('metadata', {}).get('charts', []),
                "news": None # Default to None
            }
            
            # Check for news in function calls and add to message
            news_function_call = next((fc for fc in response_data.get('function_calls', []) if fc['name'] == 'get_stock_news'), None)
            if news_function_call:
                # Re-fetch news to get the displayable content
                tickers = news_function_call['arguments'].get('tickers', [])
                if tickers:
                    news_result = services['agent']._func_get_stock_news(tickers=tickers)
                    assistant_message["news"] = news_result.get("news")

            st.session_state.chat_history.append(assistant_message)
            st.rerun()

with tab2:
    st.subheader("💼 Portfolio Manager")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### ➕ Add New Stock")
        with st.form("add_stock_form"):
            new_ticker = st.text_input("Stock Ticker", placeholder="e.g., AAPL, RELIANCE.NS")
            new_amount = st.number_input("Invested Amount (₹)", min_value=0.0, step=1000.0, format="%.2f")
            new_date = st.date_input("Purchase Date")
            submitted = st.form_submit_button("➕ Add Stock", width="stretch")
            
            if submitted:
                if new_ticker and new_amount > 0:
                    stock_info = services['yfinance'].get_stock_info(new_ticker.upper())
                    if not stock_info:
                        st.error(f"❌ Could not find {new_ticker}. Please check the ticker symbol.")
                    else:
                        result = services['portfolio_mgr'].add_stock(new_ticker.upper(), new_amount, new_date.strftime("%Y-%m-%d"))
                        if result['status'] == 'success':
                            st.success(f"{result['message']} - {stock_info['name']}")
                            st.rerun()
                        else:
                            st.error(result['message'])
                else:
                    st.warning("Please enter a valid ticker and amount.")

    with col_right:
        st.markdown("### 📋 Your Portfolio")
        if summary['stocks']:
            for stock in summary['stocks']:
                with st.container():
                    st.markdown(f"""
                    <div class="stock-card">
                        <h4 style="margin:0;">{stock['ticker']}</h4>
                        <p style="margin:0.5rem 0; font-size:1.1rem;">Invested: ₹{stock['invested_amount']:,.2f}</p>
                        <small>Purchase Date: {stock.get('purchase_date', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🗑️ Remove", key=f"del_{stock['ticker']}", type="secondary"):
                        result = services['portfolio_mgr'].delete_stock(stock['ticker'])
                        st.rerun()
        else:
            st.info("Your portfolio is empty. Add your first stock to get started!")