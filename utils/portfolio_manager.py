"""
Portfolio Management System with JSON Storage
"""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class PortfolioManager:
    """Manage user's stock portfolio with JSON storage"""
    
    def __init__(self, filepath: str = "data/portfolio.json"):
        self.filepath = filepath
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create portfolio file and directory if they don't exist"""
        # Create directory if needed
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # Create file with empty portfolio if it doesn't exist
        if not os.path.exists(self.filepath):
            self._save_portfolio([])
    
    def _save_portfolio(self, portfolio: List[Dict]):
        """Save portfolio to JSON file"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(portfolio, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving portfolio: {e}")
            return False
    
    def _load_portfolio(self) -> List[Dict]:
        """Load portfolio from JSON file"""
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return []
    
    def add_stock(self, ticker: str, invested_amount: float, purchase_date: str = None) -> Dict:
        """
        Add a stock to the portfolio
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.BSE')
            invested_amount: Amount invested in rupees
            purchase_date: Optional purchase date (YYYY-MM-DD)
        
        Returns:
            dict: Result with status and message
        """
        try:
            # Validate inputs
            if not ticker or len(ticker.strip()) == 0:
                return {"status": "error", "message": "Ticker cannot be empty"}
            
            if invested_amount <= 0:
                return {"status": "error", "message": "Invested amount must be positive"}
            
            ticker = ticker.strip().upper()
            
            # Load current portfolio
            portfolio = self._load_portfolio()
            
            # Check if stock already exists
            existing = next((s for s in portfolio if s['ticker'] == ticker), None)
            
            if existing:
                return {
                    "status": "error", 
                    "message": f"{ticker} already exists. Use update instead."
                }
            
            # Create new stock entry
            new_stock = {
                "ticker": ticker,
                "invested_amount": round(invested_amount, 2),
                "purchase_date": purchase_date or datetime.now().strftime("%Y-%m-%d"),
                "added_on": datetime.now().isoformat()
            }
            
            # Add to portfolio
            portfolio.append(new_stock)
            
            # Save
            if self._save_portfolio(portfolio):
                return {
                    "status": "success",
                    "message": f"✅ Added {ticker} with ₹{invested_amount:,.2f}",
                    "stock": new_stock
                }
            else:
                return {"status": "error", "message": "Failed to save portfolio"}
            
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    def get_all_stocks(self) -> List[Dict]:
        """Get all stocks in portfolio"""
        return self._load_portfolio()
    
    def get_stock(self, ticker: str) -> Optional[Dict]:
        """Get a specific stock by ticker"""
        portfolio = self._load_portfolio()
        ticker = ticker.strip().upper()
        return next((s for s in portfolio if s['ticker'] == ticker), None)
    
    def update_stock(self, ticker: str, invested_amount: float) -> Dict:
        """
        Update invested amount for a stock
        
        Args:
            ticker: Stock ticker
            invested_amount: New invested amount
        
        Returns:
            dict: Result with status
        """
        try:
            portfolio = self._load_portfolio()
            ticker = ticker.strip().upper()
            
            stock = next((s for s in portfolio if s['ticker'] == ticker), None)
            
            if not stock:
                return {"status": "error", "message": f"{ticker} not found in portfolio"}
            
            # Update amount
            stock['invested_amount'] = round(invested_amount, 2)
            stock['updated_on'] = datetime.now().isoformat()
            
            if self._save_portfolio(portfolio):
                return {
                    "status": "success",
                    "message": f"✅ Updated {ticker} to ₹{invested_amount:,.2f}"
                }
            else:
                return {"status": "error", "message": "Failed to save changes"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def delete_stock(self, ticker: str) -> Dict:
        """
        Delete a stock from portfolio
        
        Args:
            ticker: Stock ticker to delete
        
        Returns:
            dict: Result with status
        """
        try:
            portfolio = self._load_portfolio()
            ticker = ticker.strip().upper()
            
            # Find and remove stock
            initial_length = len(portfolio)
            portfolio = [s for s in portfolio if s['ticker'] != ticker]
            
            if len(portfolio) == initial_length:
                return {"status": "error", "message": f"{ticker} not found in portfolio"}
            
            if self._save_portfolio(portfolio):
                return {
                    "status": "success",
                    "message": f"✅ Deleted {ticker} from portfolio"
                }
            else:
                return {"status": "error", "message": "Failed to save changes"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_total_invested(self) -> float:
        """Calculate total amount invested across all stocks"""
        portfolio = self._load_portfolio()
        return sum(stock['invested_amount'] for stock in portfolio)
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary statistics of portfolio"""
        portfolio = self._load_portfolio()
        
        if not portfolio:
            return {
                "total_stocks": 0,
                "total_invested": 0,
                "stocks": []
            }
        
        return {
            "total_stocks": len(portfolio),
            "total_invested": self.get_total_invested(),
            "stocks": portfolio,
            "tickers": [s['ticker'] for s in portfolio]
        }
    
    def clear_portfolio(self) -> Dict:
        """Clear all stocks from portfolio"""
        try:
            if self._save_portfolio([]):
                return {"status": "success", "message": "Portfolio cleared"}
            else:
                return {"status": "error", "message": "Failed to clear portfolio"}
        except Exception as e:
            return {"status": "error", "message": str(e)}