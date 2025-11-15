"""
Unified Data Loader for Expense Tracker
Fetches expense data from Google Sheets for ML predictions
"""
import pandas as pd
import streamlit as st
from datetime import datetime
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
from utils.logging_utils import setup_logging
import json
import logging
from typing import Any

log = setup_logging("data_loader")

# Load environment variables
load_dotenv()

@st.cache_resource
def get_google_sheets_service():
    """Cache Google Sheets credentials and service"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                st.secrets['gcp_service_account'],
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
        else:
            # Check if credentials are provided as JSON string in environment
            creds_json = os.getenv('GOOGLE_SHEETS_CREDENTIALS_JSON')
            if creds_json:
                try:
                    creds_info = json.loads(creds_json)
                    creds = service_account.Credentials.from_service_account_info(
                        creds_info,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in GOOGLE_SHEETS_CREDENTIALS_JSON: {e}")
            else:
                # Local development - use file path from .env
                creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
                if not creds_path:
                    raise ValueError("Neither GOOGLE_SHEETS_CREDENTIALS_JSON nor GOOGLE_SHEETS_CREDENTIALS is set")
                if not os.path.exists(creds_path):
                    raise FileNotFoundError(f"Credentials file not found: {creds_path}")
                
                creds = service_account.Credentials.from_service_account_file(
                    creds_path,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
        
        service: Any = build('sheets', 'v4', credentials=creds)
        return service

    except Exception as e:
        logging.exception("Error creating Google Sheets service")
        st.error(f"Failed to create Google Sheets service: {str(e)}")
        raise e

def get_expense_data_from_sheets():
    """
    Fetch expense data from Google Sheets using existing integration
    
    Returns:
        pd.DataFrame: Expense data with columns [date, amount, category]
    """
    try:
        service = get_google_sheets_service()
        SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
        
        log.debug("Fetching expense data from Google Sheets")
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A1:F'
        ).execute()
        
        values = result.get('values', [])
        if not values:
            log.warning("No expense data found in sheet")
            return pd.DataFrame(columns=['date', 'amount', 'category'])
        
        # Convert to DataFrame (skip header row)
        df = pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
        
        if len(df) == 0:
            st.info("💡 No expenses found. Add expenses in the Home page first.")
            return pd.DataFrame(columns=['date', 'amount', 'category'])
        
        # Filter only Expense type transactions
        df = df[df['Type'] == 'Expense'].copy()
        
        if len(df) == 0:
            st.info("💡 No expense transactions found. Add some expenses first.")
            return pd.DataFrame(columns=['date', 'amount', 'category'])
        
        # Standardize column names
        df = df.rename(columns={
            'Date': 'date',
            'Amount': 'amount',
            'Category': 'category'
        })
        
        # Data type conversions
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=['date', 'amount'])
        
        # Filter out negative or zero amounts
        df = df[df['amount'] > 0]
        
        # Remove future dates (data quality check)
        df = df[df['date'] <= datetime.now()]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Log success
        date_range = (df['date'].max() - df['date'].min()).days
        log.info(f"✅ Loaded {len(df)} expenses spanning {date_range} days from Google Sheets")
        
        return df[['date', 'amount', 'category']]
        
    except Exception as e:
        log.error(f"❌ Error loading from Google Sheets: {str(e)}")
        st.error(f"Failed to load data from Google Sheets: {str(e)}")
        return pd.DataFrame(columns=['date', 'amount', 'category'])


def validate_expense_data(df):
    """
    Validate expense data for ML model training
    
    Args:
        df: DataFrame with expense data
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if df is None or len(df) == 0:
        return False, "No expense data available. Add expenses in the Home page first."
    
    if len(df) < 7:
        return False, f"Need at least 7 days of data for ML predictions. Found: {len(df)} records."
    
    # Check for required columns
    required = ['date', 'amount']
    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    # Check date range
    date_range = (df['date'].max() - df['date'].min()).days
    if date_range < 7:
        return False, f"Date range too small: {date_range} days. Need at least 7 days of history."
    
    # Check for data quality issues
    if df['amount'].isna().any():
        return False, "Found missing amounts in data"
    
    if (df['amount'] <= 0).any():
        return False, "Found negative or zero amounts in data"
    
    # Check data freshness
    days_since_last = (datetime.now() - df['date'].max()).days
    if days_since_last > 30:
        message = f"⚠️ Data is {days_since_last} days old. Predictions may be less accurate."
        return True, message
    
    return True, f"✅ Data valid: {len(df)} records spanning {date_range} days"


def get_data_summary(df):
    """
    Get summary statistics of expense data
    
    Args:
        df: DataFrame with expense data
    
    Returns:
        dict: Summary statistics
    """
    if df is None or len(df) == 0:
        return {
            'total_records': 0,
            'date_range': '0 days',
            'total_amount': 0,
            'daily_average': 0,
            'categories': []
        }
    
    date_range_days = (df['date'].max() - df['date'].min()).days
    
    summary = {
        'total_records': len(df),
        'date_range': f"{date_range_days} days",
        'start_date': df['date'].min().strftime('%Y-%m-%d'),
        'end_date': df['date'].max().strftime('%Y-%m-%d'),
        'total_amount': df['amount'].sum(),
        'daily_average': df['amount'].mean(),
        'max_amount': df['amount'].max(),
        'min_amount': df['amount'].min(),
        'categories': df['category'].unique().tolist() if 'category' in df.columns else []
    }
    
    return summary


# Fallback mock data generator (for testing without Google Sheets)
def generate_mock_data(days=60):
    """
    Generate mock expense data for testing
    Only used when Google Sheets is not available
    
    Args:
        days: Number of days of data to generate
    
    Returns:
        pd.DataFrame: Mock expense data
    """
    from datetime import timedelta
    import random
    
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    
    # Generate realistic spending patterns
    amounts = []
    categories = []
    
    for i, date in enumerate(dates):
        # Weekend spending is higher
        is_weekend = date.weekday() >= 5
        base_amount = 800 if is_weekend else 500
        
        # Add some randomness
        amount = base_amount + random.randint(-200, 300)
        amounts.append(max(amount, 100))  # Minimum ₹100
        
        # Distribute categories
        category_pool = [
            'Food',
            'Transportation',
            'Shopping',
            'Other'
        ]
        categories.append(random.choice(category_pool))
    
    df = pd.DataFrame({
        'date': dates,
        'amount': amounts,
        'category': categories
    })
    
    return df.sort_values('date').reset_index(drop=True)


# Main function to be used by ML predictor
def get_expense_data(use_mock=False):
    """
    Main function to get expense data
    Attempts Google Sheets first, falls back to mock if needed
    
    Args:
        use_mock: Force use of mock data (for testing)
    
    Returns:
        pd.DataFrame: Expense data
    """
    if use_mock:
        st.info("📊 Using mock data for testing")
        return generate_mock_data()
    
    # Try Google Sheets first
    df = get_expense_data_from_sheets()
    
    if len(df) == 0:
        st.warning("⚠️ No data from Google Sheets. Using mock data for demonstration.")
        df = generate_mock_data()
    
    return df
