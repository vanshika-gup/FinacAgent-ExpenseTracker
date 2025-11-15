import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import streamlit as st
from typing import Any
import json

@st.cache_resource
def get_sheets_service():
    """Cache Google Sheets service configuration"""
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