from logging import Logger
from typing import Any
import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sys
from dateutil import parser
import re
from config.constants import TRANSACTION_TYPES, CATEGORIES
from services.google_sheets import get_sheets_service
from utils.logging_utils import setup_logging
from streamlit_mic_recorder import mic_recorder

log: Logger = setup_logging("expense_tracker")

# Load environment variables
load_dotenv()
log.info("✨ Environment variables loaded")

st.set_page_config(layout='wide')

# Configure Gemini AI
@st.cache_resource
def get_gemini_model() -> Any:
    """Cache Gemini AI configuration"""
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model: Any = genai.GenerativeModel('models/gemini-2.5-flash')
        log.info("🤖 Gemini AI configured successfully")
        return model
    except Exception as e:
        log.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        raise


# Replace the direct configuration with cached versions
try:
    model = get_gemini_model()
    service = get_sheets_service()
    SHEET_ID: str | None = os.getenv('GOOGLE_SHEET_ID')
    log.info("📊 Google Sheets API connected successfully")
except Exception as e:
    log.error(f"❌ Failed to connect to Google Sheets: {str(e)}")
    log.error(f"❌ Failed to initialize services: {str(e)}")
    sys.exit(1)


@st.cache_data(ttl=300)
def get_categories() -> dict[str, dict[str, list[str]]]:
    """Cache the categories dictionary to prevent reloading"""
    return CATEGORIES

@st.cache_data
def get_transaction_types() -> list[str]:
    """Cache the transaction types to prevent reloading"""
    return TRANSACTION_TYPES

def init_session_state() -> None:
    """
    Initialize Streamlit session state variables with default values.
    Sets up necessary state variables for the application.
    """
    defaults: dict[str, Any] = {
        'messages': [],
        'save_clicked': False,
        'current_amount': None,
        'current_type': None,
        'current_category': None,
        'current_subcategory': None,
        'form_submitted': False,
        'show_analytics': False,
        'current_transaction': None,
        'conversation_history': [],
        'pending_transaction_data': {},
        'awaiting_user_input': False,
        'last_processed_audio_bytes': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio to text using Google's Gemini AI.

    Args:
        audio_bytes: Audio data in bytes

    Returns:
        str: Transcribed text
    """
    try:
        log.info("🎤 Starting audio transcription")

        # Upload audio file to Gemini
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        try:
            # Upload the audio file
            audio_file = genai.upload_file(temp_audio_path)
            log.debug(f"Audio file uploaded: {audio_file.name}")

            # Create a prompt for transcription
            prompt = """
            Please transcribe this audio accurately. Only return the transcribed text,
            nothing else. If you cannot understand the audio, return 'TRANSCRIPTION_FAILED'.
            """

            # Generate transcription
            response = model.generate_content([prompt, audio_file])
            transcription = response.text.strip()

            # Clean up
            genai.delete_file(audio_file.name)
            os.unlink(temp_audio_path)

            if transcription == 'TRANSCRIPTION_FAILED':
                log.warning("⚠️ Audio transcription failed")
                return ""

            log.info(f"✅ Audio transcribed successfully: {transcription}")
            return transcription

        except Exception as e:
            log.error(f"❌ Transcription error: {str(e)}")
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return ""

    except Exception as e:
        log.error(f"❌ Failed to transcribe audio: {str(e)}")
        return ""

def parse_date_from_text(text: str) -> datetime:
    """
    Extract and parse date from input text.

    Args:
        text (str): Input text containing date information

    Returns:
        str: Parsed date in YYYY-MM-DD format
    """
    current_date: datetime = datetime.now()
    try:
        text = text.lower()

        relative_dates: dict[str, datetime] = {
            'today': current_date,
            'yesterday': current_date - timedelta(days=1),
            'tomorrow': current_date + timedelta(days=1),
            'day before yesterday': current_date - timedelta(days=2),
        }

        for phrase, date in relative_dates.items():
            if phrase in text:
                return date

        last_pattern: str = r'last (\d+) (day|week|month)s?'
        match: re.Match[str] | None = re.search(last_pattern, text)
        if match:
            number: int = int(match.group(1))
            unit: str | Any = match.group(2)
            if unit == 'day':
                return current_date - timedelta(days=number)
            elif unit == 'week':
                return current_date - timedelta(weeks=number)
            elif unit == 'month':
                return current_date - timedelta(days=number * 30)

        next_pattern = r'next (\d+) (day|week|month)s?'
        match = re.search(next_pattern, text)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            if unit == 'day':
                return current_date + timedelta(days=number)
            elif unit == 'week':
                return current_date + timedelta(weeks=number)
            elif unit == 'month':
                return current_date + timedelta(days=number * 30)

        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        match = re.search(date_pattern, text)
        if match:
            return parser.parse(match.group())
        words: list[str] = text.split()
        for i in range(len(words)-2):
            possible_date: str = ' '.join(words[i:i+3])
            try:
                return parser.parse(possible_date)
            except Exception as e:
                log.error(f"❌ Failed to parse date from text: {str(e)}")
                continue

        return current_date

    except Exception as e:
        log.warning(f"Failed to parse date from text, using current date. Error: {str(e)}")
        return current_date

def test_sheet_access() -> bool:
    """
    Test Google Sheets API connection.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        test_values: list[list[str]] = [['TEST', 'TEST', 'TEST', 'TEST', 'TEST', 'TEST']]
        result: Any = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Expenses',
            valueInputOption='RAW',
            body={'values': test_values}
        ).execute()

        updated_range:str = result['updates']['updatedRange']

        service.spreadsheets().values().clear(
            spreadsheetId=SHEET_ID,
            range=updated_range,
            body={}
        ).execute()

        log.info("✅ Sheet access test successful")
        return True
    except Exception as e:
        log.error(f"❌ Sheet access test failed: {str(e)}")
        return False

# def initialize_sheet() -> None:
#     try:
#         sheet_metadata: Any = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute()
#         sheets: list[Any] = sheet_metadata.get('sheets', '')
#         existing_sheets: set[Any] = {s.get("properties", {}).get("title") for s in sheets}

#         if 'Expenses' not in existing_sheets:
#             log.info("Creating new Expenses sheet...")
#             body: dict[str, Any] = {
#                 'requests': [{
#                     'addSheet': {
#                         'properties': {
#                             'title': 'Expenses'
#                         }
#                     }
#                 }]
#             }
#             service.spreadsheets().batchUpdate(
#                 spreadsheetId=SHEET_ID,
#                 body=body
#             ).execute()

#             # Added 'emotion_state' to the headers
#             headers: list[list[str]] = [['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description', 'emotion_state']]
#             service.spreadsheets().values().update(
#                 spreadsheetId=SHEET_ID,
#                 range='Expenses!A1:G1', # Range extended to column G
#                 valueInputOption='RAW',
#                 body={'values': headers}
#             ).execute()

#         if 'Pending' not in existing_sheets:
#             log.info("Creating new Pending sheet...")
#             body: dict[str, Any] = {
#                 'requests': [{
#                     'addSheet': {
#                         'properties': {
#                             'title': 'Pending'
#                         }
#                     }
#                 }]
#             }
#             service.spreadsheets().batchUpdate(
#                 spreadsheetId=SHEET_ID,
#                 body=body
#             ).execute()

#             headers: list[list[str]] = [['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status']]
#             service.spreadsheets().values().update(
#                 spreadsheetId=SHEET_ID,
#                 range='Pending!A1:G1',
#                 valueInputOption='RAW',
#                 body={'values': headers}
#             ).execute()

#         if not test_sheet_access():
#             raise Exception("Failed to verify sheet access")

#         log.info("✨ Sheets initialized and verified")
#     except Exception as e:
#         log.error(f"❌ Failed to initialize sheets: {str(e)}")
#         raise

def add_transaction_to_sheet(date: str, amount: float, trans_type: str,
                           category: str, subcategory: str, description: str, emotion_state: str = "Neutral") -> bool:
    """
    Add a new transaction to Google Sheet.

    Args:
        date (str): Transaction date in YYYY-MM-DD format
        amount (float): Transaction amount
        trans_type (str): Type of transaction (Income/Expense)
        category (str): Transaction category
        subcategory (str): Transaction subcategory
        description (str): Transaction description
        emotion_state (str): The emotional category of the spending

    Returns:
        bool: True if transaction added successfully, False otherwise
    """
    try:
        log.info(f"Starting transaction save: {date}, {amount}, {trans_type}, {category}, {subcategory}, {description}, {emotion_state}")

        date_str:Any = date
        amount_str: str = str(float(amount))
        # Added emotion_state to the values list
        values: list[list[str]] = [[str(date_str), amount_str, trans_type, category, subcategory, description, emotion_state]]

        result: Any = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Expenses',
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': values}
        ).execute()

        log.info(f"✅ Transaction saved successfully: {result}")
        return True

    except Exception as e:
        log.error(f"❌ Failed to save transaction: {str(e)}")
        return False

@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
def get_transactions_data() -> pd.DataFrame:
    """
    Fetch and process all transactions from Google Sheet.

    Returns:
        pd.DataFrame: Processed transactions data
    """
    try:
        log.debug("Fetching transactions data from Google Sheets")
        result: Any = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A1:G' # Range extended to column G
        ).execute()

        values: list[list[str]] = result.get('values', [])
        columns = ['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description', 'emotion_state']
        if not values or len(values) < 1:
            log.warning("No transaction data found in sheet")
            return pd.DataFrame(columns=columns)

        # Pad rows that are shorter than the number of columns to prevent errors
        num_columns = len(columns)
        padded_values = []
        for row in values[1:]:
            padded_values.append(row + [''] * (num_columns - len(row)))

        log.info(f"📈 Retrieved {len(values)-1} transaction records")
        return pd.DataFrame(padded_values, columns=columns)
    except Exception as e:
        log.error(f"❌ Failed to fetch transactions data: {str(e)}")
        raise

def validate_amount(amount_str: str) -> float:
    """
    Validate and convert amount string to float.

    Args:
        amount_str: String representation of amount

    Returns:
        float: Validated amount

    Raises:
        ValueError: If amount is invalid
    """
    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return amount
    except ValueError as e:
        log.error(f"❌ Invalid amount: {amount_str}")
        raise ValueError(f"Invalid amount: {amount_str}") from e

def extract_transaction_info_conversational(user_message: str, conversation_history: list[dict[str, str]], pending_data: dict[str, Any]) -> dict[str, Any]:
    """
    Use conversational AI to extract transaction information, asking for missing details and probing for emotional state.
    Returns a dict with 'status', 'message', 'data', and 'complete' flag.
    """
    try:
        log.info("🔍 Starting conversational transaction extraction")
        log.debug(f"User message: {user_message}")
        log.debug(f"Pending data: {pending_data}")

        gemini_history = []
        for msg in conversation_history:
            gemini_history.append({
                'role': msg['role'],
                'parts': [msg['content']]
            })

        chat = model.start_chat(history=gemini_history)

       # Format categories and history for the prompt
        expense_categories_str = ", ".join(CATEGORIES.get('Expense', {}).keys())
        income_categories_str = ", ".join(CATEGORIES.get('Income', {}).keys())
        
        # Correctly format the conversation history for the AI to understand context
        history_log = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[:-1]]) # All but the latest message

        prompt = f"""
You are a highly specialized financial assistant AI. Your SOLE purpose is to extract transaction details from a user's conversation and format them into a strict JSON object. You must follow the rules below without deviation.

### Core Rules
1.  **Strict JSON Output:** You MUST ONLY respond with a valid JSON object. Do not include any other text.
2.  **Use Provided Categories:** You are strictly limited to the categories provided. Do NOT invent new ones.
    - **Valid Expense Categories:** `{expense_categories_str}`
    - **Valid Income Categories:** `{income_categories_str}`
3.  **Rigid Two-Step Process for Expenses:**
    - **Step 1:** First, collect all basic information (`Amount`, `Category`, `Description`, `Date`).
    - **Step 2:** ONLY after all basic info is collected for an EXPENSE, ask exactly ONE emotion-probing question.
4.  **Default to "Other":** If the user's category is ambiguous, you MUST classify it as the most logical parent category (e.g., "Shopping") or default to "Other".

### Logic Flow
Analyze the **Current User Message** by using the **Full Conversation History** as context. Your goal is to fill the fields in `extracted_data`. If a field is missing, ask a question to get it. When all fields are present, set `"status": "complete"`.

### Emotion Categories
- Coping / Stress Spending
- Routine / Neutral Spending
- Impulse / Boredom Spending
- Social Comparison Spending
- Aspirational / Identity Spending

**JSON RESPONSE FORMAT:**
```json
{{
  "status": "complete" or "incomplete",
  "transaction_type": "EXPENSE_NORMAL/INCOME_NORMAL/PENDING_TO_RECEIVE/PENDING_TO_PAY/PENDING_RECEIVED",
  "extracted_data": {{
    "amount": "number or null",
    "category": "category or null",
    "description": "text or null",
    "date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "emotion_state": "classified emotion or null"
  }},
  "missing_fields": ["list of missing required fields"],
  "assistant_message": "Your friendly response to the user. This is where you will ask your probing questions."
}}

---
### ANALYSIS TASK

**Full Conversation History (for context):**
{history_log if history_log else "No previous conversation."}

**Current User Message (process this now):**
user: "{user_message}"

**Information Collected So Far:**
{pending_data if pending_data else "None"}

**Your Task:**
Based on the history and the current message, generate the next required JSON response.
"""

        log.debug("🤖 Sending conversational prompt to Gemini")
        response = chat.send_message(prompt)
        response_text = response.text.strip()

        log.debug(f"Gemini response: {response_text}")

        import json
        try:
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()

            result = json.loads(response_text)
        except json.JSONDecodeError:
            log.warning("Failed to parse JSON, using fallback extraction")
            result = {
                "status": "incomplete",
                "assistant_message": response_text,
                "extracted_data": {},
                "missing_fields": ["amount"]
            }

        log.info(f"📋 Extraction status: {result.get('status')}")

        return result

    except Exception as e:
            log.error(f"❌ Failed conversational extraction: {str(e)}")
            return {
                "status": "error",
                "assistant_message": "I'm having trouble understanding. Could you please rephrase your transaction? For example: 'Spent 500 on groceries' or 'Received salary of 50000'",
                "extracted_data": {},
                "missing_fields": []
            }

def handle_received_pending_transaction(amount: float, description: str) -> tuple[bool, dict[str, Any] | None]:
    """
    Handle a pending transaction that has been received.
    """
    try:
        if amount <= 0:
            raise ValueError("Amount must be positive")

        log.info(f"💫 Processing received pending transaction: amount={amount}")

        log.debug("Checking for existing received transactions today")
        today = datetime.now().strftime('%Y-%m-%d')

        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A:F'
        ).execute()

        values = result.get('values', [])
        if values and len(values) > 1:
            for row in values[1:]:
                if (len(row) >= 6 and
                    row[0] == today and
                    abs(float(row[1]) - amount) < 0.01 and
                    row[2] == 'Income' and
                    row[3] == 'Other' and
                    row[4] == 'Pending Received' and
                    'received pending' in row[5].lower()):
                    log.warning("⚠️ This pending transaction was already processed today")
                    return False, None

        log.debug("Searching for matching pending transaction")
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Pending!A:G'
        ).execute()

        values = result.get('values', [])
        if not values:
            log.warning("❗ No pending transactions found in sheet")
            return False, None

        matching_rows: list[int] = []

        if len(values[0]) < 7:
            log.error("❌ Invalid sheet structure: missing required columns")
            return False, None

        for i, row in enumerate(values[1:], start=1):
            try:
                if len(row) < 7:
                    log.warning(f"⚠️ Skipping row {i+1}: insufficient columns")
                    continue

                row_amount = float(row[1])
                if (abs(row_amount - amount) < 0.01 and
                    row[6] == 'Pending' and
                    row[2] == 'To Receive'):
                    matching_rows.append(i)
                    log.debug(f"Found potential match at row {i+1}: amount={row_amount}")
            except (ValueError, IndexError) as e:
                log.warning(f"⚠️ Error processing row {i+1}: {str(e)}")
                continue

        if len(matching_rows) > 1:
            log.warning(f"⚠️ Multiple matching pending transactions found for amount {amount}")
            row_index: int = matching_rows[-1]
            log.info(f"Selected most recent match at row {row_index+1}")
        elif len(matching_rows) == 1:
            row_index = matching_rows[0]
            log.info(f"✅ Found matching pending transaction at row {row_index+1}")
        else:
            log.warning(f"❗ No matching pending transaction found for amount {amount}")
            return False, None

        log.debug(f"Updating status to Received for row {row_index+1}")
        range_name = f'Pending!G{row_index + 1}'
        try:
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range=range_name,
                valueInputOption='RAW',
                body={'values': [['Received']]}
            ).execute()
        except Exception as e:
            log.error(f"❌ Failed to update pending transaction status: {str(e)}")
            return False, None

        original_row = values[row_index]
        original_date = original_row[0]
        original_description = original_row[4] if len(original_row) > 4 else ''

        transaction_info = {
            'type': 'Income',
            'amount': str(amount),
            'category': 'Other',
            'subcategory': 'Pending Received',
            'description': f"Received pending payment ({original_date}): {original_description}",
            'date': today
        }

        log.debug("Creating new Income transaction")
        success = add_transaction_to_sheet(
            transaction_info['date'],
            amount,
            transaction_info['type'],
            transaction_info['category'],
            transaction_info['subcategory'],
            transaction_info['description']
        )

        if success:
            log.info("✨ Successfully processed received pending transaction")
        else:
            log.error("❌ Failed to create Income transaction")

        return success, transaction_info if success else None

    except Exception as e:
        log.error(f"❌ Failed to handle received pending transaction: {str(e)}")
        return False, None

def process_user_input_conversational(user_message: str) -> dict[str, Any]:
    """
    Process user input conversationally, handling incomplete information gracefully.
    Returns status and either asks for more info or returns complete transaction data.
    """
    try:
        if not user_message or not user_message.strip():
            return {
                "status": "error",
                "message": "Please tell me about your transaction. For example: 'Spent 500 on groceries' or 'Need to pay electricity bill of 2000 next week'"
            }

        log.info("🎯 Starting conversational transaction processing")

        conversation_history = st.session_state.conversation_history
        pending_data = st.session_state.pending_transaction_data

        conversation_history.append({
            'role': 'user',
            'content': user_message
        })

        result = extract_transaction_info_conversational(
            user_message,
            conversation_history,
            pending_data
        )

        if result.get('assistant_message'):
            conversation_history.append({
                'role': 'model',
                'content': result['assistant_message']
            })

        st.session_state.conversation_history = conversation_history

        if result['status'] == 'complete':
            if result.get('transaction_type') == 'PENDING_RECEIVED':
                amount = float(result['extracted_data']['amount'])
                success, transaction_info = handle_received_pending_transaction(amount, user_message)
                if success and transaction_info:
                    transaction_info['auto_processed'] = True
                    st.session_state.conversation_history = []
                    st.session_state.pending_transaction_data = {}
                    return {
                        "status": "complete",
                        "data": transaction_info
                    }
                else:
                    return {
                        "status": "error",
                        "message": "I couldn't find a matching pending transaction for that amount. Please check and try again."
                    }

            extracted = result['extracted_data']

            transaction_date = extracted.get('date')
            if not transaction_date:
                transaction_date = datetime.now().strftime('%Y-%m-%d')

            transaction_data = {
                'type': result['transaction_type'].replace('_NORMAL', '').replace('PENDING_TO_', 'To ').replace('_', ' ').title(),
                'amount': extracted['amount'],
                'category': extracted.get('category', 'Other'),
                'description': extracted.get('description', ''),
                'date': transaction_date,
                'emotion_state': extracted.get('emotion_state', 'Neutral')
            }

            if 'To Receive' in transaction_data['type'] or 'To Pay' in transaction_data['type']:
                transaction_data['due_date'] = extracted.get('due_date',
                    (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'))

            st.session_state.conversation_history = []
            st.session_state.pending_transaction_data = {}

            return {
                "status": "complete",
                "data": transaction_data,
                "message": result['assistant_message']
            }

        elif result['status'] == 'incomplete':
            if result.get('extracted_data'):
                # Merge new data with pending data
                for key, value in result['extracted_data'].items():
                    if value is not None:
                         pending_data[key] = value
                st.session_state.pending_transaction_data = pending_data

            return {
                "status": "incomplete",
                "message": result['assistant_message']
            }

        else:
            return {
                "status": "error",
                "message": result.get('assistant_message', 'Something went wrong. Please try again.')
            }

    except Exception as e:
        log.error(f"❌ Failed to process user input: {str(e)}", exc_info=True)
        st.session_state.conversation_history = []
        st.session_state.pending_transaction_data = {}
        return {
            "status": "error",
            "message": "I encountered an error. Let's start over. Please tell me about your transaction."
        }

def show_analytics() -> None:
    """
    Display analytics dashboard with transaction visualizations.
    Shows pie charts and trends for income and expenses.
    """
    try:
        log.info("Generating financial analytics")
        df = get_transactions_data()

        if df.empty:
            st.info("No transactions recorded yet. Add some transactions to see analytics!")
            return

        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        total_income = df[df['Type'] == 'Income']['Amount'].sum()
        total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()
        net_balance = total_income - total_expenses

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"Rs. {total_income:,.2f}", delta=None)
        with col2:
            st.metric("Total Expenses", f"Rs. {total_expenses:,.2f}", delta=None)
        with col3:
            st.metric("Net Balance", f"Rs. {net_balance:,.2f}",
                     delta=f"Rs. {net_balance:,.2f}",
                     delta_color="normal" if net_balance >= 0 else "inverse")

        if len(df) > 1:
            df_grouped = df.groupby(['Date', 'Type'])['Amount'].sum().unstack(fill_value=0)
            fig_timeline = px.line(df_grouped,
                                 title='Income vs Expenses Over Time',
                                 labels={'value': 'Amount (Rs. )', 'variable': 'Type'})
            st.plotly_chart(fig_timeline)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Income Breakdown")
                income_df = df[df['Type'] == 'Income']
                if not income_df.empty:
                    fig_income = px.pie(income_df, values='Amount', names='Category',
                                      title='Income by Category')
                    st.plotly_chart(fig_income)
                else:
                    st.info("No income transactions recorded yet.")

            with col2:
                st.subheader("Expense Breakdown")
                expense_df = df[df['Type'] == 'Expense']
                if not expense_df.empty:
                    fig_expense = px.pie(expense_df, values='Amount', names='Category',
                                       title='Expenses by Category')
                    st.plotly_chart(fig_expense)
                else:
                    st.info("No expense transactions recorded yet.")

            st.subheader("Monthly Summary")
            monthly_summary = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Type'])['Amount'].sum().unstack(fill_value=0)
            monthly_summary['Net'] = monthly_summary.get('Income', 0) - monthly_summary.get('Expense', 0)
            st.dataframe(monthly_summary.style.format("Rs. {:,.2f}"))

        log.info("✅ Analytics visualizations generated successfully")
    except Exception as e:
        log.error(f"❌ Failed to generate analytics: {str(e)}")
        st.error("Failed to generate analytics. Please try again later.")

@st.cache_data(ttl=3600)
def get_sheet_url() -> str:
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"

@st.cache_resource
def initialize_gemini() -> Any:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    return genai.GenerativeModel('models/gemini-2.5-flash')

@st.cache_data
def get_subcategories(trans_type: str, category: str) -> list[str]:
    return CATEGORIES[trans_type][category]

def on_save_click():
    st.session_state.save_clicked = True

# <-- FIX: This function is now the single source of truth for creating AND fixing sheets.
def verify_sheets_setup():
    """
    Verify and correct both Expenses and Pending sheets, including headers.
    This robust function creates sheets if they don't exist and fixes headers if they are incorrect.
    """
    try:
        log.info("Verifying Google Sheets setup...")
        sheet_metadata = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute()
        sheets = sheet_metadata.get('sheets', [])
        existing_sheets = {s.get("properties", {}).get("title") for s in sheets}

        # Define the CORRECT headers
        expected_exp_headers = ['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description', 'emotion_state']
        expected_pend_headers = ['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status']

        def create_or_correct_sheet(sheet_title, expected_headers):
            if sheet_title not in existing_sheets:
                service.spreadsheets().batchUpdate(
                    spreadsheetId=SHEET_ID,
                    body={'requests': [{'addSheet': {'properties': {'title': sheet_title}}}]}
                ).execute()
                log.info(f"Created new sheet: {sheet_title}")
                service.spreadsheets().values().update(
                    spreadsheetId=SHEET_ID, range=f"'{sheet_title}'!A1",
                    valueInputOption='RAW', body={'values': [expected_headers]}
                ).execute()
                log.info(f"Set headers for new sheet: {sheet_title}")
            else:
                headers_result = service.spreadsheets().values().get(spreadsheetId=SHEET_ID, range=f"'{sheet_title}'!A1:1").execute()
                current_headers = headers_result.get('values', [[]])[0]
                if current_headers != expected_headers:
                    log.warning(f"{sheet_title} sheet has incorrect headers. Correcting them.")
                    service.spreadsheets().values().update(
                        spreadsheetId=SHEET_ID, range=f"'{sheet_title}'!A1",
                        valueInputOption='RAW', body={'values': [expected_headers]}
                    ).execute()

        # Run verification for both sheets
        create_or_correct_sheet('Expenses', expected_exp_headers)
        create_or_correct_sheet('Pending', expected_pend_headers)

        log.info("✨ Sheets verified and initialized successfully")
        return True
    except Exception as e:
        log.error(f"❌ Failed to verify/initialize sheets: {str(e)}")
        st.error(f"Critical error setting up Google Sheets: {e}")
        return False

def show_success_message(transaction_date: datetime | str, subcategory: str | None) -> None:
    """
    Display success message after transaction is saved.

    Args:
        transaction_date: Date of the transaction
        subcategory: Transaction subcategory, if applicable
    """
    emoji = "💰" if st.session_state.current_transaction['type'] == "Income" else "💸"
    confirmation_message = (
        f"{emoji} Transaction recorded:\n\n"
        f"Date: {transaction_date}\n"
        f"Amount: Rs. {float(st.session_state.current_transaction['amount']):,.2f}\n"
        f"Type: {st.session_state.current_transaction['type']}\n"
        f"Category: {st.session_state.current_transaction['category']}\n"
        f"Subcategory: {subcategory if subcategory else 'N/A'}"
    )
    st.success(confirmation_message)
    st.session_state.messages.append({"role": "assistant", "content": confirmation_message})
    log.info("✅ Transaction saved and analytics updated")

def show_transaction_form():
    """Separate function to handle transaction form display and processing"""
    extracted_info = st.session_state.current_transaction

    if extracted_info.get('auto_processed'):
        log.debug("Showing feedback for auto-processed transaction")

        st.success("✅ Transaction Processed Successfully")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Transaction Details:**")
            st.write(f"📅 Date: {extracted_info.get('date')}")
            st.write(f"💰 Amount: Rs. {float(extracted_info.get('amount', 0)):,.2f}")
            st.write(f"📝 Type: {extracted_info.get('type')}")

        with col2:
            st.write(f"🏷️ Category: {extracted_info.get('category')}")
            st.write(f"📑 Subcategory: {extracted_info.get('subcategory')}")
            st.write(f"📌 Description: {extracted_info.get('description')}")

        st.divider()

        if st.button("Clear Message", key="clear_feedback"):
            st.session_state.current_transaction = None
            st.rerun()
        return

    if 'amount' in extracted_info and 'type' in extracted_info:
        st.success("✅ All Transaction Information Collected!")

        # --- START: MODIFICATION ---
        # Gracefully handle cases where the AI hallucinates a category not in our predefined list.
        categories = get_categories()
        trans_type = extracted_info.get('type')
        current_category = extracted_info.get('category')

        if trans_type not in categories or current_category not in categories.get(trans_type, {}):
            log.warning(f"AI-extracted category '{current_category}' not found for type '{trans_type}'. Defaulting to 'Other'.")
            st.warning(f"The category '{current_category}' isn't standard. We've defaulted to 'Other', but please select a more appropriate one if available.")
            extracted_info['category'] = 'Other'
            st.session_state.current_transaction['category'] = 'Other' # Ensure session state is also updated
        # --- END: MODIFICATION ---

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(f"**Type:** {extracted_info.get('type')}")
            st.write(f"**Amount:** Rs. {float(extracted_info.get('amount', 0)):,.2f}")
            if extracted_info.get('type') == 'Expense':
                 st.write(f"**Emotion:** {extracted_info.get('emotion_state', 'Neutral')}")

        with summary_col2:
            st.write(f"**Category:** {extracted_info.get('category')}")
            st.write(f"**Description:** {extracted_info.get('description', 'N/A')}")

        st.divider()
        st.markdown("### 📝 Confirm Transaction Details")

        with st.form(key="transaction_form"):
            col1, col2 = st.columns([3, 1])

            with col1:
                if extracted_info['type'] in ['To Receive', 'To Pay']:
                    try:
                        if 'due_date' in extracted_info and extracted_info['due_date']:
                            default_due_date = datetime.strptime(extracted_info['due_date'], '%Y-%m-%d')
                        else:
                            default_due_date = datetime.now() + timedelta(days=7)
                    except ValueError:
                        default_due_date = datetime.now() + timedelta(days=7)

                    due_date = st.date_input(
                        "Due date",
                        value=default_due_date,
                        key="due_date"
                    )
                else:
                    # This line is now safe because of the check above
                    subcategories = categories[extracted_info['type']][extracted_info['category']]
                    subcategory = st.selectbox(
                        "Select subcategory",
                        subcategories,
                        key="subcategory_select"
                    )

                try:
                    if extracted_info.get('date'):
                        default_date = datetime.strptime(extracted_info['date'], '%Y-%m-%d')
                    else:
                        default_date = datetime.now()
                except (ValueError, TypeError):
                    default_date = datetime.now()

                transaction_date = st.date_input(
                    "Transaction date",
                    value=default_date,
                    key="transaction_date"
                )

            with col2:
                st.write("")
                st.write("")

            submitted = st.form_submit_button(
                "💾 Save Transaction",
                type="primary",
                width="stretch"
            )

        if submitted:
                try:
                    if extracted_info['type'] in ['To Receive', 'To Pay']:
                        success = add_pending_transaction_to_sheet(
                            transaction_date.strftime('%Y-%m-%d'),
                            extracted_info['amount'],
                            extracted_info['type'],
                            extracted_info['category'],
                            extracted_info.get('description', ''),
                            due_date.strftime('%Y-%m-%d')
                        )
                    else:
                        success = add_transaction_to_sheet(
                            transaction_date.strftime('%Y-%m-%d'),
                            extracted_info['amount'],
                            extracted_info['type'],
                            extracted_info['category'],
                            subcategory,
                            extracted_info.get('description', ''),
                            extracted_info.get('emotion_state', 'Neutral')
                        )

                    if success:
                        show_success_message(
                            transaction_date.strftime('%Y-%m-%d'),
                            subcategory if 'subcategory' in locals() else None
                        )
                        st.session_state.current_transaction = None
                        st.rerun()
                    else:
                        st.error("Failed to save transaction. Please try again.")
                except Exception as e:
                    log.error(f"Failed to save transaction: {str(e)}", exc_info=True)
                    st.error(f"An error occurred while saving the transaction: {str(e)}")

def add_pending_transaction_to_sheet(date, amount, trans_type, category, description, due_date):
    try:
        if not verify_sheets_setup():
            raise Exception("Failed to verify sheets setup")

        log.info(f"Starting pending transaction save: {date}, {amount}, {trans_type}, {category}, {description}, {due_date}")

        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        if isinstance(due_date, datetime):
            due_date = due_date.strftime('%Y-%m-%d')

        amount = str(float(amount))

        values = [[str(date), amount, trans_type, category, description, str(due_date), 'Pending']]

        result = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Pending!A1:G1',
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': values}
        ).execute()

        log.info(f"✅ Pending transaction saved successfully: {result}")
        return True

    except Exception as e:
        log.error(f"❌ Failed to save pending transaction: {str(e)}")
        return False

# def verify_sheets_setup():
#     """Verify both Expenses and Pending sheets exist with correct headers"""
#     try:
#         sheet_metadata = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute()
#         sheets = sheet_metadata.get('sheets', '')
#         existing_sheets = {s.get("properties", {}).get("title") for s in sheets}

#         if 'Expenses' not in existing_sheets:
#             log.info("Creating new Expenses sheet...")
#             body = {
#                 'requests': [{
#                     'addSheet': {
#                         'properties': {
#                             'title': 'Expenses'
#                         }
#                     }
#                 }]
#             }
#             service.spreadsheets().batchUpdate(
#                 spreadsheetId=SHEET_ID,
#                 body=body
#             ).execute()

#             headers = [['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description']]
#             service.spreadsheets().values().update(
#                 spreadsheetId=SHEET_ID,
#                 range='Expenses!A1:F1',
#                 valueInputOption='RAW',
#                 body={'values': headers}
#             ).execute()

#         if 'Pending' not in existing_sheets:
#             log.info("Creating new Pending sheet...")
#             body = {
#                 'requests': [{
#                     'addSheet': {
#                         'properties': {
#                             'title': 'Pending'
#                         }
#                     }
#                 }]
#             }
#             service.spreadsheets().batchUpdate(
#                 spreadsheetId=SHEET_ID,
#                 body=body
#             ).execute()

#             headers = [['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status']]
#             service.spreadsheets().values().update(
#                 spreadsheetId=SHEET_ID,
#                 range='Pending!A1:G1',
#                 valueInputOption='RAW',
#                 body={'values': headers}
#             ).execute()

#         log.info("✨ Sheets verified and initialized")
#         return True
#     except Exception as e:
#         log.error(f"❌ Failed to verify/initialize sheets: {str(e)}")
#         return False

def main():
    """
    Main application function.
    Handles the core application flow and user interface.
    """
    try:
        log.info("🚀 Starting Finance Tracker application")

        if 'sheets_verified' not in st.session_state:
            st.session_state.sheets_verified = False

        if not st.session_state.sheets_verified:
            verify_sheets_setup()
            st.session_state.sheets_verified = True

        st.title("💰 Smart Finance Tracker")
        st.markdown(f"📊 [View Google Sheet]({get_sheet_url()})")

        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🔄 Reset", help="Clear conversation and start fresh"):
                st.session_state.conversation_history = []
                st.session_state.pending_transaction_data = {}
                st.session_state.current_transaction = None
                st.session_state.messages = []
                st.session_state.last_processed_audio_bytes = None
                st.rerun()

        st.divider()

        init_session_state()

        if not st.session_state.messages:
            st.markdown("### 💬 Chat with your Finance Tracker")
            st.markdown("I can help you log transactions naturally. Try saying:")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Expenses:**
                - "Spent 500 on groceries"
                - "Bought coffee for 150"
                - "Paid 2000 for electricity"
                """)
            with col2:
                st.markdown("""
                **Income & Pending:**
                - "Got salary of 50000"
                - "Need to pay rent of 15000 next week"
                - "Will receive 5000 from friend tomorrow"
                - "Received pending payment of 1275"
                """)

            st.info("💡 Don't worry if you forget details - I'll ask you for them!")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Text input
        if prompt := st.chat_input("Tell me about your income or expense..."):
            log.debug(f"Received user input: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = process_user_input_conversational(prompt)

                if result['status'] == 'complete':
                    st.session_state.current_transaction = result['data']
                    if result.get('message'):
                        st.markdown(result['message'])
                        st.session_state.messages.append({"role": "assistant", "content": result['message']})
                    st.rerun()

                elif result['status'] == 'incomplete':
                    st.markdown(result['message'])
                    st.session_state.messages.append({"role": "assistant", "content": result['message']})
                    st.session_state.awaiting_user_input = True

                else:
                    st.error(result['message'])
                    st.session_state.messages.append({"role": "assistant", "content": result['message']})
                    st.session_state.conversation_history = []
                    st.session_state.pending_transaction_data = {}

        audio_data = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=False,
            key='audio_recorder'
        )

        # Process audio if available and not already processed
        if audio_data and audio_data.get('bytes') != st.session_state.last_processed_audio_bytes:
            with st.spinner("🎧 Transcribing audio..."):
                transcribed_text = transcribe_audio(audio_data['bytes'])
                st.session_state.last_processed_audio_bytes = audio_data['bytes']

                if transcribed_text:
                    log.info(f"Audio transcribed: {transcribed_text}")

                    # Add transcribed text to messages
                    st.session_state.messages.append({"role": "user", "content": f"🎤 {transcribed_text}"})

                    # Process the transcribed text
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            result = process_user_input_conversational(transcribed_text)

                        if result['status'] == 'complete':
                            st.session_state.current_transaction = result['data']
                            if result.get('message'):
                                st.markdown(result['message'])
                                st.session_state.messages.append({"role": "assistant", "content": result['message']})
                            st.rerun()

                        elif result['status'] == 'incomplete':
                            st.markdown(result['message'])
                            st.session_state.messages.append({"role": "assistant", "content": result['message']})
                            st.session_state.awaiting_user_input = True

                        else:
                            st.error(result['message'])
                            st.session_state.messages.append({"role": "assistant", "content": result['message']})
                            st.session_state.conversation_history = []
                            st.session_state.pending_transaction_data = {}
                else:
                    st.warning("⚠️ Could not transcribe audio. Please try again or use text input.")

        if st.session_state.current_transaction:
            show_transaction_form()

    except Exception as e:
        log.error(f"❌ Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()