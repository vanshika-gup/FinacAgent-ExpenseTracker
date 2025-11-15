import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build # type: ignore
from dotenv import load_dotenv
from utils.logging_utils import setup_logging

log = setup_logging("expense_tracker_analytics")

st.set_page_config (layout='wide')

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
            # Local development - use file path from .env
            creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
            if not creds_path:
                raise ValueError("GOOGLE_SHEETS_CREDENTIALS not set")
            if not os.path.exists(creds_path):
                raise FileNotFoundError(f"Credentials file not found: {creds_path}")
            
            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
        
        service = build('sheets', 'v4', credentials=creds)
        return service
        
    except Exception as e:
        log.error(f"❌ Failed to connect to Google Sheets: {str(e)}")
        st.error(f"Failed to create Google Sheets service: {str(e)}")
        raise

# Initialize service and sheet ID
try:
    service = get_google_sheets_service()
    SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
except Exception:
    st.error("Failed to connect to Google Sheets. Please check your credentials.")
    sys.exit(1)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_transactions_data():
    try:
        log.debug("Fetching transactions data from Google Sheets")
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A1:G'  # Changed from A1:F to A1:G to include emotion_state
        ).execute()
        
        values = result.get('values', [])
        if not values:
            log.warning("No transaction data found in sheet")
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description', 'emotion_state'])
        
        log.info(f" Retrieved {len(values)-1} transaction records")
        return pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description', 'emotion_state'])
    except Exception as e:
        log.error(f"❌ Failed to fetch transactions data: {str(e)}")
        raise

@st.cache_data(ttl=300)
def get_pending_transactions() -> pd.DataFrame:
    """
    Fetch pending transactions from Google Sheet.
    Only returns transactions with status 'Pending'.
    """
    try:
        log.debug("Fetching pending transactions data")
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Pending!A1:G'  # Include status column
        ).execute()
        
        values = result.get('values', [])
        if not values:
            log.warning("No data found in Pending sheet")
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status'])
        
        log.debug(f"Raw data from sheet: {values[:5]}")  # Log first few rows
        
        # Convert to DataFrame
        df = pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status'])
        log.debug(f"Initial DataFrame shape: {df.shape}")
        
        # Log unique values in Status column
        log.debug(f"Unique Status values: {df['Status'].unique()}")
        
        # Filter only pending transactions
        df = df[df['Status'].str.strip().str.upper() == 'PENDING']
        log.debug(f"DataFrame shape after status filter: {df.shape}")
        
        if df.empty:
            log.warning("No pending transactions after filtering")
            return df
        
        # Convert Amount to numeric
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Convert dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Due Date'] = pd.to_datetime(df['Due Date'], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=['Amount', 'Type', 'Status'])
        log.debug(f"Final DataFrame shape: {df.shape}")
        
        log.info(f"📊 Retrieved {len(df)} pending transactions")
        return df
    except Exception as e:
        log.error(f"❌ Failed to fetch pending transactions: {str(e)}")
        raise

def initialize_filters():
    """Initialize filter values in session state if they don't exist"""
    if 'global_filter_type' not in st.session_state:
        st.session_state.global_filter_type = "All Time"
    if 'global_selected_year' not in st.session_state:
        st.session_state.global_selected_year = datetime.now().year
    if 'global_selected_month' not in st.session_state:
        st.session_state.global_selected_month = datetime.now().month
    if 'global_start_date' not in st.session_state:
        st.session_state.global_start_date = datetime.now() - timedelta(days=30)
    if 'global_end_date' not in st.session_state:
        st.session_state.global_end_date = datetime.now()
    if 'filter_container_created' not in st.session_state:
        st.session_state.filter_container_created = False

def get_date_filters(key:str="unique_global_filter"):
    """Common date filter UI component for all analytics"""
    initialize_filters()
    
    st.sidebar.subheader("📅 Date Filter")
    
    # Get min and max dates from the data
    df = get_transactions_data()
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        max_date = df['Date'].max()
    else:
        min_date = max_date = datetime.now()
    
    # Filter type selection with unique key
    st.session_state.global_filter_type = st.sidebar.radio(
        "Select Time Period",
        ["All Time", "Year", "Month", "Custom Range"],
        key=key
    )
    
    if st.session_state.global_filter_type == "Year":
        st.session_state.global_selected_year = st.sidebar.selectbox(
            "Select Year",
            sorted(df['Date'].dt.year.unique(), reverse=True),
            key="unique_global_year"
        )
        start_date = datetime(st.session_state.global_selected_year, 1, 1)
        end_date = datetime(st.session_state.global_selected_year, 12, 31)
    
    elif st.session_state.global_filter_type == "Month":
        st.session_state.global_selected_year = st.sidebar.selectbox(
            "Select Year",
            sorted(df['Date'].dt.year.unique(), reverse=True),
            key="unique_global_month_year"
        )
        st.session_state.global_selected_month = st.sidebar.selectbox(
            "Select Month",
            range(1, 13),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            key="unique_global_month"
        )
        start_date = datetime(st.session_state.global_selected_year, st.session_state.global_selected_month, 1)
        end_date = (datetime(st.session_state.global_selected_year, st.session_state.global_selected_month + 1, 1) 
                   if st.session_state.global_selected_month < 12 
                   else datetime(st.session_state.global_selected_year + 1, 1, 1)) - timedelta(days=1)
    
    elif st.session_state.global_filter_type == "Custom Range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.global_start_date = st.date_input(
                "Start Date", 
                min_date,
                key="unique_global_start_date"
            )
        with col2:
            st.session_state.global_end_date = st.date_input(
                "End Date", 
                max_date,
                key="unique_global_end_date"
            )
        
        start_date = datetime.combine(st.session_state.global_start_date, datetime.min.time()) # type: ignore
        end_date = datetime.combine(st.session_state.global_end_date, datetime.max.time()) # type: ignore
    
    else:  # All Time
        start_date = min_date
        end_date = max_date
    
    return start_date, end_date

def filter_dataframe(df, start_date, end_date):
    """Filter dataframe based on date range"""
    if df.empty:
        return df
    
    df['Date'] = pd.to_datetime(df['Date'])
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

def show_overview_analytics(df, start_date, end_date):
    st.subheader("📈 Financial Overview")
    if df.empty:
        st.info("No transactions found for the selected period.")
        return
    
    # Filter data
    df = filter_dataframe(df, start_date, end_date)
    
    # Display selected period
    # st.caption(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Calculate key metrics
    total_income = df[df['Type'] == 'Income']['Amount'].sum()
    total_expense = df[df['Type'] == 'Expense']['Amount'].sum()
    net_savings = total_income - total_expense
    saving_rate = (net_savings / total_income * 100) if total_income > 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Income", f"Rs. {total_income:,.2f}", delta=None)
    with col2:
        st.metric("Total Expenses", f"Rs. {total_expense:,.2f}", delta=None)
    with col3:
        st.metric("Net Savings", f"Rs. {net_savings:,.2f}", 
                 delta=f"Rs. {net_savings:,.2f}",
                 delta_color="normal" if net_savings >= 0 else "inverse")
    with col4:
        st.metric("Saving Rate", f"{saving_rate:.1f}%",
                 delta=None)
    
    # Monthly Summary
    st.subheader("Monthly Summary")
    monthly_summary = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Type'])['Amount'].sum().unstack(fill_value=0)
    monthly_summary['Net'] = monthly_summary.get('Income', 0) - monthly_summary.get('Expense', 0)
    
    # Monthly trend chart
    fig_monthly = px.bar(monthly_summary, 
                        title='Monthly Income vs Expenses',
                        barmode='group',
                        labels={'value': 'Amount (Rs.)', 'index': 'Month'})
    st.plotly_chart(fig_monthly)
    
    # Show monthly summary table
    st.dataframe(
        monthly_summary.style.format({
            'Income': 'Rs. {:,.2f}',
            'Expense': 'Rs. {:,.2f}',
            'Net': 'Rs. {:,.2f}'
        }),
        width="stretch",
        height=200
    )
    
    # Recent Transactions
    st.subheader("Recent Transactions")
    recent_df = df.sort_values('Date', ascending=False).head(5)
    st.dataframe(
        recent_df[['Date', 'Type', 'Category', 'Subcategory', 'Amount', 'Description']].style.format({
            'Amount': 'Rs. {:,.2f}',
            'Date': lambda x: x.strftime('%Y-%m-%d')
        }),
        hide_index=True
    )
    
    # Category-wise Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Income Categories")
        income_by_category = df[df['Type'] == 'Income'].groupby('Category')['Amount'].sum().sort_values(ascending=False).head(5)
        fig_income = px.pie(values=income_by_category.values, 
                          names=income_by_category.index,
                          title='Top Income Sources')
        st.plotly_chart(fig_income)
    
    with col2:
        st.subheader("Top Expense Categories")
        expense_by_category = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False).head(5)
        fig_expense = px.pie(values=expense_by_category.values, 
                           names=expense_by_category.index,
                           title='Top Expense Categories')
        st.plotly_chart(fig_expense)
    
    # Add Spending Patterns Analysis
    st.subheader("💡 Spending Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekday vs Weekend spending
        df['Day_Type'] = df['Date'].dt.dayofweek.map(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        daily_avg = df[df['Type'] == 'Expense'].groupby('Day_Type')['Amount'].agg(['sum', 'count'])
        daily_avg['avg'] = daily_avg['sum'] / daily_avg['count']
        
        st.caption("Weekday vs Weekend Spending")
        st.dataframe(
            daily_avg.style.format({
                'sum': 'Rs. {:,.2f}',
                'avg': 'Rs. {:,.2f}/day'
            })
        )
    
    with col2:
        # Week of month analysis
        df['Week_of_Month'] = df['Date'].dt.day.map(lambda x: (x-1)//7 + 1)
        weekly_spending = df[df['Type'] == 'Expense'].groupby('Week_of_Month')['Amount'].mean()
        
        fig_weekly = px.bar(weekly_spending,
                          title='Average Spending by Week of Month',
                          labels={'value': 'Amount (Rs.)', 'Week_of_Month': 'Week'})
        st.plotly_chart(fig_weekly)

def show_income_analytics(df, start_date, end_date):
    st.subheader("💰 Income Analytics")
    income_df = df[df['Type'] == 'Income'].copy()
    if income_df.empty:
        st.info("No income transactions found for the selected period.")
        return
    
    # Filter data
    df = filter_dataframe(df, start_date, end_date)
    
    # Display selected period
    # st.caption(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Monthly Income Trend
    monthly_income = income_df.groupby(income_df['Date'].dt.strftime('%Y-%m'))['Amount'].sum()
    fig_monthly = px.bar(monthly_income, 
                        title='Monthly Income Trend',
                        labels={'value': 'Amount (Rs.)', 'index': 'Month'})
    st.plotly_chart(fig_monthly)
    
    # Category Analysis
    col1, col2 = st.columns(2)
    with col1:
        # Category Breakdown
        fig_category = px.pie(income_df, 
                            values='Amount', 
                            names='Category',
                            title='Income by Category')
        st.plotly_chart(fig_category)
    
    with col2:
        # Top Income Sources
        st.subheader("Top Income Sources")
        top_sources = income_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        st.dataframe(
            top_sources.to_frame().style.format({
                'Amount': 'Rs. {:,.2f}'
            }),
            width="stretch",
            height=300
        )
    
    # Subcategory Analysis
    st.subheader("Income by Subcategory")
    subcategory_income = income_df.groupby('Subcategory')['Amount'].sum().sort_values(ascending=False)
    fig_subcategory = px.bar(subcategory_income,
                            title='Income by Subcategory',
                            labels={'value': 'Amount (Rs.)', 'index': 'Subcategory'})
    st.plotly_chart(fig_subcategory)
    
    # Add Income Stability Analysis
    st.subheader("💰 Income Stability Analysis")
    monthly_income = df[df['Type'] == 'Income'].groupby(df['Date'].dt.strftime('%Y-%m'))['Amount'].sum()
    
    income_stats = {
        'Average Monthly Income': monthly_income.mean(),
        'Income Volatility': monthly_income.std() / monthly_income.mean(),
        'Highest Income Month': monthly_income.max(),
        'Lowest Income Month': monthly_income.min(),
        'Months with Income': len(monthly_income)
    }
    
    st.dataframe(
        pd.Series(income_stats).to_frame('Value').style.format({
            'Value': lambda x: f"Rs. {x:,.2f}" if isinstance(x, (int, float)) and x > 100 else f"{x:.2%}" if isinstance(x, float) else x
        }) # type: ignore
    )

def show_emotion_analytics(df, start_date, end_date):
    """Display emotional spending analytics"""
    st.subheader("🎭 Emotional Spending Analytics")
    
    # Filter only expenses
    expense_df = df[df['Type'] == 'Expense'].copy()
    
    if expense_df.empty:
        st.info("No expense transactions found for the selected period.")
        return
    
    # Check if emotion_state column exists and has data
    if 'emotion_state' not in expense_df.columns:
        st.warning("No emotional data available. Emotion tracking was recently added.")
        return
    
    # Replace empty/null values with 'Missed Emotion' instead of 'Neutral'
    expense_df['emotion_state'] = expense_df['emotion_state'].fillna('Missed Emotion')
    expense_df['emotion_state'] = expense_df['emotion_state'].replace('', 'Missed Emotion')
    expense_df['emotion_state'] = expense_df['emotion_state'].str.strip()  # Remove any extra spaces
    
    # Filter data by date
    expense_df = filter_dataframe(expense_df, start_date, end_date)
    
    if expense_df.empty:
        st.info("No expense transactions found for the selected period.")
        return
    
    # Color mapping for emotions
    emotion_colors = {
        'Coping / Stress Spending': '#FF6B6B',  # Red
        'Routine / Neutral Spending': '#4ECDC4',  # Teal
        'Impulse / Boredom Spending': '#FFE66D',  # Yellow
        'Social Comparison Spending': '#A8E6CF',  # Light Green
        'Aspirational / Identity Spending': '#95E1D3',  # Mint
        'Missed Emotion': '#B0B0B0'  # Gray for missed emotions
    }
    
    # Check if all emotions are 'Missed Emotion'
    if (expense_df['emotion_state'] == 'Missed Emotion').all():
        st.warning("No emotional data available for this period. Emotion tracking was recently added.")
        return
    
    # Overall emotion distribution
    st.subheader("💰 Total Spending by Emotion")
    emotion_totals = expense_df.groupby('emotion_state')['Amount'].sum().sort_values(ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_emotion_pie = px.pie(
            values=emotion_totals.values,
            names=emotion_totals.index,
            title='Spending Distribution by Emotional State',
            color=emotion_totals.index,
            color_discrete_map=emotion_colors
        )
        fig_emotion_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_emotion_pie, width="stretch")
    
    with col2:
        st.write("### Summary")
        for emotion, amount in emotion_totals.items():
            percentage = (amount / emotion_totals.sum()) * 100
            st.metric(
                emotion.split('/')[0].strip(),
                f"Rs. {amount:,.2f}",
                f"{percentage:.1f}%"
            )
    
    st.divider()
    
    # Monthly trend by emotion
    st.subheader("📊 Monthly Emotional Spending Trends")
    monthly_emotion = expense_df.groupby([
        expense_df['Date'].dt.strftime('%Y-%m'),
        'emotion_state'
    ])['Amount'].sum().reset_index()
    
    fig_monthly_emotion = px.bar(
        monthly_emotion,
        x='Date',
        y='Amount',
        color='emotion_state',
        title='Monthly Spending by Emotion',
        labels={'Amount': 'Amount (Rs.)', 'Date': 'Month'},
        color_discrete_map=emotion_colors,
        barmode='stack'
    )
    st.plotly_chart(fig_monthly_emotion, width="stretch")
    
    st.divider()
    
    # Category breakdown by emotion
    st.subheader("🏷️ Categories by Emotional State")
    
    category_emotion = expense_df.groupby(['Category', 'emotion_state'])['Amount'].sum().reset_index()
    
    fig_category_emotion = px.bar(
        category_emotion,
        x='Category',
        y='Amount',
        color='emotion_state',
        title='Spending Categories by Emotional State',
        labels={'Amount': 'Amount (Rs.)'},
        color_discrete_map=emotion_colors,
        barmode='group'
    )
    fig_category_emotion.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_category_emotion, width="stretch")
    
    st.divider()
    
    # Insights section
    st.subheader("💡 Emotional Spending Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Most common emotion (excluding Missed Emotion)
        emotions_without_missed = expense_df[expense_df['emotion_state'] != 'Missed Emotion']
        if not emotions_without_missed.empty:
            most_common_emotion = emotions_without_missed['emotion_state'].mode()[0]
            most_common_count = len(emotions_without_missed[emotions_without_missed['emotion_state'] == most_common_emotion])
            st.metric(
                "Most Common Emotion",
                most_common_emotion.split('/')[0].strip(),
                f"{most_common_count} transactions"
            )
        else:
            st.metric("Most Common Emotion", "N/A", "No data")
    
    with col2:
        # Highest spending emotion
        highest_spending_emotion = emotion_totals.idxmax()
        highest_amount = emotion_totals.max()
        st.metric(
            "Highest Spending Emotion",
            highest_spending_emotion.split('/')[0].strip(),
            f"Rs. {highest_amount:,.2f}"
        )
    
    with col3:
        # Average transaction by emotion
        avg_per_emotion = expense_df.groupby('emotion_state')['Amount'].mean()
        highest_avg_emotion = avg_per_emotion.idxmax()
        highest_avg = avg_per_emotion.max()
        st.metric(
            "Highest Avg Transaction",
            highest_avg_emotion.split('/')[0].strip(),
            f"Rs. {highest_avg:,.2f}"
        )
    
    st.divider()
    
    # Detailed breakdown table
    st.subheader("📋 Detailed Emotion Breakdown")
    
    emotion_details = expense_df.groupby('emotion_state').agg({
        'Amount': ['sum', 'mean', 'count']
    }).round(2)
    
    emotion_details.columns = ['Total Spent', 'Average Transaction', 'Number of Transactions']
    emotion_details = emotion_details.sort_values('Total Spent', ascending=False)
    
    st.dataframe(
        emotion_details.style.format({
            'Total Spent': 'Rs. {:,.2f}',
            'Average Transaction': 'Rs. {:,.2f}',
            'Number of Transactions': '{:.0f}'
        }),
        width="stretch"
    )
    
    # Recent emotional transactions
    st.subheader("🕐 Recent Emotional Transactions")
    recent_emotion = expense_df.sort_values('Date', ascending=False).head(10)
    st.dataframe(
        recent_emotion[['Date', 'Amount', 'Category', 'Description', 'emotion_state']].style.format({
            'Amount': 'Rs. {:,.2f}',
            'Date': lambda x: x.strftime('%Y-%m-%d')
        }),
        hide_index=True,
        width="stretch"
    )

    
def show_expense_analytics(df, start_date, end_date):
    st.subheader("💸 Expense Analytics")
    expense_df = df[df['Type'] == 'Expense'].copy()
    if expense_df.empty:
        st.info("No expense transactions found for the selected period.")
        return
    
    # Filter data
    df = filter_dataframe(df, start_date, end_date)
    
    # Display selected period
    # st.caption(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Monthly Expense Trend
    monthly_expense = expense_df.groupby(expense_df['Date'].dt.strftime('%Y-%m'))['Amount'].sum()
    fig_monthly = px.bar(monthly_expense, 
                        title='Monthly Expense Trend',
                        labels={'value': 'Amount (Rs.)', 'index': 'Month'})
    st.plotly_chart(fig_monthly)
    
    # Category Analysis
    col1, col2 = st.columns(2)
    with col1:
        # Category Breakdown
        fig_category = px.pie(expense_df, 
                            values='Amount', 
                            names='Category',
                            title='Expenses by Category')
        st.plotly_chart(fig_category)
    
    with col2:
        # Top Expense Categories
        st.subheader("Top Expense Categories")
        top_expenses = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        st.dataframe(
            top_expenses.to_frame().style.format({
                'Amount': 'Rs. {:,.2f}'
            }),
            width="stretch",
            height=300
        )
    
    # Subcategory Analysis
    st.subheader("Expenses by Subcategory")
    subcategory_expenses = expense_df.groupby('Subcategory')['Amount'].sum().sort_values(ascending=False)
    fig_subcategory = px.bar(subcategory_expenses,
                            title='Expenses by Subcategory',
                            labels={'value': 'Amount (Rs.)', 'index': 'Subcategory'})
    st.plotly_chart(fig_subcategory)
    
    # Daily Average Spending
    avg_daily = expense_df.groupby(expense_df['Date'].dt.strftime('%Y-%m'))['Amount'].sum() / 30
    st.subheader("Average Daily Spending by Month")
    st.dataframe(
        avg_daily.to_frame().style.format({
            'Amount': 'Rs. {:,.2f}'
        })
    )
    
    # Add Fixed vs Variable Expenses
    st.subheader("📊 Fixed vs Variable Expenses")
    monthly_category = df[df['Type'] == 'Expense'].groupby(['Category', df['Date'].dt.strftime('%Y-%m')])['Amount'].sum()
    category_consistency = monthly_category.groupby('Category').agg(['mean', 'std'])
    category_consistency['variation'] = (category_consistency['std'] / category_consistency['mean']).fillna(0)
    
    # Categories with low variation are likely fixed expenses
    fixed_expenses = category_consistency[category_consistency['variation'] < 0.2]
    variable_expenses = category_consistency[category_consistency['variation'] >= 0.2]
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Fixed Expenses (Low Variation)")
        st.dataframe(
            fixed_expenses.style.format({
                'mean': 'Rs. {:,.2f}',
                'std': 'Rs. {:,.2f}',
                'variation': '{:.2%}'
            })
        )
    
    with col2:
        st.caption("Variable Expenses (High Variation)")
        st.dataframe(
            variable_expenses.style.format({
                'mean': 'Rs. {:,.2f}',
                'std': 'Rs. {:,.2f}',
                'variation': '{:.2%}'
            })
        )
    

def show_pending_transactions():
    """Display pending transactions section"""
    st.subheader("📋 Pending Transactions")
    
    try:
        df = get_pending_transactions()
        
        if df.empty:
            st.info("No pending transactions found.")
            return
        
        # Create tabs for To Receive and To Pay
        to_receive = df[df['Type'] == 'To Receive'].copy()
        to_pay = df[df['Type'] == 'To Pay'].copy()
        
        # Show summary metrics first
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_to_receive = to_receive['Amount'].sum()
            st.metric("To Receive", f"Rs. {total_to_receive:,.2f}")
            
        with col2:
            total_to_pay = to_pay['Amount'].sum()
            st.metric("To Pay", f"Rs. {total_to_pay:,.2f}")
            
        with col3:
            net_pending = total_to_receive - total_to_pay
            st.metric("Net Pending", 
                     f"Rs. {net_pending:,.2f}",
                     delta=f"Rs. {net_pending:,.2f}",
                     delta_color="normal" if net_pending >= 0 else "inverse")
        
        st.divider()
        
        tab1, tab2 = st.tabs(["💰 To Receive", "💸 To Pay"])
        
        with tab1:
            if to_receive.empty:
                st.info("No pending receipts.")
            else:
                st.write("### Pending Receipts")
                # Format amount with currency
                to_receive['Amount'] = to_receive['Amount'].apply(lambda x: f"Rs. {x:,.2f}")
                # Format dates
                to_receive['Date'] = to_receive['Date'].dt.strftime('%Y-%m-%d')
                to_receive['Due Date'] = to_receive['Due Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    to_receive[['Date', 'Amount', 'Category', 'Description', 'Due Date']],
                    width="stretch"
                )
        
        with tab2:
            if to_pay.empty:
                st.info("No pending payments.")
            else:
                st.write("### Pending Payments")
                # Format amount with currency
                to_pay['Amount'] = to_pay['Amount'].apply(lambda x: f"Rs. {x:,.2f}")
                # Format dates
                to_pay['Date'] = to_pay['Date'].dt.strftime('%Y-%m-%d')
                to_pay['Due Date'] = to_pay['Due Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    to_pay[['Date', 'Amount', 'Category', 'Description', 'Due Date']],
                    width="stretch"
                )
    except Exception as e:
        log.error(f"Error displaying pending transactions: {str(e)}")
        st.error("Failed to load pending transactions. Please check the logs for details.")

def show_analytics():
    try:
        st.title(" Financial Analytics")
        
        # Add regenerate button in the header
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🔄 Refresh", width="stretch"):
                get_transactions_data.clear()
                get_pending_transactions.clear()
                st.rerun()
        
        # Get date filters once for all tabs
        start_date, end_date = get_date_filters(key="global_analytics_filter")
        
        # Get and filter data
        df = get_transactions_data()
        if not df.empty:
            df['Amount'] = pd.to_numeric(df['Amount'])
            df['Date'] = pd.to_datetime(df['Date'])
            filtered_df = filter_dataframe(df, start_date, end_date)
        else:
            filtered_df = df
        
        # Display selected period
        st.caption(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Show tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Income Analytics", "Expense Analytics", "Emotion Analytics", "Pending Transactions"])
        
        with tab1:
            show_overview_analytics(filtered_df, start_date, end_date)
        with tab2:
            show_income_analytics(filtered_df, start_date, end_date)
        with tab3:
            show_expense_analytics(filtered_df, start_date, end_date)
        with tab4:
            show_emotion_analytics(filtered_df, start_date, end_date)
        with tab5:
            show_pending_transactions()
        
        log.info("📊 Analytics visualizations generated successfully")
    except Exception as e:
        log.error(f"❌ Failed to generate analytics: {str(e)}")
        st.error("Failed to generate analytics. Please try again later.")
        
if __name__ == "__main__":
    show_analytics() 