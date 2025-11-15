"""
🤖 ML-Powered Expense Predictions
Using Facebook Prophet Time Series Forecasting with Google Sheets Integration
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.ml_predictor import ExpenseMLPredictor
from utils.data_loader import get_expense_data, validate_expense_data, get_data_summary

# Page config
st.set_page_config(
    page_title="ML Predictions",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .ml-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .insight-card {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="ml-header">
    <h1>🤖 ML Expense Forecasting</h1>
    <p style="font-size: 1.1rem; margin-bottom: 0;">
        • Machine Learning Time Series Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize ML predictor in session state
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = ExpenseMLPredictor()

if 'expenses_df' not in st.session_state:
    st.session_state.expenses_df = None

if 'ml_trained' not in st.session_state:
    st.session_state.ml_trained = False

# Sidebar controls
with st.sidebar:
    st.header("🎛️ ML Model Settings")

    forecast_days = st.slider(
        "Forecast Period (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        help="Number of days to predict into the future"
    )

    st.divider()

    # Model status
    st.subheader("📊 Model Status")

    if st.session_state.ml_trained and st.session_state.ml_predictor.is_trained():
        training_info = st.session_state.ml_predictor.get_training_info()

        st.success("✅ Model is trained")
        st.caption(f"🕐 Trained: {training_info.get('training_date', 'Unknown')}")
        st.caption(f"📊 Data points: {training_info.get('data_points', 0)}")
        st.caption(f"📅 Date range: {training_info.get('date_range_days', 0)} days")
        st.caption(f"📌 Latest data: {training_info.get('last_data_date', 'Unknown')}")

        # Check data freshness
        if training_info.get('last_data_date'):
            try:
                last_date = datetime.strptime(training_info['last_data_date'], '%Y-%m-%d')
                days_old = (datetime.now() - last_date).days

                if days_old > 7:
                    st.warning(f"⚠️ Data is {days_old} days old")
                    st.caption("Consider retraining with recent data")
            except:
                pass
    else:
        st.info("⏳ Model not trained yet")
        st.caption("Click 'Train ML Model' below to start")

    st.divider()

    # Data source toggle
    st.subheader("🔄 Data Source")
    use_mock_data = st.checkbox(
        "Use Mock Data (for testing)",
        value=False,
        help="Enable this to test with generated data instead of Google Sheets"
    )

    st.divider()

    # Retrain button
    if st.button("🔄 Retrain Model", width="stretch", type="primary"):
        with st.spinner("📊 Loading fresh data from Google Sheets..."):
            # Get fresh data
            expenses_df = get_expense_data(use_mock=use_mock_data)

            # Validate
            is_valid, message = validate_expense_data(expenses_df)

            if not is_valid:
                st.error(f"❌ {message}")
            else:
                st.success("✅ Data loaded successfully")

                with st.spinner("🔬 Retraining ML model..."):
                    # Create fresh predictor
                    st.session_state.ml_predictor = ExpenseMLPredictor()

                    # Train
                    success = st.session_state.ml_predictor.train_model(expenses_df)

                    if success:
                        # Generate predictions
                        prediction = st.session_state.ml_predictor.predict_future(days=forecast_days)

                        if 'error' not in prediction:
                            # Save to session state
                            st.session_state.prediction = prediction
                            st.session_state.expenses_df = expenses_df
                            st.session_state.ml_trained = True

                            st.success("✅ Model retrained successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ Prediction failed: {prediction['error']}")
                    else:
                        st.error("❌ Training failed")

    # Clear model button
    if st.button("🗑️ Clear Model", width="stretch"):
        st.session_state.ml_predictor = ExpenseMLPredictor()
        st.session_state.ml_trained = False
        if 'prediction' in st.session_state:
            del st.session_state.prediction
        if 'expenses_df' in st.session_state:
            del st.session_state.expenses_df
        st.success("✅ Model cleared")
        st.rerun()


# Main content
tab1, tab2, tab3 = st.tabs([
    "📈 Generate Forecast",
    "📊 Category Analysis",
    "ℹ️ About ML Model"
])

# TAB 1: Main Forecasting
with tab1:
    st.subheader("🎯 ML Expense Predictions")

    # Load data section
    with st.expander("📊 View Data Summary", expanded=not st.session_state.ml_trained):
        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("📥 Load Data from Google Sheets", width="stretch", type="primary"):
                with st.spinner("🔄 Fetching data..."):
                    expenses_df = get_expense_data(use_mock=use_mock_data)

                    if len(expenses_df) > 0:
                        st.session_state.expenses_df = expenses_df
                        summary = get_data_summary(expenses_df)

                        # Display summary
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.metric("📅 Records", summary['total_records'])
                            st.metric("💰 Total Spent", f"₹{summary['total_amount']:,.2f}")

                        with col_b:
                            st.metric("📊 Date Range", summary['date_range'])
                            st.metric("📈 Daily Avg", f"₹{summary['daily_average']:,.2f}")

                        with col_c:
                            st.metric("📌 Start", summary['start_date'])
                            st.metric("📌 End", summary['end_date'])

                        st.success("✅ Data loaded successfully!")
                    else:
                        st.error("❌ No data available")

        with col2:
            if st.session_state.expenses_df is not None:
                st.write("#### 📋 Categories Found:")
                summary = get_data_summary(st.session_state.expenses_df)
                for cat in summary['categories']:
                    st.write(f"• {cat}")

    st.divider()

    # Train button
    if not st.session_state.ml_trained:
        st.info("👆 Load data first, then train the ML model")

    if st.session_state.expenses_df is not None:
        expenses_df = st.session_state.expenses_df

        # Validate before training
        is_valid, validation_msg = validate_expense_data(expenses_df)

        if not is_valid:
            st.error(f"❌ {validation_msg}")
        else:
            st.success(validation_msg)

            if st.button("🤖 Train ML Model & Generate Predictions", type="primary", width="stretch"):
                with st.spinner("🔬 Training machine learning model..."):
                    success = st.session_state.ml_predictor.train_model(expenses_df)

                    if success:
                        st.success("✅ Model trained successfully!")

                        with st.spinner(f"🔮 Generating {forecast_days}-day forecast..."):
                            prediction = st.session_state.ml_predictor.predict_future(days=forecast_days)

                            if 'error' not in prediction:
                                st.session_state.prediction = prediction
                                st.session_state.ml_trained = True

                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ {prediction['error']}")

    # Display predictions if available
    if 'prediction' in st.session_state and st.session_state.get('ml_trained'):
        pred = st.session_state.prediction

        st.divider()
        st.subheader("🔮 ML Forecast Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric-box">
                <h4 style="margin: 0; color: #667eea;">Total Predicted</h4>
                <h2 style="margin: 0.5rem 0;">₹{:,.2f}</h2>
                <p style="margin: 0; color: #666;">Next {} days</p>
            </div>
            """.format(pred['total_predicted'], forecast_days), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-box">
                <h4 style="margin: 0; color: #667eea;">Daily Average</h4>
                <h2 style="margin: 0.5rem 0;">₹{:,.2f}</h2>
                <p style="margin: 0; color: #666;">Per day</p>
            </div>
            """.format(pred['daily_average']), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-box">
                <h4 style="margin: 0; color: #28a745;">Lowest Day</h4>
                <h2 style="margin: 0.5rem 0;">₹{:,.2f}</h2>
                <p style="margin: 0; color: #666;">{}</p>
            </div>
            """.format(pred['min_day']['amount'], pred['min_day']['date']), unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-box">
                <h4 style="margin: 0; color: #dc3545;">Highest Day</h4>
                <h2 style="margin: 0.5rem 0;">₹{:,.2f}</h2>
                <p style="margin: 0; color: #666;">{}</p>
            </div>
            """.format(pred['max_day']['amount'], pred['max_day']['date']), unsafe_allow_html=True)

        st.divider()

        # Chart
        st.subheader("📊 Interactive Forecast Visualization")
        fig = st.session_state.ml_predictor.create_forecast_chart(expenses_df, pred)
        st.plotly_chart(fig, width="stretch")

        st.divider()

        # AI Insights
        st.subheader("🧠 ML-Powered Insights")

        current_total = expenses_df['amount'].sum()
        insights = st.session_state.ml_predictor.generate_insights(
            current_total,
            pred['total_predicted']
        )

        for insight in insights:
            if '🚨' in insight or '⚠️' in insight:
                st.markdown(f"""
                <div class="warning-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="insight-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)

        # Comparison with historical
        with st.expander("📊 Compare with Historical Data"):
            col1, col2 = st.columns(2)

            hist_total = expenses_df['amount'].sum()
            hist_avg = expenses_df['amount'].mean()

            with col1:
                st.write("### 📅 Historical")
                st.metric("Total Spent", f"₹{hist_total:,.2f}")
                st.metric("Daily Average", f"₹{hist_avg:,.2f}")
                st.metric("Highest Day", f"₹{expenses_df['amount'].max():,.2f}")
                st.metric("Lowest Day", f"₹{expenses_df['amount'].min():,.2f}")

            with col2:
                st.write("### 🔮 Predicted")
                st.metric("Total Predicted", f"₹{pred['total_predicted']:,.2f}")
                st.metric("Daily Average", f"₹{pred['daily_average']:,.2f}")
                st.metric("Highest Day", f"₹{pred['max_day']['amount']:,.2f}")
                st.metric("Lowest Day", f"₹{pred['min_day']['amount']:,.2f}")

            # Calculate differences
            total_diff = pred['total_predicted'] - hist_total
            avg_diff = pred['daily_average'] - hist_avg

            st.divider()
            st.write("### 📈 Change Analysis")

            col_a, col_b = st.columns(2)
            with col_a:
                change_pct = (total_diff / hist_total * 100) if hist_total > 0 else 0
                st.metric(
                    "Total Change",
                    f"₹{abs(total_diff):,.2f}",
                    f"{change_pct:+.1f}%"
                )

            with col_b:
                avg_change_pct = (avg_diff / hist_avg * 100) if hist_avg > 0 else 0
                st.metric(
                    "Avg Change",
                    f"₹{abs(avg_diff):,.2f}",
                    f"{avg_change_pct:+.1f}%"
                )

# TAB 2: Category Analysis
with tab2:
    st.subheader("📊 Category-wise ML Predictions")

    if st.session_state.expenses_df is not None:
        expenses_df = st.session_state.expenses_df

        if 'category' in expenses_df.columns:
            categories = expenses_df['category'].unique()

            st.write(f"Found **{len(categories)}** expense categories")

            selected_category = st.selectbox(
                "Select Category to Analyze",
                options=categories,
                help="Choose a category to get ML predictions"
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                if st.button("🔮 Predict Category Spending", width="stretch", type="primary"):
                    with st.spinner(f"Analyzing {selected_category}..."):
                        result = st.session_state.ml_predictor.get_category_forecast(
                            expenses_df,
                            selected_category,
                            days=30
                        )

                        if 'error' not in result:
                            st.success("✅ Prediction generated!")

                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.metric(
                                    f"Predicted 30-Day: {selected_category}",
                                    f"₹{result['predicted_total']:,.2f}"
                                )

                            with col_b:
                                st.metric(
                                    "Daily Average",
                                    f"₹{result['daily_average']:,.2f}"
                                )

                            # Historical comparison
                            category_data = expenses_df[expenses_df['category'] == selected_category]
                            hist_total = category_data['amount'].sum()

                            st.divider()
                            st.write("### 📊 Historical vs Predicted")

                            comparison_df = pd.DataFrame({
                                'Metric': ['Total', 'Daily Average'],
                                'Historical': [
                                    f"₹{hist_total:,.2f}",
                                    f"₹{category_data['amount'].mean():,.2f}"
                                ],
                                'Predicted (30 days)': [
                                    f"₹{result['predicted_total']:,.2f}",
                                    f"₹{result['daily_average']:,.2f}"
                                ]
                            })

                            st.dataframe(comparison_df, width="stretch", hide_index=True)

                            # --- MODIFICATION START ---
                            # Show forecast chart instead of simple historical trend
                            st.divider()
                            st.write(f"### 📈 Forecast Visualization for {selected_category}")
                            
                            # Create and display the interactive forecast chart
                            category_fig = st.session_state.ml_predictor.create_category_forecast_chart(
                                category_name=selected_category,
                                historical_df=result['historical_df'],
                                forecast_df=result['forecast_df']
                            )
                            st.plotly_chart(category_fig, width="stretch")
                            # --- MODIFICATION END ---

                        else:
                            st.warning(f"⚠️ {result['error']}")

            with col2:
                # Category statistics
                if selected_category:
                    category_data = expenses_df[expenses_df['category'] == selected_category]

                    st.write("#### 📊 Stats")
                    st.metric("Transactions", len(category_data))
                    st.metric("Total Spent", f"₹{category_data['amount'].sum():,.2f}")
                    st.metric("Average", f"₹{category_data['amount'].mean():,.2f}")
                    st.metric("Max", f"₹{category_data['amount'].max():,.2f}")

            st.divider()

            # All categories summary
            st.subheader("📋 All Categories Overview")

            category_summary = expenses_df.groupby('category').agg({
                'amount': ['sum', 'mean', 'count']
            }).round(2)

            category_summary.columns = ['Total Spent', 'Average', 'Count']
            category_summary = category_summary.sort_values('Total Spent', ascending=False)

            # Format currency
            category_summary['Total Spent'] = category_summary['Total Spent'].apply(lambda x: f"₹{x:,.2f}")
            category_summary['Average'] = category_summary['Average'].apply(lambda x: f"₹{x:,.2f}")

            st.dataframe(category_summary, width="stretch")
        else:
            st.warning("⚠️ No category information available in data")
    else:
        st.info("👆 Load data first in the 'Generate Forecast' tab")

# TAB 3: About ML Model
with tab3:
    st.subheader("ℹ️ About the Machine Learning Model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🤖 Facebook Prophet
        
        **Prophet** is an open-source forecasting tool developed by Facebook's Data Science team.
        
        #### Key Features:
        - **Automatic seasonality detection**
        - **Trend analysis**
        - **Holiday effects**
        - **Confidence intervals**
        - **Robust to missing data**
        
        #### How It Works:
        1. **Decomposition**: Breaks down data into trend, seasonality, and holidays
        2. **Pattern Learning**: Identifies recurring patterns in your spending
        3. **Forecasting**: Projects patterns into the future
        4. **Uncertainty**: Provides confidence intervals (95% by default)
        """)

    with col2:
        st.markdown("""
        ### 📊 Model Performance
        
        #### Training Data:
        - Minimum: 7 days of expenses
        - Recommended: 30+ days
        - Optimal: 60+ days
        
        #### Accuracy Factors:
        - **Data Quality**: More consistent data = better predictions
        - **Seasonality**: Weekly/monthly patterns improve accuracy
        - **Outliers**: Unusual expenses may affect predictions
        
        #### Confidence Interval:
        The shaded area in the forecast represents the **95% confidence interval** - 
        meaning there's a 95% probability the actual value will fall within this range.
        
        #### Use Cases:
        - **Budget Planning**: Plan future budgets based on ML predictions
        - **Trend Detection**: Identify spending trends early
        - **Anomaly Detection**: Spot unusual spending patterns
        - **Goal Setting**: Set realistic saving goals
        """)

    st.divider()

    st.markdown("""
    ### 📚 Technical Details
    
    **Algorithm**: Additive Regression Model
    
    **Formula**: `y(t) = g(t) + s(t) + h(t) + ε(t)`
    
    Where:
    - `g(t)`: Trend (growth)
    - `s(t)`: Seasonality (daily, weekly)
    - `h(t)`: Holiday effects
    - `ε(t)`: Error term
    
    **Library**: `prophet` by Facebook Research
    
    **Model Training Time**: 2-5 seconds (depending on data size)
    
    **Prediction Speed**: <1 second for 90-day forecast
    """)

    st.info("💡 **Pro Tip**: The model gets more accurate as you add more expense data over time!")

# Footer
st.divider()
st.caption("📊 Advanced Time Series Forecasting | 🔬 95% Confidence Intervals")