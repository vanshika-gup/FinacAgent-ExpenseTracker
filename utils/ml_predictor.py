"""
ML-Powered Expense Prediction using Facebook Prophet
Time Series Forecasting Model
"""
import pandas as pd
from prophet import Prophet
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go

class ExpenseMLPredictor:
    """Machine Learning model for expense forecasting"""
    
    def __init__(self):
        self.model = None
        self.forecast_data = None
        self.training_metadata = {
            'trained': False,
            'training_date': None,
            'data_points': 0,
            'date_range_days': 0,
            'last_data_date': None
        }
    
    def prepare_data(self, expenses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert expense data to Prophet format
        
        Args:
            expenses_df: DataFrame with 'date' and 'amount' columns
        
        Returns:
            DataFrame in Prophet format (ds, y)
        """
        # Make a copy to avoid modifying original
        df = expenses_df.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and sum amounts
        daily_expenses = df.groupby('date')['amount'].sum().reset_index()
        
        # Rename columns for Prophet (ds = datestamp, y = value)
        daily_expenses.columns = ['ds', 'y']
        
        # Sort by date
        daily_expenses = daily_expenses.sort_values('ds').reset_index(drop=True)
        
        return daily_expenses
    
    def train_model(self, expenses_df: pd.DataFrame) -> bool:
        """
        Train Prophet ML model on historical expense data
        
        Args:
            expenses_df: Historical expense data
        
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            df = self.prepare_data(expenses_df)
            
            if len(df) < 7:
                st.warning("⚠️ Need at least 7 days of expense data for ML predictions")
                return False
            
            # Initialize Prophet with optimized parameters
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,  # Flexibility of trend changes
                seasonality_prior_scale=10,     # Strength of seasonality
                interval_width=0.95            # 95% confidence interval
            )
            
            # Train the model
            with st.spinner("🤖 Training ML model on your expense history..."):
                self.model.fit(df)
            
            # Store training metadata
            self.training_metadata = {
                'trained': True,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(df),
                'date_range_days': (df['ds'].max() - df['ds'].min()).days,
                'last_data_date': df['ds'].max().strftime('%Y-%m-%d')
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error training ML model: {e}")
            return False
    
    def predict_future(self, days: int = 30) -> dict:
        """
        Generate ML predictions for future expenses
        
        Args:
            days: Number of days to predict
        
        Returns:
            dict: Prediction results with metrics and forecast data
        """
        if not self.model:
            return {"error": "Model not trained. Please train the model first."}
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=days, freq='D')
            
            # Generate predictions
            forecast = self.model.predict(future)
            
            # Get future predictions only
            future_forecast = forecast.tail(days).copy()
            
            # Calculate metrics
            total_predicted = future_forecast['yhat'].sum()
            daily_avg = future_forecast['yhat'].mean()
            max_day = future_forecast.loc[future_forecast['yhat'].idxmax()]
            min_day = future_forecast.loc[future_forecast['yhat'].idxmin()]
            
            self.forecast_data = forecast
            
            return {
                'success': True,
                'total_predicted': round(total_predicted, 2),
                'daily_average': round(daily_avg, 2),
                'max_day': {
                    'date': max_day['ds'].strftime('%Y-%m-%d'),
                    'amount': round(max_day['yhat'], 2)
                },
                'min_day': {
                    'date': min_day['ds'].strftime('%Y-%m-%d'),
                    'amount': round(min_day['yhat'], 2)
                },
                'forecast_df': future_forecast,
                'full_forecast': forecast
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def generate_insights(self, current_spending: float, predicted_spending: float) -> list:
        """
        Generate ML-powered insights based on predictions
        
        Args:
            current_spending: Current period total spending
            predicted_spending: Predicted period total spending
        
        Returns:
            list: Actionable insights
        """
        insights = []
        
        # Calculate change percentage
        if current_spending > 0:
            change_pct = ((predicted_spending / current_spending) - 1) * 100
            
            if change_pct > 10:
                insights.append(f"🚨 **ML Alert**: Spending predicted to increase by {change_pct:.1f}%")
                insights.append(f"💰 Expected additional spending: ₹{abs(predicted_spending - current_spending):,.2f}")
                insights.append("💡 **Recommendation**: Review discretionary expenses and set spending limits")
            
            elif change_pct < -10:
                insights.append(f"✅ **Good Trend**: Spending predicted to decrease by {abs(change_pct):.1f}%")
                insights.append(f"💵 Expected savings: ₹{abs(predicted_spending - current_spending):,.2f}")
                insights.append("💡 **Recommendation**: Consider allocating savings to investments")
            
            else:
                insights.append(f"📊 **Stable Pattern**: Spending expected to remain consistent (±{abs(change_pct):.1f}%)")
                insights.append("💡 **Recommendation**: Maintain current spending habits")
        
        # Trend analysis
        if self.forecast_data is not None:
            trend = self.forecast_data['trend'].tail(30).mean()
            if trend > current_spending / 30 * 1.05:
                insights.append("📈 **ML Insight**: Upward spending trend detected")
            elif trend < current_spending / 30 * 0.95:
                insights.append("📉 **ML Insight**: Downward spending trend detected")
        
        return insights
    
    def create_forecast_chart(self, expenses_df: pd.DataFrame, forecast_result: dict):
        """
        Create interactive Plotly chart showing historical and predicted expenses
        
        Args:
            expenses_df: Historical expense data
            forecast_result: Result from predict_future()
        
        Returns:
            plotly.graph_objects.Figure
        """
        historical = self.prepare_data(expenses_df)
        forecast_df = forecast_result['forecast_df']
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['ds'],
            y=historical['y'],
            mode='lines+markers',
            name='Historical Expenses',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Predicted data
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines+markers',
            name='ML Prediction',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Upper confidence bound
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            name='Upper Bound (95% CI)',
            line=dict(color='rgba(255,127,14,0.3)', width=0),
            showlegend=True,
            hovertemplate='Upper: ₹%{y:,.2f}<extra></extra>'
        ))
        
        # Lower confidence bound with fill
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            name='Lower Bound (95% CI)',
            fill='tonexty',
            line=dict(color='rgba(255,127,14,0.3)', width=0),
            hovertemplate='Lower: ₹%{y:,.2f}<extra></extra>'
        ))
        
        # --- MODIFICATION START ---
        
        # Get the last date and convert to a Python datetime object
        last_historical_date = historical['ds'].max().to_pydatetime()
        
        # 1. Add the vertical line WITHOUT annotation text
        fig.add_vline(
            x=last_historical_date,
            line_dash="dot",
            line_color="gray"
        )
        
        # 2. Add the annotation for the line separately
        fig.add_annotation(
            x=last_historical_date,
            y=1.05,
            yref="paper", # Positions the 'y' value relative to the plotting area
            text="Today",
            showarrow=False,
            font=dict(
                color="gray"
            )
        )
        
        # --- MODIFICATION END ---
        
        # Update layout
        fig.update_layout(
            title='🤖 ML-Powered Expense Forecast (Facebook Prophet)',
            xaxis_title='Date',
            yaxis_title='Amount (₹)',
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        return fig
    
    def get_category_forecast(self, expenses_df: pd.DataFrame, category: str, days: int = 30) -> dict:
        """
        Predict spending for a specific category
        
        Args:
            expenses_df: Expense data with category column
            category: Category to predict
            days: Days to forecast
        
        Returns:
            dict: Category-specific prediction
        """
        try:
            # Filter by category
            category_df = expenses_df[expenses_df['category'] == category].copy()
            
            if len(category_df) < 5:
                return {"error": f"Not enough data for {category} (need at least 5 records)"}
            
            # Train model
            df = self.prepare_data(category_df)
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            model.fit(df)
            
            # Predict
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            future_forecast = forecast.tail(days)
            
            total_predicted = future_forecast['yhat'].sum()
            
            return {
                'category': category,
                'predicted_total': round(total_predicted, 2),
                'daily_average': round(total_predicted / days, 2),
                'historical_df': df,         # Add historical data
                'forecast_df': forecast,     # Add the full forecast data
                'success': True
            }

            
        except Exception as e:
            return {"error": str(e)}

    def create_category_forecast_chart(self, category_name: str, historical_df: pd.DataFrame, forecast_df: pd.DataFrame):
        """
        Create interactive Plotly chart for a single category's forecast
        
        Args:
            category_name: The name of the category
            historical_df: The historical data for the category
            forecast_df: The forecasted data for the category
        
        Returns:
            plotly.graph_objects.Figure
        """
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_df['ds'],
            y=historical_df['y'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=5)
        ))
        
        # Predicted data
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Prediction',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 127, 14, 0.2)',
            showlegend=False
        ))

        fig.update_layout(
            title=f'Forecast for {category_name}',
            xaxis_title='Date',
            yaxis_title='Amount',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self.training_metadata['trained']
    
    def get_training_info(self) -> dict:
        """Get training metadata"""
        return self.training_metadata