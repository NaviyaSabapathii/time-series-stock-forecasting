import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

# Suppress all warnings including TensorFlow
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid warnings

# Model imports with error handling
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    # Additional TensorFlow warning suppression
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecasting Dashboard",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_stock_data(symbol, start_date, end_date):
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data):
    """Preprocess the data for modeling"""
    try:
        # Reset index to make Date a column
        df = data.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have the right columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = ['Date'] + [col for col in df.columns if col != 'Date']
        
        # Map available columns to required columns
        if len(available_columns) >= 6:
            df.columns = required_columns[:len(available_columns)]
        
        # Remove any missing values
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    try:
        # Ensure arrays are the same length
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        
        # Avoid division by zero
        actual_nonzero = np.where(actual != 0, actual, 1)
        mape = np.mean(np.abs((actual - predicted) / actual_nonzero)) * 100
        
        return {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'MAPE': round(mape, 2)
        }
    except Exception as e:
        return {'RMSE': 0, 'MAE': 0, 'MAPE': 0}

def fit_arima_model(data, order=(5,1,0)):
    """Fit ARIMA model with error handling"""
    if not ARIMA_AVAILABLE:
        return None, None, "ARIMA library not available"
    
    try:
        series = data['Close']
        
        # Split data
        train_size = int(len(series) * 0.8)
        train_data = series[:train_size]
        test_data = series[train_size:]
        
        if len(train_data) < 10:
            return None, None, "Not enough data for ARIMA"
        
        # Fit model
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        
        # Forecast
        forecast_steps = min(len(test_data), 30)
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Calculate metrics
        test_actual = test_data[:forecast_steps]
        metrics = calculate_metrics(test_actual.values, forecast)
        
        return {
            'model': model_fit,
            'forecast': forecast,
            'test_actual': test_actual,
            'train_data': train_data,
            'metrics': metrics
        }, None, None
        
    except Exception as e:
        return None, None, f"ARIMA error: {str(e)}"

def fit_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Fit SARIMA model with error handling"""
    if not SARIMA_AVAILABLE:
        return None, None, "SARIMA library not available"
    
    try:
        series = data['Close']
        
        # Split data
        train_size = int(len(series) * 0.8)
        train_data = series[:train_size]
        test_data = series[train_size:]
        
        if len(train_data) < 24:  # Need more data for seasonal
            return None, None, "Not enough data for SARIMA"
        
        # Fit model
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        
        # Forecast
        forecast_steps = min(len(test_data), 30)
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Calculate metrics
        test_actual = test_data[:forecast_steps]
        metrics = calculate_metrics(test_actual.values, forecast)
        
        return {
            'model': model_fit,
            'forecast': forecast,
            'test_actual': test_actual,
            'train_data': train_data,
            'metrics': metrics
        }, None, None
        
    except Exception as e:
        return None, None, f"SARIMA error: {str(e)}"

def fit_prophet_model(data):
    """Fit Prophet model with error handling"""
    if not PROPHET_AVAILABLE:
        return None, None, "Prophet library not available"
    
    try:
        # Prepare data
        df_prophet = data[['Date', 'Close']].copy()
        df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Split data
        train_size = int(len(df_prophet) * 0.8)
        train_data = df_prophet[:train_size]
        test_data = df_prophet[train_size:]
        
        if len(train_data) < 10:
            return None, None, "Not enough data for Prophet"
        
        # Fit model
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        model.fit(train_data)
        
        # Forecast
        forecast_steps = min(len(test_data), 30)
        test_dates = test_data['ds'].head(forecast_steps)
        future_df = pd.DataFrame({'ds': test_dates})
        forecast = model.predict(future_df)
        
        # Calculate metrics
        test_actual = test_data['y'].head(forecast_steps)
        metrics = calculate_metrics(test_actual.values, forecast['yhat'].values)
        
        return {
            'model': model,
            'forecast': forecast['yhat'],
            'test_actual': test_actual,
            'train_data': train_data['y'],
            'metrics': metrics
        }, None, None
        
    except Exception as e:
        return None, None, f"Prophet error: {str(e)}"

def fit_lstm_model(data, sequence_length=60, epochs=20):
    """Fit LSTM model with error handling"""
    if not LSTM_AVAILABLE:
        return None, None, "LSTM libraries not available"
    
    try:
        # Prepare data
        close_data = data['Close'].values.reshape(-1, 1)
        
        if len(close_data) < sequence_length + 50:
            return None, None, "Not enough data for LSTM"
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model with additional warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            
            # Make predictions
            test_predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_actual = scaler.inverse_transform(y_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test_actual.flatten(), test_predictions.flatten())
        
        return {
            'model': model,
            'forecast': test_predictions.flatten(),
            'test_actual': y_test_actual.flatten(),
            'scaler': scaler,
            'metrics': metrics
        }, None, None
        
    except Exception as e:
        return None, None, f"LSTM error: {str(e)}"

def plot_forecast(data, results, model_name, symbol):
    """Create forecast plot"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data (last 100 points)
        recent_data = data['Close'].tail(100)
        ax.plot(range(len(recent_data)), recent_data.values, 
                label='Historical Price', color='blue', linewidth=2)
        
        # Plot predictions if available
        if results and 'forecast' in results:
            forecast = results['forecast']
            forecast_x = range(len(recent_data), len(recent_data) + len(forecast))
            ax.plot(forecast_x, forecast, 
                    label=f'{model_name} Forecast', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f'{symbol} Stock Price Forecast - {model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header"> Stock Price Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header(" Configuration")
        
        # Stock selection
        symbol = st.selectbox(
            "Select Stock Symbol",
            ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            index=0
        )
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        
        start_date = st.date_input("Start Date", value=start_date)
        end_date = st.date_input("End Date", value=end_date)
        
        # Model selection
        st.subheader(" Model Selection")
        use_arima = st.checkbox("ARIMA", value=ARIMA_AVAILABLE)
        use_sarima = st.checkbox("SARIMA", value=SARIMA_AVAILABLE)
        use_prophet = st.checkbox("Prophet", value=PROPHET_AVAILABLE)
        use_lstm = st.checkbox("LSTM", value=LSTM_AVAILABLE)
        
        # Run analysis button
        run_analysis = st.button(" Run Analysis", type="primary")
    
    # Main content
    if run_analysis:
        # Load data
        with st.spinner(" Loading stock data..."):
            raw_data = load_stock_data(symbol, start_date, end_date)
            
            if raw_data is None or raw_data.empty:
                st.error(" Failed to load data. Please check your inputs.")
                return
            
            data = preprocess_data(raw_data)
            if data is None:
                st.error(" Failed to preprocess data.")
                return
        
        st.success(" Data loaded successfully!")
        
        # Display stock info
        st.subheader(f" {symbol} Stock Information")
        
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}")
        with col2:
            st.metric("Price Change %", f"{price_change_pct:+.2f}%")
        with col3:
            st.metric("52W High", f"${data['Close'].max():.2f}")
        with col4:
            st.metric("52W Low", f"${data['Close'].min():.2f}")
        
        # Price chart
        st.subheader(" Price History")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Date'], data['Close'], linewidth=2, color='#1f77b4')
        ax.set_title(f'{symbol} Stock Price History', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Model results
        st.subheader(" Model Results")
        
        results = {}
        model_tabs = []
        
        # ARIMA
        if use_arima:
            with st.spinner("Training ARIMA model..."):
                arima_results, arima_model, arima_error = fit_arima_model(data)
                if arima_error:
                    st.warning(f"ARIMA: {arima_error}")
                else:
                    results['ARIMA'] = arima_results
                    model_tabs.append('ARIMA')
        
        # SARIMA
        if use_sarima:
            with st.spinner("Training SARIMA model..."):
                sarima_results, sarima_model, sarima_error = fit_sarima_model(data)
                if sarima_error:
                    st.warning(f"SARIMA: {sarima_error}")
                else:
                    results['SARIMA'] = sarima_results
                    model_tabs.append('SARIMA')
        
        # Prophet
        if use_prophet:
            with st.spinner("Training Prophet model..."):
                prophet_results, prophet_model, prophet_error = fit_prophet_model(data)
                if prophet_error:
                    st.warning(f"Prophet: {prophet_error}")
                else:
                    results['Prophet'] = prophet_results
                    model_tabs.append('Prophet')
        
        # LSTM
        if use_lstm:
            with st.spinner("Training LSTM model..."):
                lstm_results, lstm_model, lstm_error = fit_lstm_model(data, epochs=20)
                if lstm_error:
                    st.warning(f"LSTM: {lstm_error}")
                else:
                    results['LSTM'] = lstm_results
                    model_tabs.append('LSTM')
        
        # Display results
        if results:
            # Performance comparison
            st.subheader(" Model Performance Comparison")
            
            performance_data = []
            for model_name, result in results.items():
                metrics = result['metrics']
                performance_data.append({
                    'Model': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'MAPE': f"{metrics['MAPE']:.2f}%"
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Best model
            best_model = min(results.items(), key=lambda x: x[1]['metrics']['RMSE'])
            st.success(f" Best Model: {best_model[0]} (RMSE: {best_model[1]['metrics']['RMSE']:.4f})")
            
            # Model tabs
            if model_tabs:
                tabs = st.tabs(model_tabs)
                
                for i, (model_name, tab) in enumerate(zip(model_tabs, tabs)):
                    with tab:
                        result = results[model_name]
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RMSE", f"{result['metrics']['RMSE']:.4f}")
                        with col2:
                            st.metric("MAE", f"{result['metrics']['MAE']:.4f}")
                        with col3:
                            st.metric("MAPE", f"{result['metrics']['MAPE']:.2f}%")
                        
                        # Plot
                        fig = plot_forecast(data, result, model_name, symbol)
                        if fig:
                            st.pyplot(fig)
                        
                        # Forecast summary
                        if 'forecast' in result:
                            forecast_mean = np.mean(result['forecast'])
                            st.info(f" Average Forecast: ${forecast_mean:.2f}")
        
        else:
            st.warning(" No models were successfully trained. Please check your data and try again.")
    
    else:
        # Initial page
        st.info(" Configure your settings in the sidebar and click 'Run Analysis' to start!")
        
        st.subheader(" Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ** Data Analysis:**
            - Real-time stock data from Yahoo Finance
            - Historical price trends and statistics
            - Interactive visualizations
            
            ** Machine Learning Models:**
            - ARIMA: Classical time series model
            - SARIMA: Seasonal ARIMA model
            - Prophet: Facebook's forecasting tool
            - LSTM: Deep learning neural network
            """)
        
        with col2:
            st.markdown("""
            ** Performance Metrics:**
            - RMSE: Root Mean Square Error
            - MAE: Mean Absolute Error
            - MAPE: Mean Absolute Percentage Error
            
            ** Forecasting:**
            - Future price predictions
            - Model comparison
            - Best model recommendations
            """)
        
        # Library status
        st.subheader(" Library Status")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.write(" ARIMA Available" if ARIMA_AVAILABLE else " ARIMA Not Available")
            st.write(" SARIMA Available" if SARIMA_AVAILABLE else " SARIMA Not Available")
        
        with status_col2:
            st.write(" Prophet Available" if PROPHET_AVAILABLE else " Prophet Not Available")
            st.write(" LSTM Available" if LSTM_AVAILABLE else " LSTM Not Available")
        
        if not any([ARIMA_AVAILABLE, SARIMA_AVAILABLE, PROPHET_AVAILABLE, LSTM_AVAILABLE]):
            st.error(" No forecasting libraries available. Please install required packages.")
            st.code("""
            pip install yfinance pandas numpy matplotlib seaborn scikit-learn
            pip install statsmodels prophet tensorflow streamlit
            """)

if __name__ == "__main__":
    main()


    
