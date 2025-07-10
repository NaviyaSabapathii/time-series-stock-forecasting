import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Load raw CSV (skip first 2 rows for actual data)
df = pd.read_csv('../data/stock_data.csv', skiprows=2)

#  Rename columns (based on your cleaned structure)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert date and set index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Convert Close column to numeric safely
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Extract close price series and drop any missing values
series = df['Close'].dropna()

print(f"\n Series length: {len(series)}")
print(f"  Date range: {series.index.min().date()} to {series.index.max().date()}")
print(f" Price range: ${series.min():.2f} - ${series.max():.2f}")

# Split train/test (last 30 days as test)
train_size = len(series) - 30
train_data = series[:train_size]
test_data = series[train_size:]

print(f"\n Training data: {len(train_data)} days")
print(f" Testing data: {len(test_data)} days")

# Fit ARIMA model
print("\n Fitting ARIMA(5,1,0) model...")
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit()

# Forecast for test period
forecast_test = model_fit.forecast(steps=len(test_data))

# Forecast next 30 days into the future
forecast_future = model_fit.forecast(steps=30)
last_date = series.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, forecast_test))
print(f"\n Validation RMSE: ${rmse:.2f}")

# Plot the results
plt.figure(figsize=(15, 8))

plt.plot(train_data.index, train_data, label='Training Data', color='blue', alpha=0.7)
plt.plot(test_data.index, test_data, label='Actual Test Data', color='green', linewidth=2)
plt.plot(test_data.index, forecast_test, label='Test Forecast', color='orange', linestyle='--', linewidth=2)
plt.plot(future_dates, forecast_future, label='Future Forecast', color='red', linestyle='--', linewidth=2)

# Optional: Confidence interval (mocked as ±5%)
plt.fill_between(future_dates, forecast_future * 0.95, forecast_future * 1.05, 
                 alpha=0.2, color='red', label='Confidence Band')

plt.title('AAPL Stock Price Forecast - ARIMA Model', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Ensure reports folder exists
os.makedirs('../reports', exist_ok=True)
plt.savefig('../reports/arima_forecast.png', dpi=300)
plt.show()

# Print summary
print("\n ARIMA Model Summary:")
print("=" * 50)
print(f"Model Order: (5,1,0)")
print(f"AIC: {model_fit.aic:.2f}")
print(f"BIC: {model_fit.bic:.2f}")
print(f"RMSE on test data: ${rmse:.2f}")

# Print next 5-day forecast
print(f"\n Next 5 days forecast:")
for i in range(5):
    print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${forecast_future.iloc[i]:.2f}")

# Additional metrics
print(f"\n 30-day average forecast: ${forecast_future.mean():.2f}")
print(f" Current price: ${series.iloc[-1]:.2f}")
change = ((forecast_future.mean() - series.iloc[-1]) / series.iloc[-1]) * 100
print(f" Predicted change over next 30 days: {change:.1f}%")
