import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os

# Load the CSV, skipping the first 2 rows
df = pd.read_csv('../data/stock_data.csv', skiprows=2)

# Rename columns manually
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert 'Date' to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Convert all columns to numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
series = df['Close'].dropna()

# Fit SARIMA model
model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecast the next 30 days
forecast = results.forecast(steps=30)

# Create future date range
last_date = series.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Plot the result
plt.figure(figsize=(12, 6))
plt.plot(series[-100:], label='Original (last 100 days)', color='blue')
plt.plot(future_dates, forecast, label='SARIMA Forecast (next 30 days)', color='green', linestyle='--')
plt.title("SARIMA Model Forecast - AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Ensure the 'reports' directory exists
os.makedirs('../reports', exist_ok=True)

# Save the plot
plt.savefig('../reports/sarima_forecast.png', dpi=300)
plt.show()

# Print forecast summary
print("Next 5-day forecast:")
for i in range(5):
    print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${forecast.iloc[i]:.2f}")

print(f"\n30-day average forecast: ${forecast.mean():.2f}")
print(f"Last known price: ${series.iloc[-1]:.2f}")
change = ((forecast.mean() - series.iloc[-1]) / series.iloc[-1]) * 100
print(f"Predicted % change over next 30 days: {change:.2f}%")
