from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load CSV and skip the first 2 rows
df = pd.read_csv('../data/stock_data.csv', skiprows=2)

# Step 2: Rename columns correctly
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Step 3: Convert date column and keep only Date and Close
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']].dropna()

# Step 4: Rename for Prophet
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Step 5: Fit Prophet model
model = Prophet()
model.fit(df)

# Step 6: Forecast next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Step 7: Plot forecast
fig = model.plot(forecast)
# Plot the forecast
fig = model.plot(forecast)
fig.tight_layout()

# Add custom title and labels
plt.title("AAPL Stock Price Forecast using Prophet", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Close Price ($)", fontsize=12)

fig.tight_layout()

# Ensure the reports folder exists
os.makedirs('../reports', exist_ok=True)

# Save plot
fig.savefig('../reports/prophet_forecast.png', dpi=300)
plt.show()

# Step 8: Print forecast summary
print("Next 5-day forecast:")

print(forecast[['ds', 'yhat']].tail(35).head(5).to_string(index=False))

print(f"\n30-day average forecast: ${forecast['yhat'][-30:].mean():.2f}")
print(f"Last known price: ${df['y'].iloc[-1]:.2f}")
change = ((forecast['yhat'][-30:].mean() - df['y'].iloc[-1]) / df['y'].iloc[-1]) * 100
print(f"Predicted % change over next 30 days: {change:.2f}%")
