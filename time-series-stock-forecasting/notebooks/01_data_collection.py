import yfinance as yf
import pandas as pd

# Download stock data (e.g., Apple)
data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')

# Save as CSV
data.to_csv('../data/stock_data.csv')
print("Data collected and saved to data/stock_data.csv")
