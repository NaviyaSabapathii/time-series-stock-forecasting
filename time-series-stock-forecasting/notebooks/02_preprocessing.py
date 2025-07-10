import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Show original structure for debug
df_raw = pd.read_csv('../data/stock_data.csv')
print("Original CSV structure:")
print(df_raw.head())

# Step 2: Skip first 2 rows (Ticker/Label Info) to access actual data
df_clean = pd.read_csv('../data/stock_data.csv', skiprows=2)
print("\nAfter skipping first 2 rows:")
print(df_clean.head())

# Step 3: Rename columns manually
df_clean.rename(columns={
    df_clean.columns[0]: "Date",
    df_clean.columns[1]: "Close",
    df_clean.columns[2]: "High",
    df_clean.columns[3]: "Low",
    df_clean.columns[4]: "Open",
    df_clean.columns[5]: "Volume"
}, inplace=True)

# Step 4: Convert 'Date' to datetime and set as index
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean.set_index('Date', inplace=True)

# Step 5: Ensure all columns are numeric (safe parsing)
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Step 6: Display structure after cleaning
print("\nAfter processing:")
print(df_clean.head())
print(f"\nData types:\n{df_clean.dtypes}")
print(f"\nMissing values:\n{df_clean.isnull().sum()}")

# Step 7: Drop rows with missing 'Close' values
df_clean = df_clean.dropna(subset=['Close'])

# Step 8: Plot the Closing Price trend
plt.figure(figsize=(12, 6))
plt.plot(df_clean['Close'], label='Close Price', color='blue', linewidth=1.5)
plt.title('AAPL Stock Closing Price Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Step 8.1: Ensure 'reports' directory exists before saving
os.makedirs('../reports', exist_ok=True)

# Save the plot
plt.savefig('../reports/closing_price_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 9: Summary stats
print(f"\n Stock price range: ${df_clean['Close'].min():.2f} - ${df_clean['Close'].max():.2f}")
print(f" Total trading days: {len(df_clean)}")

# Step 10: Save cleaned data for modeling
df_clean.to_csv('../data/cleaned_stock_data.csv')
print("\n Cleaned data saved to: data/cleaned_stock_data.csv")
