import pandas as pd

# Sample comparison
data = {
    'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
    'RMSE': [1.23, 1.15, 1.10, 1.08],
    'MAE': [0.89, 0.85, 0.80, 0.75],
    'MAPE': ['3.5%', '3.0%', '2.8%', '2.6%']
}

df = pd.DataFrame(data)
df.to_csv('../reports/model_comparison.csv', index=False)
print(df)
