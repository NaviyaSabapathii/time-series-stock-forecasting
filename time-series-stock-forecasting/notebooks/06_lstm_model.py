import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Try importing TensorFlow/Keras with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    print("Please follow the installation steps below.")
    exit(1)

def create_lstm_model():
    """Create and train LSTM model for stock price forecasting"""
    
    # Step 1: Load raw CSV, skip extra rows
    try:
        df = pd.read_csv('../data/stock_data.csv', skiprows=2)
    except FileNotFoundError:
        print("Error: stock_data.csv not found in ../data/ directory")
        return
    
    # Step 2: Rename columns for consistency
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Step 3: Use only 'Close' prices
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    close_data = df['Close'].dropna().values.reshape(-1, 1)
    
    print(f"Data shape: {close_data.shape}")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    
    # Step 4: Normalize the close prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_data)
    
    # Step 5: Prepare training sequences (60-day windows)
    sequence_length = 60
    X, y = [], []
    
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i])
    
    X, y = np.array(X), np.array(y)
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Step 6: Split data into train/test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Step 7: Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Model Summary:")
    model.summary()
    
    # Step 8: Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Step 9: Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    
    # Step 10: Plot the results
    plt.figure(figsize=(15, 8))
    
    # Create arrays for plotting
    train_plot = np.empty_like(close_data)
    train_plot[:, :] = np.nan
    train_plot[sequence_length:sequence_length+len(train_predictions), :] = train_predictions
    
    # For test plot, calculate the correct starting position
    test_start_idx = sequence_length + len(train_predictions)
    test_end_idx = test_start_idx + len(test_predictions)
    
    test_plot = np.empty_like(close_data)
    test_plot[:, :] = np.nan
    
    # Only assign if we have enough space in the array
    if test_end_idx <= len(close_data):
        test_plot[test_start_idx:test_end_idx, :] = test_predictions
    else:
        # Trim test predictions to fit
        available_space = len(close_data) - test_start_idx
        test_plot[test_start_idx:, :] = test_predictions[:available_space]
    
    plt.plot(close_data, label='Original Closing Price', color='blue', linewidth=1)
    plt.plot(train_plot, label='Training Predictions', color='orange', linewidth=1)
    plt.plot(test_plot, label='Test Predictions', color='red', linewidth=1)
    
    plt.title("AAPL Stock Price Forecast using LSTM", fontsize=16)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Step 11: Ensure 'reports' directory exists and save the plot
    os.makedirs('../reports', exist_ok=True)
    plt.savefig('../reports/lstm_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 12: Print model performance
    actual_train = close_data[sequence_length:sequence_length+len(train_predictions)]
    actual_test = close_data[test_start_idx:test_start_idx+len(test_predictions)]
    
    train_rmse = np.sqrt(np.mean((train_predictions - actual_train)**2))
    test_rmse = np.sqrt(np.mean((test_predictions[:len(actual_test)] - actual_test)**2))
    
    print(f"\nModel Performance:")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Training samples: {len(train_predictions)}")
    print(f"Test samples: {len(test_predictions)}")
    print(f"Total data points: {len(close_data)}")
    
    return model, scaler, history

if __name__ == "__main__":
    model, scaler, history = create_lstm_model()