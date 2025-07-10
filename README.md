Time Series Stock Forecasting:
--This project focuses on predicting stock prices using various time series forecasting models, including ARIMA, SARIMA, Prophet, and LSTM. It is built as part of a data analytics internship to demonstrate real-world applications of forecasting techniques on historical stock data.

--The project includes a Streamlit-based interactive web application that allows users to visualize stock trends and compare forecasts across different models.

Project Structure:
Data Collection: Uses Yahoo Finance API via yfinance to fetch historical stock data.

Exploratory Analysis: Conducts time series visualization, decomposition, and stationarity checks.

Forecasting Models:

ARIMA & SARIMA: Classical statistical models used for linear trend predictions.

Prophet: Facebook's model used for seasonality and trend detection.

LSTM: Deep learning-based approach using TensorFlow for sequence modeling.

Evaluation Metrics: MAE, RMSE, and MAPE are used to compare model performance.

Streamlit App: Built for user interaction, allowing users to select a stock symbol, time range, and forecasting model.

How to Run Locally
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/time-series-stock-forecasting.git
cd time-series-stock-forecasting

Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt

Run the Streamlit app:
bash
Copy
Edit
streamlit run app.py

Tools and Libraries Used:
Python (Pandas, NumPy, Matplotlib, Seaborn)
Scikit-learn
Statsmodels (for ARIMA, SARIMA)
Facebook Prophet
TensorFlow (for LSTM)
Streamlit
Yahoo Finance (yfinance)

Project Goals:
--Understand and apply different time series models to real stock data
--Build a comparative framework for evaluating forecast accuracy
--Deliver an interactive forecasting tool that is easy to use and visually informative
--Future Enhancements
--Add multivariate time series support
--Deploy the Streamlit app on the cloud (e.g., Streamlit Cloud or Heroku)
--Include real-time data updates and sentiment analysis integration

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/992e7db5-9b5e-4b52-acf1-bba4da81507d" />
