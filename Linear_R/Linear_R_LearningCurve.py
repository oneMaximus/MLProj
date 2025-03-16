import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

# Import stock data function from your existing script
import yfinance as yf
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg, average_true_range
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price

# Function to fetch stock data
def fetch_data(ticker="^GSPC", start="2018-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# Fetch stock data
sp500_data = fetch_data(ticker="^GSPC")
vix_data = fetch_data(ticker="^VIX")
vix_close = vix_data['Close'].reindex(sp500_data.index, method='ffill')

# Compute technical indicators
close = sp500_data['Close']
high = sp500_data['High']
low = sp500_data['Low']
volume = sp500_data['Volume']

indicators = pd.DataFrame(index=sp500_data.index)
indicators['Y'] = close.shift(-1)  # Predict next day's closing price

# Select technical indicators (Modify as needed)
indicators['SMA'] = sma_indicator(close, window=9)
indicators['EMA'] = ema_indicator(close, window=9)
indicators['MACD'] = macd_diff(close)
indicators['RSI'] = rsi(close, window=9)
indicators['VWAP'] = volume_weighted_average_price(high, low, close, volume, window=14)

# Drop NaN values
indicators.dropna(inplace=True)

# Select features and target variable
X = indicators[['SMA', 'EMA', 'MACD', 'RSI', 'VWAP']]
y = indicators['Y']

# Time-based split for stock data
train_size = int(len(indicators) * 0.6)  # 60% for training
val_size = int(len(indicators) * 0.2)    # 20% for validation

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Generate Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Convert negative MSE to positive for readability
train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

# Plot Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Error", marker='o')
plt.plot(train_sizes, val_scores_mean, label="Validation Error", marker='s')
plt.xlabel("Training Set Size")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve - Stock Price Prediction")
plt.legend()
plt.grid(True)
plt.show()