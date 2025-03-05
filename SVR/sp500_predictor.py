import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator, ichimoku_a
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg, average_true_range
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Data fetching function
def fetch_data(ticker, start, end):
    """Download historical data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# Custom WMA function
def wma(prices, window):
    weights = np.arange(1, window + 1)
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Custom Momentum Indicator
def compute_momentum(close, window):
    return close - close.shift(window)

# Main prediction logic
start_date = "2018-01-01"  # Historical start date
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')  # Today, March 5, 2025

print(f"Fetching data from {start_date} to {end_date}")

# Fetch S&P 500 and VIX data
sp500_data = fetch_data("^GSPC", start_date, end_date)
vix_data = fetch_data("^VIX", start_date, end_date)

# Debug: Print the last date in sp500_data
print(f"Last date in S&P 500 data: {sp500_data.index[-1]}")

# Prepare data for indicators
close = sp500_data['Close']
high = sp500_data['High']
low = sp500_data['Low']
volume = sp500_data['Volume']
vix_close = vix_data['Close'].reindex(sp500_data.index, method='ffill')

# Create indicators DataFrame
indicators = pd.DataFrame(index=sp500_data.index)
indicators['Time'] = (sp500_data.index - sp500_data.index[0]).days
indicators['Y'] = sp500_data['Close'].shift(-1)  # Next day's close as target

# Technical Indicators
window = 9
indicators['SMA'] = sma_indicator(close, window=window)
indicators['EMA'] = ema_indicator(close, window=window)
indicators['WMA'] = wma(close, window=window)
indicators['MACD'] = macd_diff(close)
indicators['Parabolic_SAR'] = psar_down_indicator(high, low, close)
indicators['Ichimoku_Tenkan_sen'] = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
indicators['RSI'] = rsi(close, window=window)
indicators['Stochastic_Oscillator'] = stoch(high, low, close, window=window, smooth_window=3)
indicators['ROC'] = roc(close, window=window)
indicators['MOM'] = compute_momentum(close, window=window)
indicators['Williams_%R'] = williams_r(high, low, close, lbp=window)
indicators['Bollinger_Mavg'] = bollinger_mavg(close, window=window)
indicators['OBV'] = on_balance_volume(close, volume)
indicators['MFI'] = money_flow_index(high, low, close, volume, window=14)
indicators['VWAP'] = volume_weighted_average_price(high, low, close, volume, window=14)
indicators['Approx_AD'] = (sp500_data['Close'] - sp500_data['Open']) * volume.cumsum()
indicators['VIX'] = vix_close

# Clean data and check for NaNs
data = indicators.ffill().bfill()
print("NaNs in indicators before dropna:", data.isna().sum().sum())
data = data.dropna()
print("NaNs in indicators after dropna:", data.isna().sum().sum())

# Features list
features = [
    'Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_%R', 'Bollinger_Mavg',
    'OBV', 'MFI', 'VWAP', 'Approx_AD', 'VIX'
]

# 100% Training
X_full = data[features]
y_full = data['Y']
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_full_scaled = scaler_X.fit_transform(X_full)
y_full_scaled = scaler_Y.fit_transform(y_full.values.reshape(-1, 1)).ravel()

# Use your best parameters from GridSearchCV
best_params = {
    'C': 1000,
    'epsilon': 0.1,
    'gamma': 'scale',
    'kernel': 'rbf'
}

svr = SVR(**best_params)
svr.fit(X_full_scaled, y_full_scaled)

# Predict today's closing price (March 5, 2025) using the last row (March 4, 2025)
last_date = sp500_data.index[-1]  # Last date in data (e.g., March 4, 2025)

print(f"Adjusted last trading day: {last_date}")

# Get today's date (March 5, 2025)
today = pd.Timestamp.now().strftime('%Y-%m-%d')

# Use the last row of data to predict today's closing price
last_row = data.iloc[-1:]
X_last = last_row[features]

# Check for NaNs in last_row before scaling
print("NaNs in last_row before scaling:", X_last.isna().sum().sum())
X_last_scaled = scaler_X.transform(X_last)

# Predict today's closing price (March 5, 2025)
y_pred_scaled = svr.predict(X_last_scaled)
y_pred = scaler_Y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]

print(f"Predicted S&P 500 Close for {today}: {y_pred:.2f}")