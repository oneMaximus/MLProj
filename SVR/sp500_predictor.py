import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator, ichimoku_a
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg, average_true_range
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Updated Data fetching function
def fetch_data(ticker="^GSPC", start="2018-01-01", end=None):
    """
    Download historical data for the given ticker up to the latest available date.
    
    Parameters:
    - ticker (str): Stock ticker symbol (default: "^GSPC" for S&P 500).
    - start (str): Start date in 'YYYY-MM-DD' format (default: "2018-01-01").
    - end (str or None): End date in 'YYYY-MM-DD' format. If None, uses today's date (default: None).
    
    Returns:
    - pd.DataFrame: Historical data for the ticker.
    """
    if end is None:
        end = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# Custom WMA function
def wma(prices, window):
    weights = np.arange(1, window + 1)  # Weights: 1, 2, ..., 9 for a 9-day WMA
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Custom Momentum Indicator (MOM)
def compute_momentum(close, window):
    return close - close.shift(window)

# Compute all indicators
start_date = "2018-01-01"

print(f"Fetching data from {start_date} to today ({pd.Timestamp.now().strftime('%Y-%m-%d')})")

# Fetch S&P 500 data (end date defaults to today)
sp500_data = fetch_data(ticker="^GSPC", start=start_date)
close = sp500_data['Close']
high = sp500_data['High']
low = sp500_data['Low']
volume = sp500_data['Volume']

# Fetch VIX data (end date defaults to today)
vix_data = fetch_data(ticker="^VIX", start=start_date)
vix_close = vix_data['Close'].reindex(sp500_data.index, method='ffill')

# Create DataFrame for indicators
indicators = pd.DataFrame(index=sp500_data.index)

# Time feature
indicators['Time'] = (sp500_data.index - sp500_data.index[0]).days

# Target (next day's close)
indicators['Y'] = sp500_data['Close'].shift(-1)

# Technical Indicators
window = 9  # Default window for most indicators
indicators['SMA'] = sma_indicator(close, window=window)
indicators['EMA'] = ema_indicator(close, window=window)
indicators['WMA'] = wma(close, window=window)
indicators['MACD'] = macd_diff(close)  # Default uses 12, 26, 9; we use diff
indicators['Parabolic_SAR'] = psar_down_indicator(high, low, close)  # Default AF (0.02, max 0.2)
indicators['Ichimoku_Tenkan_sen'] = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2  # 9-period Tenkan-sen
indicators['RSI'] = rsi(close, window=window)  # 9-period RSI
indicators['Stochastic_Oscillator'] = stoch(high, low, close, window=window, smooth_window=3)  # 9-period %K
indicators['ROC'] = roc(close, window=window)  # 9-period Rate of Change
indicators['MOM'] = compute_momentum(close, window=window)  # 9-period Momentum
indicators['Williams_%R'] = williams_r(high, low, close, lbp=window)  # 9-period Williams %R
indicators['Bollinger_Mavg'] = bollinger_mavg(close, window=window)  # 9-period middle Bollinger Band
indicators['OBV'] = on_balance_volume(close, volume)  # On-Balance Volume
indicators['MFI'] = money_flow_index(high, low, close, volume, window=14)  # 14-period MFI
indicators['VWAP'] = volume_weighted_average_price(high, low, close, volume, window=14)  # 14-period VWAP
indicators['Approx_AD'] = (sp500_data['Close'] - sp500_data['Open']) * volume.cumsum()  # Approximated A/D Line
indicators['VIX'] = vix_close  # Volatility Index

# Drop NaN values
indicators.dropna(inplace=True)

# Prepare data for SVR
data = indicators

# Define features (all indicators + Time)
features = [
    'Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_%R', 'Bollinger_Mavg',
    'OBV', 'MFI', 'VWAP', 'Approx_AD', 'VIX'
]

# Prepare X (features) and y (actual closing prices)
X = data[features]
y = data['Y']

# Check for NaNs or infinite values
print("NaNs in X:", X.isna().sum().sum())
print("Infinites in X:", np.isinf(X).sum().sum())
print("NaNs in y:", y.isna().sum())
print("Infinites in y:", np.isinf(y).sum())

# Time-based split for 60% train, 20% validation, 20% test
total_size = len(data)
train_size = int(total_size * 0.6)  # 60% for training
val_size = int(total_size * 0.2)    # 20% for validation
test_size = total_size - train_size - val_size  # 20% for testing

# Split the data
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

# Scale features (X) and target (Y) for train and test
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit scaler on training data only, transform all sets
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_Y.transform(y_test.values.reshape(-1, 1)).ravel()

# Hyperparameter tuning using scaled train data
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000], 
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],  
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  
    'kernel': ['rbf', 'linear']  
}

grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train_scaled)

print(f"Best Parameters (All Indicators, Train): {grid.best_params_}")

# Train final model with best parameters on train, predict on test
best_svr = grid.best_estimator_
y_pred_test_scaled = best_svr.predict(X_test_scaled)
y_pred_test = scaler_Y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
y_test_actual = y_test.values  # Actual test prices

# Evaluate best model (on test set)
test_mse = mean_squared_error(y_test_actual, y_pred_test)
test_r2 = r2_score(y_test_actual, y_pred_test)
print(f"Test MSE (Test, All Indicators): {test_mse:.4f}")
print(f"Test R^2 (Test, All Indicators): {test_r2:.4f}")

# Predict today's closing price using the last row
last_date = sp500_data.index[-1]  # Last date in data (e.g., March 14, 2025)
print(f"Last date in data: {last_date}")

# Use the last row of data to predict today's closing price
today = pd.Timestamp.now().strftime('%Y-%m-%d')
last_row = data.iloc[-1:]
X_last = last_row[features]

# Check for NaNs in last_row before scaling
print("NaNs in last_row before scaling:", X_last.isna().sum().sum())
X_last_scaled = scaler_X.transform(X_last)

# Predict today's closing price
y_pred_scaled = best_svr.predict(X_last_scaled)
y_pred = scaler_Y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]

print(f"Predicted S&P 500 Close for {today}: {y_pred:.2f}")