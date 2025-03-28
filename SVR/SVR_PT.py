# =============================================================================
# PART 1: Data Fetching and Technical Indicator Calculation (Time, SMA, EMA, WMA, MACD, Parabolic_SAR, Ichimoku Cloud)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator, ichimoku_a
import matplotlib.pyplot as plt
import seaborn as sns
import itertools, random

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def fetch_data(ticker="^GSPC", start="2018-01-01", end="2023-01-01"):
    """
    Download historical data for the given ticker.
    """
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def wma(prices, window):
    """
    Calculate Weighted Moving Average, giving higher weights to recent prices.
    """
    weights = np.arange(1, window + 1)  # Weights: 1, 2, ..., 9 for a 9-day WMA
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def compute_indicators(data, window=9):
    """
    Set up time as X, add SMA, EMA, WMA, MACD, Parabolic_SAR, and Ichimoku Cloud (Tenkan-sen) as technical indicators, with actual closing prices as Y.
    """
    # Create time feature
    data['Time'] = (data.index - data.index[0]).days  # Days since first date

    # Target: Next day's closing price
    data['Y'] = data['Close'].shift(-1)  # Predict next day's closing price

    # Technical indicators: SMA, EMA, WMA, MACD, Parabolic_SAR, and Ichimoku Cloud (Tenkan-sen)
    close = data['Close']
    high = data['High']
    low = data['Low']
    data['SMA'] = sma_indicator(close, window=window)
    data['EMA'] = ema_indicator(close, window=window)
    data['WMA'] = wma(close, window=window)
    data['MACD'] = macd_diff(close)  # Default uses 12, 26, 9 (fast, slow, signal); we use diff for simplicity
    data['Parabolic_SAR'] = psar_down_indicator(high, low, close)  # Uses default AF (0.02, max 0.2)
    # Ichimoku Cloud: Tenkan-sen (9-period average, as per document)
    data['Ichimoku_Tenkan_sen'] = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2

    # Drop NaN values
    data.dropna(inplace=True)
    return data

# Download data and compute indicators
data = fetch_data()
data = compute_indicators(data, window=9)

# =============================================================================
# PART 2: SVR Model Training, Tuning, and Time-Series Visualization with Indicators Plot (Test Period Only)
# =============================================================================

# Define features (Time, SMA, EMA, WMA, MACD, Parabolic_SAR, Ichimoku_Tenkan_sen)
features = ['Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen']

# Prepare X (features) and y (actual closing prices)
X = data[features]
y = data['Y']

# Time-based split for time series
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale features (X) and target (Y)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_Y.transform(y_test.values.reshape(-1, 1)).ravel()

# Train initial SVR
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
svr.fit(X_train_scaled, y_train_scaled)

# Predict for test set
y_pred_test_scaled = svr.predict(X_test_scaled)

# Inverse transform predictions to get actual prices for test set
y_pred_test = scaler_Y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
y_test_actual = y_test.values  # Actual test prices

# Evaluate initial model (on test set)
mse = mean_squared_error(y_test_actual, y_pred_test)
r2 = r2_score(y_test_actual, y_pred_test)
print(f"Initial Model MSE (Actual Prices, Time + SMA + EMA + WMA + MACD + Parabolic_SAR + Ichimoku_Tenkan_sen): {mse:.4f}")
print(f"Initial Model R^2 (Actual Prices, Time + SMA + EMA + WMA + MACD + Parabolic_SAR + Ichimoku_Tenkan_sen): {r2:.4f}")

# Expanded hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train_scaled)
print(f"Best Parameters (Time + SMA + EMA + WMA + MACD + Parabolic_SAR + Ichimoku_Tenkan_sen): {grid.best_params_}")

# Train final model with best parameters
best_svr = grid.best_estimator_
y_pred_test_best_scaled = best_svr.predict(X_test_scaled)
y_pred_test_best = scaler_Y.inverse_transform(y_pred_test_best_scaled.reshape(-1, 1)).ravel()

# Evaluate best model (on test set)
best_mse = mean_squared_error(y_test_actual, y_pred_test_best)
best_r2 = r2_score(y_test_actual, y_pred_test_best)
print(f"Best Model MSE (Actual Prices, Time + SMA + EMA + WMA + MACD + Parabolic_SAR + Ichimoku_Tenkan_sen): {best_mse:.4f}")
print(f"Best Model R^2 ( Actual Prices, Time + SMA + EMA + WMA + MACD + Parabolic_SAR + Ichimoku_Tenkan_sen): {best_r2:.4f}")

# Create stacked plots: Indicators above, Prices below
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Plot 1: Technical Indicators (normalized for visualization) for test period
test_indicators = data[features][train_size:]  # Use all features, including 'Time', for test period only
scaler_vis = StandardScaler()  # Normalize for visualization
normalized_indicators = scaler_vis.fit_transform(test_indicators)

for i, indicator in enumerate(features):  # Include all features, including 'Time'
    ax1.plot(data.index[train_size:], normalized_indicators[:, i], label=indicator, alpha=0.7, linewidth=1.0)
ax1.set_title('Normalized Technical Indicators (Test Period, 2022–2023)')
ax1.set_ylabel('Normalized Value (Standardized)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax1.grid(True)

# Plot 2: Actual vs. Predicted Prices (test period only)
ax2.plot(data.index[train_size:], y_test_actual, label='Actual S&P 500 Closing Price (Test)', color='blue')
ax2.plot(data.index[train_size:], y_pred_test_best, label='Predicted S&P 500 Closing Price (Test, Time + SMA + EMA + WMA + MACD + Parabolic_SAR + Ichimoku_Tenkan_sen)', color='red', linestyle='--')
ax2.set_xlabel('Date')
ax2.set_ylabel('S&P 500 Closing Price')
ax2.set_title('SVR: Actual vs Predicted S&P 500 Closing Prices (Test Period, Time + SMA + EMA + WMA + MACD + Parabolic_SAR + Ichimoku_Tenkan_sen, Tuned)')
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()