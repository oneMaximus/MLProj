# =============================================================================
# PART 1: Data Fetching and Technical Indicator Calculation (Time, OBV, MFI, VWAP)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price
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

def compute_indicators(data, mfi_window=14, vwap_window=14):
    """
    Set up time as X, add OBV, MFI, and VWAP as technical indicators, with actual closing prices as Y.
    """
    # Create time feature
    data['Time'] = (data.index - data.index[0]).days  # Days since first date

    # Target: Next day's closing price
    data['Y'] = data['Close'].shift(-1)  # Predict next day's closing price

    # Technical indicators: OBV, MFI, and VWAP
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    data['OBV'] = on_balance_volume(close, volume)  # On-Balance Volume
    data['MFI'] = money_flow_index(high, low, close, volume, window=mfi_window)  # 14-period MFI
    data['VWAP'] = volume_weighted_average_price(high, low, close, volume, window=vwap_window)  # 14-period VWAP

    # Drop NaN values
    data.dropna(inplace=True)
    return data

# Download data and compute indicators
data = fetch_data()
data = compute_indicators(data, mfi_window=14, vwap_window=14)

# =============================================================================
# PART 2: SVR Model Training, Tuning, and Time-Series Visualization (Test Period Only)
# =============================================================================

# Define features (Time, OBV, MFI, VWAP)
features = ['Time', 'OBV', 'MFI', 'VWAP']

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
print(f"Initial Model MSE (Actual Prices, Time + OBV + MFI + VWAP): {mse:.4f}")
print(f"Initial Model R^2 (Actual Prices, Time + OBV + MFI + VWAP): {r2:.4f}")

# Expanded hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train_scaled)
print(f"Best Parameters (Time + OBV + MFI + VWAP): {grid.best_params_}")

# Train final model with best parameters
best_svr = grid.best_estimator_
y_pred_test_best_scaled = best_svr.predict(X_test_scaled)
y_pred_test_best = scaler_Y.inverse_transform(y_pred_test_best_scaled.reshape(-1, 1)).ravel()

# Evaluate best model (on test set)
best_mse = mean_squared_error(y_test_actual, y_pred_test_best)
best_r2 = r2_score(y_test_actual, y_pred_test_best)
print(f"Best Model MSE (Actual Prices, Time + OBV + MFI + VWAP): {best_mse:.4f}")
print(f"Best Model R^2 (Actual Prices, Time + OBV + MFI + VWAP): {best_r2:.4f}")

# Create time-series plot for test period only
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size:], y_test_actual, label='Actual S&P 500 Closing Price (Test)', color='blue')
plt.plot(data.index[train_size:], y_pred_test_best, label='Predicted S&P 500 Closing Price (Test, Time + OBV + MFI + VWAP)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('S&P 500 Closing Price')
plt.title('SVR: Actual vs Predicted S&P 500 Closing Prices (Test Period, Time + OBV + MFI + VWAP, Tuned)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()