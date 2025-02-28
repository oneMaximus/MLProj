# =============================================================================
# PART 1: Data Fetching and Technical Indicator Calculation (Time, Approx_AD, VIX)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
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

def compute_indicators(data, start="2018-01-01", end="2023-01-01"):
    """
    Set up time as X, add an approximated Advance-Decline Line and VIX as technical indicators, with actual closing prices as Y.
    Note: Approx_AD is a proxy, not a true A/D Line, due to lack of advancing/declining stock data.
    """
    # Create time feature
    data['Time'] = (data.index - data.index[0]).days  # Days since first date

    # Target: Next day's closing price
    data['Y'] = data['Close'].shift(-1)  # Predict next day's closing price

    # Approximated Advance-Decline Line (proxy using price direction and volume)
    data['Price_Change'] = data['Close'] - data['Open']
    data['Approx_AD'] = (data['Price_Change'] * data['Volume']).cumsum()  # Cumulative sum of price change * volume

    # Add VIX (Volatility Index)
    vix = yf.download('^VIX', start=start, end=end)
    data['VIX'] = vix['Close'].reindex(data.index, method='ffill')  # Forward-fill to match S&P 500 dates

    # Drop NaN values
    data.dropna(inplace=True)
    return data

# Download data and compute indicators
start_date = "2018-01-01"
end_date = "2023-01-01"
data = fetch_data(start=start_date, end=end_date)
data = compute_indicators(data, start=start_date, end=end_date)

# =============================================================================
# PART 2: SVR Model Training, Tuning, and Time-Series Visualization with Indicators Plot (Test Period Only)
# =============================================================================

# Define features (Time, Approx_AD, VIX)
features = ['Time', 'Approx_AD', 'VIX']

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
print(f"Initial Model MSE (Actual Prices, Time + Approx_AD + VIX): {mse:.4f}")
print(f"Initial Model R^2 (Actual Prices, Time + Approx_AD + VIX): {r2:.4f}")

# Expanded hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train_scaled)
print(f"Best Parameters (Time + Approx_AD + VIX): {grid.best_params_}")

# Train final model with best parameters
best_svr = grid.best_estimator_
y_pred_test_best_scaled = best_svr.predict(X_test_scaled)
y_pred_test_best = scaler_Y.inverse_transform(y_pred_test_best_scaled.reshape(-1, 1)).ravel()

# Evaluate best model (on test set)
best_mse = mean_squared_error(y_test_actual, y_pred_test_best)
best_r2 = r2_score(y_test_actual, y_pred_test_best)
print(f"Best Model MSE (Actual Prices, Time + Approx_AD + VIX): {best_mse:.4f}")
print(f"Best Model R^2 (Actual Prices, Time + Approx_AD + VIX): {best_r2:.4f}")

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
ax2.plot(data.index[train_size:], y_pred_test_best, label='Predicted S&P 500 Closing Price (Test, Time + Approx_AD + VIX)', color='red', linestyle='--')
ax2.set_xlabel('Date')
ax2.set_ylabel('S&P 500 Closing Price')
ax2.set_title('SVR: Actual vs Predicted S&P 500 Closing Prices (Test Period, Time + Approx_AD + VIX, Tuned)')
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()