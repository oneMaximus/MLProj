# =============================================================================
# PART 1: Data Fetching and Technical Indicator Calculation (All Indicators)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator, ichimoku_a
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg, average_true_range
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

# Custom WMA function
def wma(prices, window):
    weights = np.arange(1, window + 1)  # Weights: 1, 2, ..., 9 for a 9-day WMA
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Custom Momentum Indicator (MOM)
def compute_momentum(close, window):
    return close - close.shift(window)

# Compute all indicators
start_date = "2018-01-01"
end_date = "2023-01-01"

# Fetch S&P 500 data
sp500_data = fetch_data(ticker="^GSPC", start=start_date, end=end_date)
close = sp500_data['Close']
high = sp500_data['High']
low = sp500_data['Low']
volume = sp500_data['Volume']

# Fetch VIX data
vix_data = fetch_data(ticker="^VIX", start=start_date, end=end_date)
vix_close = vix_data['Close'].reindex(sp500_data.index, method='ffill')

# Create DataFrame for indicators
indicators = pd.DataFrame(index=sp500_data.index)

# Time feature
indicators['Time'] = (sp500_data.index - sp500_data.index[0]).days

# Target
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

# =============================================================================
# PART 2: SVR Model Training, Tuning, and Time-Series Visualization (Test Period Only)
# =============================================================================

# Define features (all indicators + Time)
features = [
    'Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_%R', 'Bollinger_Mavg',
    'OBV', 'MFI', 'VWAP', 'Approx_AD', 'VIX'
]

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
print(f"Initial Model MSE (Actual Prices, All Indicators): {mse:.4f}")
print(f"Initial Model R^2 (Actual Prices, All Indicators): {r2:.4f}")

# Expanded hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train_scaled)
print(f"Best Parameters (All Indicators): {grid.best_params_}")

# Train final model with best parameters
best_svr = grid.best_estimator_
y_pred_test_best_scaled = best_svr.predict(X_test_scaled)
y_pred_test_best = scaler_Y.inverse_transform(y_pred_test_best_scaled.reshape(-1, 1)).ravel()

# Evaluate best model (on test set)
best_mse = mean_squared_error(y_test_actual, y_pred_test_best)
best_r2 = r2_score(y_test_actual, y_pred_test_best)
print(f"Best Model MSE (Actual Prices, All Indicators): {best_mse:.4f}")
print(f"Best Model R^2 (Actual Prices, All Indicators): {best_r2:.4f}")

# Extract best epsilon for boundaries
best_epsilon = grid.best_params_['epsilon']
print(f"Best Epsilon Value: {best_epsilon:.4f}")

# Create stacked plots: Indicators above, Prices below with Epsilon Boundaries
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Plot 1: Technical Indicators (normalized for visualization) for test period
test_indicators = data[features[1:]][train_size:]  # Exclude 'Time', use test period only
scaler_vis = StandardScaler()  # Normalize for visualization
normalized_indicators = scaler_vis.fit_transform(test_indicators)

for i, indicator in enumerate(features[1:]):  # Skip 'Time'
    ax1.plot(data.index[train_size:], normalized_indicators[:, i], label=indicator, alpha=0.5, linewidth=0.5)
ax1.set_title('Normalized Technical Indicators (Test Period, 2022–2023)')
ax1.set_ylabel('Normalized Value (Standardized)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax1.grid(True)

# Plot 2: Actual vs. Predicted Prices (test period only) with Epsilon Boundaries
ax2.plot(data.index[train_size:], y_test_actual, label='Actual S&P 500 Closing Price (Test)', color='blue')
ax2.plot(data.index[train_size:], y_pred_test_best, label='Predicted S&P 500 Closing Price (Test, All Indicators, Tuned)', color='red', linestyle='--')

# Calculate epsilon boundaries
epsilon_upper = y_pred_test_best + best_epsilon
epsilon_lower = y_pred_test_best - best_epsilon

# Add epsilon boundaries as a shaded region
ax2.fill_between(data.index[train_size:], epsilon_lower, epsilon_upper, color='black', alpha=0.2, label=f'Epsilon Tube (±{best_epsilon:.4f})')

ax2.set_xlabel('Date')
ax2.set_ylabel('S&P 500 Closing Price')
ax2.set_title('SVR: Actual vs Predicted S&P 500 Closing Prices with Epsilon Boundaries (Test Period, All Indicators, Tuned)')
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()