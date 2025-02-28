# =============================================================================
# PART 1: Data Fetching and Technical Indicator Calculation (All Indicators for MARA)
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
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Function to fetch data with manual trading day adjustment
def fetch_data(ticker="MARA", start="2015-02-28", end="2025-03-03"):  # Manually set end to March 3, 2025
    """
    Download historical data for the given ticker with a fixed end date.
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

# Compute all indicators for MARA
start_date = "2015-02-28"
end_date = "2025-03-03"  # Manually set to next trading day

# Fetch MARA data
mara_data = fetch_data(ticker="MARA", start=start_date, end=end_date)
close = mara_data['Close']
high = mara_data['High']
low = mara_data['Low']
volume = mara_data['Volume']

#Feth Bitcoin data
btc_data = fetch_data(ticker="BTC-USD", start="2015-02-28", end="2025-03-03")

# Fetch VIX data for additional volatility context (optional, as MARA is crypto-related)
vix_data = fetch_data(ticker="^VIX", start=start_date, end=end_date)
vix_close = vix_data['Close'].reindex(mara_data.index, method='ffill')

# Create DataFrame for indicators
indicators = pd.DataFrame(index=mara_data.index)

# Time feature
indicators['Time'] = (mara_data.index - mara_data.index[0]).days

# Target (next day's closing price)
indicators['Y'] = mara_data['Close'].shift(-1)

# Technical Indicators (same as S&P 500, adjusted for MARA volatility)
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
indicators['Approx_AD'] = (mara_data['Close'] - mara_data['Open']) * volume.cumsum()  # Approximated A/D Line
indicators['VIX'] = vix_close  # Volatility Index for additional context (optional)
indicators['BTC_Close'] = btc_data['Close'].reindex(mara_data.index, method='ffill').ffill().bfill()

# Drop NaN values, but keep the last row (March 3, 2025) for prediction
# First, check for NaNs and fill or drop as needed
print("NaNs in indicators before cleaning:", indicators.isna().sum())
# Fill NaNs with forward fill (ffill) or backward fill (bfill), then drop remaining NaNs
indicators = indicators.ffill().bfill()  # Forward and backward fill to handle missing values
indicators = indicators.dropna()  # Drop any remaining NaNs, but ensure the last row is preserved if possible
print("NaNs in indicators after cleaning:", indicators.isna().sum())

# Prepare data for SVR, ensuring no NaNs
data = indicators

# =============================================================================
# PART 2: SVR Model Training, Tuning, and Time-Series Visualization with Indicators Plot, Epsilon Boundaries, Confidence Intervals, and Next Day Prediction (Test Period Only)
# =============================================================================

# Define features (all indicators + Time)
features = [
    'Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_%R', 'Bollinger_Mavg',
    'OBV', 'MFI', 'VWAP', 'Approx_AD', 'VIX', 'BTC_Close'
]

# Prepare X (features) and y (actual closing prices)
X = data[features]
y = data['Y']

# Check for NaNs or infinite values again before splitting
print("NaNs in X before split:", X.isna().sum().sum())
print("Infinites in X:", np.isinf(X).sum().sum())
print("NaNs in y:", y.isna().sum())
print("Infinites in y:", np.isinf(y).sum())

# Ensure no NaNs remain by dropping any rows with NaNs in X or y
X = X.dropna()
y = y[X.index]  # Align y with X after dropping NaNs
print("NaNs in X after drop:", X.isna().sum().sum())
print("NaNs in y after drop:", y.isna().sum())

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
X_test = X[train_size + val_size:-1]  # Exclude the last row for test, keep for prediction
y_test = y[train_size + val_size:-1]

# Prepare the next day's data (last row of the dataset, March 3, 2025)
next_day_data = X.iloc[-1:]  # Last row for prediction (March 3, 2025)

# Scale features (X) and target (Y) for train, val, and test
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit scaler on training data only, transform all sets
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = scaler_Y.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_Y.transform(y_test.values.reshape(-1, 1)).ravel()

# Scale the next day's data (transform only, using scaler_X fitted on training)
next_day_scaled = scaler_X.transform(next_day_data)

# Concatenate and scale training and validation data for tuning
X_train_val_scaled = np.vstack((X_train_scaled, X_val_scaled))
y_train_val_scaled = np.concatenate((y_train_scaled, y_val_scaled))

# RandomizedSearchCV for broader hyperparameter exploration (linear kernel only)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

param_dist = {
    'C': loguniform(0.001, 1000),  # Broader C range
    'epsilon': uniform(0.001, 0.48),  # Keep epsilon range
    'kernel': ['linear']  # Focus on linear kernel for efficiency
}

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

random_search = RandomizedSearchCV(SVR(), param_dist, n_iter=30, cv=tscv, scoring='neg_mean_squared_error', n_jobs=4, random_state=42, verbose=2)
random_search.fit(X_train_val_scaled, y_train_val_scaled)

print(f"Random Search Best Parameters: {random_search.best_params_}")

# Train final model with best parameters on train + val, predict on test and next day
best_svr = random_search.best_estimator_
y_pred_test_best_scaled = best_svr.predict(X_test_scaled)
y_pred_test_best = scaler_Y.inverse_transform(y_pred_test_best_scaled.reshape(-1, 1)).ravel()

# Predict the next day's price (March 3, 2025)
next_day_pred_scaled = best_svr.predict(next_day_scaled)
next_day_pred = scaler_Y.inverse_transform(next_day_pred_scaled.reshape(-1, 1)).ravel()[0]  # Single value for next day
print(f"Predicted MARA Closing Price for March 3, 2025: {next_day_pred:.2f}")

# Evaluate best model (on test set)
best_test_mse = mean_squared_error(y_test, y_pred_test_best)
best_test_r2 = r2_score(y_test, y_pred_test_best)
print(f"Best Model MSE (Test, All Indicators): {best_test_mse:.4f}")
print(f"Best Model R^2 (Test, All Indicators): {best_test_r2:.4f}")

# Extract best epsilon for boundaries
best_epsilon = random_search.best_params_['epsilon']
print(f"Best Epsilon Value: {best_epsilon:.4f}")

# Bootstrap to estimate confidence intervals for predictions (test and next day)
n_bootstraps = 100  # Number of bootstrap samples
bootstrap_predictions_test = []
bootstrap_predictions_next_day = []

# Resample training data with replacement and retrain/predict
for _ in range(n_bootstraps):
    # Sample indices with replacement from training data
    bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_train_boot = X_train.iloc[bootstrap_indices]
    y_train_boot = y_train.iloc[bootstrap_indices]

    # Scale bootstrapped training data
    X_train_boot_scaled = scaler_X.transform(X_train_boot)  # Use transform, not fit_transform, to match scaler_X
    y_train_boot_scaled = scaler_Y.transform(y_train_boot.values.reshape(-1, 1)).ravel()

    # Train SVR on bootstrapped data
    boot_svr = SVR(**random_search.best_params_)
    boot_svr.fit(X_train_boot_scaled, y_train_boot_scaled)

    # Predict on test set
    y_pred_boot_test_scaled = boot_svr.predict(X_test_scaled)
    y_pred_boot_test = scaler_Y.inverse_transform(y_pred_boot_test_scaled.reshape(-1, 1)).ravel()
    bootstrap_predictions_test.append(y_pred_boot_test)

    # Predict on next day's data
    y_pred_boot_next_day_scaled = boot_svr.predict(next_day_scaled)
    y_pred_boot_next_day = scaler_Y.inverse_transform(y_pred_boot_next_day_scaled.reshape(-1, 1)).ravel()[0]
    bootstrap_predictions_next_day.append(y_pred_boot_next_day)

# Convert bootstrap predictions to numpy arrays
bootstrap_predictions_test = np.array(bootstrap_predictions_test)
bootstrap_predictions_next_day = np.array(bootstrap_predictions_next_day)

# Calculate 95% confidence intervals (2.5th and 97.5th percentiles) for test predictions
ci_test_lower = np.percentile(bootstrap_predictions_test, 2.5, axis=0)
ci_test_upper = np.percentile(bootstrap_predictions_test, 97.5, axis=0)

# Calculate 95% confidence interval for next day prediction (single value, so use percentiles directly)
ci_next_day_lower = np.percentile(bootstrap_predictions_next_day, 2.5)
ci_next_day_upper = np.percentile(bootstrap_predictions_next_day, 97.5)
print(f"95% Confidence Interval for March 3, 2025 MARA Prediction: [{ci_next_day_lower:.2f}, {ci_next_day_upper:.2f}]")

# Create stacked plots: Indicators above, Prices below with Epsilon Boundaries, Confidence Intervals, and Next Day Prediction
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Plot 1: Technical Indicators (normalized for visualization) for test period
test_indicators = data[features[1:]][train_size + val_size:-1]  # Exclude 'Time', use test period only (up to last day)
scaler_vis = StandardScaler()  # Normalize for visualization
normalized_indicators = scaler_vis.fit_transform(test_indicators)

for i, indicator in enumerate(features[1:]):  # Skip 'Time'
    ax1.plot(data.index[train_size + val_size:-1], normalized_indicators[:, i], label=indicator, alpha=0.5, linewidth=0.5)
ax1.set_title('Normalized Technical Indicators (Test Period, 2022–2025)')
ax1.set_ylabel('Normalized Value (Standardized)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax1.grid(True)

# Plot 2: Actual vs. Predicted Prices (test period only) with Epsilon Boundaries, Confidence Intervals, and Next Day Prediction
ax2.plot(data.index[train_size + val_size:-1], y_test, label='Actual MARA Closing Price (Test)', color='blue')
ax2.plot(data.index[train_size + val_size:-1], y_pred_test_best, label='Predicted MARA Closing Price (Test, All Indicators, Tuned)', color='red', linestyle='--')

# Calculate epsilon boundaries
epsilon_upper = y_pred_test_best + best_epsilon
epsilon_lower = y_pred_test_best - best_epsilon

# Add epsilon boundaries as a shaded region
ax2.fill_between(data.index[train_size + val_size:-1], epsilon_lower, epsilon_upper, color='black', alpha=0.2, label=f'Epsilon Tube (±{best_epsilon:.4f})')

# Add 95% confidence intervals for test predictions as a shaded region
ax2.fill_between(data.index[train_size + val_size:-1], ci_test_lower, ci_test_upper, color='lightblue', alpha=0.3, label='95% Confidence Interval (Test)')

# Add next day prediction as a point with confidence interval
next_day_date = data.index[-1]  # March 3, 2025 (next trading day after Feb 28, 2025)
ax2.plot(next_day_date, next_day_pred, 'go', label='Predicted Next Day Price (Mar 3, 2025)', markersize=10)
ax2.errorbar(next_day_date, next_day_pred, yerr=[[next_day_pred - ci_next_day_lower], [ci_next_day_upper - next_day_pred]], color='green', capsize=5, alpha=0.7, label='95% CI (Next Day)')

ax2.set_xlabel('Date')
ax2.set_ylabel('MARA Closing Price')
ax2.set_title('SVR: Actual vs Predicted MARA Closing Prices with Epsilon, Confidence Intervals, and Next Day Prediction (Test Period, All Indicators, Tuned)')
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()