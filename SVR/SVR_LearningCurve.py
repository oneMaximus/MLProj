import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator, ichimoku_a
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg, average_true_range
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

# Data fetching function
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
    weights = np.arange(1, window + 1)
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Custom Momentum Indicator
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

# Train initial SVR
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
svr.fit(X_train_scaled, y_train_scaled)

# Predict for validation set (initial model)
y_pred_val_scaled = svr.predict(X_val_scaled)
y_pred_val = scaler_Y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).ravel()

# Evaluate initial model (on validation set)
val_mse = mean_squared_error(y_val, y_pred_val)
val_r2 = r2_score(y_val, y_pred_val)
print(f"Initial Model MSE (Validation, All Indicators): {val_mse:.4f}")
print(f"Initial Model R^2 (Validation, All Indicators): {val_r2:.4f}")

# Predict for test set (initial model)
y_pred_test_scaled = svr.predict(X_test_scaled)
y_pred_test = scaler_Y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
y_test_actual = y_test.values

# Evaluate initial model (on test set)
test_mse = mean_squared_error(y_test_actual, y_pred_test)
test_r2 = r2_score(y_test_actual, y_pred_test)
print(f"Initial Model MSE (Test, All Indicators): {test_mse:.4f}")
print(f"Initial Model R^2 (Test, All Indicators): {test_r2:.4f}")

# Expanded hyperparameter tuning using scaled train and validation data
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Concatenate scaled training and validation data for tuning
X_train_val_scaled = np.vstack((X_train_scaled, X_val_scaled))
y_train_val_scaled = np.concatenate((y_train_scaled, y_val_scaled))

grid = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid.fit(X_train_val_scaled, y_train_val_scaled)

print(f"Best Parameters (All Indicators, Train + Val): {grid.best_params_}")

# Train final model with best parameters on train + val, predict on test
best_svr = grid.best_estimator_
y_pred_test_best_scaled = best_svr.predict(X_test_scaled)
y_pred_test_best = scaler_Y.inverse_transform(y_pred_test_best_scaled.reshape(-1, 1)).ravel()

# Evaluate best model (on test set)
best_test_mse = mean_squared_error(y_test_actual, y_pred_test_best)
best_test_r2 = r2_score(y_test_actual, y_pred_test_best)
print(f"Best Model MSE (Test, All Indicators): {best_test_mse:.4f}")
print(f"Best Model R^2 (Test, All Indicators): {best_test_r2:.4f}")

# Generate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    best_svr, X_train_val_scaled, y_train_val_scaled, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error'
)

# Calculate mean and standard deviation for training and validation scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = -np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot the Learning Curve
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_sizes, train_mean, label='Training MSE (Train + Val Data)', color='blue')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
ax.plot(train_sizes, val_mean, label='Validation MSE (Cross-Validation)', color='orange')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.1)
ax.set_title('Learning Curve for SVR (Mean Squared Error vs. Training Size, Train + Val: 80%)')
ax.set_xlabel('Training Size (Number of Samples)')
ax.set_ylabel('Mean Squared Error')
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()