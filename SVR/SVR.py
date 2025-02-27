# =============================================================================
# PART 1: Data Fetching and Technical Indicator Calculation
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import ta
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
    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def compute_indicators(data, window=9):
    """
    Compute a full set of technical indicators and add them to the DataFrame.
    Also add the target variable for SVR (percentage price change).
    """
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # --- Target Variable for SVR ---
    # Percentage change in Close price (next day's change for prediction)
    data['Y'] = data['Close'].pct_change().shift(-1) * 100  # Shift -1 to predict next day

    # --- Price Trend Indicators ---
    data['SMA'] = ta.trend.sma_indicator(close, window=window)
    data['EMA'] = ta.trend.ema_indicator(close, window=window)
    data['WMA'] = ta.trend.wma_indicator(close, window=window)
    data['MACD'] = ta.trend.macd_diff(close)
    data['Parabolic_SAR'] = ta.trend.psar_down_indicator(high, low, close)
    data['Ichimoku'] = ta.trend.ichimoku_a(high, low)

    # --- Momentum Indicators ---
    data['RSI'] = ta.momentum.rsi(close, window=window)
    data['Stochastic_Oscillator'] = ta.momentum.stoch(high, low, close)
    data['ROC'] = ta.momentum.roc(close, window=window)
    data['MOM'] = ta.momentum.roc(close, window=window)  # Same as ROC
    data['Williams_R'] = ta.momentum.williams_r(high, low, close)

    # --- Volatility Indicators ---
    data['Bollinger_Mavg'] = ta.volatility.bollinger_mavg(close)
    data['ATR'] = ta.volatility.average_true_range(high, low, close)

    # --- Volume Indicators ---
    data['OBV'] = ta.volume.on_balance_volume(close, volume)
    data['Accum_Dist'] = ta.volume.acc_dist_index(high, low, close, volume)
    data['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=window)
    data['VWAP'] = ta.volume.volume_weighted_average_price(high, low, close, volume, window=window)

    # Drop rows with NaN values (from indicators and target)
    data.dropna(inplace=True)
    return data

# Download data and compute all indicators once
data = fetch_data()
data = compute_indicators(data, window=9)

# Optional: Print to verify
print(data[['Close', 'Y', 'SMA', 'EMA', 'RSI']].head())

# =============================================================================
# Visualization: Technical Indicators Before and After Normalization
# =============================================================================
features = [
    'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_R',
    'Bollinger_Mavg', 'ATR', 'OBV', 'Accum_Dist', 'MFI', 'VWAP'
]

# Plot 1: Raw (Unscaled) Technical Indicators
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(5, 4, i)  # 5 rows, 4 columns (adjust if needed for 17 features)
    plt.plot(data.index, data[feature], label=feature)
    plt.title(f'Raw {feature}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scale the features for visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Convert scaled array back to DataFrame with original feature names and index
data_scaled = pd.DataFrame(X_scaled, columns=features, index=data.index)

# Plot 2: Scaled (Normalized) Technical Indicators
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(5, 4, i)  # 5 rows, 4 columns (adjust if needed)
    plt.plot(data_scaled.index, data_scaled[feature], label=feature)
    plt.title(f'Scaled {feature}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (z-score)')
    plt.legend()
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# PART 2: SVR Model Training and Evaluation
# =============================================================================
# [Your existing Part 2 code remains unchanged]
# Define feature columns (all technical indicators)
features = [
    'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_R',
    'Bollinger_Mavg', 'ATR', 'OBV', 'Accum_Dist', 'MFI', 'VWAP'
]

# Prepare X (features) and y (target)
X = data[features]
y = data['Y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (SVR is sensitive to scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train SVR model
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
svr.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svr.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Optional: Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual % Change')
plt.ylabel('Predicted % Change')
plt.title('SVR: Predicted vs Actual S&P 500 % Change')
plt.legend()
plt.show()

# Optional: Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto', 0.1]
}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid.best_params_}")
print(f"Best MSE (negative): {grid.best_score_:.4f}")

# Train final model with best parameters
best_svr = grid.best_estimator_
y_pred_best = best_svr.predict(X_test_scaled)
best_mse = mean_squared_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)
print(f"Best Model MSE: {best_mse:.4f}")
print(f"Best Model R^2: {best_r2:.4f}")