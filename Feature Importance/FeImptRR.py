# =============================================================================
# PART 1: Data Fetching and Technical Indicator Calculation (All Indicators)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price

# Custom WMA function
def wma(prices, window):
    weights = np.arange(1, window + 1)
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Custom Momentum Indicator (MOM)
def compute_momentum(close, window):
    return close - close.shift(window)

# Fetch data function
def fetch_data(ticker="^GSPC", start="2018-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# Compute all indicators
start_date = "2018-01-01"
end_date = "2023-01-01"

# Fetch S&P 500 and VIX data
sp500_data = fetch_data(ticker="^GSPC", start=start_date, end=end_date)
vix_data = fetch_data(ticker="^VIX", start=start_date, end=end_date)
close = sp500_data['Close']
high = sp500_data['High']
low = sp500_data['Low']
volume = sp500_data['Volume']
vix_close = vix_data['Close'].reindex(sp500_data.index, method='ffill')

# Create DataFrame for indicators
indicators = pd.DataFrame(index=sp500_data.index)

# Time feature and Target
indicators['Time'] = (sp500_data.index - sp500_data.index[0]).days
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
data = indicators

# Define features
features = [
    'Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_%R', 'Bollinger_Mavg',
    'OBV', 'MFI', 'VWAP', 'Approx_AD', 'VIX'
]

# Prepare X and y
X = data[features]
y = data['Y']

# =============================================================================
# PART 2: Feature Importance Using Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Time-series, no shuffle

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print("\nFeature Importance (Random Forest):")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance for S&P 500 Prediction (Random Forest)')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

# Optional: Evaluate the model
y_pred = rf.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nRandom Forest MSE (Test Set): {mse:.4f}")
print(f"Random Forest R^2 (Test Set): {r2:.4f}")