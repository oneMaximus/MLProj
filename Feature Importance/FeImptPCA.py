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

# Prepare X (features only, no target needed for PCA yet)
X = data[features]

# =============================================================================
# PART 2: Feature Importance Using PCA
# =============================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Standardize the features (PCA requires this due to scale differences)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
n_components = min(X.shape[1], 5)  # Use 5 components or fewer if features < 5
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained Variance Ratio per Principal Component:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
print(f"Total Explained Variance: {sum(explained_variance_ratio):.4f} ({sum(explained_variance_ratio)*100:.2f}%)")

# Feature loadings (contribution of each feature to the principal components)
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=features)

# Calculate feature importance as the absolute sum of loadings across components
feature_importance = np.abs(loadings).sum(axis=1)
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print("\nFeature Importance (Sum of Absolute PCA Loadings):")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lightcoral')
plt.xlabel('Sum of Absolute Loadings Across PCs')
plt.title('Feature Importance for S&P 500 Indicators (PCA)')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

# Optional: Plot loadings for the first two principal components
plt.figure(figsize=(10, 6))
plt.bar(features, loadings['PC1'], alpha=0.5, label='PC1', color='blue')
plt.bar(features, loadings['PC2'], alpha=0.5, label='PC2', color='orange')
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Loadings')
plt.title('Feature Loadings for PC1 and PC2')
plt.legend()
plt.tight_layout()
plt.show()