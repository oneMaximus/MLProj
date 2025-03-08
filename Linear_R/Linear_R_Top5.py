import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg, average_true_range
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Function to fetch data
def fetch_data(ticker="^GSPC", start="2018-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# Fetch S&P 500 and VIX data
sp500_data = fetch_data(ticker="^GSPC")
vix_data = fetch_data(ticker="^VIX")
vix_close = vix_data['Close'].reindex(sp500_data.index, method='ffill')

# Compute technical indicators
close = sp500_data['Close']
high = sp500_data['High']
low = sp500_data['Low']
volume = sp500_data['Volume']

indicators = pd.DataFrame(index=sp500_data.index)
indicators['Y'] = close.shift(-1)  # Predict next day's closing price

# Trend Indicators
indicators['SMA'] = sma_indicator(close, window=9)
indicators['EMA'] = ema_indicator(close, window=9)

# Momentum Indicators
indicators['ROC'] = roc(close, window=9)

# Volatility Indicators
indicators['Bollinger_Mavg'] = bollinger_mavg(close, window=9)

# Volume Indicators
indicators['VWAP'] = volume_weighted_average_price(high, low, close, volume, window=14)

# Drop NaN values
indicators.dropna(inplace=True)

# Select the 5 specific features
selected_features = ['SMA', 'EMA', 'ROC', 'Bollinger_Mavg', 'VWAP']
X = indicators[selected_features]
y = indicators['Y']

# Time-based split for time series
train_size = int(len(indicators) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predict test data
y_pred = lin_reg.predict(X_test_scaled)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Print evaluation results
print("\n   Evaluation: Using Selected 5 Features (EMA, VWAP, SMA, Bollinger, ROC) ")
print("-" * 120)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.6f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print("-" * 120)

# Feature Importance Calculation
feature_importance = abs(lin_reg.coef_)
feature_importance_df = pd.DataFrame({"Feature": selected_features, "Importance": feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

# Display feature importance
print("\nFeature Importance (Higher = More Impactful):")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Linear Regression (Selected 5 Features)")
plt.grid(True)
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual Prices", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted Prices", color='red', linestyle='--')
plt.xlabel("Date")
plt.ylabel("S&P 500 Closing Price")
plt.title("Linear Regression: Actual vs Predicted Prices (Selected 5 Features)")
plt.legend()
plt.grid(True)
plt.show()