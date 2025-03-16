#Importing all Libraries,
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


#Function for Fetching Data,
def fetch_data(ticker="^GSPC", start="2018-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


#Fetch all S&P 500 Data,
data = fetch_data()
print(data.head())


#Calculate Technical Indicators,
def compute_indicators(data, window=9):
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

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
    data['MOM'] = ta.momentum.roc(close, window=window)  # same as ROC
    data['Williams_R'] = ta.momentum.williams_r(high, low, close)

    # --- Volatility Indicators ---
    data['Bollinger_Mavg'] = ta.volatility.bollinger_mavg(close)
    data['ATR'] = ta.volatility.average_true_range(high, low, close)

    # --- Volume Indicators ---
    data['OBV'] = ta.volume.on_balance_volume(close, volume)
    data['Accum_Dist'] = ta.volume.acc_dist_index(high, low, close, volume)
    data['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=window)
    data['VWAP'] = ta.volume.volume_weighted_average_price(high, low, close, volume, window=window)


    #Drop NaN Values,
    data.dropna(inplace=True)
    return data

# Compute indicators
data = compute_indicators(data)
print(data.head())


#Define Target Variable,
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)  #Drop last row with NaN target


#Define Features Columns,
full_features = ['SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku',
                 'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_R',
                 'Bollinger_Mavg', 'ATR', 'OBV', 'Accum_Dist', 'MFI', 'VWAP']
X = data[full_features]
y = data['Target']


#Standardize features,
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Splitting Dataset,
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#Training using Logistic Regression,
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)


#Model Predictions,
y_pred = log_reg.predict(X_test)


#Model Accuracy,
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


#Feature Importance of Contribution (Absolute Values of Coefficients),
feature_importance = abs(log_reg.coef_[0])


#Visualising Feature Data,
importance_df = pd.DataFrame({'Feature': full_features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


#Printing Feature Data Ranking,
print("Feature Importance Ranking:")
print(importance_df.to_string(index=False))


#Plotting Feature Data,
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance in Logistic Regression")
plt.show()
