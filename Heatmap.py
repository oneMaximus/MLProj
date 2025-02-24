import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Fetch S&P 500 data
sp500 = yf.download('^GSPC', start='2023-01-01', end='2025-02-23')  # Adjust end date as needed
data = sp500[['Close', 'High', 'Low']].copy()

# Step 2: Calculate Technical Indicators
# Price & Trend Indicators
data['SMA_9'] = data['Close'].rolling(window=9).mean()  # Simple Moving Average
data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()  # Exponential Moving Average
data['WMA_9'] = data['Close'].rolling(window=9).apply(lambda x: np.sum(x * np.arange(1, 10)) / 45, raw=True)  # Weighted Moving Average
data['MACD'] = data['Close'].ewm(span=9, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()  # MACD Line
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()  # MACD Signal Line
data['Tenkan_sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2  # Ichimoku Tenkan-sen

# Momentum Indicators
# RSI (9-day)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
rs = gain / loss
data['RSI_9'] = 100 - (100 / (1 + rs))

# Stochastic Oscillator (%K, 9-day)
lowest_low = data['Low'].rolling(window=9).min()
highest_high = data['High'].rolling(window=9).max()
data['Stoch_%K'] = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)

# Rate of Change (9-day)
data['ROC_9'] = ((data['Close'] - data['Close'].shift(9)) / data['Close'].shift(9)) * 100

# Momentum (9-day)
data['Momentum_9'] = data['Close'] - data['Close'].shift(9)

# Williams %R (9-day)
data['Williams_%R'] = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)

# Volatility Indicators
# Bollinger Bands (9-day) - Fixed Version
data['BollingerBands_Middle'] = data['Close'].rolling(window=9).mean().squeeze()
std_dev = data['Close'].rolling(window=9).std().squeeze()  # This is a Series
data['BollingerBands_Upper'] = data['BollingerBands_Middle'] + (2 * std_dev)  # Explicitly compute as Series
data['BollingerBands_Lower'] = data['BollingerBands_Middle'] - (2 * std_dev)  # Explicitly compute as Series

# Average True Range (14-day)
tr1 = data['High'] - data['Low']
tr2 = abs(data['High'] - data['Close'].shift())
tr3 = abs(data['Low'] - data['Close'].shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
data['ATR_14'] = tr.rolling(window=14).mean()

# Step 3: Normalize S&P 500 Price Movement
data['Price_Movement_%'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100

# Step 4: Select indicators for correlation
indicators = ['SMA_9', 'EMA_9', 'WMA_9', 'MACD', 'Signal_Line', 'Tenkan_sen', 
              'RSI_9', 'Stoch_%K', 'ROC_9', 'Momentum_9', 'Williams_%R', 
              'BollingerBands_Upper', 'BollingerBands_Lower', 'ATR_14', 'Price_Movement_%']
df = data[indicators].dropna()

# Compute correlation matrix
correlation_matrix = df.corr()

# Step 5: Create Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Technical Indicators Heatmap for S&P 500 (Short-Term)')
plt.show()