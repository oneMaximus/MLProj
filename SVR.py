import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, MFIIndicator, VolumeWeightedAveragePrice  # Fixed imports
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Step 1: Fetch S&P 500 data
sp500 = yf.download('^GSPC', start='2015-01-01', end='2025-02-23')  # Today is Feb 25, 2025, so this is fine
data = sp500[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Step 2: Calculate Target Variable (Y)
data['Y'] = data['Close'].pct_change().shift(-1) * 100  # Predict next day's % change
data = data.dropna()  # Remove NaN values

# Step 3: Calculate Technical Indicators

# Price and Trend Indicators
data['SMA_9'] = SMAIndicator(data['Close'], window=9).sma_indicator()
data['EMA_9'] = EMAIndicator(data['Close'], window=9).ema_indicator()

#ta doesn't have WMAIndicator directly; we'll use a custom function instead
def wma(prices, window):
    weights = np.arange(1, window + 1)
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
data['WMA_9'] = wma(data['Close'], 9)

macd = MACD(data['Close'], window_slow=26, window_fast=9, window_sign=9)
data['MACD'] = macd.macd()
data['Signal_Line'] = macd.macd_signal()

psar = PSARIndicator(data['High'], data['Low'], data['Close'])
data['PSAR'] = psar.psar()

# IchimokuIndicator in ta requires more setup; let's just do Tenkan-sen for simplicity
data['Tenkan_sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2

# Momentum Indicators
data['RSI_9'] = RSIIndicator(data['Close'], window=9).rsi()

stoch = StochasticOscillator(data['High'], data['Low'], data['Close'], window=9, smooth_window=3)
data['Stoch_%K'] = stoch.stoch()

data['ROC_9'] = ROCIndicator(data['Close'], window=9).roc()

data['Momentum'] = data['Close'] - data['Close'].shift(9)

data['Williams_%R'] = WilliamsRIndicator(data['High'], data['Low'], data['Close'], lbp=9).williams_r()

# Volatility Indicators
bb = BollingerBands(data['Close'], window=9, window_dev=2)
data['BollingerBands_Middle'] = bb.bollinger_mavg()
data['BollingerBands_Upper'] = bb.bollinger_hband()
data['BollingerBands_Lower'] = bb.bollinger_lband()

data['ATR_14'] = AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()

# Volume Indicators (Fixed with proper imports)
data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
data['AD'] = AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
data['MFI'] = MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=14).money_flow_index()
data['VWAP'] = VolumeWeightedAveragePrice(data['Close'], data['Volume'], window=14).volume_weighted_average_price()

# Support and Resistance Indicators
# Fibonacci Retracement
def fib_levels(high, low):
    diff = high - low
    return {
        'Fib_23.6': high - diff * 0.236,
        'Fib_38.2': high - diff * 0.382,
        'Fib_50.0': high - diff * 0.500,
        'Fib_61.8': high - diff * 0.618
    }
rolling_high = data['High'].rolling(window=9).max()
rolling_low = data['Low'].rolling(window=9).min()
fib = pd.DataFrame([fib_levels(h, l) for h, l in zip(rolling_high, rolling_low)], index=data.index)
data = pd.concat([data, fib], axis=1)

# Pivot Points
data['Pivot_Point'] = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
data['S1'] = 2 * data['Pivot_Point'] - data['High'].shift(1)
data['R1'] = 2 * data['Pivot_Point'] - data['Low'].shift(1)

# Market Breadth Indicators
# Advance/Decline Line (Simplified for index; real ADL needs stock-level data)
data['ADL'] = data['Close'] - data['Open']  # Placeholder; not true ADL

# Volatility Index VIX
vix = yf.download('^VIX', start='2015-01-01', end='2025-02-23')
data['VIX'] = vix['Close'].reindex(data.index, method='ffill')