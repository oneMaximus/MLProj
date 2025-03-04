# =============================================================================
# PART 1: Data Fetching & Technical Indicator Calculation (All Indicators)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator, ichimoku_a
from ta.momentum import rsi, stoch, roc, williams_r
from ta.volatility import bollinger_mavg, average_true_range
from ta.volume import on_balance_volume, money_flow_index, volume_weighted_average_price
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fetch S&P 500 data from Yahoo Finance
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
    """
    Calculate Weighted Moving Average, giving higher weights to recent prices.
    """
    weights = np.arange(1, window + 1)  # Weights: 1, 2, ..., 9 for a 9-day WMA
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Custom Momentum Indicator (MOM)
def compute_momentum(close, window):
    """
    Calculate Momentum Indicator.
    """
    return close - close.shift(window)

# Compute all indicators
start_date = "2018-01-01"
end_date = "2023-01-01"

# Fetch S&P 500 data
sp500_data = fetch_data(ticker="^GSPC", start=start_date, end=end_date)

# Extract data
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

# Trend Indicators
indicators['SMA'] = sma_indicator(close, window=window)
indicators['EMA'] = ema_indicator(close, window=window)
indicators['WMA'] = wma(close, window=window)
indicators['MACD'] = macd_diff(close)  # Default uses 12, 26, 9; we use diff
indicators['Parabolic_SAR'] = psar_down_indicator(high, low, close)  # Default AF (0.02, max 0.2)
indicators['Ichimoku_Tenkan_sen'] = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2  # 9-period Tenkan-sen

# Momentum Indicators
indicators['RSI'] = rsi(close, window=window)  # 9-period RSI
indicators['Stochastic_Oscillator'] = stoch(high, low, close, window=window, smooth_window=3)  # 9-period %K
indicators['ROC'] = roc(close, window=window)  # 9-period Rate of Change
indicators['MOM'] = compute_momentum(close, window=window)  # 9-period Momentum
indicators['Williams_%R'] = williams_r(high, low, close, lbp=window)  # 9-period Williams %R

# Volatility & Volume Indicators
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
# PART 2: Data Pre-processing
# =============================================================================

# Define features (all indicators + Time)
features = [
    'Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen',
    'RSI', 'Stochastic_Oscillator', 'ROC', 'MOM', 'Williams_%R', 'Bollinger_Mavg',
    'OBV', 'MFI', 'VWAP', 'Approx_AD', 'VIX'
]

# Prepare X (features) and y (actual closing prices)
X = data[features]
y = data['Y'].values.reshape(-1, 1)  # Ensure y is 2D

# Time-based split for time series
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalize the training and testing data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit scaler on training data only and transform both train and test sets
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# PART 3: Shallow Neural Network Model Creation, Training & Testing
# =============================================================================

# Define the Neural Network for Stock Prediction
class StockPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(StockPredictionNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size1, out_features=hidden_size2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size2, out_features=output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Hyperparameters
input_size = len(features)  # Number of input features
hidden_size1 = 128 # First hidden layer neurons
hidden_size2 = 64  # Second hidden layer neurons
output_size = 1  # Predicting one value (e.g., stock price)

# Initialize model
model = StockPredictionNN(input_size, hidden_size1, hidden_size2, output_size)

# Print model architecture
print(model)

# Initialize loss function & optimizer
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):

    model.train()

    for X_train, y_train in train_loader:

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted outputs by passing inputs to the model
        y_pred = model(X_train)

        # Compute the loss
        loss = criterion(y_pred, y_train)

        # Backward pass: Compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Set the model to evaluation mode
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_test, y_test in test_loader:

        # Forward pass: Compute predicted outputs by passing inputs to the model
        y_pred = model(X_test)

        predictions.extend(y_pred.numpy())

        actuals.extend(y_test.numpy())

# Convert predictions and actuals back to original scale
predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))

# =============================================================================
# PART 4: Evaluation
# =============================================================================

# Compute evaluation metrics
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

# Print results
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.6f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Actual Prices", linestyle='solid', alpha=0.8)
plt.plot(predictions, label="Predicted Prices", linestyle='dashed', alpha=0.8)
plt.xlabel("Time (Test Data Points)")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction: Actual vs Predicted")
plt.legend()
plt.show()