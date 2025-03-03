# =============================================================================
# PART 1: Data Fetching & Technical Indicator Calculation (Time, SMA, EMA, WMA, MACD, Parabolic_SAR, Ichimoku Cloud)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ta.trend import sma_indicator, ema_indicator, macd_diff, psar_down_indicator, ichimoku_a
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def fetch_data(ticker="^GSPC", start="2018-01-01", end="2023-01-01"):
    """
    Download historical data for the given ticker.
    """
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def wma(prices, window):
    """
    Calculate Weighted Moving Average, giving higher weights to recent prices.
    """
    weights = np.arange(1, window + 1)  # Weights: 1, 2, ..., 9 for a 9-day WMA
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def compute_indicators(data, window=9):
    """
    Set up time as X, add SMA, EMA, WMA, MACD, Parabolic_SAR, and Ichimoku Cloud (Tenkan-sen) as technical indicators, with actual closing prices as Y.
    """
    # Create time feature
    data['Time'] = (data.index - data.index[0]).days  # Days since first date

    # Target: Next day's closing price
    data['Y'] = data['Close'].shift(-1)  # Predict next day's closing price

    # Technical indicators: SMA, EMA, WMA, MACD, Parabolic_SAR, and Ichimoku Cloud (Tenkan-sen)
    close = data['Close']
    high = data['High']
    low = data['Low']
    data['SMA'] = sma_indicator(close, window=window)
    data['EMA'] = ema_indicator(close, window=window)
    data['WMA'] = wma(close, window=window)
    data['MACD'] = macd_diff(close)  # Default uses 12, 26, 9 (fast, slow, signal); we use diff for simplicity
    data['Parabolic_SAR'] = psar_down_indicator(high, low, close)  # Uses default AF (0.02, max 0.2)
    # Ichimoku Cloud: Tenkan-sen (9-period average, as per document)
    data['Ichimoku_Tenkan_sen'] = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2

    # Drop NaN values
    data.dropna(inplace=True)
    return data

# Download data and compute indicators
data = fetch_data()
data = compute_indicators(data, window=9)

# =============================================================================
# PART 2: Data Pre-processing
# =============================================================================

# Define features (Time, SMA, EMA, WMA, MACD, Parabolic_SAR, Ichimoku_Tenkan_sen)
features = ['Time', 'SMA', 'EMA', 'WMA', 'MACD', 'Parabolic_SAR', 'Ichimoku_Tenkan_sen']
target = ['Close']

# Prepare X (features) and y (actual closing prices)
X = data[features].values
y = data[target].values.reshape(-1, 1)  # Ensure y is 2D

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

predictions = scaler_y.inverse_transform(np.array(predictions).flatten().reshape(-1, 1))
actuals = scaler_y.inverse_transform(np.array(actuals).flatten().reshape(-1, 1))

# =============================================================================
# PART 4: Evaluation
# =============================================================================

# Compute evaluation metrics
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

# Print results
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.6f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Actual Prices", linestyle='solid', alpha=0.8)
plt.plot(predictions, label="Predicted Prices", linestyle='dashed', alpha=0.8)
plt.xlabel("Time (Test Data Points)")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction: Actual vs Predicted")
plt.legend()
plt.show()

# Ensure model is in evaluation mode
model.eval()

# =============================================================================
# PART 5: Permutation Feature Importance
# =============================================================================

# Permutation Importance Calculation
def permutation_importance(model, X_test, y_test, metric=mean_squared_error, n_repeats=10):
    y_test_np = y_test.numpy()  # Convert tensor to NumPy
    baseline_score = metric(y_test_np, model(X_test).detach().numpy())
    importance_scores = np.zeros(X_test.shape[1])

    for i in range(X_test.shape[1]):
        permuted_scores = []
        for _ in range(n_repeats):
            X_permuted = X_test.clone()
            X_permuted[:, i] = X_test[torch.randperm(X_test.shape[0]), i]  # Shuffle only one column
            permuted_score = metric(y_test_np, model(X_permuted).detach().numpy())
            permuted_scores.append(permuted_score)

        importance_scores[i] = np.mean(permuted_scores) - baseline_score

    return importance_scores

# Compute permutation importance
perm_importance = permutation_importance(model, X_test_tensor, y_test_tensor)

# Rank features based on permutation importance
features = np.array(features)  # Ensure feature names are in array format
perm_feature_importance = sorted(zip(features, perm_importance), key=lambda x: x[1], reverse=True)

# Display ranked features
print("Feature ranking based on Permutation Importance:")
for i, (feature, importance) in enumerate(perm_feature_importance, 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Bar plot of permutation importance
plt.figure(figsize=(8,5))
plt.barh([f[0] for f in perm_feature_importance], [f[1] for f in perm_feature_importance])
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Ranking (Permutation)")
plt.gca().invert_yaxis()
plt.show()