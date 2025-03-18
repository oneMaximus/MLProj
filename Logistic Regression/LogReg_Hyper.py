#Importing all Libraries,
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve


#Function for Fetching Data,
def fetch_data(ticker="^GSPC", start="2018-01-01", end="2023-01-01"):
    try:
        data = yf.download(ticker, start=start, end=end)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


#Fetch all S&P 500 Data,
data = fetch_data()


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


#Obtaining Absolute Values,
feature_importance = abs(log_reg.coef_[0])


#Display Importance of each Feature,
importance_df = pd.DataFrame({'Feature': full_features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


#Printing Feature Ranking,
print(importance_df)


#Use Recursive Feature Elimination (RFE) with Logistic Regression,
rfe = RFE(log_reg, n_features_to_select=5)  #Select Best 5 Features,
rfe.fit(X_train, y_train)


#Print Features that are Selected,
print()
selected_features = pd.DataFrame({'Feature': full_features, 'Selected': rfe.support_})
print(selected_features[selected_features['Selected'] == True])


#User can Select Feature to Train the Model,
X_train_rfe = X_train[:, rfe.support_]
X_test_rfe = X_test[:, rfe.support_]


#Re-train Model with Selected Feature,
log_reg.fit(X_train_rfe, y_train)
y_pred_rfe = log_reg.predict(X_test_rfe)


#Model Evaluation,
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print()
print(f"Accuracy with RFE selected features: {accuracy_rfe}")


#Calculate Correlation Matrix (Using Feature Selected),
correlation_matrix = data[full_features + ['Target']].corr()


#Correlation with Target Variable,
target_correlation = correlation_matrix['Target'].sort_values(ascending=False)


#Display Top Features with Target Variable,
print()
print(target_correlation)


#Drop Highly Correlated Features (If correlation > 0.9),
correlated_features = target_correlation[target_correlation > 0.9].index
print()
print("Highly correlated features:", correlated_features)


#Remove Correlated Feature from Dataset,
data_filtered = data.drop(columns=correlated_features)


#Hyperparameter Tuning with GridSearchCV,
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


#Best Parameters,
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


#Train Logistic Regression with Best Parameters,
log_reg = LogisticRegression(**best_params, max_iter=1000)
log_reg.fit(X_train, y_train)


#Model Predictions,
y_pred = log_reg.predict(X_test)


#Model Evaluation,
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


#Evaluation Metrics using Probability Predictions
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability for class 1


#R-Squared, MSE, and MAE
r2 = r2_score(y_test, y_prob)
mse = mean_squared_error(y_test, y_prob)
mae = mean_absolute_error(y_test, y_prob)

print()
print(f"R-Squared: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print()


#Feature Importance
feature_importance = abs(log_reg.coef_[0])
importance_df = pd.DataFrame({'Feature': full_features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)


#Function to plot learning curve with MSE comparison
def plot_learning_curve_mse(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))

    #Generate learning curve data with MSE as the metric
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_squared_error', train_sizes=train_sizes, n_jobs=-1
    )

    #Calculate mean and standard deviation for smooth curves
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    #Plot training curve (MSE, lower is better)
    plt.plot(train_sizes, -train_mean, 'o-', color='blue', label='Training MSE')
    plt.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, color='blue', alpha=0.2)

    #Plot validation curve (MSE, lower is better)
    plt.plot(train_sizes, -test_mean, 'o-', color='green', label='Validation MSE')
    plt.fill_between(train_sizes, -test_mean - test_std, -test_mean + test_std, color='green', alpha=0.2)

    #Curve details
    plt.title("Learning Curve - Mean Squared Error")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

#Plot learning curve for the optimized model with MSE
plot_learning_curve_mse(log_reg, X_scaled, y)


#Plotting Feature Data,
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance from Logistic Regression")
plt.show()

