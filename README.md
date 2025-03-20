# 📊 MLProj: Financial Market Prediction with Machine Learning

## 📌 Project Overview
This repository contains a **Financial Market Prediction** project using **Machine Learning models**. The goal is to to find a beginner-friendly machine learning model to start out stock prediction!


The models explored include:
- **Shallow Neural Network** 🧠
- **Linear Regression** 📉
- **Logistic Regression**📊
- **Random Forest** 🌲
- **AdaBoost (XGBoost Extension)** 🚀
- **Support Vector Regression (SVR)** 🏎️
## 📊 Dataset Used
**Data Source:** S&P 500 Stock Market Index 📊

### Features Used
| Category                      | Indicators                                        |
|-------------------------------|---------------------------------------------------|
| **Price & Trend Indicators**  | SMA, EMA, WMA, MACD, Parabolic SAR                |
| **Momentum Indicators**       | RSI, Stochastic Oscillator, ROC, MOM, Williams %R |
| **Volatility Indicators**     | Bollinger Mavg,                                   |
| **Volume Indicators**         | OBV, MFI, VWAP                                    |
| **Market Breadth Indicators** | Approx AD, VIX                                    |

## 📂 Repository Structure
```
MLProj/
│── Adaboost/
│   ├── AdaBoost.ipynb               # AdaBoost model development
│
│── Assets/
│   ├── Cleaned_CPI_Report.xlsx       # Cleaned Consumer Price Index data
│   ├── Cleaned_CPI.csv               # Cleaned CPI in CSV format
│   ├── Heatmap.png                    # Correlation heatmap of features
│   ├── RawData.png                     # Original raw data visualization
│   ├── StandardScaled.png             # Standardized feature comparison
│
│── Feature Importance/
│   ├── Felmpt.Test                   # Feature importance analysis (Random Forest)
│   ├── FelmptPCA.py                   # PCA for feature selection
│   ├── FelmptRR.py                    # Ridge Regression feature importance
│
│── Linear_R/
│   ├── Linear_Regression_Analysis/    # Various linear regression models
│
│── RandomForest/
│   ├── RF_Main/                       # Random Forest model analysis
│   ├── Results/                        # Results & model performance
│
│── SVR/
│   ├── SVR_Combined.py                 # SVR model combination
│   ├── SVR_LearningCurve.py            # SVR model learning curve analysis
│   ├── SVR_MBI.py                       # SVR for Market-Based Indicators
│   ├── sp500_predictor.py               # SP500 Market Prediction using SVR
│
│── Shallow Neural Network/
│   ├── SNN.ipynb                      # SNN Model Implementation
│   ├── SNNReport.test                 # SNN Model Results & Evaluation
│
│── .gitignore
│── README.md
```



## 🤝 Contributions
### Team Members & Assigned Models
| Name     | Model Used                     |
|----------|--------------------------------|
| Max      | Support Vector Regression      |
| Shawn    | Logistic Regression            |
| Daniel   | Linear Regression              |
| Gregory  | Neural Network (Shallow)       |
| Roy      | Random Forest                  |
| Zul      | AdaBoost                       |
