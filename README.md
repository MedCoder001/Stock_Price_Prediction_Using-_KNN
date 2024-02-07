## Stock Price Prediction using K-Nearest Neighbors (KNN)

This project explores the application of K-Nearest Neighbors (KNN) algorithm in two scenarios: classification and regression, for stock price prediction. KNN is a simple yet powerful supervised machine learning algorithm and can be used for both classification and regression tasks.
Stock market prediction is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. The successful prediction of a stock's future price could yield significant profit.

### Project Overview

#### Objective
The objective of this project is two-fold:
1. Classification: Determine whether a customer should buy (+1) or sell (-1) a stock based on the comparison of the 'Close' prices of the current day and the next day.
2. Regression: Predict the stock prices using the KNN regression algorithm.

### Dataset
The dataset used in this project is obtained from Quandl and contains historical stock price data for Tata Global Beverages (TATAGLOBAL) from the National Stock Exchange (NSE).

### Approach

#### Classification:
1. Data was preprocessed and I calculated the 'Open-Close' and 'High-Low' features from the raw data.
2. Created Binary Classification Target Variable to compare the 'Close' prices of the current day and the next day to determine whether to buy or sell the stock.
3. Splited the dataset into training and testing sets.
4. Implemented KNN Classifier (KNeighborsClassifier) from the scikit-learn library.
5. For Hyperparameter tuning to find the best parameter (number of neighbors), I used GridSearchCV.
6. Evaluated the model by calculating accuracy scores on both training and testing sets. Generated predictions and compared actual vs. predicted classes.

#### Regression:
1. Prepared data and splitted the dataset into features (X) and target variable (y).
2. Splitted the dataset into training and testing sets.
3. Implemented KNN Regressor (KNeighborsRegressor) from scikit-learn library.
4. For Hyperparameter tuning to find the best parameter (number of neighbors), I used GridSearchCV.
5. Fitted the model on the training data and predicted stock prices on the testing data.
6. Evaluated the model by calculating Root Mean Square Error (RMSE) to assess the accuracy of the regression model. Compare actual vs. predicted stock prices.

### Conclusion
This project demonstrates the application of K-Nearest Neighbors algorithm in both classification and regression tasks for stock price prediction. The classification model helps determine whether to buy or sell a stock based on historical data, while the regression model predicts future stock prices. By leveraging machine learning techniques like KNN, investors and traders can make informed decisions in the financial markets.

**Note**: Ensure you have the required libraries installed (`pandas`, `numpy`, `matplotlib`, `quandl`, `scikit-learn`). Download the dataset from Quandl and run the provided Python script to execute the prediction models.
