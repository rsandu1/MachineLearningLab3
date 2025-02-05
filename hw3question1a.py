# CISC 5800 Homework 3 Question 1a
# Robert Sandu

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Given training and test data
Xtr = np.array([...])  # Training features (NxP matrix)
ytr = np.array([...])  # Training labels (Nx1 vector)
Xts = np.array([...])  # Test features
yts = np.array([...])  # Test labels
best_feature = None
best_mse = float('inf')

# Iterate over each feature (column)
for i in range(Xtr.shape[1]):
    X_train_i = Xtr[:, i].reshape(-1, 1)
    X_test_i = Xts[:, i].reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train_i, ytr)
    y_pred = model.predict(X_test_i)

    mse = mean_squared_error(yts, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_feature = i

print(f"Best feature index: {best_feature}, MSE: {best_mse}")