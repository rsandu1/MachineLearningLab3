# CISC 5800 Homework 3 Question 1b
# Robert Sandu

from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Given training and test data
Xtr = np.array([...])  # Training features (NxP matrix)
ytr = np.array([...])  # Training labels (Nx1 vector)
Xts = np.array([...])  # Test features
yts = np.array([...])  # Test labels
best_features = None
best_mse = float('inf')

# Iterate over all pairs of features
for i, j in combinations(range(Xtr.shape[1]), 2):
    X_train_ij = Xtr[:, [i, j]]
    X_test_ij = Xts[:, [i, j]]

    model = LinearRegression()
    model.fit(X_train_ij, ytr)
    y_pred = model.predict(X_test_ij)

    mse = mean_squared_error(yts, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_features = (i, j)

print(f"Best feature pair: {best_features}, MSE: {best_mse}")
