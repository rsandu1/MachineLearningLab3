# CISC 5800 Homework 3 Question 4
# Robert Sandu

import numpy as np
from sklearn.preprocessing import StandardScaler

Xtr = np.array([...])  # Training features (NxP matrix)
ytr = np.array([...])  # Training labels (Nx1 vector)
Xts = np.array([...])  # Test features
yts = np.array([...])  # Test labels

# Normalize features
scaler_X = StandardScaler()
Xtr_norm = scaler_X.fit_transform(Xtr)
Xts_norm = scaler_X.transform(Xts)

# Normalize target variable
y_mean, y_std = np.mean(ytr), np.std(ytr)
ytr_norm = (ytr - y_mean) / y_std

# Train model
model = SomeModel() #Don't know which library is from 
model.fit(Xtr_norm, ytr_norm)

# Predict and compute RSS
yhat_norm = model.predict(Xts_norm)
yhat = yhat_norm * y_std + y_mean  # Convert back to original scale
rss = np.sum((yts - yhat) ** 2)

print(f"RSS on test data: {rss}")
