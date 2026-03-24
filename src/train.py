# train.py — Agent-editable model file
#
# THIS IS THE ONLY FILE THE AGENT EDITS.
#
# The agent must expose exactly one function:
#
#   train_and_predict(train_X, train_y, val_X) -> np.ndarray
#
#   Inputs:
#     train_X : np.ndarray, shape (80, 2) — training features [x1, x2]
#     train_y : np.ndarray, shape (80,)  — training targets
#     val_X   : np.ndarray, shape (20, 2) — validation features
#
#   Output:
#     np.ndarray, shape (20,) — predicted values for val_X
#
# Rules:
#   - Do not read files, make network requests, or spawn subprocesses.
#   - Do not import evaluate.py or any project-local module.
#   - Set random seeds for reproducibility.
#   - Keep this function self-contained.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def train_and_predict(train_X, train_y, val_X):
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    param_grid = {
        "C": [1, 10, 100, 1000],
        "gamma": ["scale", "auto", 0.1, 1.0],
        "epsilon": [0.01, 0.05, 0.1],
    }
    svr = SVR(kernel="rbf")
    gs = GridSearchCV(svr, param_grid, cv=5, scoring="neg_mean_squared_error")
    gs.fit(train_X_scaled, train_y)
    return gs.predict(val_X_scaled)
