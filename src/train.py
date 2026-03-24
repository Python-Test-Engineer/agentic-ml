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
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def train_and_predict(train_X, train_y, val_X):
    # Model 1: degree-5 polynomial OLS
    poly = PolynomialFeatures(degree=5, include_bias=False)
    train_poly = poly.fit_transform(train_X)
    val_poly = poly.transform(val_X)
    lr = LinearRegression()
    lr.fit(train_poly, train_y)
    pred_lr = lr.predict(val_poly)

    # Scaled raw features for kernel/neural models
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    val_scaled = scaler.transform(val_X)

    # Model 2: SVR with RBF kernel
    svr_grid = {
        "C": [1, 10, 100, 1000],
        "gamma": ["scale", "auto", 0.1, 1.0],
        "epsilon": [0.01, 0.05, 0.1],
    }
    gs_svr = GridSearchCV(SVR(kernel="rbf"), svr_grid, cv=5,
                          scoring="neg_mean_squared_error")
    gs_svr.fit(train_scaled, train_y)
    pred_svr = gs_svr.predict(val_scaled)

    # Model 3: MLP neural network
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation="relu",
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )
    mlp.fit(train_scaled, train_y)
    pred_mlp = mlp.predict(val_scaled)

    return (pred_lr + pred_svr + pred_mlp) / 3
