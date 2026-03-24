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
from sklearn.preprocessing import PolynomialFeatures


def train_and_predict(train_X, train_y, val_X):
    poly = PolynomialFeatures(degree=5, include_bias=False)
    train_X_poly = poly.fit_transform(train_X)
    val_X_poly = poly.transform(val_X)
    model = LinearRegression()
    model.fit(train_X_poly, train_y)
    return model.predict(val_X_poly)
