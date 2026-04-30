"""
Linear Regression FROM SCRATCH (clean + stable)
"""

import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iter):

            y_pred = self._predict(X)
            error = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

    def _predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._predict(X)

    def evaluate(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / ss_tot)

        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        return {
            "R2": float(round(r2, 4)),
            "MAE": float(round(mae, 2)),
            "RMSE": float(round(rmse, 2))
        }