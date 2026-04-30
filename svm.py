"""
svm.py (FIXED VERSION)
Linear SVM from scratch (stable + corrected logic)
"""

import numpy as np


class LinearSVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def _convert_labels(self, y):
        return np.where(y == 1, 1, -1)

    def fit(self, X, y):

        n_samples, n_features = X.shape
        y_svm = self._convert_labels(y)

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iter):

            for idx, x_i in enumerate(X):

                condition = y_svm[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1

                if condition:
                    dW = 2 * self.lambda_param * self.weights
                    db = 0
                else:
                    dW = 2 * self.lambda_param * self.weights - y_svm[idx] * x_i
                    db = -y_svm[idx]

                self.weights -= self.lr * dW
                self.bias    -= self.lr * db

            # monitoring
            if (i + 1) % 200 == 0:
                acc = self.evaluate(X, y)['Accuracy']
                print(f"Iteration {i+1}/{self.n_iter} | Accuracy: {acc:.2%}")

    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        scores = self.decision_function(X)

        # FIX: avoid np.sign(0) problem
        return (scores >= 0).astype(int)

    def evaluate(self, X, y):

        y_pred = self.predict(X)

        accuracy = np.mean(y_pred == y)

        TP = np.sum((y_pred == 1) & (y == 1))
        FP = np.sum((y_pred == 1) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))

        precision = TP / (TP + FP) if TP + FP else 0
        recall    = TP / (TP + FN) if TP + FN else 0
        f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0

        return {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4)
        }

    def confusion_matrix(self, X, y):

        y_pred = self.predict(X)

        TN = np.sum((y_pred == 0) & (y == 0))
        FP = np.sum((y_pred == 1) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))
        TP = np.sum((y_pred == 1) & (y == 1))

        return np.array([[TN, FP], [FN, TP]])