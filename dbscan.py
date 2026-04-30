"""
dbscan.py
---------
Simplified DBSCAN (From Scratch using NumPy)
Used for anomaly detection (noise = -1)
"""

import numpy as np
from collections import deque


class DBSCAN:
    def __init__(self, epsilon=0.5, min_samples=5):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        # -2 = unvisited, -1 = noise
        labels = np.full(n_samples, -2)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -2:
                continue

            neighbors = self._find_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1

        self.labels = labels
        return self

    def _find_neighbors(self, X, idx):
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.epsilon)[0]

    def _expand_cluster(self, X, labels, core_idx, neighbors, cluster_id):
        labels[core_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            point = queue.popleft()

            if labels[point] == -1:
                labels[point] = cluster_id

            if labels[point] == -2:
                labels[point] = cluster_id

                new_neighbors = self._find_neighbors(X, point)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)

    def get_anomalies(self):
        if self.labels is None:
            raise ValueError("Model not fitted yet")
        return np.where(self.labels == -1)[0]

    def summary(self, df=None):
        if self.labels is None:
            print("Model not fitted yet")
            return

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = np.sum(self.labels == -1)

        print("\nDBSCAN Results:")
        print(f"Clusters: {n_clusters}")
        print(f"Anomalies: {n_noise}")

        if df is not None and n_noise > 0:
            idx = self.get_anomalies()
            print("\nAnomaly Rows:")
            print(df.iloc[idx].to_string(index=False))


# ── TEST ──────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_data
    from preprocessing import preprocess

    df = generate_data()
    data = preprocess(df)

    X = data["X_full"][:, :2]

    model = DBSCAN(epsilon=0.5, min_samples=5)
    model.fit(X)
    model.summary(df)