"""
kmeans.py
---------
K-Means FROM SCRATCH (clean + stable)
"""

import numpy as np


class KMeans:

    def __init__(self, k=3, max_iterations=100, random_seed=42):
        self.k = k
        self.max_iter = max_iterations
        self.random_seed = random_seed
        self.centroids = None
        self.labels = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        np.random.seed(self.random_seed)

        n_samples = X.shape[0]

        # Init centroids
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx].copy()

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)

            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i)
                else self.centroids[i]
                for i in range(self.k)
            ])

            # stop condition
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # final labels
        self.labels = self._assign_clusters(X)
        return self

    def _assign_clusters(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids,
            axis=2
        )
        return np.argmin(distances, axis=1)

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model not fitted yet")
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X)

    def inertia(self, X):
        if self.labels is None:
            raise ValueError("Model not fitted yet")

        total = 0
        for i in range(self.k):
            points = X[self.labels == i]
            if len(points) > 0:
                diff = points - self.centroids[i]
                total += np.sum(diff ** 2)

        return float(total)

    def cluster_summary(self, X, profit_values):
        if self.labels is None:
            raise ValueError("Model not fitted yet")

        cluster_info = []

        for i in range(self.k):
            mask = self.labels == i
            avg = profit_values[mask].mean() if np.any(mask) else 0
            count = np.sum(mask)
            cluster_info.append((i, avg, count))

        cluster_info.sort(key=lambda x: x[1], reverse=True)

        labels_map = {
            0: "High",
            1: "Medium",
            2: "Low"
        }

        summary = []
        for rank, (cid, avg, count) in enumerate(cluster_info):
            summary.append({
                "cluster_id": cid,
                "performance": labels_map[rank],
                "avg_profit": float(avg),
                "count": int(count)
            })

        return summary


# TEST
if __name__ == "__main__":
    from data_generator import generate_data
    from preprocessing import preprocess

    df = generate_data()
    data = preprocess(df)

    model = KMeans(k=3)
    model.fit(data["X_full"])

    print("Labels:", model.labels[:10])
    print("Inertia:", model.inertia(data["X_full"]))
    print(model.cluster_summary(data["X_full"], data["y_reg"]))