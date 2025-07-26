import numpy as np
from sklearn.utils import check_random_state

class CBMModel:
    def __init__(self, n_bins=16, n_rounds=5, learning_rate=0.1, random_state=None):
        self.n_bins = n_bins
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.random_state = check_random_state(random_state)
    
    def _bin_features(self, X):
        # Quantile binning for each feature
        n_samples, n_features = X.shape
        self.bin_edges_ = []
        X_binned = np.zeros_like(X, dtype=np.int32)
        for j in range(n_features):
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(X[:, j], quantiles))
            self.bin_edges_.append(edges)
            X_binned[:, j] = np.digitize(X[:, j], edges)
        return X_binned

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        X_binned = self._bin_features(X)
        # f_j[k]: contribution for feature j, bin k
        self.f_jk_ = [np.zeros(self.n_bins) for _ in range(n_features)]
        self.base_ = y.mean()
        pred = np.full(n_samples, self.base_)

        for epoch in range(self.n_rounds):
            for j in range(n_features):
                # compute residuals (holding other features fixed)
                residual = y - pred
                # For each bin, fit mean residual
                for k in range(self.n_bins):
                    mask = (X_binned[:, j] == k)
                    if np.any(mask):
                        update = residual[mask].mean()
                        # Update the model (additive)
                        self.f_jk_[j][k] += self.learning_rate * update
                        pred[mask] += self.learning_rate * update
                # (Optional: shrinkage, regularization, etc.)
    
    def predict(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        out = np.full(X.shape[0], self.base_)
        for j in range(n_features):
            # Bin X[:, j] using bin_edges_
            edges = self.bin_edges_[j]
            binned = np.digitize(X[:, j], edges)
            out += self.f_jk_[j][binned]
        return out

    def feature_importances_(self):
        # Simple importances: sum of absolute bin effects for each feature
        return np.array([np.abs(f_j).sum() for f_j in self.f_jk_])

# Example usage
if __name__ == '__main__':
    # Fake data: y = x0 + 0.5*x1 + noise
    X = np.random.randn(1000, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.2 * np.random.randn(1000)
    model = CyclicAdditiveBoosting(n_bins=10, n_rounds=5, learning_rate=0.2)
    model.fit(X, y)
    preds = model.predict(X)
    print("R2:", 1 - np.sum((preds - y) ** 2) / np.sum((y - y.mean()) ** 2))
    print("Feature importances:", model.feature_importances_())