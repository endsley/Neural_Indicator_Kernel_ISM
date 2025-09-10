import numpy as np
import sklearn.metrics

class SORF:
    """
    Structured Orthogonal Random Features (SORF) for RBF kernel approximation.
    """
    def __init__(self, n_components=2048, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma
        self.proj_matrix = None

    def _hadamard(self, x):
        """In-place Fast Walsh-Hadamard Transform."""
        h = 1
        n = x.shape[1]
        assert (n & (n - 1)) == 0, "SORF requires input dimension to be power of 2"
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x[:, j], x[:, j + h] = x[:, j] + x[:, j + h], x[:, j] - x[:, j + h]
            h *= 2
        return x

    def _generate_projection(self, d):
        """Generate SORF projection matrix."""
        # Make sure input dimension is power of 2
        next_pow2 = 1 << (d - 1).bit_length()
        if next_pow2 != d:
            raise ValueError("SORF requires input dimension to be power of 2")

        H = lambda x: self._hadamard(x.copy())
        D1 = np.random.choice([-1, 1], size=(d,))
        D2 = np.random.choice([-1, 1], size=(d,))
        D3 = np.random.choice([-1, 1], size=(d,))
        def sorf_proj(X):
            # Apply SORF: X * D1 * H * D2 * H * D3 * H
            X = X * D1
            X = H(X)
            X = X * D2
            X = H(X)
            X = X * D3
            X = H(X)
            return X
        return sorf_proj

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        # Ensure input dimension is power of 2
        next_pow2 = 1 << (n_features - 1).bit_length()
        if next_pow2 != n_features:
            # pad with zeros
            X = np.hstack([X, np.zeros((n_samples, next_pow2 - n_features))])
            n_features = next_pow2

        sorf_proj = self._generate_projection(n_features)
        Z = sorf_proj(X)
        Z = np.sqrt(2.0 / self.n_components) * np.cos(np.sqrt(2 * self.gamma) * Z[:, :self.n_components])
        return Z
