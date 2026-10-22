"""Median imputer, standard scaler and polynomial features, from scratch in NumPy.

Statistics are learned in ``fit`` on the TRAIN split only and reused on val/test.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def _check_2d(X: np.ndarray, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d), got shape {X.shape}")
    return X


class MedianImputer:
    """Replace NaN by the per-column training median. ``median_: (d,)``."""

    def fit(self, X: np.ndarray) -> MedianImputer:
        X = _check_2d(X)
        if np.isnan(X).all(axis=0).any():
            raise ValueError("cannot impute: a column is entirely NaN in the fit data")
        self.median_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of ``X: (n, d)`` with NaN filled."""
        X = _check_2d(X).copy()
        if X.shape[1] != self.median_.shape[0]:
            raise ValueError(f"expected {self.median_.shape[0]} columns, got {X.shape[1]}")
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.median_[cols]
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScaler:
    r"""Zero mean, unit variance per column: :math:`z = (x - \mu) / \sigma` (``ddof=0``).

    Constant columns get scale 1. Scaling makes the MSE contours near-circular, which is what
    lets gradient descent use a large learning rate.
    """

    def fit(self, X: np.ndarray) -> StandardScaler:
        X = _check_2d(X)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """``(X - mean_) / scale_``, shape ``(n, d)``."""
        return (_check_2d(X) - self.mean_) / self.scale_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """``Z * scale_ + mean_``."""
        return _check_2d(Z, "Z") * self.scale_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PolynomialFeatures:
    r"""All monomials up to ``degree``, no constant column, in sklearn's column order.

    ``X: (n, d)`` with degree 2 gives :math:`d + d(d+1)/2` columns: :math:`x_i`, then
    :math:`x_i x_j` for :math:`i \le j`.
    """

    def __init__(self, degree: int = 2):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree

    def fit(self, X: np.ndarray) -> PolynomialFeatures:
        X = _check_2d(X)
        self.n_input_features_ = X.shape[1]
        self.powers_ = [
            combo
            for deg in range(1, self.degree + 1)
            for combo in combinations_with_replacement(range(self.n_input_features_), deg)
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _check_2d(X)
        if X.shape[1] != self.n_input_features_:
            raise ValueError(f"expected {self.n_input_features_} columns, got {X.shape[1]}")
        # Loop is over feature combinations (tens), never over samples.
        return np.column_stack([np.prod(X[:, list(c)], axis=1) for c in self.powers_])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
