"""Shared plumbing for the linear models: input validation, ``predict`` and ``score``."""

from __future__ import annotations

import numpy as np

from aqlinreg.metrics import r2


class NotFittedError(RuntimeError):
    """Raised when ``predict``/``score`` is called before ``fit``."""


def check_Xy(X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate ``X: (n, d)`` and ``y: (n,)``; return float arrays."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d), got shape {X.shape}")
    if y is None:
        return X, None
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
    return X, y


class LinearModel:
    r"""Base for :math:`\hat y = Xw + b`; subclasses set ``coef_: (d,)`` and ``intercept_``."""

    coef_: np.ndarray
    intercept_: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return :math:`Xw + b`, shape ``(n,)``."""
        if not hasattr(self, "coef_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit first")
        X, _ = check_Xy(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"expected {self.coef_.shape[0]} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² of ``predict(X)`` against ``y``."""
        return r2(y, self.predict(X))
