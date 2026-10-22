"""Loss functions and their hand-derived gradients."""

from __future__ import annotations

import numpy as np


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    r""":math:`L = \frac1n \sum_i (\hat y_i - y_i)^2`. ``y, y_hat: (n,)``."""
    r = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    return float(np.mean(r * r))


def mse_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    r"""Gradient of the MSE of :math:`\hat y = Xw + b`.

    With residuals :math:`r = Xw + b - y`: :math:`\nabla_w L = \frac2n X^\top r` and
    :math:`\partial L/\partial b = \frac2n \sum_i r_i`.

    ``X: (n, d)``, ``y: (n,)``, ``w: (d,)``, ``b: float``. Returns ``(grad_w: (d,), grad_b)``.
    """
    n = X.shape[0]
    r = X @ w + b - y
    return (2.0 / n) * (X.T @ r), float((2.0 / n) * r.sum())
