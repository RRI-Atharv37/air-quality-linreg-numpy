"""Closed-form (normal equation) linear regression, with two numerical solvers."""

from __future__ import annotations

import numpy as np

from aqlinreg.models.base import LinearModel, check_Xy


def condition_number_xtx(X: np.ndarray) -> float:
    r""":math:`\kappa(X^\top X) = \kappa(X)^2`, computed from the SVD of ``X: (n, d)``."""
    s = np.linalg.svd(np.asarray(X, dtype=float), compute_uv=False)
    return float((s[0] / s[-1]) ** 2) if s[-1] > 0 else float("inf")


class LinearRegressionNormalEq(LinearModel):
    r"""Least squares with optional L2 penalty, solved in closed form.

    Solves :math:`(X_c^\top X_c + \lambda I)\,w = X_c^\top y_c` on mean-centred data and sets
    :math:`b = \bar y - \bar x^\top w`, so the bias is never penalised.

    Parameters
    ----------
    alpha : ridge strength :math:`\lambda \ge 0` (0 = OLS).
    solver : ``"lstsq"`` solves the ridge-augmented system with ``np.linalg.lstsq`` (SVD).
        ``"inverse"`` uses the textbook explicit ``inv(XᵀX + λI) @ Xᵀy``. Forming XᵀX squares
        the condition number, so lstsq is more accurate, and it also handles rank-deficient X.

    Attributes
    ----------
    coef_ : (d,)    intercept_ : float
    """

    def __init__(self, alpha: float = 0.0, solver: str = "lstsq"):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if solver not in {"lstsq", "inverse"}:
            raise ValueError("solver must be 'lstsq' or 'inverse'")
        self.alpha = alpha
        self.solver = solver

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionNormalEq:
        """Fit on ``X: (n, d)``, ``y: (n,)``."""
        X, y = check_Xy(X, y)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        d = X.shape[1]

        if self.solver == "lstsq":
            if self.alpha > 0:
                Xc = np.vstack([Xc, np.sqrt(self.alpha) * np.eye(d)])
                yc = np.concatenate([yc, np.zeros(d)])
            w = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        else:
            gram = Xc.T @ Xc + self.alpha * np.eye(d)
            w = np.linalg.inv(gram) @ (Xc.T @ yc)

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self
