import numpy as np
import pytest

from aqlinreg.losses import mse, mse_grad


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences: (f(x+eps e_i) - f(x-eps e_i)) / 2 eps."""
    g = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        g[i] = (f(x + step) - f(x - step)) / (2 * eps)
    return g


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mse_grad_matches_numerical(seed):
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(30, 5)), rng.normal(size=30)
    w, b = rng.normal(size=5), float(rng.normal())

    gw, gb = mse_grad(X, y, w, b)
    num_gw = numerical_grad(lambda v: mse(y, X @ v + b), w)
    num_gb = numerical_grad(lambda v: mse(y, X @ w + v[0]), np.array([b]))[0]

    assert rel_error(gw, num_gw) < 1e-6
    assert abs(gb - num_gb) / max(abs(gb) + abs(num_gb), 1e-12) < 1e-6


def test_mse_value():
    assert mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])) == pytest.approx(4 / 3)


def test_grad_zero_at_optimum(synthetic_regression):
    X, _, w, b = synthetic_regression
    y = X @ w + b
    gw, gb = mse_grad(X, y, w, b)
    np.testing.assert_allclose(gw, 0.0, atol=1e-12)
    assert gb == pytest.approx(0.0, abs=1e-12)


def test_grad_shapes():
    X, y = np.ones((6, 3)), np.zeros(6)
    gw, gb = mse_grad(X, y, np.zeros(3), 0.0)
    assert gw.shape == (3,)
    assert isinstance(gb, float)
