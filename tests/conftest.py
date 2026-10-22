"""Shared fixtures: tiny synthetic datasets with fixed seeds."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_regression(rng):
    """Well-conditioned problem: y = X @ w + b + small noise. Returns (X, y, w, b)."""
    n, d = 200, 4
    X = rng.normal(size=(n, d))
    w = np.array([2.0, -1.0, 0.5, 3.0])
    b = 1.5
    y = X @ w + b + 0.01 * rng.normal(size=n)
    return X, y, w, b
