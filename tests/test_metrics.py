import numpy as np
import pytest

from aqlinreg import metrics
from helpers import requires_sklearn


def test_metrics_hand_computed():
    y = np.array([3.0, -0.5, 2.0, 7.0])
    p = np.array([2.5, 0.0, 2.0, 8.0])
    assert metrics.mse(y, p) == pytest.approx(0.375)
    assert metrics.rmse(y, p) == pytest.approx(0.375**0.5)
    assert metrics.mae(y, p) == pytest.approx(0.5)
    assert metrics.r2(y, p) == pytest.approx(1 - 1.5 / 29.1875)


@requires_sklearn
def test_metrics_match_sklearn(rng):
    from sklearn import metrics as skm

    y = rng.normal(size=100)
    p = y + rng.normal(scale=0.5, size=100)
    assert metrics.mse(y, p) == pytest.approx(skm.mean_squared_error(y, p))
    assert metrics.rmse(y, p) == pytest.approx(skm.mean_squared_error(y, p) ** 0.5)
    assert metrics.mae(y, p) == pytest.approx(skm.mean_absolute_error(y, p))
    assert metrics.r2(y, p) == pytest.approx(skm.r2_score(y, p))


def test_r2_baselines():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.r2(y, y) == 1.0
    assert metrics.r2(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert metrics.r2(y, y[::-1]) < 0


def test_r2_constant_target():
    y = np.full(4, 2.0)
    assert metrics.r2(y, y) == 1.0
    assert metrics.r2(y, y + 1) == 0.0


def test_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.mse(np.zeros(3), np.zeros(4))
