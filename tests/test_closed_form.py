import numpy as np
import pytest

from aqlinreg.models.base import NotFittedError
from aqlinreg.models.closed_form import LinearRegressionNormalEq, condition_number_xtx
from helpers import requires_sklearn


def augmented_solution(X, y, alpha):
    """Independent reference: solve (A^T A + alpha*D) theta = A^T y with A = [X, 1] and
    D = diag(1,...,1,0) so the bias is not penalised."""
    A = np.hstack([X, np.ones((len(X), 1))])
    D = np.diag([1.0] * X.shape[1] + [0.0])
    theta = np.linalg.solve(A.T @ A + alpha * D, A.T @ y)
    return theta[:-1], theta[-1]


@pytest.mark.parametrize("solver", ["lstsq", "inverse"])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0, 50.0])
def test_matches_augmented_normal_equations(synthetic_regression, solver, alpha):
    X, y, _, _ = synthetic_regression
    y = y + 100.0  # large intercept: it must NOT be shrunk
    mine = LinearRegressionNormalEq(alpha=alpha, solver=solver).fit(X, y)
    w, b = augmented_solution(X, y, alpha)
    np.testing.assert_allclose(mine.coef_, w, atol=1e-8)
    assert mine.intercept_ == pytest.approx(b, abs=1e-8)


@requires_sklearn
@pytest.mark.parametrize("solver", ["lstsq", "inverse"])
def test_matches_sklearn_ols(synthetic_regression, solver):
    from sklearn.linear_model import LinearRegression

    X, y, _, _ = synthetic_regression
    mine = LinearRegressionNormalEq(solver=solver).fit(X, y)
    ref = LinearRegression().fit(X, y)
    np.testing.assert_allclose(mine.coef_, ref.coef_, atol=1e-8)
    assert mine.intercept_ == pytest.approx(ref.intercept_, abs=1e-8)
    np.testing.assert_allclose(mine.predict(X), ref.predict(X), atol=1e-8)


@requires_sklearn
@pytest.mark.parametrize("solver", ["lstsq", "inverse"])
@pytest.mark.parametrize("alpha", [0.1, 1.0, 50.0])
def test_ridge_matches_sklearn_and_bias_unpenalised(synthetic_regression, solver, alpha):
    from sklearn.linear_model import Ridge

    X, y, _, _ = synthetic_regression
    y = y + 100.0
    mine = LinearRegressionNormalEq(alpha=alpha, solver=solver).fit(X, y)
    ref = Ridge(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(mine.coef_, ref.coef_, atol=1e-8)
    assert mine.intercept_ == pytest.approx(ref.intercept_, abs=1e-8)


def test_recovers_true_parameters(synthetic_regression):
    X, y, w, b = synthetic_regression
    m = LinearRegressionNormalEq().fit(X, y)
    np.testing.assert_allclose(m.coef_, w, atol=0.01)
    assert m.intercept_ == pytest.approx(b, abs=0.01)


def test_score_is_r2(synthetic_regression):
    X, y, _, _ = synthetic_regression
    m = LinearRegressionNormalEq().fit(X, y)
    r = y - m.predict(X)
    assert m.score(X, y) == pytest.approx(1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())))


def test_shapes_and_fit_returns_self(synthetic_regression):
    X, y, _, _ = synthetic_regression
    m = LinearRegressionNormalEq()
    assert m.fit(X, y) is m
    assert m.coef_.shape == (X.shape[1],)
    assert m.predict(X[:7]).shape == (7,)


def test_edge_cases(synthetic_regression):
    X, y, _, _ = synthetic_regression
    with pytest.raises(NotFittedError):
        LinearRegressionNormalEq().predict(X)
    with pytest.raises(ValueError):
        LinearRegressionNormalEq().fit(X[:, 0], y)
    with pytest.raises(ValueError):
        LinearRegressionNormalEq().fit(X, y[:-1])
    with pytest.raises(ValueError):
        LinearRegressionNormalEq(alpha=-1.0)
    with pytest.raises(ValueError):
        LinearRegressionNormalEq(solver="nope")
    m = LinearRegressionNormalEq().fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :2])


def test_single_feature_and_single_sample_dimension():
    x = np.arange(10.0).reshape(-1, 1)
    m = LinearRegressionNormalEq().fit(x, 3.0 * x[:, 0] + 2.0)
    assert m.coef_[0] == pytest.approx(3.0)
    assert m.intercept_ == pytest.approx(2.0)


def test_collinear_features_lstsq_still_fits_but_inverse_cannot_be_trusted(rng):
    a = rng.normal(size=(50, 2))
    X = np.column_stack([a, a[:, 0]])  # third column duplicates the first -> rank 2
    y = a @ np.array([1.0, 2.0])
    m = LinearRegressionNormalEq(solver="lstsq").fit(X, y)
    np.testing.assert_allclose(m.predict(X), y, atol=1e-8)
    np.testing.assert_allclose(m.coef_[[0, 2]], [0.5, 0.5], atol=1e-8)


def test_condition_number_is_squared(rng):
    X = rng.normal(size=(40, 3)) * np.array([1.0, 10.0, 100.0])
    s = np.linalg.svd(X, compute_uv=False)
    assert condition_number_xtx(X) == pytest.approx((s[0] / s[-1]) ** 2)
    assert condition_number_xtx(X) == pytest.approx(np.linalg.cond(X.T @ X), rel=1e-4)


def test_lstsq_more_accurate_than_inverse_when_ill_conditioned():
    """Near-collinear columns: kappa(X) ~ 1e6 so kappa(X^T X) ~ 1e12."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(200, 1))
    X = np.hstack([base, base + 1e-6 * rng.normal(size=(200, 1)), rng.normal(size=(200, 1))])
    w_true = np.array([1.0, -1.0, 2.0])
    y = X @ w_true
    assert condition_number_xtx(X) > 1e10

    err = {}
    for solver in ("lstsq", "inverse"):
        m = LinearRegressionNormalEq(solver=solver).fit(X, y)
        err[solver] = np.linalg.norm(m.predict(X) - y)
    assert err["lstsq"] < err["inverse"]
