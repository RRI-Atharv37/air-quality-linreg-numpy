import numpy as np
import pytest

from aqlinreg.preprocessing import MedianImputer, PolynomialFeatures, StandardScaler
from helpers import requires_sklearn


@pytest.fixture
def X(rng):
    return rng.normal(loc=[5.0, -3.0, 100.0], scale=[1.0, 10.0, 0.01], size=(50, 3))


def test_scaler_matches_numpy_formula(X):
    mine = StandardScaler().fit(X)
    np.testing.assert_allclose(mine.mean_, np.mean(X, axis=0), atol=1e-12)
    np.testing.assert_allclose(mine.scale_, np.std(X, axis=0, ddof=0), atol=1e-12)
    np.testing.assert_allclose(
        mine.transform(X), (X - np.mean(X, axis=0)) / np.std(X, axis=0), atol=1e-12
    )


@requires_sklearn
def test_scaler_matches_sklearn(X):
    from sklearn.preprocessing import StandardScaler as SkScaler

    mine, ref = StandardScaler().fit(X), SkScaler().fit(X)
    np.testing.assert_allclose(mine.mean_, ref.mean_, atol=1e-12)
    np.testing.assert_allclose(mine.scale_, ref.scale_, atol=1e-12)
    np.testing.assert_allclose(mine.transform(X), ref.transform(X), atol=1e-12)


def test_scaler_output_is_standardised(X):
    Z = StandardScaler().fit_transform(X)
    # column 3 has mean 100, std 0.01: rounding of ~1e-14 relative is amplified by 1e4
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_scaler_inverse_round_trip(X):
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-10)


def test_scaler_constant_column_maps_to_zero_not_nan():
    X = np.column_stack([np.arange(5.0), np.full(5, 7.0)])
    Z = StandardScaler().fit_transform(X)
    assert np.isfinite(Z).all()
    np.testing.assert_array_equal(Z[:, 1], 0.0)


def test_scaler_uses_train_statistics_only(X):
    sc = StandardScaler().fit(X[:30])
    np.testing.assert_allclose(sc.transform(X[30:]), (X[30:] - X[:30].mean(0)) / X[:30].std(0))


def test_scaler_rejects_1d():
    with pytest.raises(ValueError):
        StandardScaler().fit(np.arange(5.0))


def test_imputer_fills_with_column_medians(rng):
    X = rng.normal(size=(40, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    mine = MedianImputer().fit(X).transform(X)
    assert not np.isnan(mine).any()
    for j in range(4):
        med = np.median(X[~np.isnan(X[:, j]), j])
        np.testing.assert_array_equal(mine[np.isnan(X[:, j]), j], med)
        np.testing.assert_array_equal(mine[~np.isnan(X[:, j]), j], X[~np.isnan(X[:, j]), j])


@requires_sklearn
def test_imputer_matches_sklearn(rng):
    from sklearn.impute import SimpleImputer

    X = rng.normal(size=(40, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    mine = MedianImputer().fit(X).transform(X)
    np.testing.assert_allclose(mine, SimpleImputer(strategy="median").fit_transform(X))


def test_imputer_uses_train_medians_and_does_not_mutate():
    train = np.array([[1.0], [2.0], [3.0], [np.nan]])
    test = np.array([[np.nan], [100.0]])
    imp = MedianImputer().fit(train)
    out = imp.transform(test)
    assert out[0, 0] == 2.0
    assert np.isnan(test[0, 0])


def test_imputer_rejects_all_nan_column():
    with pytest.raises(ValueError):
        MedianImputer().fit(np.array([[1.0, np.nan], [2.0, np.nan]]))


def test_poly_degree2_columns_explicit():
    X = np.array([[2.0, 3.0], [-1.0, 4.0]])
    out = PolynomialFeatures(2).fit_transform(X)
    expected = np.array([[2, 3, 4, 6, 9], [-1, 4, 1, -4, 16]], dtype=float)
    np.testing.assert_array_equal(out, expected)


@requires_sklearn
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_poly_matches_sklearn(rng, degree):
    from sklearn.preprocessing import PolynomialFeatures as SkPoly

    X = rng.normal(size=(20, 3))
    mine = PolynomialFeatures(degree).fit_transform(X)
    ref = SkPoly(degree, include_bias=False).fit_transform(X)
    assert mine.shape == ref.shape
    np.testing.assert_allclose(mine, ref, atol=1e-12)


def test_poly_degree2_shape(rng):
    d = 15
    out = PolynomialFeatures(2).fit_transform(rng.normal(size=(7, d)))
    assert out.shape == (7, d + d * (d + 1) // 2)


def test_poly_wrong_width_and_bad_degree(rng):
    p = PolynomialFeatures(2).fit(rng.normal(size=(5, 3)))
    with pytest.raises(ValueError):
        p.transform(rng.normal(size=(5, 4)))
    with pytest.raises(ValueError):
        PolynomialFeatures(0)
