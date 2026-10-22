"""Test helpers. sklearn is a reference-only dependency, so tests that need it skip cleanly
when it cannot be imported (e.g. a locked-down machine that blocks its compiled extensions)."""

import pytest

try:
    import sklearn.impute
    import sklearn.linear_model
    import sklearn.metrics
    import sklearn.preprocessing  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn is not importable")
