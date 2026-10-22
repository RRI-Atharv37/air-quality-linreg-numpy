import numpy as np
import pandas as pd
import pytest

from aqlinreg.data import (
    add_time_features,
    clean,
    drop_missing_target,
    load_raw,
    time_split,
)

# Mimics the real file: ';' separator, ',' decimal mark, 2 trailing empty columns,
# trailing empty rows, -200 sentinel, DD/MM/YYYY dates and HH.MM.SS times.
RAW_CSV = (
    "Date;Time;CO(GT);NMHC(GT);NO2(GT);T;Unnamed1;Unnamed2\n"
    "10/03/2004;18.00.00;2,6;150;113;13,6;;\n"
    "10/03/2004;19.00.00;-200;-200;-200;13,3;;\n"
    "11/03/2004;00.00.00;1,2;-200;92;11,5;;\n"
    ";;;;;;;\n"
    ";;;;;;;\n"
)


@pytest.fixture
def raw_path(tmp_path):
    p = tmp_path / "AirQualityUCI.csv"
    p.write_text(RAW_CSV, encoding="utf-8")
    return p


def test_load_raw_parses_decimal_comma_and_drops_empties(raw_path):
    df = load_raw(raw_path)
    assert df.shape == (3, 6)
    assert df["CO(GT)"].iloc[0] == pytest.approx(2.6)
    assert df["T"].iloc[2] == pytest.approx(11.5)


def test_clean_converts_sentinel_to_nan_and_drops_column(raw_path):
    df = clean(load_raw(raw_path), drop_columns=["NMHC(GT)"])
    assert "NMHC(GT)" not in df.columns
    assert np.isnan(df["CO(GT)"].iloc[1])
    assert np.isnan(df["NO2(GT)"].iloc[1])
    assert not (df == -200).any().any()
    assert df["NO2(GT)"].iloc[0] == 113


def test_clean_builds_datetime_index(raw_path):
    df = clean(load_raw(raw_path))
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2004-03-10 18:00:00")  # day-first, not March->October
    assert df.index[-1] == pd.Timestamp("2004-03-11 00:00:00")
    assert df.index.is_monotonic_increasing


def test_time_features_cyclical_and_weekend():
    idx = pd.to_datetime(["2004-03-13 00:00", "2004-03-13 06:00", "2004-03-15 23:00"])
    out = add_time_features(pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx))
    assert out["hour_sin"].iloc[1] == pytest.approx(1.0)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0)
    np.testing.assert_allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)
    assert list(out["is_weekend"]) == [1.0, 1.0, 0.0]
    assert out["month_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 2 / 12))


def test_23h_and_0h_are_neighbours():
    idx = pd.to_datetime(["2004-03-10 23:00", "2004-03-11 00:00"])
    out = add_time_features(pd.DataFrame({"x": [0.0, 0.0]}, index=idx))
    gap = np.hypot(
        out["hour_sin"].iloc[0] - out["hour_sin"].iloc[1],
        out["hour_cos"].iloc[0] - out["hour_cos"].iloc[1],
    )
    assert gap < 0.3


def test_drop_missing_target():
    df = pd.DataFrame({"y": [1.0, np.nan, 3.0], "x": [np.nan, 1.0, 2.0]})
    out = drop_missing_target(df, "y")
    assert len(out) == 2
    assert out["x"].isna().sum() == 1


def test_time_split_sizes_and_no_overlap():
    idx = pd.date_range("2004-01-01", periods=100, freq="h")
    df = pd.DataFrame({"x": np.arange(100.0)}, index=idx)
    train, val, test = time_split(df, 0.70, 0.15)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) == len(df)


def test_time_split_sorts_shuffled_input():
    idx = pd.date_range("2004-01-01", periods=50, freq="h")
    df = pd.DataFrame({"x": np.arange(50.0)}, index=idx).sample(frac=1.0, random_state=0)
    train, val, test = time_split(df)
    assert train.index.max() < val.index.min() < test.index.min()


@pytest.mark.parametrize("tr,va", [(0.9, 0.2), (0.0, 0.5), (0.5, 0.5)])
def test_time_split_rejects_bad_fractions(tr, va):
    df = pd.DataFrame({"x": range(10)}, index=pd.date_range("2004-01-01", periods=10, freq="h"))
    with pytest.raises(ValueError):
        time_split(df, tr, va)
