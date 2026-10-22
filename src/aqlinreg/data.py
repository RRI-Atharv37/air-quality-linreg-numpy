"""Download, load, clean and split the UCI Air Quality dataset (id 360)."""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MISSING_SENTINEL = -200.0  # the dataset encodes "no measurement" as -200


def download_zip(url: str, dest_dir: str | Path, csv_name: str, force: bool = False) -> Path:
    """Download the UCI zip and extract only ``csv_name`` into ``dest_dir``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / csv_name
    if csv_path.exists() and not force:
        logger.info("%s already present, skipping download", csv_path)
        return csv_path

    zip_path = dest_dir / "air_quality.zip"
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp, zip_path.open("wb") as out:
        shutil.copyfileobj(resp, out)

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if Path(m).name == csv_name]
        if not members:
            raise FileNotFoundError(f"{csv_name} not found in archive; has {zf.namelist()}")
        with zf.open(members[0]) as src, csv_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    zip_path.unlink()
    return csv_path


def load_raw(path: str | Path) -> pd.DataFrame:
    """Parse the ``;``-separated, decimal-comma CSV and drop empty columns and rows."""
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    """-200 -> NaN, drop ``drop_columns``, index by datetime (Date DD/MM/YYYY, Time HH.MM.SS)."""
    out = df.copy()
    ts = pd.to_datetime(
        out["Date"].astype(str) + " " + out["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
    )
    out = out.drop(columns=["Date", "Time", *(drop_columns or [])], errors="ignore")
    out = out.astype(float).replace(MISSING_SENTINEL, np.nan)
    out.index = pd.DatetimeIndex(ts, name="datetime")
    return out.sort_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings of hour (24), day of week (7), month (12) and ``is_weekend``.

    A cycle of period P becomes ``(sin(2*pi*t/P), cos(2*pi*t/P))`` so 23:00 and 00:00 are close.
    """
    idx = df.index
    out = df.copy()
    for name, value, period in (
        ("hour", idx.hour, 24),
        ("dow", idx.dayofweek, 7),
        ("month", idx.month - 1, 12),
    ):
        angle = 2.0 * np.pi * np.asarray(value) / period
        out[f"{name}_sin"] = np.sin(angle)
        out[f"{name}_cos"] = np.cos(angle)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    return out


def drop_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop rows whose target is NaN (feature NaNs are imputed later, from train only)."""
    return df.loc[df[target].notna()]


def time_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split; a random split would leak neighbouring hours."""
    if not (0 < train_frac and 0 < val_frac and train_frac + val_frac < 1):
        raise ValueError("need train_frac > 0, val_frac > 0 and train_frac + val_frac < 1")
    df = df.sort_index()
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return df.iloc[:n_train], df.iloc[n_train : n_train + n_val], df.iloc[n_train + n_val :]
