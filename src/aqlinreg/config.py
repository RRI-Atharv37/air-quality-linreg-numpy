"""Load ``configs/*.yaml`` into typed dataclasses (no magic numbers buried in code)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from aqlinreg.utils import resolve_path


@dataclass
class DataConfig:
    url: str
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    csv_name: str = "AirQualityUCI.csv"
    target: str = "NO2(GT)"
    drop_columns: list[str] = field(default_factory=lambda: ["NMHC(GT)"])
    train_frac: float = 0.70
    val_frac: float = 0.15


@dataclass
class FeatureConfig:
    raw: list[str]
    time: list[str]
    polynomial_degree: int = 1

    @property
    def all(self) -> list[str]:
        return [*self.raw, *self.time]


@dataclass
class GDConfig:
    learning_rate: float = 0.05
    epochs: int = 200
    batch_size: int | None = None
    schedule: str = "constant"
    decay: float = 0.1
    step_size: int = 50
    momentum: float = 0.0
    early_stopping_patience: int | None = None
    shuffle: bool = True


@dataclass
class RidgeConfig:
    alpha: float = 1.0
    alphas_sweep: dict[str, float] = field(
        default_factory=lambda: {"start": -3.0, "stop": 4.0, "num": 50}
    )


@dataclass
class PathsConfig:
    figures_dir: str = "reports/figures"
    results_md: str = "reports/results.md"


@dataclass
class Config:
    seed: int
    data: DataConfig
    features: FeatureConfig
    gd: GDConfig = field(default_factory=GDConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build(cls: type, raw: dict[str, Any]):
    """Instantiate dataclass ``cls`` from ``raw``, rejecting unknown keys (typo protection)."""
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Read a YAML file into a :class:`Config`."""
    with resolve_path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(
        seed=raw["seed"],
        data=_build(DataConfig, raw["data"]),
        features=_build(FeatureConfig, raw["features"]),
        gd=_build(GDConfig, raw.get("gd", {})),
        ridge=_build(RidgeConfig, raw.get("ridge", {})),
        paths=_build(PathsConfig, raw.get("paths", {})),
    )
