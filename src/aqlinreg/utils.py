"""Seeding, logging setup and project paths."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to ``PROJECT_ROOT`` (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global RNGs and return a fresh ``Generator``.

    Library code should take the returned ``Generator`` (or build one from the config seed)
    rather than rely on the global NumPy state.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global state, seeded for third-party code
    return np.random.default_rng(seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
