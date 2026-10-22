"""Download the UCI Air Quality dataset into data/raw/."""

from __future__ import annotations

import argparse
import logging

from aqlinreg.config import load_config
from aqlinreg.data import download_zip
from aqlinreg.utils import resolve_path, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    cfg = load_config(args.config)
    path = download_zip(
        cfg.data.url, resolve_path(cfg.data.raw_dir), cfg.data.csv_name, force=args.force
    )
    print(f"Dataset ready at {path}")


if __name__ == "__main__":
    main()
