# Data

## Source

**UCI Machine Learning Repository: Air Quality** (dataset id 360)

- Page: <https://archive.ics.uci.edu/dataset/360/air+quality>
- Download: <https://archive.ics.uci.edu/static/public/360/air+quality.zip>
- License: **CC BY 4.0**
- 9,357 hourly rows, 2004-03-10 to 2005-04-04, from a multisensor device deployed in a polluted
  area of an Italian city. Reference analyzers provide the ground truth (`*(GT)` columns); the
  `PT08.*` columns are cheap metal-oxide sensor responses.

**Citation:** S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, "On field
calibration of an electronic nose for benzene estimation in an urban pollution monitoring
scenario", *Sensors and Actuators B: Chemical*, 129(2), 750-757, 2008.

## How to download

```bash
uv run python scripts/download_data.py
```

This puts `AirQualityUCI.csv` in `data/raw/`. Nothing under `data/` except this file is
committed.

## Gotchas handled in `src/aqlinreg/data.py`

| Gotcha | Handling |
|---|---|
| `;` separator and `,` as the **decimal** mark (`2,6` means 2.6) | `pd.read_csv(sep=";", decimal=",")` in `load_raw` |
| Two trailing empty columns and empty trailing rows (from the trailing `;;`) | dropped in `load_raw` (`dropna(how="all")` on both axes) |
| Missing values are encoded as **-200** | replaced by `NaN` in `clean` (a -200 left in would be treated as a real reading and wreck every mean and fit) |
| `NMHC(GT)` is ~90% missing | column dropped in `clean` (`data.drop_columns` in the config) |
| `Date` is `DD/MM/YYYY`, `Time` is `HH.MM.SS` (dots, not colons) | parsed with an explicit format into a single `DatetimeIndex`; day-first matters, as `10/03/2004` is 10 March |

## Processing decisions

- **Target:** `NO2(GT)`. About 17.5% of rows have no target and are dropped (after cleaning,
  before splitting).
- **Feature NaNs** (about 3.9% for the sensor and weather columns) are imputed with the
  **train-split medians only**, so no information from validation or test leaks in.
- **Time split**, not random: first 70% of rows train, next 15% validation, last 15% test. Hourly
  air quality is strongly autocorrelated; a random split would place neighbouring hours of
  every test row in the training set and so measure interpolation, not forecasting.
- The other ground-truth gases (`CO(GT)`, `C6H6(GT)`, `NOx(GT)`) are **not** used as features:
  they are lab-grade measurements of co-emitted pollutants, which is not the "cheap sensors
  plus weather plus calendar" setting this project targets.
