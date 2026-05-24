from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PAGE = (
    "https://openenergyhub.ornl.gov/explore/dataset/"
    "renewable-energy-and-electricity-demand-time-series-dataset-with-exogenous-varia/"
)
MENDELEY_PAGE = "https://data.mendeley.com/datasets/fdfftr3tc2/1"
MENDELEY_FILES_API = (
    "https://data.mendeley.com/public-api/datasets/fdfftr3tc2/files"
    "?folder_id=root&version=1"
)
RAW_DIR = Path("data/openenergyhub_caiso_raw")
PROCESSED_DIR = Path("data/processed")
RAW_CSV_PATH = RAW_DIR / "Database.csv"
FIVE_MIN_PATH = PROCESSED_DIR / "openenergyhub_caiso_5min_wind_solar_load_weather.csv"
HOURLY_PATH = PROCESSED_DIR / "openenergyhub_caiso_hourly_wind_solar_load_weather.csv"
REPORT_PATH = PROCESSED_DIR / "openenergyhub_caiso_data_quality_report.md"
DOWNLOAD_META_PATH = RAW_DIR / "download_metadata.json"


@dataclass
class DownloadInfo:
    success: bool
    source_url: str
    file_size_bytes: int | None
    note: str


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(url: str) -> list[dict]:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_raw_dataset() -> DownloadInfo:
    ensure_dirs()
    if RAW_CSV_PATH.exists() and RAW_CSV_PATH.stat().st_size > 0:
        return DownloadInfo(
            success=True,
            source_url="existing-local-file",
            file_size_bytes=RAW_CSV_PATH.stat().st_size,
            note="Raw CSV already existed locally and was reused.",
        )

    files = fetch_json(MENDELEY_FILES_API)
    if not files:
        return DownloadInfo(
            success=False,
            source_url=MENDELEY_PAGE,
            file_size_bytes=None,
            note="Mendeley files API returned an empty file list.",
        )

    file_info = files[0]
    download_url = file_info["content_details"]["download_url"]
    req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, RAW_CSV_PATH.open("wb") as f:
        f.write(resp.read())

    return DownloadInfo(
        success=True,
        source_url=download_url,
        file_size_bytes=RAW_CSV_PATH.stat().st_size,
        note=f"Downloaded {file_info['filename']} from the public Mendeley file endpoint.",
    )


def get_nh_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def rank_spike_count(series: pd.Series) -> tuple[float, int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan, 0
    q99 = clean.quantile(0.99)
    iqr = clean.quantile(0.75) - clean.quantile(0.25)
    threshold = q99 + 3.0 * iqr
    return float(threshold), int((clean > threshold).sum())


def numeric_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "q95": math.nan,
            "q99": math.nan,
        }
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "q95": float(clean.quantile(0.95)),
        "q99": float(clean.quantile(0.99)),
    }


def load_and_standardize() -> tuple[pd.DataFrame, dict[str, str], dict[str, int | str | float]]:
    raw = pd.read_csv(RAW_CSV_PATH)
    raw_columns = list(raw.columns)
    field_mapping = {
        "Time": "time",
        "Electric_demand": "load",
        "Wind_production": "wind_power",
        "PV_production": "solar_power",
        "Temperature": "temp",
        "Humidity": "humidity",
        "Wind_speed": "wind_speed",
        "DHI": "dhi",
        "DNI": "dni",
        "GHI": "ghi",
        "Season": "season_code_raw",
        "Day_of_the_week": "day_of_week_raw",
    }

    if raw_columns and str(raw_columns[0]).startswith("Unnamed"):
        raw = raw.rename(columns={raw_columns[0]: "source_row"})
    elif raw_columns and raw_columns[0] == "":
        raw = raw.rename(columns={raw_columns[0]: "source_row"})

    for original, renamed in field_mapping.items():
        if original in raw.columns:
            raw = raw.rename(columns={original: renamed})

    raw["time"] = pd.to_datetime(raw["time"], format="%Y-%m-%d-T%H:%M", errors="coerce")
    raw = raw.sort_values("time").reset_index(drop=True)

    invalid_time_count = int(raw["time"].isna().sum())
    duplicate_count = int(raw["time"].duplicated().sum())

    dedup = raw.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="first").copy()
    dedup = dedup.set_index("time").sort_index()

    full_index = pd.date_range(dedup.index.min(), dedup.index.max(), freq="5min")
    aligned = dedup.reindex(full_index)
    aligned.index.name = "time"
    missing_timestamp_count = int(len(full_index) - len(dedup.index))

    aligned = aligned.reset_index()
    aligned["season"] = aligned["time"].dt.month.map(get_nh_season)
    aligned["day_of_week"] = aligned["time"].dt.dayofweek

    raw_negative_counts = {}
    for col in ["load", "wind_power", "solar_power"]:
        raw_negative_counts[f"raw_negative_{col}"] = int(
            (pd.to_numeric(aligned[col], errors="coerce") < 0).sum()
        )

    # Build a cleaned processing view:
    # - power and irradiance variables are non-negative
    # - humidity is clipped into [0, 100]
    # - temperature is left unchanged
    non_negative_cols = ["load", "wind_power", "solar_power", "wind_speed", "dhi", "dni", "ghi"]
    for col in non_negative_cols:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").clip(lower=0)
    aligned["humidity"] = pd.to_numeric(aligned["humidity"], errors="coerce").clip(lower=0, upper=100)
    aligned["temp"] = pd.to_numeric(aligned["temp"], errors="coerce")

    standard_front = [
        "time",
        "load",
        "wind_power",
        "solar_power",
        "temp",
        "humidity",
        "wind_speed",
        "dhi",
        "dni",
        "ghi",
        "season",
        "day_of_week",
    ]
    for col in standard_front:
        if col not in aligned.columns:
            aligned[col] = np.nan

    extra_cols = [col for col in aligned.columns if col not in standard_front]
    aligned = aligned[standard_front + extra_cols]

    diagnostics = {
        "raw_row_count": int(len(raw)),
        "invalid_time_count": invalid_time_count,
        "duplicate_timestamp_count": duplicate_count,
        "missing_timestamp_count": missing_timestamp_count,
        "distinct_timestamp_count_after_dedup": int(len(dedup.index)),
        "aligned_row_count": int(len(aligned)),
        "time_start": aligned["time"].min().isoformat(),
        "time_end": aligned["time"].max().isoformat(),
    }
    diagnostics.update(raw_negative_counts)
    return aligned, field_mapping, diagnostics


def resample_hourly(aligned_5min: pd.DataFrame) -> pd.DataFrame:
    hourly = aligned_5min.set_index("time").copy()
    numeric_cols = [
        "load",
        "wind_power",
        "solar_power",
        "temp",
        "humidity",
        "wind_speed",
        "dhi",
        "dni",
        "ghi",
    ]
    extra_numeric = [
        col
        for col in hourly.columns
        if col not in numeric_cols + ["season", "day_of_week"]
        and pd.api.types.is_numeric_dtype(hourly[col])
    ]
    agg_cols = numeric_cols + extra_numeric
    hourly_out = hourly[agg_cols].resample("1h").mean()
    hourly_out["season"] = hourly_out.index.month.map(get_nh_season)
    hourly_out["day_of_week"] = hourly_out.index.dayofweek
    hourly_out = hourly_out.reset_index()

    front = [
        "time",
        "load",
        "wind_power",
        "solar_power",
        "temp",
        "humidity",
        "wind_speed",
        "dhi",
        "dni",
        "ghi",
        "season",
        "day_of_week",
    ]
    for col in front:
        if col not in hourly_out.columns:
            hourly_out[col] = np.nan
    rest = [col for col in hourly_out.columns if col not in front]
    return hourly_out[front + rest]


def build_report(
    aligned_5min: pd.DataFrame,
    hourly: pd.DataFrame,
    field_mapping: dict[str, str],
    diagnostics: dict[str, int | str | float],
    download_info: DownloadInfo,
) -> str:
    time_series = aligned_5min["time"].dropna().sort_values()
    diffs = time_series.diff().dropna()
    strict_5min = bool((diffs == pd.Timedelta(minutes=5)).all())
    has_timezone = time_series.dt.tz is not None

    missing_rates = (
        aligned_5min.isna().mean().sort_values(ascending=False).rename("missing_rate")
    )
    load_stats = numeric_summary(aligned_5min["load"])
    wind_stats = numeric_summary(aligned_5min["wind_power"])
    solar_stats = numeric_summary(aligned_5min["solar_power"])

    negative_counts = {
        "load": int((pd.to_numeric(aligned_5min["load"], errors="coerce") < 0).sum()),
        "wind_power": int(
            (pd.to_numeric(aligned_5min["wind_power"], errors="coerce") < 0).sum()
        ),
        "solar_power": int(
            (pd.to_numeric(aligned_5min["solar_power"], errors="coerce") < 0).sum()
        ),
    }

    night_mask = pd.to_numeric(aligned_5min["ghi"], errors="coerce").fillna(0) <= 1.0
    night_solar = pd.to_numeric(aligned_5min.loc[night_mask, "solar_power"], errors="coerce")
    night_solar_mean = float(night_solar.dropna().mean()) if not night_solar.dropna().empty else math.nan
    night_solar_max = float(night_solar.dropna().max()) if not night_solar.dropna().empty else math.nan

    load_spike_threshold, load_spike_count = rank_spike_count(aligned_5min["load"])
    wind_spike_threshold, wind_spike_count = rank_spike_count(aligned_5min["wind_power"])
    solar_spike_threshold, solar_spike_count = rank_spike_count(aligned_5min["solar_power"])

    weather_ranges = {
        "temp": numeric_summary(aligned_5min["temp"]),
        "humidity": numeric_summary(aligned_5min["humidity"]),
        "wind_speed": numeric_summary(aligned_5min["wind_speed"]),
        "dhi": numeric_summary(aligned_5min["dhi"]),
        "dni": numeric_summary(aligned_5min["dni"]),
        "ghi": numeric_summary(aligned_5min["ghi"]),
    }

    suitability = []
    if diagnostics["missing_timestamp_count"] == 0 and diagnostics["duplicate_timestamp_count"] == 0:
        suitability.append("The time axis is complete and is suitable for event window extraction.")
    else:
        suitability.append("The time axis contains gaps or duplicates and needs careful masking or imputation.")
    if max(negative_counts.values()) == 0:
        suitability.append("After cleaning, the main power variables do not contain negative values.")
    else:
        suitability.append("Some negative values remain after processing and would need extra clipping.")
    if pd.to_numeric(aligned_5min["load"], errors="coerce").notna().mean() > 0.95:
        suitability.append("Key fields are highly complete and the dataset is usable for external validation.")
    else:
        suitability.append("Key fields have noticeable incompleteness, so external validation should be cautious.")
    if night_solar_max > 100:
        suitability.append(
            "However, night-time low-GHI periods still show non-trivial solar output, which suggests possible timezone or irradiance-power alignment issues."
        )

    mapping_lines = "\n".join(
        f"| `{raw}` | `{std}` |" for raw, std in field_mapping.items()
    )
    missing_lines = "\n".join(
        f"| `{col}` | {rate:.6f} |" for col, rate in missing_rates.items()
    )

    def stats_table_row(name: str, stats: dict[str, float]) -> str:
        return (
            f"| {name} | {stats['min']:.6f} | {stats['max']:.6f} | "
            f"{stats['mean']:.6f} | {stats['q95']:.6f} | {stats['q99']:.6f} |"
        )

    report = f"""# OpenEnergyHub CAISO Data Quality Report

## 1. Download Status
- Download success: `{download_info.success}`
- OpenEnergyHub dataset page: {DATASET_PAGE}
- Linked Mendeley page: {MENDELEY_PAGE}
- Raw file source used: `{download_info.source_url}`
- Raw file size (bytes): `{download_info.file_size_bytes}`
- Note: {download_info.note}

## 2. Raw-to-Standard Field Mapping
| Original field | Standard field |
|---|---|
{mapping_lines}

## 3. Time Coverage
- Start time: `{diagnostics['time_start']}`
- End time: `{diagnostics['time_end']}`
- Raw record count: `{diagnostics['raw_row_count']}`
- Distinct timestamps after de-duplication: `{diagnostics['distinct_timestamp_count_after_dedup']}`
- 5 min aligned record count: `{diagnostics['aligned_row_count']}`
- Hourly resampled record count: `{len(hourly)}`
- Strict 5-minute interval before reindexing: `{strict_5min}`
- Duplicate timestamps: `{diagnostics['duplicate_timestamp_count']}`
- Missing timestamps in full 5-minute grid: `{diagnostics['missing_timestamp_count']}`
- Invalid timestamps: `{diagnostics['invalid_time_count']}`
- Timezone info present: `{has_timezone}`
- Note on timezone: {"Timezone-aware timestamps were preserved." if has_timezone else "The source timestamps are naive strings without explicit timezone metadata."}

## 4. Missing Rate by Field
| Field | Missing rate |
|---|---:|
{missing_lines}

## 5. Wind/Solar/Load Summary
| Variable | min | max | mean | q95 | q99 |
|---|---:|---:|---:|---:|---:|
{stats_table_row("load", load_stats)}
{stats_table_row("wind_power", wind_stats)}
{stats_table_row("solar_power", solar_stats)}

## 6. Physical Checks
- Raw negative values before cleaning:
  - `load`: {diagnostics.get('raw_negative_load', 0)}
  - `wind_power`: {diagnostics.get('raw_negative_wind_power', 0)}
  - `solar_power`: {diagnostics.get('raw_negative_solar_power', 0)}
- Cleaning applied in processed outputs:
  - `load`, `wind_power`, `solar_power`, `wind_speed`, `dhi`, `dni`, `ghi` clipped at lower bound `0`
  - `humidity` clipped to `[0, 100]`
- Negative value count:
  - `load`: {negative_counts['load']}
  - `wind_power`: {negative_counts['wind_power']}
  - `solar_power`: {negative_counts['solar_power']}
- Night solar proxy used: `ghi <= 1 W/m^2`
- Night solar mean: `{night_solar_mean:.6f}`
- Night solar max: `{night_solar_max:.6f}`
- Spike check:
  - `load` threshold `{load_spike_threshold:.6f}`, count `{load_spike_count}`
  - `wind_power` threshold `{wind_spike_threshold:.6f}`, count `{wind_spike_count}`
  - `solar_power` threshold `{solar_spike_threshold:.6f}`, count `{solar_spike_count}`

## 7. Weather Variable Ranges
| Variable | min | max | mean | q95 | q99 |
|---|---:|---:|---:|---:|---:|
{stats_table_row("temp", weather_ranges["temp"])}
{stats_table_row("humidity", weather_ranges["humidity"])}
{stats_table_row("wind_speed", weather_ranges["wind_speed"])}
{stats_table_row("dhi", weather_ranges["dhi"])}
{stats_table_row("dni", weather_ranges["dni"])}
{stats_table_row("ghi", weather_ranges["ghi"])}

Interpretation:
- `DHI/DNI/GHI` are irradiance-style variables in `W/m^2`, so hourly resampling used arithmetic mean rather than energy integration.
- `season` in the standardized outputs is recomputed from timestamp month using Northern Hemisphere California seasons because this dataset comes from CAISO.

## 8. Suitability for Extreme Scenario Generation
"""
    for line in suitability:
        report += f"- {line}\n"

    report += """
## 9. Preliminary Event Identification Suggestions
- Heat-wave events: start with `temp` monthly or seasonal `q90 / q95`.
- Strong-wind events: start with `wind_speed` `q90 / q95`.
- Low-irradiance events: start with `ghi` `q10`.
- Sharp irradiance drop: use 3-hour `ghi` decline magnitude.
- Net-load extremes: define `net_load = load - wind_power - solar_power`.
- Joint imbalance risk: combine `cum_deficit`, 3-hour net-load ramp, and imbalance duration after choosing a month-specific net-load threshold.
"""
    return report


def save_outputs(
    aligned_5min: pd.DataFrame,
    hourly: pd.DataFrame,
    report_text: str,
    download_info: DownloadInfo,
) -> None:
    aligned_5min.to_csv(FIVE_MIN_PATH, index=False)
    hourly.to_csv(HOURLY_PATH, index=False)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    DOWNLOAD_META_PATH.write_text(
        json.dumps(download_info.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    download_info = download_raw_dataset()
    if not download_info.success:
        ensure_dirs()
        REPORT_PATH.write_text(
            "\n".join(
                [
                    "# OpenEnergyHub CAISO Data Quality Report",
                    "",
                    "## Download Failure",
                    f"- OpenEnergyHub dataset page: {DATASET_PAGE}",
                    f"- Linked Mendeley page: {MENDELEY_PAGE}",
                    f"- Failure reason: {download_info.note}",
                ]
            ),
            encoding="utf-8",
        )
        raise RuntimeError(download_info.note)

    aligned_5min, field_mapping, diagnostics = load_and_standardize()
    hourly = resample_hourly(aligned_5min)
    report_text = build_report(aligned_5min, hourly, field_mapping, diagnostics, download_info)
    save_outputs(aligned_5min, hourly, report_text, download_info)


if __name__ == "__main__":
    main()
