from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


EVENT_COLD = "寒潮"
EVENT_RAIN = "暴雨/强降水"
EVENT_WIND = "大风/沙尘暴"
EVENT_HEAT = "高温"
# Kept for backward compatibility only. The default main event set now uses EVENT_RAIN.
EVENT_SNOW = "暴雪/风吹雪"


@dataclass
class DetectConfig:
    time_col: str = "time"
    temp_col: str = "temp"
    wind_speed_col: str = "wind_speed"
    irradiance_col: str = "irradiance"
    snowfall_col: str = "snowfall"
    visibility_col: str = "visibility"
    precipitation_col: str = "precipitation"
    precipitation_fallback_col: str = "precipitation"

    freq_hours: int = 1
    min_event_hours: int = 6
    low_resource_min_hours: int = 3

    cold_temp_threshold: float = -10.0
    cold_drop_24h_threshold: float = 8.0
    heat_temp_threshold: float = 35.0
    high_wind_speed_threshold: float = 12.0
    low_visibility_threshold: float = 3000.0
    blowing_snow_wind_threshold: float = 8.0

    # Heavy rain / strong precipitation event. Prefer adaptive quantile for Xinjiang-style local extremes.
    replace_snow_with_heavy_rain: bool = True
    use_adaptive_rain_threshold: bool = True
    heavy_rain_quantile: float = 0.98
    heavy_rain_rolling_quantile: float = 0.97
    rain_rolling_window_hours: int = 3
    heavy_rain_threshold: Optional[float] = None
    rain_min_event_hours: int = 2
    rain_min_total_precip: Optional[float] = None
    rain_require_power_impact: bool = True

    # Legacy snow parameters retained for compatibility when replace_snow_with_heavy_rain=False.
    snowfall_threshold: float = 1.0
    snowfall_quantile: float = 0.97
    snow_temp_margin: float = 2.0
    require_wind_for_snow_event: bool = False
    snow_precip_relax_ratio_with_wind: float = 0.7

    use_adaptive_thresholds: bool = False
    cold_temp_quantile: float = 0.10
    cold_drop_24h_quantile: float = 0.95
    heat_temp_quantile: float = 0.98
    high_wind_speed_quantile: float = 0.99
    adaptive_cold_drop_min: float = 4.5

    # Resource-state labels. They are process tags, not main event types.
    low_irr_quantile: float = 0.35
    low_wind_quantile: float = 0.30
    daylight_irradiance_min: float = 30.0


def _prepare_dataframe(df: pd.DataFrame, cfg: DetectConfig) -> pd.DataFrame:
    out = df.copy()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col])
    out = out.sort_values(cfg.time_col).reset_index(drop=True)
    return out


def _find_true_segments(flag: pd.Series, time: pd.Series, min_len: int) -> List[Dict]:
    flag = flag.fillna(False).astype(bool).reset_index(drop=True)
    time = pd.to_datetime(time).reset_index(drop=True)
    segments: List[Dict] = []
    start_idx: Optional[int] = None

    for i, value in enumerate(flag):
        if value and start_idx is None:
            start_idx = i

        is_terminal = (not value) or (i == len(flag) - 1)
        if is_terminal and start_idx is not None:
            end_idx = i if value and i == len(flag) - 1 else i - 1
            n_steps = end_idx - start_idx + 1
            if n_steps >= min_len:
                segments.append(
                    {
                        "start_time": time.iloc[start_idx],
                        "end_time": time.iloc[end_idx],
                        "n_steps": int(n_steps),
                    }
                )
            start_idx = None
    return segments


def _add_buffer(start_time: pd.Timestamp, end_time: pd.Timestamp, buffer_hours: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return start_time - pd.Timedelta(hours=buffer_hours), end_time + pd.Timedelta(hours=buffer_hours)


def _merge_overlapping_windows(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return windows.copy()
    windows = windows.sort_values(["start_time", "end_time"]).reset_index(drop=True)
    merged = [windows.iloc[0].to_dict()]
    for _, row in windows.iloc[1:].iterrows():
        last = merged[-1]
        if row["start_time"] <= last["end_time"]:
            last["end_time"] = max(last["end_time"], row["end_time"])
            last["core_start_time"] = min(last["core_start_time"], row["core_start_time"])
            last["core_end_time"] = max(last["core_end_time"], row["core_end_time"])
            last["event_type"] = last["event_type"] + "+" + row["event_type"]
            last["low_irradiance_flag"] = max(last["low_irradiance_flag"], row["low_irradiance_flag"])
            last["low_wind_flag"] = max(last["low_wind_flag"], row["low_wind_flag"])
        else:
            merged.append(row.to_dict())
    out = pd.DataFrame(merged)
    out["duration_hours"] = (
        (pd.to_datetime(out["end_time"]) - pd.to_datetime(out["start_time"])) / pd.Timedelta(hours=1)
    ).astype(int) + 1
    return out


def _safe_quantile(series: pd.Series, q: float, default: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float(default)
    return float(clean.quantile(q))


def _resolve_precip_col(df: pd.DataFrame, cfg: DetectConfig) -> Optional[str]:
    if cfg.precipitation_col in df.columns:
        return cfg.precipitation_col
    if cfg.precipitation_fallback_col in df.columns:
        return cfg.precipitation_fallback_col
    if cfg.snowfall_col in df.columns:
        # For legacy datasets this may be the only precipitation-like variable.
        return cfg.snowfall_col
    return None


def _resolve_thresholds(df: pd.DataFrame, cfg: DetectConfig) -> dict[str, float]:
    thresholds = {
        "cold_temp_threshold": float(cfg.cold_temp_threshold),
        "cold_drop_24h_threshold": float(cfg.cold_drop_24h_threshold),
        "heat_temp_threshold": float(cfg.heat_temp_threshold),
        "high_wind_speed_threshold": float(cfg.high_wind_speed_threshold),
        "snowfall_threshold": float(cfg.snowfall_threshold),
        "blowing_snow_wind_threshold": float(cfg.blowing_snow_wind_threshold),
        "low_visibility_threshold": float(cfg.low_visibility_threshold),
        "heavy_rain_threshold": float(cfg.heavy_rain_threshold or 0.0),
    }

    if cfg.use_adaptive_thresholds and cfg.temp_col in df.columns:
        temp = df[cfg.temp_col]
        thresholds["cold_temp_threshold"] = max(
            float(cfg.cold_temp_threshold),
            _safe_quantile(temp, cfg.cold_temp_quantile, cfg.cold_temp_threshold),
        )
        thresholds["heat_temp_threshold"] = min(
            float(cfg.heat_temp_threshold),
            _safe_quantile(temp, cfg.heat_temp_quantile, cfg.heat_temp_threshold),
        )
        temp_drop = pd.to_numeric(df.get("temp_drop_24h"), errors="coerce")
        drop_q = _safe_quantile(temp_drop, cfg.cold_drop_24h_quantile, cfg.cold_drop_24h_threshold)
        thresholds["cold_drop_24h_threshold"] = min(
            float(cfg.cold_drop_24h_threshold),
            max(float(cfg.adaptive_cold_drop_min), drop_q),
        )

    if cfg.use_adaptive_thresholds and cfg.wind_speed_col in df.columns:
        thresholds["high_wind_speed_threshold"] = min(
            float(cfg.high_wind_speed_threshold),
            _safe_quantile(df[cfg.wind_speed_col], cfg.high_wind_speed_quantile, cfg.high_wind_speed_threshold),
        )

    precip_col = _resolve_precip_col(df, cfg)
    if precip_col is not None:
        if cfg.use_adaptive_rain_threshold:
            thresholds["heavy_rain_threshold"] = _safe_quantile(
                df[precip_col], cfg.heavy_rain_quantile, cfg.heavy_rain_threshold or cfg.snowfall_threshold
            )
        elif cfg.heavy_rain_threshold is not None:
            thresholds["heavy_rain_threshold"] = float(cfg.heavy_rain_threshold)
        if cfg.use_adaptive_thresholds:
            thresholds["snowfall_threshold"] = max(
                float(cfg.snowfall_threshold),
                _safe_quantile(df[precip_col], cfg.snowfall_quantile, cfg.snowfall_threshold),
            )
    return thresholds


def _build_low_irr_flag(df: pd.DataFrame, cfg: DetectConfig) -> pd.Series:
    if cfg.irradiance_col not in df.columns:
        return pd.Series(False, index=df.index)
    daylight_mask = df[cfg.irradiance_col] > cfg.daylight_irradiance_min
    if daylight_mask.sum() == 0:
        return pd.Series(False, index=df.index)
    irr_q = df.loc[daylight_mask, cfg.irradiance_col].quantile(cfg.low_irr_quantile)
    return (daylight_mask & (df[cfg.irradiance_col] <= irr_q)).reindex(df.index, fill_value=False)


def _build_low_wind_flag(df: pd.DataFrame, cfg: DetectConfig) -> pd.Series:
    if cfg.wind_speed_col not in df.columns:
        return pd.Series(False, index=df.index)
    wind_q = df[cfg.wind_speed_col].quantile(cfg.low_wind_quantile)
    return (df[cfg.wind_speed_col] <= wind_q).reindex(df.index, fill_value=False)


def detect_extreme_samples(
    df: pd.DataFrame,
    cfg: Optional[DetectConfig] = None,
    add_buffer_hours: int = 0,
    merge_overlap: bool = True,
) -> pd.DataFrame:
    cfg = cfg or DetectConfig()
    df = _prepare_dataframe(df, cfg)
    min_len = max(1, int(cfg.min_event_hours / cfg.freq_hours))
    rain_min_len = max(1, int(cfg.rain_min_event_hours / cfg.freq_hours))
    low_res_len = max(1, int(cfg.low_resource_min_hours / cfg.freq_hours))

    if cfg.temp_col in df.columns:
        lag_steps_24h = max(1, int(24 / cfg.freq_hours))
        df["temp_drop_24h"] = df[cfg.temp_col].shift(lag_steps_24h) - df[cfg.temp_col]
    else:
        df["temp_drop_24h"] = np.nan

    thresholds = _resolve_thresholds(df, cfg)
    heavy_rain_warning = ""
    heavy_rain_enabled = bool(cfg.replace_snow_with_heavy_rain)
    heavy_rain_instant_threshold = float("nan")
    heavy_rain_rolling_threshold = float("nan")
    low_irr_flag = _build_low_irr_flag(df, cfg)
    low_wind_flag = _build_low_wind_flag(df, cfg)

    flags: dict[str, pd.Series] = {}
    event_min_lens: dict[str, int] = {}

    if cfg.temp_col in df.columns:
        flags[EVENT_COLD] = (
            (df[cfg.temp_col] <= thresholds["cold_temp_threshold"])
            & (df["temp_drop_24h"] >= thresholds["cold_drop_24h_threshold"])
        )
        flags[EVENT_HEAT] = df[cfg.temp_col] >= thresholds["heat_temp_threshold"]
    else:
        flags[EVENT_COLD] = pd.Series(False, index=df.index)
        flags[EVENT_HEAT] = pd.Series(False, index=df.index)
    event_min_lens[EVENT_COLD] = min_len
    event_min_lens[EVENT_HEAT] = min_len

    if cfg.wind_speed_col in df.columns:
        strong_wind = df[cfg.wind_speed_col] >= thresholds["high_wind_speed_threshold"]
        if cfg.visibility_col in df.columns:
            dust_like = (df[cfg.wind_speed_col] >= thresholds["blowing_snow_wind_threshold"]) & (
                df[cfg.visibility_col] <= thresholds["low_visibility_threshold"]
            )
            flags[EVENT_WIND] = strong_wind | dust_like
        else:
            flags[EVENT_WIND] = strong_wind
    else:
        flags[EVENT_WIND] = pd.Series(False, index=df.index)
    event_min_lens[EVENT_WIND] = min_len

    precip_col = _resolve_precip_col(df, cfg)
    if cfg.replace_snow_with_heavy_rain:
        if precip_col is None:
            heavy_rain_warning = "missing_precipitation_column"
            flags[EVENT_RAIN] = pd.Series(False, index=df.index)
        else:
            precip_raw = pd.to_numeric(df[precip_col], errors="coerce")
            valid_precip = precip_raw.dropna()
            precip = precip_raw.fillna(0.0)
            if valid_precip.empty or float(precip.max()) <= 0.0 or int((precip > 0).sum()) < 2:
                heavy_rain_warning = "insufficient_or_zero_precipitation"
                flags[EVENT_RAIN] = pd.Series(False, index=df.index)
            else:
                freq_hours = max(1, int(cfg.freq_hours))
                roll_win = max(1, int(np.ceil(float(cfg.rain_rolling_window_hours) / float(freq_hours))))
                heavy_rain_instant_threshold = float(precip.quantile(cfg.heavy_rain_quantile))
                rolling_precip = precip.rolling(window=roll_win, min_periods=1).sum()
                heavy_rain_rolling_threshold = float(rolling_precip.quantile(cfg.heavy_rain_rolling_quantile))
                rain_flag = (precip >= heavy_rain_instant_threshold) | (rolling_precip >= heavy_rain_rolling_threshold)

                if cfg.rain_min_total_precip is not None:
                    min_total = float(cfg.rain_min_total_precip)
                    kept_flag = pd.Series(False, index=df.index)
                    segments = _find_true_segments(rain_flag, df[cfg.time_col], rain_min_len)
                    for segment in segments:
                        seg_mask = (df[cfg.time_col] >= segment["start_time"]) & (df[cfg.time_col] <= segment["end_time"])
                        seg_precip_sum = float(precip.loc[seg_mask].sum())
                        seg_roll_max = float(rolling_precip.loc[seg_mask].max()) if int(seg_mask.sum()) > 0 else 0.0
                        if max(seg_precip_sum, seg_roll_max) >= min_total:
                            kept_flag.loc[seg_mask] = True
                    rain_flag = kept_flag
                flags[EVENT_RAIN] = rain_flag.fillna(False)
        event_min_lens[EVENT_RAIN] = rain_min_len
    else:
        if precip_col is not None:
            snow_like = pd.to_numeric(df[precip_col], errors="coerce").fillna(0.0) >= thresholds["snowfall_threshold"]
            if cfg.temp_col in df.columns:
                snow_temp_ceiling = thresholds["cold_temp_threshold"] + float(cfg.snow_temp_margin)
                snow_like = snow_like & (df[cfg.temp_col] <= snow_temp_ceiling)
            if cfg.wind_speed_col in df.columns:
                wind_boost_like = (
                    (pd.to_numeric(df[precip_col], errors="coerce").fillna(0.0) >= thresholds["snowfall_threshold"] * float(cfg.snow_precip_relax_ratio_with_wind))
                    & (df[cfg.wind_speed_col] >= thresholds["blowing_snow_wind_threshold"])
                )
                if cfg.temp_col in df.columns:
                    wind_boost_like = wind_boost_like & (df[cfg.temp_col] <= snow_temp_ceiling)
                snow_like = wind_boost_like if cfg.require_wind_for_snow_event else (snow_like | wind_boost_like)
            flags[EVENT_SNOW] = snow_like
        else:
            flags[EVENT_SNOW] = pd.Series(False, index=df.index)
        event_min_lens[EVENT_SNOW] = min_len

    windows = []
    for event_name, flag in flags.items():
        segments = _find_true_segments(flag, df[cfg.time_col], event_min_lens.get(event_name, min_len))
        for segment in segments:
            core_start = segment["start_time"]
            core_end = segment["end_time"]
            start_time, end_time = core_start, core_end
            if add_buffer_hours > 0:
                start_time, end_time = _add_buffer(core_start, core_end, add_buffer_hours)

            # Resource-state labels describe the process around the event, so use the buffered window.
            event_sub = df[(df[cfg.time_col] >= start_time) & (df[cfg.time_col] <= end_time)]
            low_irr_segments = _find_true_segments(low_irr_flag.loc[event_sub.index], event_sub[cfg.time_col], low_res_len)
            low_wind_segments = _find_true_segments(low_wind_flag.loc[event_sub.index], event_sub[cfg.time_col], low_res_len)

            windows.append(
                {
                    "event_type": event_name,
                    "core_start_time": core_start,
                    "core_end_time": core_end,
                    "start_time": start_time,
                    "end_time": end_time,
                    "core_n_steps": segment["n_steps"],
                    "duration_hours": int((core_end - core_start) / pd.Timedelta(hours=1)) + 1,
                    "window_duration_hours": int((end_time - start_time) / pd.Timedelta(hours=1)) + 1,
                    "month": int(pd.Timestamp(core_start).month),
                    "low_irradiance_flag": int(bool(low_irr_segments)),
                    "low_wind_flag": int(bool(low_wind_segments)),
                }
            )

    out = pd.DataFrame(windows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "sample_id",
                "event_type",
                "core_start_time",
                "core_end_time",
                "start_time",
                "end_time",
                "core_n_steps",
                "duration_hours",
                "window_duration_hours",
                "month",
                "low_irradiance_flag",
                "low_wind_flag",
            ]
        )
        thresholds.update(
            {
                "heavy_rain_instant_threshold": heavy_rain_instant_threshold,
                "heavy_rain_rolling_threshold": heavy_rain_rolling_threshold,
                "heavy_rain_quantile": float(cfg.heavy_rain_quantile),
                "heavy_rain_rolling_quantile": float(cfg.heavy_rain_rolling_quantile),
                "rain_rolling_window_hours": int(cfg.rain_rolling_window_hours),
                "rain_min_event_hours": int(cfg.rain_min_event_hours),
                "rain_min_total_precip": cfg.rain_min_total_precip,
                "precipitation_col_used": precip_col,
                "heavy_rain_enabled": heavy_rain_enabled,
                "heavy_rain_warning": heavy_rain_warning,
            }
        )
        out.attrs["resolved_thresholds"] = thresholds
        return out

    if merge_overlap:
        out = _merge_overlapping_windows(out)
    out = out.reset_index(drop=True)
    out.insert(0, "sample_id", [f"S{i:04d}" for i in range(1, len(out) + 1)])
    thresholds.update(
        {
            "heavy_rain_instant_threshold": heavy_rain_instant_threshold,
            "heavy_rain_rolling_threshold": heavy_rain_rolling_threshold,
            "heavy_rain_quantile": float(cfg.heavy_rain_quantile),
            "heavy_rain_rolling_quantile": float(cfg.heavy_rain_rolling_quantile),
            "rain_rolling_window_hours": int(cfg.rain_rolling_window_hours),
            "rain_min_event_hours": int(cfg.rain_min_event_hours),
            "rain_min_total_precip": cfg.rain_min_total_precip,
            "precipitation_col_used": precip_col,
            "heavy_rain_enabled": heavy_rain_enabled,
            "heavy_rain_warning": heavy_rain_warning,
        }
    )
    out.attrs["resolved_thresholds"] = thresholds
    return out


def make_mock_data() -> pd.DataFrame:
    time = pd.date_range("2024-01-01 00:00:00", periods=24 * 15, freq="1h")
    n = len(time)
    rng = np.random.default_rng(0)
    hour = time.hour.to_numpy()
    day = np.arange(n)

    temp = 5 + rng.normal(0, 2, n)
    wind_speed = 6 + rng.normal(0, 1.0, n)
    irradiance = np.maximum(0, 500 * np.sin((hour - 6) / 12 * np.pi))
    snowfall = np.zeros(n)
    precipitation = np.zeros(n)
    visibility = np.full(n, 10000.0)

    load = 600 + 60 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 10, n)
    wind_power = np.clip(150 + 20 * np.sin(day / 24 * 2 * np.pi / 3) + rng.normal(0, 15, n), 0, None)
    solar_power = np.clip(np.maximum(0, 220 * np.sin((hour - 6) / 12 * np.pi)) + rng.normal(0, 8, n), 0, None)

    df = pd.DataFrame(
        {
            "time": time,
            "temp": temp,
            "wind_speed": wind_speed,
            "irradiance": irradiance,
            "snowfall": snowfall,
            "precipitation": precipitation,
            "visibility": visibility,
            "load": load,
            "wind_power": wind_power,
            "solar_power": solar_power,
        }
    )

    # 高温事件
    idx = (df["time"] >= "2024-01-05 10:00") & (df["time"] <= "2024-01-05 20:00")
    df.loc[idx, "temp"] = 38
    df.loc[idx, "load"] += 100
    df.loc[idx, "solar_power"] *= 0.85

    # 寒潮事件
    idx = (df["time"] >= "2024-01-09 03:00") & (df["time"] <= "2024-01-09 12:00")
    df.loc[idx, "temp"] = -12
    df.loc[idx, "load"] += 90
    df.loc[idx, "wind_power"] *= 0.55

    # 大风/沙尘暴事件
    idx = (df["time"] >= "2024-01-11 08:00") & (df["time"] <= "2024-01-11 16:00")
    df.loc[idx, "wind_speed"] = 15
    df.loc[idx, "visibility"] = 1500
    df.loc[idx, "solar_power"] *= 0.35

    # 暴雨/强降水事件
    idx = (df["time"] >= "2024-01-13 13:00") & (df["time"] <= "2024-01-13 16:00")
    df.loc[idx, "precipitation"] = 18.0
    df.loc[idx, "visibility"] = 2500
    df.loc[idx, "solar_power"] *= 0.25
    df.loc[idx, "wind_power"] *= 0.75
    return df


if __name__ == "__main__":
    df_demo = make_mock_data()
    samples = detect_extreme_samples(df_demo, DetectConfig(), add_buffer_hours=2)
    print(samples)
