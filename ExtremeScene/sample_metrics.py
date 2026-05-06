from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from risk_metrics import RiskMetricConfig, build_tau_diagnostic, hard_risk_metrics_from_net_load, month_to_season, prepare_net_load_frame, resolve_context_tau


@dataclass
class MetricConfig:
    time_col: str = "time"
    load_col: str = "load"
    wind_power_col: str = "wind_power"
    solar_power_col: str = "solar_power"
    imbalance_tau_mode: str = "monthly_quantile"
    imbalance_tau_fixed: float = 0.0
    imbalance_tau_quantile: float = 0.75
    freq_hours: Optional[float] = None


def _to_risk_config(cfg: MetricConfig) -> RiskMetricConfig:
    return RiskMetricConfig(
        time_col=cfg.time_col,
        load_col=cfg.load_col,
        wind_col=cfg.wind_power_col,
        solar_col=cfg.solar_power_col,
        tau_mode=cfg.imbalance_tau_mode,
        tau_fixed=cfg.imbalance_tau_fixed,
        tau_quantile=cfg.imbalance_tau_quantile,
        freq_hours=cfg.freq_hours,
    )


def compute_metrics_for_samples(df: pd.DataFrame, samples: pd.DataFrame, cfg: Optional[MetricConfig] = None) -> pd.DataFrame:
    cfg = cfg or MetricConfig()
    if samples.empty:
        return samples.copy()
    required_sample_cols = {"sample_id", "event_type", "start_time", "end_time"}
    missing_sample_cols = required_sample_cols - set(samples.columns)
    if missing_sample_cols:
        raise ValueError(f"samples is missing required columns: {sorted(missing_sample_cols)}")
    required_df_cols = {cfg.time_col, cfg.load_col, cfg.wind_power_col, cfg.solar_power_col}
    missing_df_cols = required_df_cols - set(df.columns)
    if missing_df_cols:
        raise ValueError(f"df is missing required columns: {sorted(missing_df_cols)}")

    risk_cfg = _to_risk_config(cfg)
    prepared_df, delta_t_hours = prepare_net_load_frame(df, risk_cfg)
    tau_diagnostic = build_tau_diagnostic(prepared_df, risk_cfg)
    out_rows = []
    for _, row in samples.iterrows():
        start_time = pd.to_datetime(row["start_time"])
        end_time = pd.to_datetime(row["end_time"])
        sub = prepared_df[(prepared_df[cfg.time_col] >= start_time) & (prepared_df[cfg.time_col] <= end_time)].copy()
        rec = row.to_dict()
        month = int(row.get("month", pd.Timestamp(row.get("core_start_time", start_time)).month))
        season = str(row.get("season", month_to_season(month)))
        tau = resolve_context_tau(prepared_df, risk_cfg, month=month, season=season)
        rec["imbalance_tau"] = tau
        rec["imbalance_tau_mode"] = cfg.imbalance_tau_mode
        rec["delta_t_hours"] = delta_t_hours
        if sub.empty:
            rec["cum_deficit"] = np.nan
            rec["netload_ramp_max"] = np.nan
            rec["imbalance_duration"] = np.nan
            rec["netload_peak"] = np.nan
            rec["netload_mean"] = np.nan
            out_rows.append(rec)
            continue
        rec.update(hard_risk_metrics_from_net_load(sub["net_load"].to_numpy(dtype=float), tau=tau, delta_t_hours=delta_t_hours))
        out_rows.append(rec)
    out = pd.DataFrame(out_rows)
    out.attrs["tau_diagnostic"] = tau_diagnostic
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    time = pd.date_range("2024-01-01 00:00:00", periods=48, freq="1h")
    df_demo = pd.DataFrame({"time": time, "load": 600 + rng.normal(0, 10, 48), "wind_power": 150 + rng.normal(0, 15, 48), "solar_power": np.maximum(0, 200 * np.sin((time.hour.to_numpy() - 6) / 12 * np.pi))})
    samples_demo = pd.DataFrame({"sample_id": ["S0001"], "event_type": ["寒潮"], "start_time": [pd.Timestamp("2024-01-01 08:00:00")], "end_time": [pd.Timestamp("2024-01-01 22:00:00")], "low_irradiance_flag": [1], "low_wind_flag": [1]})
    print(compute_metrics_for_samples(df_demo, samples_demo, MetricConfig()))
