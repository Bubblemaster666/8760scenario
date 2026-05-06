from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_TYPE_NORMAL = "常规背景"
EVENT_TYPE_NORMAL_CODE = 4


@dataclass
class PretrainWindowConfig:
    timeseries_csv: str
    out_dir: str
    seq_len: int = 36
    stride_hours: float = 6.0
    max_windows: int = 1000
    exclude_extreme_windows: bool = False
    train_ratio: float = 0.85
    val_ratio: float = 0.15
    seed: int = 42
    time_col: str = "time"
    load_col: str = "load"
    wind_col: str = "wind_power"
    solar_col: str = "solar_power"
    tau_quantile: float = 0.75
    daylight_start_hour: int = 6
    daylight_end_hour: int = 18


def _infer_season(month: int) -> str:
    month = int(month)
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _season_to_code(season: str) -> int:
    return {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}[season]


def _infer_freq_hours(time_values: pd.Series) -> float:
    diffs = pd.to_datetime(time_values).sort_values().diff().dropna()
    if diffs.empty:
        return 1.0
    hours = diffs.dt.total_seconds().median() / 3600.0
    return float(hours if np.isfinite(hours) and hours > 0 else 1.0)


def _monthly_tau(df: pd.DataFrame, q: float) -> dict[int, float]:
    net_load = df["load"].astype(float) - df["wind_power"].astype(float) - df["solar_power"].astype(float)
    tmp = pd.DataFrame({"month": pd.to_datetime(df["time"]).dt.month, "net_load": net_load})
    default_tau = float(tmp["net_load"].quantile(q))
    out: dict[int, float] = {}
    for month in range(1, 13):
        values = tmp.loc[tmp["month"] == month, "net_load"]
        out[month] = float(values.quantile(q)) if len(values) else default_tau
    return out


def _risk_metrics(window: np.ndarray, tau: float, delta_t_hours: float = 1.0) -> dict[str, float]:
    load = window[0]
    wind = window[1]
    solar = window[2]
    net = load - wind - solar
    excess = np.maximum(0.0, net - float(tau))
    ramp = np.diff(net, prepend=net[0])
    return {
        "cum_deficit": float(excess.sum() * delta_t_hours),
        "netload_ramp_max": float(ramp.max()),
        "imbalance_duration": float((net > float(tau)).sum() * delta_t_hours),
    }


def build_pretrain_windows(cfg: PretrainWindowConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(cfg.timeseries_csv)
    if not src.exists():
        raise FileNotFoundError(f"Timeseries CSV not found: {src}")

    df = pd.read_csv(src)
    rename = {
        cfg.time_col: "time",
        cfg.load_col: "load",
        cfg.wind_col: "wind_power",
        cfg.solar_col: "solar_power",
    }
    missing = [col for col in rename if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in timeseries_csv: {missing}")
    df = df.rename(columns=rename).copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["load", "wind_power", "solar_power"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "load", "wind_power", "solar_power"]).reset_index(drop=True)

    freq_hours = _infer_freq_hours(df["time"])
    stride_steps = max(1, int(round(float(cfg.stride_hours) / max(freq_hours, 1e-6))))
    seq_len = int(cfg.seq_len)
    if len(df) < seq_len:
        raise ValueError(f"Timeseries is shorter than seq_len={seq_len}.")

    starts = np.arange(0, len(df) - seq_len + 1, stride_steps, dtype=np.int64)
    warnings: list[str] = []
    if cfg.exclude_extreme_windows:
        warnings.append("exclude_extreme_windows=True was requested, but no extreme-window index was supplied; no windows were removed.")

    rng = np.random.default_rng(int(cfg.seed))
    if cfg.max_windows and len(starts) > int(cfg.max_windows):
        starts = np.sort(rng.choice(starts, size=int(cfg.max_windows), replace=False))

    tau_by_month = _monthly_tau(df, float(cfg.tau_quantile))
    x_rows: list[np.ndarray] = []
    cond_rows: list[dict] = []
    meta_rows: list[dict] = []
    day_masks: list[np.ndarray] = []
    values = df[["load", "wind_power", "solar_power"]].to_numpy(dtype=np.float32)

    for j, start in enumerate(starts):
        end = int(start + seq_len)
        window_df = df.iloc[int(start):end]
        start_time = pd.Timestamp(window_df["time"].iloc[0])
        end_time = pd.Timestamp(window_df["time"].iloc[-1])
        month = int(start_time.month)
        season = _infer_season(month)
        sample_id = f"P{j + 1:05d}"
        window = values[int(start):end].T.astype(np.float32)
        tau = float(tau_by_month[month])
        risk = _risk_metrics(window, tau=tau, delta_t_hours=freq_hours)
        hours = (int(start_time.hour) + np.arange(seq_len)) % 24
        day_mask = ((hours >= int(cfg.daylight_start_hour)) & (hours <= int(cfg.daylight_end_hour))).astype(np.float32)

        x_rows.append(window)
        day_masks.append(day_mask)
        cond_rows.append(
            {
                "sample_id": sample_id,
                "scenario_id": "",
                "scenario_seed": "",
                "event_type": EVENT_TYPE_NORMAL,
                "event_type_code": EVENT_TYPE_NORMAL_CODE,
                "month": month,
                "season": season,
                "season_code": _season_to_code(season),
                "low_wind_flag": 0,
                "low_irradiance_flag": 0,
                "duration_hours": 0.0,
                "core_n_steps": 0,
                "core_start_offset": -1,
                "core_end_offset": -1,
                "extreme_prob": 1.0,
                "tail_score": 0.0,
                "tail_score_zscore": 0.0,
                "severity_level": 0,
                "cum_deficit": risk["cum_deficit"],
                "netload_ramp_max": risk["netload_ramp_max"],
                "imbalance_duration": risk["imbalance_duration"],
                "imbalance_tau": tau,
                "imbalance_tau_mode": "monthly_quantile",
                "delta_t_hours": float(freq_hours),
            }
        )
        meta_rows.append(
            {
                "sample_id": sample_id,
                "scenario_id": "",
                "core_start_time": "",
                "core_end_time": "",
                "window_start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "window_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "original_start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "original_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "core_start_offset": -1,
                "core_end_offset": -1,
            }
        )

    x = np.stack(x_rows).astype(np.float32)
    cond_df = pd.DataFrame(cond_rows)
    meta_df = pd.DataFrame(meta_rows)
    day_mask_arr = np.stack(day_masks).astype(np.float32)
    order = np.arange(len(x))
    rng.shuffle(order)
    train_n = int(round(len(order) * float(cfg.train_ratio)))
    train_n = min(max(train_n, 1), len(order) - 1 if len(order) > 1 else 1)
    train_idx = np.sort(order[:train_n])
    val_idx = np.sort(order[train_n:])
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1] if len(train_idx) > 1 else train_idx

    np.save(out_dir / "pretrain_X_train.npy", x[train_idx])
    np.save(out_dir / "pretrain_X_val.npy", x[val_idx])
    np.save(out_dir / "pretrain_day_mask_train.npy", day_mask_arr[train_idx])
    np.save(out_dir / "pretrain_day_mask_val.npy", day_mask_arr[val_idx])
    cond_df.iloc[train_idx].reset_index(drop=True).to_csv(out_dir / "pretrain_cond_train.csv", index=False, encoding="utf-8-sig")
    cond_df.iloc[val_idx].reset_index(drop=True).to_csv(out_dir / "pretrain_cond_val.csv", index=False, encoding="utf-8-sig")
    meta_df.iloc[train_idx].reset_index(drop=True).to_csv(out_dir / "pretrain_meta_train.csv", index=False, encoding="utf-8-sig")
    meta_df.iloc[val_idx].reset_index(drop=True).to_csv(out_dir / "pretrain_meta_val.csv", index=False, encoding="utf-8-sig")

    summary = {
        "config": asdict(cfg),
        "source": str(src),
        "freq_hours": float(freq_hours),
        "stride_steps": int(stride_steps),
        "num_windows_total": int(len(x)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "x_shape": list(x.shape),
        "warnings": warnings,
        "note": "Normal-window pretraining data only; original extreme train/val/test artifacts were not modified.",
    }
    (out_dir / "pretrain_window_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> PretrainWindowConfig:
    parser = argparse.ArgumentParser(description="Build normal-window pretraining arrays from long wind/solar/load time series.")
    parser.add_argument("--timeseries-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--stride-hours", type=float, default=6.0)
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--exclude-extreme-windows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--load-col", type=str, default="load")
    parser.add_argument("--wind-col", type=str, default="wind_power")
    parser.add_argument("--solar-col", type=str, default="solar_power")
    parser.add_argument("--tau-quantile", type=float, default=0.75)
    return PretrainWindowConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    build_pretrain_windows(parse_args())
