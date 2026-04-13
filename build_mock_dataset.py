from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Extreme_Extract import DetectConfig, detect_extreme_samples
from sample_metrics import MetricConfig, compute_metrics_for_samples
from evt_fit import EVTConfig, fit_evt_and_label


@dataclass
class BatchMockConfig:
    # ===== 批量 mock 数据设置 =====
    n_scenarios: int = 60
    days_per_scenario: int = 30
    base_start: str = "2024-01-01 00:00:00"
    gap_days_between_scenarios: int = 3
    random_seed: int = 42

    # ===== 事件窗口设置 =====
    add_buffer_hours: int = 2
    fixed_window_hours: int = 24
    merge_overlap: bool = False

    # ===== 输出目录 =====
    output_dir: str = "mock_dataset_outputs"


EVENT_TYPE_TO_CODE: Dict[str, int] = {
    "寒潮": 0,
    "高温": 1,
    "大风/沙尘暴": 2,
    "暴雪/风吹雪": 3,
}


def _daylight_profile(hour: np.ndarray, peak: float = 520.0) -> np.ndarray:
    return np.maximum(0.0, peak * np.sin((hour - 6) / 12 * np.pi))


def make_random_mock_data(
    seed: int,
    start_time: str | pd.Timestamp,
    days: int = 30,
) -> pd.DataFrame:
    """
    随机生成一段包含风光荷和气象变量的 mock 数据。
    与原 make_mock_data() 相比，这里把事件注入做成了“可随机化、可重复调用”的版本，
    便于批量生成训练样本。
    """
    rng = np.random.default_rng(seed)

    time = pd.date_range(pd.Timestamp(start_time), periods=24 * days, freq="1h")
    n = len(time)
    hour = time.hour.to_numpy()
    day_index = np.arange(n) / 24.0

    # =========================
    # 1) 基础气象变量
    # =========================
    seasonal = 4.0 * np.sin(2 * np.pi * day_index / max(days, 1))
    temp = 6.0 + seasonal + rng.normal(0, 2.0, n)
    wind_speed = 6.0 + 1.2 * np.sin(2 * np.pi * day_index / 5.0) + rng.normal(0, 1.1, n)

    cloud_factor = 0.9 + 0.2 * np.sin(2 * np.pi * day_index / 7.0 + 0.8)
    irradiance = _daylight_profile(hour, peak=520.0) * cloud_factor
    irradiance += rng.normal(0, 18.0, n)
    irradiance = np.clip(irradiance, 0.0, None)

    snowfall = np.zeros(n)
    visibility = np.full(n, 10000.0)

    # =========================
    # 2) 基础风光荷功率变量
    #    尽量让功率与气象有一定相关性，避免完全脱节
    # =========================
    load = (
        620
        + 65 * np.sin((hour - 8) / 24 * 2 * np.pi)
        + 1.6 * np.maximum(0.0, 8.0 - temp)
        + 1.0 * np.maximum(0.0, temp - 28.0)
        + rng.normal(0, 10.0, n)
    )

    wind_power = 25.0 * np.maximum(0.0, wind_speed - 0.8) + rng.normal(0, 10.0, n)
    wind_power = np.clip(wind_power, 0.0, None)

    solar_power = 0.46 * irradiance + rng.normal(0, 8.0, n)
    solar_power = np.clip(solar_power, 0.0, None)

    df = pd.DataFrame(
        {
            "time": time,
            "temp": temp,
            "wind_speed": wind_speed,
            "irradiance": irradiance,
            "snowfall": snowfall,
            "visibility": visibility,
            "load": load,
            "wind_power": wind_power,
            "solar_power": solar_power,
        }
    )

    # =========================
    # 3) 注入主事件（每类至少 1 次）
    #    为避免冲突，给每类事件预留一个时间槽，再在槽内随机抖动
    # =========================
    slot_days = [4, 10, 16, 22]
    jitters = rng.integers(-18, 19, size=4)  # 小时级随机偏移

    def make_mask(start_idx: int, duration_h: int) -> pd.Series:
        start_idx = max(0, min(n - 1, start_idx))
        end_idx = max(start_idx, min(n - 1, start_idx + duration_h - 1))
        return (df.index >= start_idx) & (df.index <= end_idx)

    # ---- 寒潮 ----
    cold_duration = int(rng.integers(10, 19))
    cold_start = slot_days[0] * 24 + int(jitters[0])
    cold_mask = make_mask(cold_start, cold_duration)

    prev_start = max(0, cold_start - 24)
    prev_end = max(prev_start, min(n - 1, prev_start + min(cold_duration, 12) - 1))
    prev_mask = (df.index >= prev_start) & (df.index <= prev_end)

    cold_temp = float(rng.uniform(-16.0, -11.5))
    df.loc[prev_mask, "temp"] = rng.uniform(1.0, 5.0)
    df.loc[cold_mask, "temp"] = cold_temp
    df.loc[cold_mask, "load"] += rng.uniform(60.0, 130.0)
    df.loc[cold_mask, "irradiance"] *= rng.uniform(0.35, 0.75)
    df.loc[cold_mask, "solar_power"] *= rng.uniform(0.25, 0.65)
    if rng.random() < 0.45:
        df.loc[cold_mask, "wind_speed"] *= rng.uniform(0.25, 0.55)
        df.loc[cold_mask, "wind_power"] *= rng.uniform(0.15, 0.45)

    # ---- 高温 ----
    heat_duration = int(rng.integers(8, 15))
    heat_start = slot_days[1] * 24 + 10 + int(jitters[1])
    heat_mask = make_mask(heat_start, heat_duration)
    heat_temp = float(rng.uniform(35.5, 40.0))
    df.loc[heat_mask, "temp"] = heat_temp
    df.loc[heat_mask, "load"] += rng.uniform(80.0, 150.0)
    df.loc[heat_mask, "wind_speed"] -= rng.uniform(1.2, 2.8)
    df.loc[heat_mask, "wind_power"] *= rng.uniform(0.5, 0.8)

    # ---- 大风 / 沙尘暴 ----
    dust_duration = int(rng.integers(6, 13))
    dust_start = slot_days[2] * 24 + 6 + int(jitters[2])
    dust_mask = make_mask(dust_start, dust_duration)
    df.loc[dust_mask, "wind_speed"] = rng.uniform(13.0, 17.5)
    df.loc[dust_mask, "visibility"] = rng.uniform(800.0, 2500.0)
    df.loc[dust_mask, "irradiance"] *= rng.uniform(0.08, 0.35)
    df.loc[dust_mask, "solar_power"] *= rng.uniform(0.05, 0.30)
    if rng.random() < 0.4:
        df.loc[dust_mask, "load"] += rng.uniform(20.0, 60.0)

    # ---- 暴雪 / 风吹雪 ----
    snow_duration = int(rng.integers(7, 15))
    snow_start = slot_days[3] * 24 + 2 + int(jitters[3])
    snow_mask = make_mask(snow_start, snow_duration)
    df.loc[snow_mask, "snowfall"] = rng.uniform(1.5, 3.5)
    df.loc[snow_mask, "wind_speed"] = rng.uniform(8.5, 12.0)
    df.loc[snow_mask, "temp"] = rng.uniform(-14.0, -6.0)
    df.loc[snow_mask, "solar_power"] *= rng.uniform(0.02, 0.20)
    df.loc[snow_mask, "irradiance"] *= rng.uniform(0.10, 0.40)
    df.loc[snow_mask, "load"] += rng.uniform(50.0, 110.0)

    # =========================
    # 4) 再注入若干补充资源状态片段（不作为主事件，只增加多样性）
    # =========================
    low_wind_count = int(rng.integers(1, 3))
    for _ in range(low_wind_count):
        duration = int(rng.integers(6, 13))
        start = int(rng.integers(0, max(1, n - duration)))
        mask = make_mask(start, duration)
        df.loc[mask, "wind_speed"] = np.minimum(df.loc[mask, "wind_speed"], rng.uniform(0.8, 1.8))
        df.loc[mask, "wind_power"] *= rng.uniform(0.08, 0.35)

    low_irr_count = int(rng.integers(1, 3))
    for _ in range(low_irr_count):
        duration = int(rng.integers(6, 11))
        start_day = int(rng.integers(0, max(1, days - 1)))
        start_hour = int(rng.integers(8, 14))
        start = start_day * 24 + start_hour
        mask = make_mask(start, duration)
        daytime_mask = mask & (df["irradiance"] > 60.0)
        df.loc[daytime_mask, "irradiance"] = np.minimum(df.loc[daytime_mask, "irradiance"], rng.uniform(5.0, 35.0))
        df.loc[daytime_mask, "solar_power"] *= rng.uniform(0.05, 0.25)

    # 最终做一次裁剪
    df["wind_speed"] = np.clip(df["wind_speed"], 0.0, None)
    df["irradiance"] = np.clip(df["irradiance"], 0.0, None)
    df["load"] = np.clip(df["load"], 0.0, None)
    df["wind_power"] = np.clip(df["wind_power"], 0.0, None)
    df["solar_power"] = np.clip(df["solar_power"], 0.0, None)

    return df


def _make_unique_scenario_time(
    scenario_id: int,
    base_start: str,
    days_per_scenario: int,
    gap_days: int,
) -> pd.Timestamp:
    offset_days = scenario_id * (days_per_scenario + gap_days)
    return pd.Timestamp(base_start) + pd.Timedelta(days=offset_days)


def _extract_fixed_window(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cols: List[str],
) -> pd.DataFrame:
    time_index = pd.date_range(start_time, end_time, freq="1h")
    sub = df.set_index("time")[cols].reindex(time_index)

    # 先用前后值填补；若整段为空，再用 0 填补
    sub = sub.ffill().bfill().fillna(0.0)
    sub.index.name = "time"
    return sub.reset_index()


def build_mock_dataset(
    batch_cfg: Optional[BatchMockConfig] = None,
    detect_cfg: Optional[DetectConfig] = None,
    metric_cfg: Optional[MetricConfig] = None,
    evt_cfg: Optional[EVTConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict]:
    if batch_cfg is None:
        batch_cfg = BatchMockConfig()
    if detect_cfg is None:
        detect_cfg = DetectConfig(
            time_col="time",
            temp_col="temp",
            wind_speed_col="wind_speed",
            irradiance_col="irradiance",
            snowfall_col="snowfall",
            visibility_col="visibility",
            freq_hours=1,
            min_event_hours=6,
            low_resource_min_hours=6,
            daylight_irradiance_min=50.0,
        )
    if metric_cfg is None:
        metric_cfg = MetricConfig(
            time_col="time",
            load_col="load",
            wind_power_col="wind_power",
            solar_power_col="solar_power",
            imbalance_tau_mode="quantile",
            imbalance_tau_quantile=0.75,
        )
    if evt_cfg is None:
        evt_cfg = EVTConfig(
            metric_col="cum_deficit",
            threshold_quantile=0.90,
            severe_prob=0.01,
            moderate_prob=0.05,
            mild_prob=0.10,
            min_exceedances=8,
        )

    rng = np.random.default_rng(batch_cfg.random_seed)
    all_df: List[pd.DataFrame] = []
    all_samples: List[pd.DataFrame] = []

    for scenario_id in range(batch_cfg.n_scenarios):
        scenario_start = _make_unique_scenario_time(
            scenario_id=scenario_id,
            base_start=batch_cfg.base_start,
            days_per_scenario=batch_cfg.days_per_scenario,
            gap_days=batch_cfg.gap_days_between_scenarios,
        )
        seed = int(rng.integers(0, 1_000_000_000))
        df_one = make_random_mock_data(
            seed=seed,
            start_time=scenario_start,
            days=batch_cfg.days_per_scenario,
        )
        df_one["scenario_id"] = scenario_id
        df_one["scenario_seed"] = seed

        samples_one = detect_extreme_samples(
            df=df_one,
            cfg=detect_cfg,
            add_buffer_hours=batch_cfg.add_buffer_hours,
            merge_overlap=batch_cfg.merge_overlap,
        )

        if not samples_one.empty:
            samples_one["scenario_id"] = scenario_id
            samples_one["scenario_seed"] = seed
            all_samples.append(samples_one)

        all_df.append(df_one)

    df_all = pd.concat(all_df, ignore_index=True).sort_values("time").reset_index(drop=True)

    if all_samples:
        samples_all = pd.concat(all_samples, ignore_index=True)
        samples_all = samples_all.reset_index(drop=True)
        samples_all["sample_id"] = [f"S{i:05d}" for i in range(1, len(samples_all) + 1)]
    else:
        samples_all = pd.DataFrame()

    samples_metrics = compute_metrics_for_samples(df=df_all, samples=samples_all, cfg=metric_cfg)
    samples_labeled, evt_info = fit_evt_and_label(samples=samples_metrics, cfg=evt_cfg)

    # ===== 固定长度裁剪，构造成模型输入 =====
    half_window = batch_cfg.fixed_window_hours // 2
    cols = [metric_cfg.load_col, metric_cfg.wind_power_col, metric_cfg.solar_power_col]

    x_list: List[np.ndarray] = []
    cond_rows: List[Dict] = []
    meta_rows: List[Dict] = []

    for _, row in samples_labeled.iterrows():
        core_start = pd.to_datetime(row["core_start_time"])
        core_end = pd.to_datetime(row["core_end_time"])
        center = core_start + (core_end - core_start) / 2
        center = pd.Timestamp(center).round("1h")

        window_start = center - pd.Timedelta(hours=half_window)
        window_end = window_start + pd.Timedelta(hours=batch_cfg.fixed_window_hours - 1)

        fixed_sub = _extract_fixed_window(df_all, window_start, window_end, cols)
        seq = fixed_sub[cols].to_numpy(dtype=float).T  # [3, T]
        x_list.append(seq)

        event_type = str(row["event_type"])
        cond_rows.append(
            {
                "sample_id": row["sample_id"],
                "scenario_id": int(row["scenario_id"]),
                "event_type": event_type,
                "event_type_code": EVENT_TYPE_TO_CODE.get(event_type, -1),
                "low_wind_flag": int(row.get("low_wind_flag", 0)),
                "low_irradiance_flag": int(row.get("low_irradiance_flag", 0)),
                "severity_level": int(row.get("severity_level", 0)) if pd.notna(row.get("severity_level", 0)) else -1,
                "duration_hours": int(row.get("duration_hours", 0)),
                "month": int(row.get("month", 0)),
                "extreme_prob": float(row.get("extreme_prob", np.nan)),
                "cum_deficit": float(row.get("cum_deficit", np.nan)),
                "netload_ramp_max": float(row.get("netload_ramp_max", np.nan)),
                "imbalance_duration": float(row.get("imbalance_duration", np.nan)),
            }
        )

        meta_rows.append(
            {
                "sample_id": row["sample_id"],
                "scenario_id": int(row["scenario_id"]),
                "core_start_time": core_start,
                "core_end_time": core_end,
                "window_start_time": window_start,
                "window_end_time": window_end,
                "original_start_time": pd.to_datetime(row["start_time"]),
                "original_end_time": pd.to_datetime(row["end_time"]),
            }
        )

    X = np.stack(x_list, axis=0) if x_list else np.empty((0, 3, batch_cfg.fixed_window_hours), dtype=float)
    cond_df = pd.DataFrame(cond_rows)
    meta_df = pd.DataFrame(meta_rows)

    # ===== 保存输出 =====
    out_dir = Path(batch_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(out_dir / "mock_timeseries_all.csv", index=False, encoding="utf-8-sig")
    samples_all.to_csv(out_dir / "mock_samples_raw.csv", index=False, encoding="utf-8-sig")
    samples_metrics.to_csv(out_dir / "mock_samples_with_metrics.csv", index=False, encoding="utf-8-sig")
    samples_labeled.to_csv(out_dir / "mock_samples_evt_labeled.csv", index=False, encoding="utf-8-sig")
    cond_df.to_csv(out_dir / "cond.csv", index=False, encoding="utf-8-sig")
    meta_df.to_csv(out_dir / "meta.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "X.npy", X)

    with open(out_dir / "event_type_mapping.json", "w", encoding="utf-8") as f:
        json.dump(EVENT_TYPE_TO_CODE, f, ensure_ascii=False, indent=2)

    summary = {
        "batch_cfg": asdict(batch_cfg),
        "evt_info": evt_info,
        "n_total_timesteps": int(len(df_all)),
        "n_samples": int(len(samples_labeled)),
        "x_shape": list(X.shape),
        "event_type_counts": cond_df["event_type"].value_counts().to_dict() if not cond_df.empty else {},
        "severity_level_counts": cond_df["severity_level"].value_counts().sort_index().to_dict() if not cond_df.empty else {},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return samples_labeled, cond_df, X, summary


def main() -> None:
    batch_cfg = BatchMockConfig(
        n_scenarios=60,
        days_per_scenario=30,
        base_start="2024-01-01 00:00:00",
        gap_days_between_scenarios=3,
        random_seed=42,
        add_buffer_hours=2,
        fixed_window_hours=24,
        merge_overlap=False,
        output_dir="mock_dataset_outputs",
    )

    samples_labeled, cond_df, X, summary = build_mock_dataset(batch_cfg=batch_cfg)

    print("=== mock 数据集构建完成 ===")
    print(f"总样本数: {summary['n_samples']}")
    print(f"X.shape: {tuple(summary['x_shape'])}")
    print("\n事件类型分布:")
    print(pd.Series(summary["event_type_counts"]))
    print("\n严重等级分布:")
    print(pd.Series(summary["severity_level_counts"]))
    print("\ncond.csv 预览:")
    print(cond_df.head())
    print("\n输出目录:", Path(batch_cfg.output_dir).resolve())


if __name__ == "__main__":
    main()
