from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from Extreme_Extract import DetectConfig, detect_extreme_samples
from evt_fit import EVTConfig, fit_evt_and_label
from sample_metrics import MetricConfig, compute_metrics_for_samples


DEFAULT_EVENT_TYPE_MAPPING = {
    "寒潮": 0,
    "暴雪/风吹雪": 1,
    "大风/沙尘暴": 2,
    "高温": 3,
}

EVENT_TYPE_ALIASES = {
    "寒潮": "寒潮",
    "暴雪/风吹雪": "暴雪/风吹雪",
    "大风/沙尘暴": "大风/沙尘暴",
    "高温": "高温",
}


@dataclass
class ColumnMapping:
    time_col: str = "time"
    load_col: str = "load"
    wind_col: str = "wind_power"
    solar_col: str = "solar_power"
    temp_col: str = "temp"
    wind_speed_col: str = "wind_speed"
    irradiance_col: str = "irradiance"
    snowfall_col: str = "snowfall"
    visibility_col: str = "visibility"


@dataclass
class DatasetBuildConfig:
    seq_len: int = 24
    add_buffer_hours: int = 2
    merge_overlap: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    split_group_col: str = "sample_id"
    output_dir: str = "dataset"


def infer_season(month: int) -> str:
    month = int(month)
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def season_to_code(season: str) -> int:
    mapping = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
    return mapping[str(season).lower()]


def canonicalize_event_type(value: str) -> str:
    text = str(value).strip()
    return EVENT_TYPE_ALIASES.get(text, text)


def _extract_fixed_window(
    df: pd.DataFrame,
    time_col: str,
    start_time: pd.Timestamp,
    seq_len: int,
    value_cols: list[str],
) -> pd.DataFrame:
    time_index = pd.date_range(start_time, periods=seq_len, freq="1h")
    sub = df.set_index(time_col)[value_cols].reindex(time_index)
    sub = sub.ffill().bfill().fillna(0.0)
    sub.index.name = time_col
    return sub.reset_index()


def _build_detect_config(mapping: ColumnMapping) -> DetectConfig:
    return DetectConfig(
        time_col=mapping.time_col,
        temp_col=mapping.temp_col,
        wind_speed_col=mapping.wind_speed_col,
        irradiance_col=mapping.irradiance_col,
        snowfall_col=mapping.snowfall_col,
        visibility_col=mapping.visibility_col,
    )


def _build_metric_config(mapping: ColumnMapping) -> MetricConfig:
    return MetricConfig(
        time_col=mapping.time_col,
        load_col=mapping.load_col,
        wind_power_col=mapping.wind_col,
        solar_power_col=mapping.solar_col,
    )


def build_event_samples(
    df: pd.DataFrame,
    column_mapping: Optional[ColumnMapping] = None,
    detect_cfg: Optional[DetectConfig] = None,
    metric_cfg: Optional[MetricConfig] = None,
    evt_cfg: Optional[EVTConfig] = None,
    add_buffer_hours: int = 2,
    merge_overlap: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = column_mapping or ColumnMapping()
    detect_cfg = detect_cfg or _build_detect_config(mapping)
    metric_cfg = metric_cfg or _build_metric_config(mapping)
    evt_cfg = evt_cfg or EVTConfig(metric_col="cum_deficit")

    samples = detect_extreme_samples(
        df=df,
        cfg=detect_cfg,
        add_buffer_hours=add_buffer_hours,
        merge_overlap=merge_overlap,
    )
    samples = compute_metrics_for_samples(df=df, samples=samples, cfg=metric_cfg)
    labeled, evt_info = fit_evt_and_label(samples=samples, cfg=evt_cfg)
    return labeled, evt_info


def build_dataset_artifacts(
    df: pd.DataFrame,
    labeled_samples: pd.DataFrame,
    output_dir: str | Path,
    build_cfg: Optional[DatasetBuildConfig] = None,
    column_mapping: Optional[ColumnMapping] = None,
    event_type_mapping: Optional[dict[str, int]] = None,
    evt_info: Optional[dict[str, Any]] = None,
    extra_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    mapping = column_mapping or ColumnMapping()
    build_cfg = build_cfg or DatasetBuildConfig(output_dir=str(output_dir))
    event_type_mapping = event_type_mapping or DEFAULT_EVENT_TYPE_MAPPING.copy()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_local = df.copy()
    df_local[mapping.time_col] = pd.to_datetime(df_local[mapping.time_col])
    df_local = df_local.sort_values(mapping.time_col).reset_index(drop=True)

    if labeled_samples.empty:
        raise ValueError("No extreme-event samples were found for dataset export.")

    samples = labeled_samples.copy().reset_index(drop=True)
    if "sample_id" not in samples.columns:
        samples.insert(0, "sample_id", [f"S{i:05d}" for i in range(1, len(samples) + 1)])

    channels = [mapping.load_col, mapping.wind_col, mapping.solar_col]
    x_list: list[np.ndarray] = []
    cond_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for _, row in samples.iterrows():
        core_start = pd.to_datetime(row["core_start_time"])
        core_end = pd.to_datetime(row["core_end_time"])
        center = core_start + (core_end - core_start) / 2
        center = pd.Timestamp(center).round("1h")
        half_window = build_cfg.seq_len // 2
        window_start = center - pd.Timedelta(hours=half_window)
        window_end = window_start + pd.Timedelta(hours=build_cfg.seq_len - 1)

        fixed_sub = _extract_fixed_window(
            df_local,
            time_col=mapping.time_col,
            start_time=window_start,
            seq_len=build_cfg.seq_len,
            value_cols=channels,
        )
        seq = fixed_sub[channels].to_numpy(dtype=np.float32).T
        x_list.append(seq)

        event_type = canonicalize_event_type(row["event_type"])
        month = int(pd.Timestamp(core_start).month)
        season = infer_season(month)
        tail_score = float(row.get("tail_score", -np.log(float(row.get("extreme_prob", 1.0)) + 1e-8)))
        tail_score_z = float(row.get("tail_score_zscore", row.get("tail_score_z", tail_score)))
        cond_rows.append(
            {
                "sample_id": row["sample_id"],
                "scenario_id": row.get("scenario_id", np.nan),
                "scenario_seed": row.get("scenario_seed", np.nan),
                "event_type": event_type,
                "event_type_code": int(event_type_mapping.get(event_type, -1)),
                "month": month,
                "season": season,
                "season_code": season_to_code(season),
                "low_wind_flag": int(row.get("low_wind_flag", 0)),
                "low_irradiance_flag": int(row.get("low_irradiance_flag", 0)),
                "duration_hours": float(row.get("duration_hours", 0.0)),
                "extreme_prob": float(row.get("extreme_prob", np.nan)),
                "tail_score": tail_score,
                "tail_score_zscore": tail_score_z,
                "severity_level": int(row.get("severity_level", 0)),
                "cum_deficit": float(row.get("cum_deficit", np.nan)),
                "netload_ramp_max": float(row.get("netload_ramp_max", np.nan)),
                "imbalance_duration": float(row.get("imbalance_duration", np.nan)),
                "imbalance_tau": float(row.get("imbalance_tau", 0.0)),
            }
        )
        meta_rows.append(
            {
                "sample_id": row["sample_id"],
                "scenario_id": row.get("scenario_id", np.nan),
                "core_start_time": core_start,
                "core_end_time": core_end,
                "window_start_time": window_start,
                "window_end_time": window_end,
                "original_start_time": pd.to_datetime(row["start_time"]),
                "original_end_time": pd.to_datetime(row["end_time"]),
            }
        )

    X = np.stack(x_list, axis=0).astype(np.float32)
    cond_df = pd.DataFrame(cond_rows)
    meta_df = pd.DataFrame(meta_rows)
    split_df = make_dataset_split(cond_df, build_cfg=build_cfg)

    np.save(out_dir / "X.npy", X)
    cond_df.to_csv(out_dir / "cond.csv", index=False, encoding="utf-8-sig")
    meta_df.to_csv(out_dir / "meta.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(out_dir / "split_assignments.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "event_type_mapping.json", "w", encoding="utf-8") as f:
        json.dump(event_type_mapping, f, ensure_ascii=False, indent=2)

    split_to_idx = {
        split: split_df.index[split_df["split"] == split].to_numpy(dtype=int)
        for split in ["train", "val", "test"]
    }
    for split, idx in split_to_idx.items():
        np.save(out_dir / f"X_{split}.npy", X[idx])
        cond_df.iloc[idx].reset_index(drop=True).to_csv(out_dir / f"cond_{split}.csv", index=False, encoding="utf-8-sig")
        meta_df.iloc[idx].reset_index(drop=True).to_csv(out_dir / f"meta_{split}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "build_cfg": asdict(build_cfg),
        "column_mapping": asdict(mapping),
        "event_type_mapping": event_type_mapping,
        "evt_info": evt_info or {},
        "x_shape": list(X.shape),
        "n_samples": int(len(cond_df)),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "event_type_counts": cond_df["event_type"].value_counts().to_dict(),
        "severity_level_counts": cond_df["severity_level"].value_counts().sort_index().to_dict(),
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return {
        "X": X,
        "cond_df": cond_df,
        "meta_df": meta_df,
        "split_df": split_df,
        "summary": summary,
        "output_dir": out_dir,
    }


def make_dataset_split(cond_df: pd.DataFrame, build_cfg: DatasetBuildConfig) -> pd.DataFrame:
    if cond_df.empty:
        raise ValueError("cond_df is empty.")
    if not np.isclose(build_cfg.train_ratio + build_cfg.val_ratio + build_cfg.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    group_col = build_cfg.split_group_col if build_cfg.split_group_col in cond_df.columns else "sample_id"
    groups = cond_df[group_col].astype(str).drop_duplicates().to_list()
    rng = np.random.default_rng(build_cfg.split_seed)
    groups = list(rng.permutation(groups))

    n_groups = len(groups)
    if n_groups == 1:
        train_groups, val_groups, test_groups = set(groups), set(), set()
    elif n_groups == 2:
        train_groups, val_groups, test_groups = {groups[0]}, set(), {groups[1]}
    else:
        n_train = max(1, int(round(n_groups * build_cfg.train_ratio)))
        n_val = max(1, int(round(n_groups * build_cfg.val_ratio)))
        if n_train + n_val >= n_groups:
            n_val = max(1, n_groups - n_train - 1)
        n_test = max(1, n_groups - n_train - n_val)
        if n_train + n_val + n_test > n_groups:
            n_train = max(1, n_groups - n_val - n_test)

        train_groups = set(groups[:n_train])
        val_groups = set(groups[n_train : n_train + n_val])
        test_groups = set(groups[n_train + n_val :])
        if not test_groups and val_groups:
            moved = next(iter(val_groups))
            val_groups.remove(moved)
            test_groups.add(moved)

    split_df = cond_df[["sample_id"]].copy()
    split_df[group_col] = cond_df[group_col].astype(str)
    split_df["split"] = "test" if test_groups else "train"
    split_df.loc[split_df[group_col].isin(train_groups), "split"] = "train"
    if val_groups:
        split_df.loc[split_df[group_col].isin(val_groups), "split"] = "val"
    return split_df


def load_split_bundle(data_dir: str | Path, split: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    X = np.load(root / f"X_{split}.npy").astype(np.float32)
    cond_df = pd.read_csv(root / f"cond_{split}.csv")
    meta_df = pd.read_csv(root / f"meta_{split}.csv")
    return X, cond_df, meta_df
