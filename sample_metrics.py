from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricConfig:
    # 时间列
    time_col: str = "time"

    # 电力相关列
    load_col: str = "load"
    wind_power_col: str = "wind_power"
    solar_power_col: str = "solar_power"

    # 失衡阈值设置
    # fixed: 使用 imbalance_tau_fixed
    # quantile: 使用全样本净负荷分位数作为阈值
    imbalance_tau_mode: str = "quantile"
    imbalance_tau_fixed: float = 0.0
    imbalance_tau_quantile: float = 0.75


def _prepare_df(df: pd.DataFrame, cfg: MetricConfig) -> pd.DataFrame:
    """
    预处理原始时序数据：
    1. 时间列转 datetime
    2. 按时间排序
    3. 计算净负荷和净负荷一阶差分
    """
    out = df.copy()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col])
    out = out.sort_values(cfg.time_col).reset_index(drop=True)

    # 计算净负荷
    out["net_load"] = (
        out[cfg.load_col] - out[cfg.wind_power_col] - out[cfg.solar_power_col]
    )

    # 一阶差分（净负荷爬坡）
    out["net_load_diff"] = out["net_load"].diff().fillna(0.0)

    return out


def _resolve_imbalance_tau(df: pd.DataFrame, cfg: MetricConfig) -> float:
    """
    统一计算失衡阈值 tau。
    为了让不同样本可比较，这里默认使用全时序净负荷分位数。
    """
    mode = cfg.imbalance_tau_mode.lower().strip()

    if mode == "fixed":
        return float(cfg.imbalance_tau_fixed)

    if mode == "quantile":
        q = float(cfg.imbalance_tau_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("imbalance_tau_quantile 必须在 (0, 1) 之间")
        return float(df["net_load"].quantile(q))

    raise ValueError("imbalance_tau_mode 仅支持 'fixed' 或 'quantile'")


def compute_metrics_for_samples(
    df: pd.DataFrame,
    samples: pd.DataFrame,
    cfg: Optional[MetricConfig] = None,
) -> pd.DataFrame:
    """
    输入：
        df: 原始时序数据，必须包含:
            - time
            - load
            - wind_power
            - solar_power

        samples: 极端样本窗口表，至少包含:
            - sample_id
            - event_type
            - start_time
            - end_time

    输出：
        在样本表基础上新增以下字段：
            - cum_deficit: 超阈值净负荷累计量
            - netload_ramp_max
            - imbalance_duration: 净负荷超阈值持续时长
            - netload_peak
            - netload_mean
            - imbalance_tau
    """
    if cfg is None:
        cfg = MetricConfig()

    if samples.empty:
        return samples.copy()

    required_sample_cols = {"sample_id", "event_type", "start_time", "end_time"}
    missing_sample_cols = required_sample_cols - set(samples.columns)
    if missing_sample_cols:
        raise ValueError(f"samples 缺少必要字段: {missing_sample_cols}")

    required_df_cols = {
        cfg.time_col,
        cfg.load_col,
        cfg.wind_power_col,
        cfg.solar_power_col,
    }
    missing_df_cols = required_df_cols - set(df.columns)
    if missing_df_cols:
        raise ValueError(f"df 缺少必要字段: {missing_df_cols}")

    df = _prepare_df(df, cfg)
    imbalance_tau = _resolve_imbalance_tau(df, cfg)
    out_rows = []

    for _, row in samples.iterrows():
        start_time = pd.to_datetime(row["start_time"])
        end_time = pd.to_datetime(row["end_time"])

        sub = df[
            (df[cfg.time_col] >= start_time) &
            (df[cfg.time_col] <= end_time)
        ].copy()

        rec = row.to_dict()
        rec["imbalance_tau"] = imbalance_tau

        # 如果这个窗口没有截到数据，返回空值
        if sub.empty:
            rec["cum_deficit"] = np.nan
            rec["netload_ramp_max"] = np.nan
            rec["imbalance_duration"] = np.nan
            rec["netload_peak"] = np.nan
            rec["netload_mean"] = np.nan
            out_rows.append(rec)
            continue

        # 1) 超阈值净负荷累计量（更适合作为“联合失衡强度”主指标）
        excess = np.maximum(0.0, sub["net_load"].values - imbalance_tau)
        cum_deficit = float(excess.sum())

        # 2) 净负荷最大爬坡强度
        netload_ramp_max = float(sub["net_load_diff"].max())

        # 3) 净负荷超阈值持续时长
        imbalance_duration = int((sub["net_load"] > imbalance_tau).sum())

        # 4) 附带统计量
        netload_peak = float(sub["net_load"].max())
        netload_mean = float(sub["net_load"].mean())

        rec["cum_deficit"] = cum_deficit
        rec["netload_ramp_max"] = netload_ramp_max
        rec["imbalance_duration"] = imbalance_duration
        rec["netload_peak"] = netload_peak
        rec["netload_mean"] = netload_mean

        out_rows.append(rec)

    out = pd.DataFrame(out_rows)
    return out


if __name__ == "__main__":
    # =========================
    # 最小可运行测试
    # =========================
    rng = np.random.default_rng(0)
    time = pd.date_range("2024-01-01 00:00:00", periods=48, freq="1h")

    df_demo = pd.DataFrame({
        "time": time,
        "load": 600 + rng.normal(0, 10, 48),
        "wind_power": 150 + rng.normal(0, 15, 48),
        "solar_power": np.maximum(
            0,
            200 * np.sin((time.hour.to_numpy() - 6) / 12 * np.pi)
        ),
    })

    # 人工制造一段更极端的时间窗
    idx = (
        (df_demo["time"] >= "2024-01-01 10:00:00") &
        (df_demo["time"] <= "2024-01-01 20:00:00")
    )
    df_demo.loc[idx, "load"] += 80
    df_demo.loc[idx, "wind_power"] *= 0.5
    df_demo.loc[idx, "solar_power"] *= 0.3

    samples_demo = pd.DataFrame({
        "sample_id": ["S0001"],
        "event_type": ["寒潮"],
        "start_time": [pd.Timestamp("2024-01-01 08:00:00")],
        "end_time": [pd.Timestamp("2024-01-01 22:00:00")],
        "low_irradiance_flag": [1],
        "low_wind_flag": [1],
    })

    cfg = MetricConfig(
        time_col="time",
        load_col="load",
        wind_power_col="wind_power",
        solar_power_col="solar_power",
        imbalance_tau_mode="quantile",
        imbalance_tau_quantile=0.75,
    )

    result = compute_metrics_for_samples(df_demo, samples_demo, cfg)
    print(result)
