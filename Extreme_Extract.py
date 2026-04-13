from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class DetectConfig:
    # 列名映射
    time_col: str = "time"
    temp_col: str = "temp"
    wind_speed_col: str = "wind_speed"
    irradiance_col: str = "irradiance"
    snowfall_col: str = "snowfall"
    visibility_col: str = "visibility"

    # 数据时间分辨率（小时）
    freq_hours: int = 1

    # 主事件最短持续时间（小时）
    min_event_hours: int = 6

    # 寒潮：低温 + 24h降温
    cold_temp_threshold: float = -10.0
    cold_drop_24h_threshold: float = 8.0

    # 高温
    heat_temp_threshold: float = 35.0

    # 大风 / 沙尘暴
    high_wind_speed_threshold: float = 12.0
    low_visibility_threshold: float = 3000.0  # m

    # 暴雪 / 风吹雪
    snowfall_threshold: float = 1.0
    blowing_snow_wind_threshold: float = 8.0

    # 持续低辐照 / 持续低风
    low_irr_quantile: float = 0.2
    low_wind_quantile: float = 0.2
    low_resource_min_hours: int = 6
    daylight_irradiance_min: float = 50.0


def _prepare_dataframe(df: pd.DataFrame, cfg: DetectConfig) -> pd.DataFrame:
    out = df.copy()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col])
    out = out.sort_values(cfg.time_col).reset_index(drop=True)
    return out


def _find_true_segments(flag: pd.Series, time: pd.Series, min_len: int) -> List[Dict]:
    """
    把连续 True 区间提取成事件窗口
    """
    flag = flag.fillna(False).astype(bool).reset_index(drop=True)
    time = pd.to_datetime(time).reset_index(drop=True)

    segs = []
    start_idx = None

    for i, val in enumerate(flag):
        if val and start_idx is None:
            start_idx = i

        if (not val or i == len(flag) - 1) and start_idx is not None:
            end_idx = i if val and i == len(flag) - 1 else i - 1
            n_steps = end_idx - start_idx + 1

            if n_steps >= min_len:
                segs.append({
                    "start_time": time.iloc[start_idx],
                    "end_time": time.iloc[end_idx],
                    "n_steps": int(n_steps),
                })
            start_idx = None

    return segs


def _add_buffer(start_time: pd.Timestamp, end_time: pd.Timestamp, buffer_hours: int):
    return (
        start_time - pd.Timedelta(hours=buffer_hours),
        end_time + pd.Timedelta(hours=buffer_hours),
    )


def _merge_overlapping_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """
    如果不同规则筛出来的窗口有重叠，可合并
    """
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


def _build_low_irr_flag(df: pd.DataFrame, cfg: DetectConfig) -> pd.Series:
    if cfg.irradiance_col not in df.columns:
        return pd.Series(False, index=df.index)

    daylight_mask = df[cfg.irradiance_col] > cfg.daylight_irradiance_min
    if daylight_mask.sum() == 0:
        return pd.Series(False, index=df.index)

    irr_q = df.loc[daylight_mask, cfg.irradiance_col].quantile(cfg.low_irr_quantile)
    low_irr_flag = daylight_mask & (df[cfg.irradiance_col] <= irr_q)
    return low_irr_flag.reindex(df.index, fill_value=False)


def _build_low_wind_flag(df: pd.DataFrame, cfg: DetectConfig) -> pd.Series:
    if cfg.wind_speed_col not in df.columns:
        return pd.Series(False, index=df.index)

    wind_q = df[cfg.wind_speed_col].quantile(cfg.low_wind_quantile)
    low_wind_flag = df[cfg.wind_speed_col] <= wind_q
    return low_wind_flag.reindex(df.index, fill_value=False)


def detect_extreme_samples(
    df: pd.DataFrame,
    cfg: Optional[DetectConfig] = None,
    add_buffer_hours: int = 0,
    merge_overlap: bool = True,
) -> pd.DataFrame:
    """
    输入：
        df: 原始时序数据，至少包含 time，以及若干气象列
        cfg: 阈值配置
        add_buffer_hours: 是否在事件前后补缓冲段
        merge_overlap: 是否合并重叠窗口

    输出：
        极端样本窗口表
    """
    if cfg is None:
        cfg = DetectConfig()

    df = _prepare_dataframe(df, cfg)

    min_len = max(1, int(cfg.min_event_hours / cfg.freq_hours))
    low_res_len = max(1, int(cfg.low_resource_min_hours / cfg.freq_hours))

    # =========================
    # 1. 主事件布尔标记
    # =========================
    if cfg.temp_col in df.columns:
        lag_steps_24h = int(24 / cfg.freq_hours)
        df["temp_drop_24h"] = df[cfg.temp_col].shift(lag_steps_24h) - df[cfg.temp_col]
    else:
        df["temp_drop_24h"] = np.nan

    flags = {}

    # 寒潮
    if cfg.temp_col in df.columns:
        flags["寒潮"] = (
            (df[cfg.temp_col] <= cfg.cold_temp_threshold) &
            (df["temp_drop_24h"] >= cfg.cold_drop_24h_threshold)
        )
    else:
        flags["寒潮"] = pd.Series(False, index=df.index)

    # 高温
    if cfg.temp_col in df.columns:
        flags["高温"] = (df[cfg.temp_col] >= cfg.heat_temp_threshold)
    else:
        flags["高温"] = pd.Series(False, index=df.index)

    # 大风 / 沙尘暴
    if cfg.wind_speed_col in df.columns and cfg.visibility_col in df.columns:
        flags["大风/沙尘暴"] = (
            (df[cfg.wind_speed_col] >= cfg.high_wind_speed_threshold) |
            (
                (df[cfg.wind_speed_col] >= cfg.blowing_snow_wind_threshold) &
                (df[cfg.visibility_col] <= cfg.low_visibility_threshold)
            )
        )
    elif cfg.wind_speed_col in df.columns:
        flags["大风/沙尘暴"] = (df[cfg.wind_speed_col] >= cfg.high_wind_speed_threshold)
    else:
        flags["大风/沙尘暴"] = pd.Series(False, index=df.index)

    # 暴雪 / 风吹雪
    if cfg.snowfall_col in df.columns and cfg.wind_speed_col in df.columns:
        flags["暴雪/风吹雪"] = (
            (df[cfg.snowfall_col] >= cfg.snowfall_threshold) &
            (df[cfg.wind_speed_col] >= cfg.blowing_snow_wind_threshold)
        )
    else:
        flags["暴雪/风吹雪"] = pd.Series(False, index=df.index)

    # =========================
    # 2. 资源状态布尔标记
    # =========================
    low_irr_flag = _build_low_irr_flag(df, cfg)
    low_wind_flag = _build_low_wind_flag(df, cfg)

    # =========================
    # 3. 提取事件窗口
    # =========================
    windows = []

    for event_name, flag in flags.items():
        segs = _find_true_segments(flag, df[cfg.time_col], min_len)

        for seg in segs:
            core_start = seg["start_time"]
            core_end = seg["end_time"]

            start_time, end_time = core_start, core_end
            if add_buffer_hours > 0:
                start_time, end_time = _add_buffer(core_start, core_end, add_buffer_hours)

            # 资源状态标签尽量基于核心事件段判断，减少 buffer 段和夜间误判
            core_sub = df[(df[cfg.time_col] >= core_start) & (df[cfg.time_col] <= core_end)]

            low_irr_segs = _find_true_segments(
                low_irr_flag.loc[core_sub.index],
                core_sub[cfg.time_col],
                low_res_len,
            )
            low_wind_segs = _find_true_segments(
                low_wind_flag.loc[core_sub.index],
                core_sub[cfg.time_col],
                low_res_len,
            )

            windows.append({
                "event_type": event_name,
                "core_start_time": core_start,
                "core_end_time": core_end,
                "start_time": start_time,
                "end_time": end_time,
                "core_n_steps": seg["n_steps"],
                "duration_hours": int((end_time - start_time) / pd.Timedelta(hours=1)) + 1,
                "month": int(pd.Timestamp(core_start).month),
                "low_irradiance_flag": int(len(low_irr_segs) > 0),
                "low_wind_flag": int(len(low_wind_segs) > 0),
            })

    out = pd.DataFrame(windows)

    if out.empty:
        return out

    if merge_overlap:
        out = _merge_overlapping_windows(out)

    out = out.reset_index(drop=True)
    out.insert(0, "sample_id", [f"S{i:04d}" for i in range(1, len(out) + 1)])
    return out


# =========================
# 下面是一个可测试的小例子
# =========================
def make_mock_data() -> pd.DataFrame:
    time = pd.date_range("2024-01-01 00:00:00", periods=24 * 15, freq="1h")
    n = len(time)
    rng = np.random.default_rng(0)

    hour = time.hour.to_numpy()
    day = np.arange(n)

    # -------------------------
    # 1. 气象变量
    # -------------------------
    temp = 5 + rng.normal(0, 2, n)
    wind_speed = 6 + rng.normal(0, 1.0, n)
    irradiance = np.maximum(0, 500 * np.sin((hour - 6) / 12 * np.pi))
    snowfall = np.zeros(n)
    visibility = np.full(n, 10000.0)

    # -------------------------
    # 2. 风光荷基础时序
    # -------------------------
    load = 600 + 60 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 10, n)
    wind_power = 150 + 20 * np.sin(day / 24 * 2 * np.pi / 3) + rng.normal(0, 15, n)
    wind_power = np.clip(wind_power, 0, None)

    solar_power = np.maximum(0, 220 * np.sin((hour - 6) / 12 * np.pi))
    solar_power = solar_power + rng.normal(0, 8, n)
    solar_power = np.clip(solar_power, 0, None)

    df = pd.DataFrame({
        "time": time,
        "temp": temp,
        "wind_speed": wind_speed,
        "irradiance": irradiance,
        "snowfall": snowfall,
        "visibility": visibility,
        "load": load,
        "wind_power": wind_power,
        "solar_power": solar_power,
    })

    # =========================
    # 注入极端事件
    # =========================

    # 1) 寒潮：低温、负荷升高、光伏下降
    idx = (df["time"] >= "2024-01-05 00:00:00") & (df["time"] <= "2024-01-05 12:00:00")
    df.loc[idx, "temp"] = -12
    df.loc[idx, "load"] += 80
    df.loc[idx, "solar_power"] *= 0.4

    # 为了满足 24h 降温
    idx_prev = (df["time"] >= "2024-01-04 00:00:00") & (df["time"] <= "2024-01-04 12:00:00")
    df.loc[idx_prev, "temp"] = 2

    # 2) 高温：高温、负荷升高、风电略降
    idx = (df["time"] >= "2024-01-08 12:00:00") & (df["time"] <= "2024-01-08 20:00:00")
    df.loc[idx, "temp"] = 37
    df.loc[idx, "load"] += 90
    df.loc[idx, "wind_speed"] -= 2
    df.loc[idx, "wind_power"] *= 0.75

    # 3) 大风/沙尘暴：风速高、能见度低、辐照下降、光伏骤降
    idx = (df["time"] >= "2024-01-10 06:00:00") & (df["time"] <= "2024-01-10 15:00:00")
    df.loc[idx, "wind_speed"] = 14
    df.loc[idx, "visibility"] = 2000
    df.loc[idx, "irradiance"] *= 0.3
    df.loc[idx, "solar_power"] *= 0.2

    # 4) 暴雪/风吹雪：降雪 + 风大 + 光伏下降 + 负荷升高
    idx = (df["time"] >= "2024-01-12 03:00:00") & (df["time"] <= "2024-01-12 12:00:00")
    df.loc[idx, "snowfall"] = 2.0
    df.loc[idx, "wind_speed"] = 9
    df.loc[idx, "solar_power"] *= 0.1
    df.loc[idx, "load"] += 70

    # 5) 持续低风：只做补充资源状态
    idx = (df["time"] >= "2024-01-14 00:00:00") & (df["time"] <= "2024-01-14 10:00:00")
    df.loc[idx, "wind_speed"] = 1.2
    df.loc[idx, "wind_power"] *= 0.2

    # 6) 持续低辐照：只做补充资源状态（白天时段）
    idx = (df["time"] >= "2024-01-11 08:00:00") & (df["time"] <= "2024-01-11 16:00:00")
    df.loc[idx, "irradiance"] = 20
    df.loc[idx, "solar_power"] *= 0.15

    return df


if __name__ == "__main__":
    cfg = DetectConfig()
    df = make_mock_data()

    samples = detect_extreme_samples(
        df=df,
        cfg=cfg,
        add_buffer_hours=2,   # 事件前后各扩2小时
        merge_overlap=False,  # 先不合并，便于检查
    )

    print(samples)
    samples.to_csv("extreme_samples.csv", index=False)
