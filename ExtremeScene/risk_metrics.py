from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass
class RiskMetricConfig:
    time_col: str = "time"
    load_col: str = "load"
    wind_col: str = "wind_power"
    solar_col: str = "solar_power"
    tau_mode: str = "global_quantile"
    tau_fixed: float = 0.0
    tau_quantile: float = 0.75
    freq_hours: Optional[float] = None
    duration_temp: float = 12.0


def infer_freq_hours(time_index: pd.Series | pd.Index | Iterable) -> float:
    ts = pd.to_datetime(pd.Series(time_index)).sort_values().drop_duplicates()
    if len(ts) <= 1:
        return 1.0
    delta = ts.diff().dropna().dt.total_seconds().median() / 3600.0
    if not np.isfinite(delta) or delta <= 0:
        return 1.0
    return float(delta)


def compute_net_load(load: np.ndarray | pd.Series, wind: np.ndarray | pd.Series, solar: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(load, dtype=float) - np.asarray(wind, dtype=float) - np.asarray(solar, dtype=float)


def month_to_season(month: int) -> str:
    month = int(month)
    if month in {12, 1, 2}: return "winter"
    if month in {3, 4, 5}: return "spring"
    if month in {6, 7, 8}: return "summer"
    return "autumn"


def resolve_tau(net_load: np.ndarray | pd.Series, cfg: RiskMetricConfig) -> float:
    mode = cfg.tau_mode.strip().lower()
    if mode == "fixed":
        return float(cfg.tau_fixed)
    if mode in {"quantile", "global_quantile", "monthly_quantile", "seasonal_quantile"}:
        if not (0.0 < float(cfg.tau_quantile) < 1.0):
            raise ValueError("tau_quantile must be between 0 and 1.")
        values = np.asarray(net_load, dtype=float)
        if values.size == 0:
            return 0.0
        return float(np.quantile(values, float(cfg.tau_quantile)))
    raise ValueError("tau_mode must be one of {'fixed', 'quantile', 'global_quantile', 'monthly_quantile', 'seasonal_quantile'}.")


def prepare_net_load_frame(df: pd.DataFrame, cfg: RiskMetricConfig) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col])
    out = out.sort_values(cfg.time_col).reset_index(drop=True)
    delta_t_hours = float(cfg.freq_hours or infer_freq_hours(out[cfg.time_col]))
    out["net_load"] = compute_net_load(out[cfg.load_col], out[cfg.wind_col], out[cfg.solar_col])
    out["net_load_diff"] = out["net_load"].diff().fillna(0.0)
    out["month"] = out[cfg.time_col].dt.month.astype(int)
    out["season"] = out["month"].map(month_to_season)
    return out, delta_t_hours


def resolve_context_tau(prepared_df: pd.DataFrame, cfg: RiskMetricConfig, month: int | None = None, season: str | None = None) -> float:
    mode = cfg.tau_mode.strip().lower()
    if mode in {"fixed", "quantile", "global_quantile"}:
        return resolve_tau(prepared_df["net_load"], cfg)
    group = prepared_df
    if mode == "monthly_quantile" and month is not None and "month" in prepared_df.columns:
        group = prepared_df[prepared_df["month"].astype(int) == int(month)]
    elif mode == "seasonal_quantile" and season is not None and "season" in prepared_df.columns:
        group = prepared_df[prepared_df["season"].astype(str) == str(season)]
    if group.empty:
        group = prepared_df
    return resolve_tau(group["net_load"], RiskMetricConfig(**{**cfg.__dict__, "tau_mode": "global_quantile"}))


def build_tau_diagnostic(prepared_df: pd.DataFrame, cfg: RiskMetricConfig) -> pd.DataFrame:
    mode = cfg.tau_mode.strip().lower()
    rows: list[dict[str, object]] = []
    global_tau = resolve_tau(prepared_df["net_load"], RiskMetricConfig(**{**cfg.__dict__, "tau_mode": "global_quantile"}))
    rows.append({"tau_mode": mode, "group_type": "global", "group_value": "all", "count": int(len(prepared_df)), "tau": global_tau})
    if "month" in prepared_df.columns:
        for month, sub in prepared_df.groupby("month"):
            rows.append({"tau_mode": mode, "group_type": "month", "group_value": int(month), "count": int(len(sub)), "tau": float(np.quantile(sub["net_load"].to_numpy(dtype=float), float(cfg.tau_quantile)))})
    if "season" in prepared_df.columns:
        for season, sub in prepared_df.groupby("season"):
            rows.append({"tau_mode": mode, "group_type": "season", "group_value": str(season), "count": int(len(sub)), "tau": float(np.quantile(sub["net_load"].to_numpy(dtype=float), float(cfg.tau_quantile)))})
    return pd.DataFrame(rows)


def hard_risk_metrics_from_net_load(net_load: np.ndarray | pd.Series, tau: float, delta_t_hours: float = 1.0) -> dict[str, float]:
    net = np.asarray(net_load, dtype=float)
    if net.size == 0:
        return {"cum_deficit": float("nan"), "netload_ramp_max": float("nan"), "imbalance_duration": float("nan"), "netload_peak": float("nan"), "netload_mean": float("nan")}
    excess = np.maximum(0.0, net - tau)
    cum_deficit = float(excess.sum() * delta_t_hours)
    ramp = np.diff(net, prepend=net[0])
    return {"cum_deficit": cum_deficit, "netload_ramp_max": float(np.max(ramp)), "imbalance_duration": float((net > tau).sum() * delta_t_hours), "netload_peak": float(np.max(net)), "netload_mean": float(np.mean(net))}


def hard_risk_metrics_from_frame(df: pd.DataFrame, cfg: RiskMetricConfig) -> tuple[pd.DataFrame, float, float]:
    out, delta_t_hours = prepare_net_load_frame(df, cfg)
    tau = resolve_context_tau(out, cfg)
    return out, tau, delta_t_hours


def _parse_ramp_windows(multiscale_ramp_windows: str | Iterable[float] | None) -> list[float]:
    if multiscale_ramp_windows is None:
        return [1.0, 2.0, 3.0]
    if isinstance(multiscale_ramp_windows, str):
        return [float(item.strip()) for item in multiscale_ramp_windows.split(",") if item.strip()]
    return [float(item) for item in multiscale_ramp_windows]


def batch_hard_risk_metrics(
    x: np.ndarray,
    tau: np.ndarray | float,
    delta_t_hours: float = 1.0,
    ramp_metric_mode: str = "one_step",
    ramp_window_hours: float = 1.0,
    multiscale_ramp_windows: str | Iterable[float] | None = None,
) -> dict[str, np.ndarray]:
    if x.ndim != 3 or x.shape[1] != 3:
        raise ValueError("Expected x with shape [N, 3, T].")
    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.ndim == 0:
        tau_arr = np.full((x.shape[0],), float(tau_arr), dtype=float)
    net = x[:, 0, :] - x[:, 1, :] - x[:, 2, :]
    excess = np.maximum(0.0, net - tau_arr[:, None])
    cum = excess.sum(axis=1) * delta_t_hours
    mode = str(ramp_metric_mode).strip().lower()
    if mode in {"one_step", "window_1h"}:
        ramp_max = compute_windowed_netload_ramp(x, delta_t_hours=delta_t_hours, ramp_window_hours=1.0, positive_only=True)
    elif mode == "window_2h":
        ramp_max = compute_windowed_netload_ramp(x, delta_t_hours=delta_t_hours, ramp_window_hours=2.0, positive_only=True)
    elif mode == "window_3h":
        ramp_max = compute_windowed_netload_ramp(x, delta_t_hours=delta_t_hours, ramp_window_hours=3.0, positive_only=True)
    elif mode == "custom_window":
        ramp_max = compute_windowed_netload_ramp(x, delta_t_hours=delta_t_hours, ramp_window_hours=float(ramp_window_hours), positive_only=True)
    elif mode == "multiscale":
        ramps = compute_multiscale_netload_ramp(x, delta_t_hours=delta_t_hours, ramp_window_hours_list=_parse_ramp_windows(multiscale_ramp_windows), aggregation="max")
        ramp_max = ramps["netload_ramp_multiscale"]
    else:
        raise ValueError("ramp_metric_mode must be one of one_step/window_1h/window_2h/window_3h/custom_window/multiscale.")
    duration = (net > tau_arr[:, None]).sum(axis=1) * delta_t_hours
    return {"cum_deficit": cum.astype(float), "netload_ramp_max": ramp_max.astype(float), "imbalance_duration": duration.astype(float)}


def compute_windowed_netload_ramp(
    x: np.ndarray,
    delta_t_hours: float,
    ramp_window_hours: float,
    positive_only: bool = True,
) -> np.ndarray | float:
    """Compute maximum net-load ramp over a configurable time window.

    x can be [N, 3, T] or [3, T], with channels load, wind_power, solar_power.
    net_load is load - wind_power - solar_power. ramp_window_hours is the
    ramp time window. windowed_ramp_max is the maximum net-load rise within the
    selected window and represents short-to-mid-term regulation pressure.
    """

    arr = np.asarray(x, dtype=float)
    single = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        single = True
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError("Expected x with shape [N, 3, T] or [3, T].")
    if delta_t_hours <= 0:
        raise ValueError("delta_t_hours must be positive.")
    window_steps = max(1, int(round(float(ramp_window_hours) / float(delta_t_hours))))
    t_len = arr.shape[2]
    if window_steps >= t_len:
        out = np.full((arr.shape[0],), np.nan, dtype=float)
        return float(out[0]) if single else out
    net_load = arr[:, 0, :] - arr[:, 1, :] - arr[:, 2, :]
    ramp = net_load[:, window_steps:] - net_load[:, :-window_steps]
    if positive_only:
        ramp = np.maximum(ramp, 0.0)
    out = np.nanmax(ramp, axis=1).astype(float)
    return float(out[0]) if single else out


def compute_multiscale_netload_ramp(
    x: np.ndarray,
    delta_t_hours: float,
    ramp_window_hours_list: list[float] | tuple[float, ...] = (1.0, 2.0, 3.0),
    aggregation: str = "max",
) -> dict[str, np.ndarray]:
    """Compute 1h/2h/3h and aggregated multiscale net-load ramp."""

    ramps: dict[str, np.ndarray] = {}
    stack = []
    for window in ramp_window_hours_list:
        values = np.asarray(compute_windowed_netload_ramp(x, delta_t_hours, window, positive_only=True), dtype=float)
        key = f"netload_ramp_{int(window) if float(window).is_integer() else str(window).replace('.', 'p')}h"
        ramps[key] = values
        stack.append(values)
    stacked = np.vstack(stack) if stack else np.empty((0,))
    mode = aggregation.strip().lower()
    if mode == "max":
        multi = np.nanmax(stacked, axis=0)
    elif mode == "mean":
        multi = np.nanmean(stacked, axis=0)
    else:
        raise ValueError("aggregation must be 'max' or 'mean'.")
    ramps["netload_ramp_multiscale"] = multi.astype(float)
    return ramps


def compute_jirp_ramp_tail_intensity(
    x: np.ndarray,
    delta_t_hours: float = 1.0,
    ramp_windows: list[float] | tuple[float, ...] = (1.0, 2.0, 3.0),
    topk_ratio: float = 0.10,
    definition: str = "multiscale_topk_mean",
) -> np.ndarray | float:
    """Compute JIRP-v2 burst-regulation intensity R.

    x 是风光荷序列，形状为 [N, 3, T] 或 [3, T]；通道顺序为
    load、wind_power、solar_power。net_load 为净负荷，等于负荷减去风电和
    光伏出力。jirp_ramp_tail_intensity 将 1h/2h/3h 正向净负荷爬坡合并后
    取尾部 top-k 均值，用于刻画中短时突发调节压力。
    """

    arr = np.asarray(x, dtype=float)
    single = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        single = True
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError("Expected x with shape [N, 3, T] or [3, T].")
    if delta_t_hours <= 0:
        raise ValueError("delta_t_hours must be positive.")
    net_load = arr[:, 0, :] - arr[:, 1, :] - arr[:, 2, :]
    per_window_max = []
    per_sample_values: list[list[np.ndarray]] = [[] for _ in range(arr.shape[0])]
    for window in ramp_windows:
        steps = max(1, int(round(float(window) / float(delta_t_hours))))
        if steps >= arr.shape[2]:
            continue
        ramp = np.maximum(net_load[:, steps:] - net_load[:, :-steps], 0.0)
        per_window_max.append(np.nanmax(ramp, axis=1))
        for i in range(arr.shape[0]):
            per_sample_values[i].append(ramp[i])
    mode = str(definition).strip().lower()
    if mode == "multiscale_max":
        if not per_window_max:
            out = np.zeros((arr.shape[0],), dtype=float)
        else:
            out = np.nanmax(np.vstack(per_window_max), axis=0).astype(float)
    elif mode == "multiscale_topk_mean":
        out_values = []
        for values in per_sample_values:
            merged = np.concatenate(values) if values else np.zeros((1,), dtype=float)
            if merged.size == 0:
                out_values.append(0.0)
                continue
            k = max(1, int(np.ceil(float(topk_ratio) * merged.size)))
            topk = np.partition(merged, -k)[-k:]
            out_values.append(float(np.mean(topk)))
        out = np.asarray(out_values, dtype=float)
    else:
        raise ValueError("definition must be 'multiscale_topk_mean' or 'multiscale_max'.")
    return float(out[0]) if single else out


def batch_jirp_v2_metrics(
    x: np.ndarray,
    tau: np.ndarray | float,
    delta_t_hours: float = 1.0,
    ramp_windows: list[float] | tuple[float, ...] = (1.0, 2.0, 3.0),
    topk_ratio: float = 0.10,
    ramp_definition: str = "multiscale_topk_mean",
) -> dict[str, np.ndarray]:
    """Compute JIRP-v2 C/R/D metrics for a batch.

    C 为累计失衡强度，R 为突发调节强度，D 为持续影响程度。tau 为失衡
    阈值，只使用样本自身 cond 中的 imbalance_tau，不使用测试真实曲线以外
    的额外信息。
    """

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError("Expected x with shape [N, 3, T].")
    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.ndim == 0:
        tau_arr = np.full((arr.shape[0],), float(tau_arr), dtype=float)
    net_load = arr[:, 0, :] - arr[:, 1, :] - arr[:, 2, :]
    excess = np.maximum(net_load - tau_arr[:, None], 0.0)
    c = excess.sum(axis=1) * float(delta_t_hours)
    r = compute_jirp_ramp_tail_intensity(
        arr,
        delta_t_hours=delta_t_hours,
        ramp_windows=ramp_windows,
        topk_ratio=topk_ratio,
        definition=ramp_definition,
    )
    d = (net_load > tau_arr[:, None]).sum(axis=1) * float(delta_t_hours)
    return {
        "jirp_cum_intensity": np.asarray(c, dtype=float),
        "jirp_ramp_tail_intensity": np.asarray(r, dtype=float),
        "jirp_duration": np.asarray(d, dtype=float),
    }


def batch_joint_risk_profile(
    x: np.ndarray,
    tau: np.ndarray | float,
    event_mask: np.ndarray | None = None,
    delta_t_hours: float = 1.0,
    ramp_window_hours: float = 3.0,
    resource_thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the joint risk profile G with shape [N, 4, T].

    G contains:
    - D(t): net-load exceedance depth, cumulative imbalance pressure.
    - R3h+(t): positive 3h net-load ramp, medium-short-term regulation pressure.
    - C(t): synchronized adverse resource state, high load + low wind + low solar.
    - event_mask(t): core major-weather-event period.
    """

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError("Expected x with shape [N, 3, T] or [3, T].")
    n, _, seq_len = arr.shape
    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.ndim == 0:
        tau_arr = np.full((n,), float(tau_arr), dtype=float)
    tau_arr = tau_arr.reshape(-1)
    if tau_arr.size == 1:
        tau_arr = np.full((n,), float(tau_arr[0]), dtype=float)

    net_load = arr[:, 0, :] - arr[:, 1, :] - arr[:, 2, :]
    depth = np.maximum(net_load - tau_arr[:, None], 0.0) * float(delta_t_hours)
    steps = max(1, int(round(float(ramp_window_hours) / max(float(delta_t_hours), 1e-6))))
    ramp = np.zeros((n, seq_len), dtype=float)
    if steps < seq_len:
        ramp[:, steps:] = np.maximum(net_load[:, steps:] - net_load[:, :-steps], 0.0)

    thresholds = resource_thresholds or {}
    load_high = float(thresholds.get("load_high", np.quantile(arr[:, 0, :], 0.75)))
    wind_low = float(thresholds.get("wind_low", np.quantile(arr[:, 1, :], 0.25)))
    solar_low = float(thresholds.get("solar_low", np.quantile(arr[:, 2, :], 0.25)))
    sync = (
        (arr[:, 0, :] >= load_high).astype(float)
        + (arr[:, 1, :] <= wind_low).astype(float)
        + (arr[:, 2, :] <= solar_low).astype(float)
    )

    if event_mask is None:
        mask = np.ones((n, seq_len), dtype=float)
    else:
        mask = np.asarray(event_mask, dtype=float)
        if mask.shape != (n, seq_len):
            raise ValueError(f"Expected event_mask with shape {(n, seq_len)}, got {mask.shape}.")
    return np.stack([depth, ramp, sync, mask], axis=1).astype(np.float32)


def joint_risk_profile_loss_torch(
    x_proj: torch.Tensor,
    target_profile: torch.Tensor,
    tau: torch.Tensor,
    event_mask: torch.Tensor,
    profile_norm: dict[str, torch.Tensor],
    resource_thresholds: dict[str, torch.Tensor],
    delta_t_hours: float = 1.0,
    ramp_window_hours: float = 3.0,
    indicator_temp: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable joint risk profile loss.

    The profile compares D(t), R3h+(t), and C(t). event_mask(t) is used as
    weather-process alignment information and for core-risk concentration.
    """

    if target_profile is None or target_profile.numel() == 0:
        zero = x_proj.new_tensor(0.0)
        return zero, zero, zero, zero, zero
    net_load = x_proj[:, 0, :] - x_proj[:, 1, :] - x_proj[:, 2, :]
    tau = tau.view(-1, 1).to(dtype=x_proj.dtype, device=x_proj.device)
    temp = max(float(indicator_temp), 1e-6)
    depth = F.softplus((net_load - tau) / temp) * temp * float(delta_t_hours)
    steps = max(1, int(round(float(ramp_window_hours) / max(float(delta_t_hours), 1e-6))))
    ramp = torch.zeros_like(net_load)
    if steps < net_load.shape[1]:
        ramp[:, steps:] = F.relu(net_load[:, steps:] - net_load[:, :-steps])

    load_high = resource_thresholds["load_high"].to(device=x_proj.device, dtype=x_proj.dtype)
    wind_low = resource_thresholds["wind_low"].to(device=x_proj.device, dtype=x_proj.dtype)
    solar_low = resource_thresholds["solar_low"].to(device=x_proj.device, dtype=x_proj.dtype)
    sync = (
        torch.sigmoid((x_proj[:, 0, :] - load_high) / temp)
        + torch.sigmoid((wind_low - x_proj[:, 1, :]) / temp)
        + torch.sigmoid((solar_low - x_proj[:, 2, :]) / temp)
    )

    d_norm = (depth - profile_norm["profile_d_mean"]) / profile_norm["profile_d_std"]
    r_norm = (ramp - profile_norm["profile_r_mean"]) / profile_norm["profile_r_std"]
    c_norm = (sync - profile_norm["profile_c_mean"]) / profile_norm["profile_c_std"]
    target = target_profile.to(device=x_proj.device, dtype=x_proj.dtype)
    d_loss = F.smooth_l1_loss(d_norm, target[:, 0, :])
    r_loss = F.smooth_l1_loss(r_norm, target[:, 1, :])
    c_loss = F.smooth_l1_loss(c_norm, target[:, 2, :])

    mask = event_mask.to(device=x_proj.device, dtype=x_proj.dtype)
    target_depth = target[:, 0, :] * profile_norm["profile_d_std"] + profile_norm["profile_d_mean"]
    target_depth = target_depth.clamp(min=0.0)
    p_core_pred = (depth * mask).sum(dim=1) / (depth.sum(dim=1) + 1e-6)
    p_core_true = (target_depth * mask).sum(dim=1) / (target_depth.sum(dim=1) + 1e-6)
    core_share_loss = F.smooth_l1_loss(p_core_pred, p_core_true)
    return d_loss + r_loss + c_loss, d_loss, r_loss, c_loss, core_share_loss


def _soft_windowed_ramp_torch(net: torch.Tensor, window_steps: int, duration_temp: float) -> torch.Tensor:
    if window_steps >= net.shape[1]:
        return torch.zeros((net.size(0),), device=net.device, dtype=net.dtype)
    ramp = F.relu(net[:, window_steps:] - net[:, :-window_steps])
    if ramp.shape[1] == 0:
        return torch.zeros((net.size(0),), device=net.device, dtype=net.dtype)
    return torch.logsumexp(ramp * duration_temp, dim=1) / duration_temp


def soft_jirp_ramp_tail_intensity_torch(
    x_denorm: torch.Tensor,
    delta_t_hours: float = 1.0,
    ramp_windows: str | Iterable[float] | None = None,
    topk_ratio: float = 0.10,
    definition: str = "multiscale_topk_mean",
) -> torch.Tensor:
    """Differentiable JIRP-v2 burst-regulation intensity.

    R 使用多时间窗口正向净负荷爬坡的尾部均值。torch.topk 的选择集合是
    分段可导的，足以作为 Stage 3 的轻量一致性约束。
    """

    net = x_denorm[:, 0, :] - x_denorm[:, 1, :] - x_denorm[:, 2, :]
    ramps = []
    for window in _parse_ramp_windows(ramp_windows):
        steps = max(1, int(round(float(window) / float(delta_t_hours))))
        if steps >= net.shape[1]:
            continue
        ramps.append(F.relu(net[:, steps:] - net[:, :-steps]))
    if not ramps:
        return torch.zeros((x_denorm.size(0),), device=x_denorm.device, dtype=x_denorm.dtype)
    mode = str(definition).strip().lower()
    if mode == "multiscale_max":
        values = [ramp.max(dim=1).values for ramp in ramps]
        return torch.stack(values, dim=1).max(dim=1).values
    if mode == "multiscale_topk_mean":
        merged = torch.cat(ramps, dim=1)
        k = max(1, int(np.ceil(float(topk_ratio) * merged.shape[1])))
        return torch.topk(merged, k=k, dim=1).values.mean(dim=1)
    raise ValueError("definition must be 'multiscale_topk_mean' or 'multiscale_max'.")


def soft_jirp_v2_metrics_torch(
    x_denorm: torch.Tensor,
    tau: torch.Tensor,
    delta_t_hours: float = 1.0,
    duration_temp: float = 12.0,
    ramp_windows: str | Iterable[float] | None = None,
    topk_ratio: float = 0.10,
    ramp_definition: str = "multiscale_topk_mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable JIRP-v2 C/R/D metrics for Stage 3.

    C：累计失衡强度；R：突发调节强度；D：持续影响程度。
    """

    net = x_denorm[:, 0, :] - x_denorm[:, 1, :] - x_denorm[:, 2, :]
    tau = tau.view(-1, 1)
    excess = F.softplus((net - tau) * duration_temp) / duration_temp
    c = excess.sum(dim=1) * float(delta_t_hours)
    r = soft_jirp_ramp_tail_intensity_torch(
        x_denorm,
        delta_t_hours=delta_t_hours,
        ramp_windows=ramp_windows,
        topk_ratio=topk_ratio,
        definition=ramp_definition,
    )
    d = torch.sigmoid((net - tau) * duration_temp).sum(dim=1) * float(delta_t_hours)
    return c, r, d


def soft_risk_metrics_torch(
    x_denorm: torch.Tensor,
    tau: torch.Tensor,
    delta_t_hours: float = 1.0,
    duration_temp: float = 12.0,
    ramp_metric_mode: str = "one_step",
    ramp_window_hours: float = 1.0,
    multiscale_ramp_windows: str | Iterable[float] | None = None,
) -> dict[str, torch.Tensor]:
    net = x_denorm[:, 0, :] - x_denorm[:, 1, :] - x_denorm[:, 2, :]
    tau = tau.view(-1, 1)
    excess = F.softplus((net - tau) * duration_temp) / duration_temp
    cum = excess.sum(dim=1) * delta_t_hours
    mode = str(ramp_metric_mode).strip().lower()
    if mode in {"one_step", "window_1h"}:
        ramp_max = _soft_windowed_ramp_torch(net, max(1, int(round(1.0 / float(delta_t_hours)))), duration_temp)
    elif mode == "window_2h":
        ramp_max = _soft_windowed_ramp_torch(net, max(1, int(round(2.0 / float(delta_t_hours)))), duration_temp)
    elif mode == "window_3h":
        ramp_max = _soft_windowed_ramp_torch(net, max(1, int(round(3.0 / float(delta_t_hours)))), duration_temp)
    elif mode == "custom_window":
        ramp_max = _soft_windowed_ramp_torch(net, max(1, int(round(float(ramp_window_hours) / float(delta_t_hours)))), duration_temp)
    elif mode == "multiscale":
        values = [
            _soft_windowed_ramp_torch(net, max(1, int(round(float(hours) / float(delta_t_hours)))), duration_temp)
            for hours in _parse_ramp_windows(multiscale_ramp_windows)
        ]
        ramp_max = torch.stack(values, dim=1).max(dim=1).values
    else:
        raise ValueError("ramp_metric_mode must be one of one_step/window_1h/window_2h/window_3h/custom_window/multiscale.")
    duration = torch.sigmoid((net - tau) * duration_temp).sum(dim=1) * delta_t_hours
    return cum, ramp_max, duration


def soft_core_risk_metrics_torch(
    x_denorm: torch.Tensor,
    tau: torch.Tensor,
    event_mask: torch.Tensor,
    delta_t_hours: float = 1.0,
    duration_temp: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    net = x_denorm[:, 0, :] - x_denorm[:, 1, :] - x_denorm[:, 2, :]
    tau = tau.view(-1, 1)
    mask = event_mask.to(device=x_denorm.device, dtype=x_denorm.dtype)
    if mask.ndim != 2 or mask.shape != net.shape:
        mask = torch.ones_like(net)
    empty_mask = mask.sum(dim=1, keepdim=True) <= 0.5
    mask = torch.where(empty_mask, torch.ones_like(mask), mask)

    excess = F.softplus((net - tau) * duration_temp) / duration_temp
    cum = (excess * mask).sum(dim=1) * delta_t_hours

    if net.shape[1] <= 1:
        ramp_max = torch.zeros((net.size(0),), device=net.device, dtype=net.dtype)
    else:
        ramp = net[:, 1:] - net[:, :-1]
        pair_mask = mask[:, 1:] * mask[:, :-1]
        valid_pair = pair_mask.sum(dim=1) > 0.5
        masked_ramp = ramp.masked_fill(pair_mask <= 0.5, -1e6)
        ramp_est = torch.logsumexp(masked_ramp * duration_temp, dim=1) / duration_temp
        ramp_max = torch.where(valid_pair, ramp_est, torch.zeros_like(ramp_est))

    duration = (torch.sigmoid((net - tau) * duration_temp) * mask).sum(dim=1) * delta_t_hours
    return cum, ramp_max, duration


def topk_tail_distribution_loss_torch(
    generated_cum: torch.Tensor,
    target_cum: torch.Tensor,
    topk_ratio: float = 0.10,
) -> torch.Tensor:
    if generated_cum.numel() == 0:
        return generated_cum.new_tensor(0.0)
    ratio = min(max(float(topk_ratio), 0.0), 1.0)
    k = max(1, int(generated_cum.numel() * ratio))
    k = min(k, generated_cum.numel())
    topk_idx = torch.topk(target_cum.detach(), k=k).indices
    return F.smooth_l1_loss(generated_cum[topk_idx], target_cum[topk_idx])


def net_load_delta_loss_torch(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Differentiable net-load difference-shape loss.

    输入形状为 [B, 3, T]。net_load：净负荷，等于负荷减去风电和光伏出力。
    delta_net：相邻时刻净负荷变化量，用于刻画爬坡过程。
    L_delta_net：净负荷差分形态损失，用于约束极端事件过程中的突发变化形态。
    """
    if x_pred.shape[-1] <= 1:
        return x_pred.new_tensor(0.0)
    net_pred = x_pred[:, 0, :] - x_pred[:, 1, :] - x_pred[:, 2, :]
    net_true = x_true[:, 0, :] - x_true[:, 1, :] - x_true[:, 2, :]
    delta_pred = net_pred[:, 1:] - net_pred[:, :-1]
    delta_true = net_true[:, 1:] - net_true[:, :-1]
    return F.smooth_l1_loss(delta_pred, delta_true)


def ramp_topk_loss_torch(x_pred: torch.Tensor, x_true: torch.Tensor, topk_ratio: float = 0.10) -> torch.Tensor:
    """Top-k positive net-load ramp loss.

    ramp_pos_pred / ramp_pos_true 取净负荷正向爬坡，用 top-k 平滑约束尾部爬坡，
    比单点最大爬坡更稳定。
    """
    if x_pred.shape[-1] <= 1:
        return x_pred.new_tensor(0.0)
    net_pred = x_pred[:, 0, :] - x_pred[:, 1, :] - x_pred[:, 2, :]
    net_true = x_true[:, 0, :] - x_true[:, 1, :] - x_true[:, 2, :]
    ramp_pos_pred = F.relu(net_pred[:, 1:] - net_pred[:, :-1])
    ramp_pos_true = F.relu(net_true[:, 1:] - net_true[:, :-1])
    ratio = min(max(float(topk_ratio), 0.0), 1.0)
    k = max(1, int(ramp_pos_pred.shape[1] * ratio))
    k = min(k, ramp_pos_pred.shape[1])
    top_pred = torch.topk(ramp_pos_pred, k=k, dim=1).values
    top_true = torch.topk(ramp_pos_true, k=k, dim=1).values
    return F.smooth_l1_loss(top_pred, top_true)


def highrisk_shape_moment_loss_torch(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    highrisk_mask: torch.Tensor | None = None,
    highrisk_only: bool = True,
) -> torch.Tensor:
    """Lightweight high-risk shape-statistic anchor loss.

    统计三通道均值、标准差、一阶差分均值和一阶差分标准差。该损失用于改善
    highrisk_acf_mae，不直接计算复杂 ACF，但约束高风险样本的时序形态和波动强度。
    """
    if highrisk_only and highrisk_mask is not None:
        mask = highrisk_mask.to(device=x_pred.device, dtype=torch.bool)
        if int(mask.sum().item()) >= 2:
            x_pred = x_pred[mask]
            x_true = x_true[mask]
    if x_pred.numel() == 0:
        return x_true.new_tensor(0.0)

    def _stats(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(0, 2))
        std = x.std(dim=(0, 2), unbiased=False)
        if x.shape[-1] <= 1:
            diff = torch.zeros_like(x)
        else:
            diff = x[:, :, 1:] - x[:, :, :-1]
        diff_mean = diff.mean(dim=(0, 2))
        diff_std = diff.std(dim=(0, 2), unbiased=False)
        return torch.cat([mean, std, diff_mean, diff_std], dim=0)

    return F.l1_loss(_stats(x_pred), _stats(x_true))
