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


def batch_hard_risk_metrics(x: np.ndarray, tau: np.ndarray | float, delta_t_hours: float = 1.0) -> dict[str, np.ndarray]:
    if x.ndim != 3 or x.shape[1] != 3:
        raise ValueError("Expected x with shape [N, 3, T].")
    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.ndim == 0:
        tau_arr = np.full((x.shape[0],), float(tau_arr), dtype=float)
    net = x[:, 0, :] - x[:, 1, :] - x[:, 2, :]
    excess = np.maximum(0.0, net - tau_arr[:, None])
    cum = excess.sum(axis=1) * delta_t_hours
    ramp = np.diff(net, axis=1, prepend=net[:, :1])
    ramp_max = ramp.max(axis=1)
    duration = (net > tau_arr[:, None]).sum(axis=1) * delta_t_hours
    return {"cum_deficit": cum.astype(float), "netload_ramp_max": ramp_max.astype(float), "imbalance_duration": duration.astype(float)}


def soft_risk_metrics_torch(x_denorm: torch.Tensor, tau: torch.Tensor, delta_t_hours: float = 1.0, duration_temp: float = 12.0) -> dict[str, torch.Tensor]:
    net = x_denorm[:, 0, :] - x_denorm[:, 1, :] - x_denorm[:, 2, :]
    tau = tau.view(-1, 1)
    excess = F.softplus((net - tau) * duration_temp) / duration_temp
    cum = excess.sum(dim=1) * delta_t_hours
    ramp = net[:, 1:] - net[:, :-1]
    if ramp.shape[1] == 0:
        ramp_max = torch.zeros((net.size(0),), device=net.device, dtype=net.dtype)
    else:
        ramp_max = torch.logsumexp(ramp * duration_temp, dim=1) / duration_temp
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
