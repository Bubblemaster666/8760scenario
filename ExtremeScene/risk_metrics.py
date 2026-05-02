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
    tau_mode: str = "quantile"
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


def compute_net_load(
    load: np.ndarray | pd.Series,
    wind: np.ndarray | pd.Series,
    solar: np.ndarray | pd.Series,
) -> np.ndarray:
    load_arr = np.asarray(load, dtype=float)
    wind_arr = np.asarray(wind, dtype=float)
    solar_arr = np.asarray(solar, dtype=float)
    return load_arr - wind_arr - solar_arr


def resolve_tau(net_load: np.ndarray | pd.Series, cfg: RiskMetricConfig) -> float:
    mode = cfg.tau_mode.strip().lower()
    if mode == "fixed":
        return float(cfg.tau_fixed)
    if mode == "quantile":
        if not (0.0 < float(cfg.tau_quantile) < 1.0):
            raise ValueError("tau_quantile must be between 0 and 1.")
        return float(np.quantile(np.asarray(net_load, dtype=float), float(cfg.tau_quantile)))
    raise ValueError("tau_mode must be one of {'fixed', 'quantile'}.")


def hard_risk_metrics_from_net_load(
    net_load: np.ndarray | pd.Series,
    tau: float,
    delta_t_hours: float = 1.0,
) -> dict[str, float]:
    net = np.asarray(net_load, dtype=float)
    if net.size == 0:
        return {
            "cum_deficit": float("nan"),
            "netload_ramp_max": float("nan"),
            "imbalance_duration": float("nan"),
            "netload_peak": float("nan"),
            "netload_mean": float("nan"),
        }

    excess = np.maximum(0.0, net - tau)
    # If tau == 0, this reduces to sum(max(0, net_load)) * delta_t.
    cum_deficit = float(excess.sum() * delta_t_hours)
    ramp = np.diff(net, prepend=net[0])
    ramp_max = float(np.max(ramp))
    imbalance_duration = float((net > tau).sum() * delta_t_hours)
    return {
        "cum_deficit": cum_deficit,
        "netload_ramp_max": ramp_max,
        "imbalance_duration": imbalance_duration,
        "netload_peak": float(np.max(net)),
        "netload_mean": float(np.mean(net)),
    }


def hard_risk_metrics_from_frame(df: pd.DataFrame, cfg: RiskMetricConfig) -> tuple[pd.DataFrame, float, float]:
    out = df.copy()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col])
    out = out.sort_values(cfg.time_col).reset_index(drop=True)
    delta_t_hours = float(cfg.freq_hours or infer_freq_hours(out[cfg.time_col]))
    out["net_load"] = compute_net_load(out[cfg.load_col], out[cfg.wind_col], out[cfg.solar_col])
    out["net_load_diff"] = out["net_load"].diff().fillna(0.0)
    tau = resolve_tau(out["net_load"], cfg)
    return out, tau, delta_t_hours


def batch_hard_risk_metrics(
    x: np.ndarray,
    tau: np.ndarray | float,
    delta_t_hours: float = 1.0,
) -> dict[str, np.ndarray]:
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
    return {
        "cum_deficit": cum.astype(float),
        "netload_ramp_max": ramp_max.astype(float),
        "imbalance_duration": duration.astype(float),
    }


def soft_risk_metrics_torch(
    x_denorm: torch.Tensor,
    tau: torch.Tensor,
    delta_t_hours: float = 1.0,
    duration_temp: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_denorm.ndim != 3 or x_denorm.size(1) != 3:
        raise ValueError("Expected x_denorm with shape [N, 3, T].")
    tau = tau.view(-1, 1)
    net = x_denorm[:, 0, :] - x_denorm[:, 1, :] - x_denorm[:, 2, :]
    cum = F.relu(net - tau).sum(dim=1) * delta_t_hours
    ramp = torch.diff(net, dim=1, prepend=net[:, :1])
    ramp_max = ramp.max(dim=1).values
    duration = torch.sigmoid((net - tau) / max(duration_temp, 1e-6)).sum(dim=1) * delta_t_hours
    return cum, ramp_max, duration
