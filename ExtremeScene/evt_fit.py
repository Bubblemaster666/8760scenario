from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import genpareto


@dataclass
class EVTConfig:
    metric_col: str = "cum_deficit"
    threshold_quantile: float = 0.90
    severity_mode: str = "hybrid"
    severity_q1: float = 0.60
    severity_q2: float = 0.80
    severity_q3: float = 0.92
    severity_positive_only: bool = True
    severe_prob: float = 0.01
    moderate_prob: float = 0.05
    mild_prob: float = 0.10
    min_exceedances: int = 5
    eps: float = 1e-8


def prob_to_level(prob: float, severe_prob: float, moderate_prob: float, mild_prob: float) -> int:
    if prob <= severe_prob:
        return 3
    if prob <= moderate_prob:
        return 2
    if prob <= mild_prob:
        return 1
    return 0


def empirical_exceedance_prob(x: pd.Series, eps: float = 1e-8) -> pd.Series:
    out = pd.Series(np.nan, index=x.index, dtype=float)
    valid = x.notna()
    n = int(valid.sum())
    if n == 0:
        return out
    desc_rank = x.loc[valid].rank(method="average", ascending=False)
    out.loc[valid] = desc_rank / (n + 1)
    return out.clip(eps, 1.0)


def compute_tail_score(prob: pd.Series | np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prob_arr = np.asarray(prob, dtype=float)
    score = -np.log(np.clip(prob_arr, eps, 1.0))
    mean = float(np.nanmean(score))
    std = float(np.nanstd(score) + 1e-6)
    z = (score - mean) / std
    return score.astype(float), z.astype(float), {"mean": mean, "std": std}


def metric_to_quantile_level(
    metric: pd.Series,
    q1: float = 0.60,
    q2: float = 0.80,
    q3: float = 0.92,
    positive_only: bool = True,
) -> tuple[pd.Series, dict[str, float | bool | int]]:
    metric_numeric = metric.astype(float)
    valid = metric_numeric.dropna()
    base = valid[valid > 0] if positive_only else valid
    used_positive_only = bool(positive_only and len(base) >= 4)
    if not used_positive_only:
        base = valid
    if base.empty:
        return pd.Series(0, index=metric.index, dtype=int), {"q1": float("nan"), "q2": float("nan"), "q3": float("nan"), "positive_only": used_positive_only, "n_for_quantiles": 0}
    q1_value = float(base.quantile(q1))
    q2_value = float(base.quantile(q2))
    q3_value = float(base.quantile(q3))
    levels = pd.Series(0, index=metric.index, dtype=int)
    values = metric_numeric.fillna(-np.inf)
    levels = levels.mask(values >= q1_value, 1)
    levels = levels.mask(values >= q2_value, 2)
    levels = levels.mask(values >= q3_value, 3)
    if used_positive_only:
        levels = levels.mask(values <= 0, 0)
    return levels.astype(int), {"q1": q1_value, "q2": q2_value, "q3": q3_value, "positive_only": used_positive_only, "n_for_quantiles": int(len(base))}


def fit_evt_and_label(samples: pd.DataFrame, cfg: Optional[EVTConfig] = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or EVTConfig()
    if samples.empty:
        return samples.copy(), {"method": "empty"}
    if cfg.metric_col not in samples.columns:
        raise ValueError(f"samples is missing EVT metric column: {cfg.metric_col}")
    out = samples.copy()
    metric_series = out[cfg.metric_col].astype(float)
    valid_x = metric_series.dropna().to_numpy(dtype=float)
    if valid_x.size == 0:
        raise ValueError(f"{cfg.metric_col} is empty after dropping NaNs.")

    threshold_u = float(np.quantile(valid_x, cfg.threshold_quantile))
    exceed = valid_x[valid_x > threshold_u] - threshold_u
    if exceed.size < cfg.min_exceedances:
        extreme_prob = empirical_exceedance_prob(metric_series, eps=cfg.eps)
        evt_info = {"method": "empirical_fallback", "metric_col": cfg.metric_col, "threshold_u": threshold_u, "n_total": int(valid_x.size), "n_exceed": int(exceed.size)}
    else:
        c, _, scale = genpareto.fit(exceed, floc=0)
        tail_prob_at_u = float((valid_x > threshold_u).mean())
        empirical_prob = empirical_exceedance_prob(metric_series, eps=cfg.eps)
        probs = []
        for idx, xi in enumerate(metric_series.to_numpy(dtype=float)):
            if np.isnan(xi):
                probs.append(np.nan); continue
            if xi <= threshold_u:
                p = max(float(empirical_prob.iloc[idx]), tail_prob_at_u)
            else:
                y = xi - threshold_u
                tail_cond = 1.0 - genpareto.cdf(y, c=c, loc=0, scale=scale)
                p = tail_prob_at_u * tail_cond
            probs.append(float(np.clip(p, cfg.eps, 1.0)))
        extreme_prob = pd.Series(probs, index=out.index, dtype=float)
        evt_info = {"method": "pot_gpd", "metric_col": cfg.metric_col, "threshold_u": threshold_u, "shape_c": float(c), "scale": float(scale), "tail_prob_at_u": tail_prob_at_u, "n_total": int(valid_x.size), "n_exceed": int(exceed.size)}

    tail_score, tail_score_z, tail_score_stats = compute_tail_score(extreme_prob, eps=cfg.eps)
    out["extreme_prob"] = np.asarray(extreme_prob, dtype=float)
    out["tail_score"] = tail_score
    out["tail_score_zscore"] = tail_score_z
    severity_mode = cfg.severity_mode.strip().lower()
    if severity_mode == "evt_prob":
        out["severity_level"] = out["extreme_prob"].apply(lambda p: np.nan if pd.isna(p) else prob_to_level(float(p), cfg.severe_prob, cfg.moderate_prob, cfg.mild_prob))
        severity_quantiles = {}
    elif severity_mode in {"quantile", "hybrid"}:
        levels, severity_quantiles = metric_to_quantile_level(metric_series, q1=cfg.severity_q1, q2=cfg.severity_q2, q3=cfg.severity_q3, positive_only=cfg.severity_positive_only)
        out["severity_level"] = levels
    else:
        raise ValueError("severity_mode must be one of {'evt_prob', 'quantile', 'hybrid'}.")

    evt_info["tail_score_stats"] = tail_score_stats
    evt_info["severity_mode"] = severity_mode
    evt_info["severity_quantiles"] = severity_quantiles
    evt_info["severity_level_counts"] = {str(k): int(v) for k, v in out["severity_level"].fillna(0).astype(int).value_counts().sort_index().items()}
    evt_info["severity_thresholds"] = {
        "severe_prob": cfg.severe_prob,
        "moderate_prob": cfg.moderate_prob,
        "mild_prob": cfg.mild_prob,
        "severity_q1": cfg.severity_q1,
        "severity_q2": cfg.severity_q2,
        "severity_q3": cfg.severity_q3,
    }
    return out, evt_info


if __name__ == "__main__":
    df_demo = pd.DataFrame({"sample_id": [f"S{i:03d}" for i in range(10)], "cum_deficit": np.linspace(0, 100, 10)})
    labeled, info = fit_evt_and_label(df_demo, EVTConfig(severity_mode="hybrid"))
    print(labeled)
    print(info)
