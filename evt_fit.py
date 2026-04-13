from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from scipy.stats import genpareto


@dataclass
class EVTConfig:
    # 主指标列名
    metric_col: str = "cum_deficit"

    # POT 阈值分位数
    threshold_quantile: float = 0.90

    # 极端等级划分阈值（按尾部超越概率）
    severe_prob: float = 0.01   # 重度
    moderate_prob: float = 0.05 # 中度
    mild_prob: float = 0.10     # 轻度

    # 最少超阈值样本数，太少就退化为经验分级
    min_exceedances: int = 5


def _prob_to_level(
    p: float,
    severe_prob: float,
    moderate_prob: float,
    mild_prob: float,
) -> int:
    """
    返回等级：
    0 = 非极端/尾部外
    1 = 轻度
    2 = 中度
    3 = 重度
    """
    if p <= severe_prob:
        return 3
    elif p <= moderate_prob:
        return 2
    elif p <= mild_prob:
        return 1
    else:
        return 0


def _empirical_exceedance_prob(x: pd.Series) -> pd.Series:
    """
    使用经验超越概率（descending rank / (n+1)）。
    这样最大值不会直接得到 0，更适合作为尾部概率近似。
    """
    out = pd.Series(np.nan, index=x.index, dtype=float)
    valid = x.notna()
    n = int(valid.sum())
    if n == 0:
        return out

    desc_rank = x.loc[valid].rank(method="average", ascending=False)
    out.loc[valid] = desc_rank / (n + 1)
    return out.clip(1e-8, 1.0)


def fit_evt_and_label(
    samples: pd.DataFrame,
    cfg: Optional[EVTConfig] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    对 samples[metric_col] 做 POT-GPD 拟合，并返回：
    1. 带 extreme_prob / severity_level 的样本表
    2. EVT 参数信息
    """
    if cfg is None:
        cfg = EVTConfig()

    if samples.empty:
        return samples.copy(), {"method": "empty"}

    if cfg.metric_col not in samples.columns:
        raise ValueError(f"samples 中缺少主指标列: {cfg.metric_col}")

    out = samples.copy()
    metric_series = out[cfg.metric_col].astype(float)
    x = metric_series.values

    # 去掉 nan 做阈值
    valid_x = metric_series.dropna().values
    if len(valid_x) == 0:
        raise ValueError(f"{cfg.metric_col} 全为空，无法做 EVT")

    # 1) 选阈值
    u = float(np.quantile(valid_x, cfg.threshold_quantile))

    # 2) 提取超阈值
    exceed = valid_x[valid_x > u] - u

    # 如果超阈值样本太少，退化成经验超越概率
    if len(exceed) < cfg.min_exceedances:
        out["extreme_prob"] = _empirical_exceedance_prob(metric_series)
        out["severity_level"] = out["extreme_prob"].apply(
            lambda p: np.nan if pd.isna(p) else _prob_to_level(
                p,
                cfg.severe_prob,
                cfg.moderate_prob,
                cfg.mild_prob,
            )
        )

        evt_info = {
            "method": "empirical_fallback",
            "metric_col": cfg.metric_col,
            "threshold_u": u,
            "n_total": int(len(valid_x)),
            "n_exceed": int(len(exceed)),
        }
        return out, evt_info

    # 3) GPD 拟合（固定 loc=0）
    c, loc, scale = genpareto.fit(exceed, floc=0)

    # 阈值处经验尾部概率
    p_u = float((valid_x > u).mean())

    empirical_probs = _empirical_exceedance_prob(metric_series)
    probs = []
    for idx, xi in enumerate(x):
        if np.isnan(xi):
            probs.append(np.nan)
            continue

        if xi <= u:
            # 非尾部样本用经验超越概率，且不小于阈值处尾部概率
            p = max(float(empirical_probs.iloc[idx]), p_u)
        else:
            y = xi - u
            # P(X > x) = P(X > u) * P(Y > y | X > u)
            tail_cond = 1.0 - genpareto.cdf(y, c=c, loc=0, scale=scale)
            p = p_u * tail_cond

        # 限制范围，避免 0 和 1
        p = float(np.clip(p, 1e-8, 1.0))
        probs.append(p)

    out["extreme_prob"] = probs
    out["severity_level"] = out["extreme_prob"].apply(
        lambda p: np.nan if pd.isna(p) else _prob_to_level(
            p,
            cfg.severe_prob,
            cfg.moderate_prob,
            cfg.mild_prob,
        )
    )

    evt_info = {
        "method": "pot_gpd",
        "metric_col": cfg.metric_col,
        "threshold_u": u,
        "shape_c": float(c),
        "scale": float(scale),
        "tail_prob_at_u": p_u,
        "n_total": int(len(valid_x)),
        "n_exceed": int(len(exceed)),
    }
    return out, evt_info


if __name__ == "__main__":
    # ===== 最小测试 =====
    df_demo = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(1, 16)],
        "event_type": ["寒潮"] * 15,
        "cum_deficit": [
            1200, 1350, 1400, 1500, 1600,
            1700, 1800, 1900, 2000, 2200,
            2500, 2800, 3200, 4500, 7000
        ],
    })

    cfg = EVTConfig(
        metric_col="cum_deficit",
        threshold_quantile=0.90,
        severe_prob=0.01,
        moderate_prob=0.05,
        mild_prob=0.10,
        min_exceedances=3,  # 测试时放低一点
    )

    labeled, evt_info = fit_evt_and_label(df_demo, cfg)

    print("=== EVT INFO ===")
    print(evt_info)
    print("\n=== LABELED SAMPLES ===")
    print(labeled)
