from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RISK_MAIN_METRICS = [
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

OPTIONAL_RISK_CONTROL_METRICS = [
    "extreme_degree_match_rate",
]

RISK_SCORE_WEIGHTS = {
    "q99_cum_deficit_error": 0.30,
    "core_q99_cum_deficit_error": 0.30,
    "netload_ramp_max_mae": 0.25,
    "imbalance_duration_mae": 0.15,
}

AUXILIARY_REALISM_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "highrisk_js",
    "highrisk_corr_matrix_error",
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
    "physics_violation_rate",
    "night_solar_error",
    "negative_power_rate",
]


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk_score/risk_rank using only joint imbalance risk errors.

    The four risk errors are min-max normalized column-wise. Smaller is better.
    Missing values are treated as worst available normalized value so that an
    unavailable core metric does not silently win the main risk ranking.
    """

    out = df.copy()
    used_norm_cols: list[str] = []
    for metric in RISK_MAIN_METRICS:
        norm_col = f"norm_{metric}"
        values = pd.to_numeric(out.get(metric), errors="coerce")
        valid = values.dropna()
        if valid.empty:
            out[norm_col] = np.nan
            continue
        lo = float(valid.min())
        hi = float(valid.max())
        if abs(hi - lo) <= 1e-12:
            norm = values * 0.0
        else:
            norm = (values - lo) / (hi - lo)
        out[norm_col] = norm.fillna(1.0)
        used_norm_cols.append(norm_col)

    if used_norm_cols:
        score = np.zeros((len(out),), dtype=float)
        total_weight = 0.0
        for metric, weight in RISK_SCORE_WEIGHTS.items():
            norm_col = f"norm_{metric}"
            if norm_col in out.columns:
                score += weight * pd.to_numeric(out[norm_col], errors="coerce").fillna(1.0).to_numpy()
                total_weight += weight
        out["risk_score"] = score / max(total_weight, 1e-12)
        out["risk_rank"] = pd.to_numeric(out["risk_score"], errors="coerce").rank(method="min", ascending=True).astype("Int64")
    else:
        out["risk_score"] = np.nan
        out["risk_rank"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    return out.sort_values(["risk_score", "risk_rank"], na_position="last").reset_index(drop=True)


def build_risk_main_table(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    """Return the paper main table based on joint imbalance risk metrics only."""

    work = df.copy()
    if method_col != "method" and method_col in work.columns:
        work = work.rename(columns={method_col: "method"})
    work = add_risk_score(work)
    columns = ["method", *RISK_MAIN_METRICS, "risk_score", "risk_rank"]
    if "extreme_degree_match_rate" in work.columns:
        columns.append("extreme_degree_match_rate")
    return work[[col for col in columns if col in work.columns]]


def build_auxiliary_realism_table(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    """Return auxiliary realism metrics, excluded from the main risk ranking."""

    work = df.copy()
    if method_col != "method" and method_col in work.columns:
        work = work.rename(columns={method_col: "method"})
    columns = ["method", *[col for col in AUXILIARY_REALISM_METRICS if col in work.columns]]
    table = work[columns].copy()
    if "realism_check_pass" not in table.columns:
        table["realism_check_pass"] = ""
    return table


def write_risk_tables(df: pd.DataFrame, out_dir: Path, method_col: str = "method") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write risk_main_compare.csv and auxiliary_realism_metrics.csv."""

    out_dir.mkdir(parents=True, exist_ok=True)
    risk_table = build_risk_main_table(df, method_col=method_col)
    aux_table = build_auxiliary_realism_table(df, method_col=method_col)
    risk_table.to_csv(out_dir / "risk_main_compare.csv", index=False, encoding="utf-8-sig")
    aux_table.to_csv(out_dir / "auxiliary_realism_metrics.csv", index=False, encoding="utf-8-sig")
    return risk_table, aux_table


RISK_RANKING_EXPLANATION = (
    "主排序基于联合失衡风险指标，不包含 Wasserstein、JS、ACF 等整体统计分布指标。"
)

RISK_EVALUATION_EXPLANATION = (
    "由于本文方法采用极端样本强化与风险过程条件，生成分布会向高风险尾部区域偏移。"
    "因此，整体分布距离类指标仅作为辅助真实性检验，不再作为主评价标准。"
    "本文主评价聚焦累计缺额、三小时净负荷爬坡和持续失衡时长三类联合失衡风险指标。"
)
