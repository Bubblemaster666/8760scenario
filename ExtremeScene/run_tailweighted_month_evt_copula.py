from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from risk_ranking_utils import (
    AUXILIARY_REALISM_METRICS,
    RISK_EVALUATION_EXPLANATION,
    RISK_RANKING_EXPLANATION,
    add_risk_score,
    write_risk_tables,
)
from run_copula_guided_residual_diffusion import _dataset_specs
from run_month_evt_copula_risk_selection import (
    ADAPTIVE_METHOD as MONTH_ADAPTIVE_METHOD,
    FIXED_METHOD as MONTH_FIXED_METHOD,
    MonthEvtCopulaConfig,
    _add_month_season,
    _build_train_risk_table,
    _choose_copula_model,
    _compute_monthly_tau,
    _copula_cfg,
    _evaluate_method,
    _existing_sources_for_dataset,
    _generate_candidate_pool,
    _load_split,
    _parse_weights,
    _select_adaptive_weights,
    _select_candidates,
    _write_method_report as _write_month_method_report,
    get_au_season,
)
from traditional_statistical_extreme_baseline.traditional_copula_baseline import (
    CopulaConfig,
    EmpiricalMarginal,
    flatten_nct,
    unflatten_nct,
)


BASE_DIR = Path(__file__).resolve().parent
TAIL_FIXED_METHOD = "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed"
TAIL_ADAPTIVE_METHOD = "TailWeighted_Month_EVT_Copula_Risk_Selection_Adaptive"


@dataclass
class TailWeightedConfig(MonthEvtCopulaConfig):
    out_dir: Path = BASE_DIR / "results" / "tailweighted_month_evt_copula"
    alpha_tail: float = 1.0
    tail_w_cum: float = 0.50
    tail_w_core: float = 0.20
    tail_w_ramp: float = 0.20
    tail_w_duration: float = 0.10


class TailWeightedGaussianCopulaGroupModel:
    """Gaussian Copula group model with tail-weighted Gaussian-score covariance.

    sample_weight: 训练样本权重，仅由训练集联合失衡风险 tail_score 计算。
    The empirical marginals remain unweighted for stability; the Gaussian-score
    mean/covariance is weighted so tail-risk samples influence dependence.
    """

    def __init__(self, group_name: str, x_group: np.ndarray, sample_weight: np.ndarray, cfg: CopulaConfig):
        self.group_name = group_name
        self.n_samples = int(x_group.shape[0])
        self.n_channels = int(x_group.shape[1])
        self.seq_len = int(x_group.shape[2])
        self.dim = int(self.n_channels * self.seq_len)
        self.cfg_dict = asdict(cfg)
        self.fit_diagnostics: dict[str, float | bool] = {}

        x_flat = flatten_nct(x_group)
        self.marginals = [EmpiricalMarginal(x_flat[:, j], cfg.quantile_grid_size) for j in range(self.dim)]
        z = np.zeros_like(x_flat, dtype=np.float64)
        for j, marginal in enumerate(self.marginals):
            u = marginal.cdf(x_flat[:, j])
            u = np.clip(u, 1.0 / (self.n_samples + 2), 1.0 - 1.0 / (self.n_samples + 2))
            z[:, j] = stats.norm.ppf(u)
        z = np.nan_to_num(z, nan=0.0, posinf=4.75, neginf=-4.75)

        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != self.n_samples:
            w = np.ones((self.n_samples,), dtype=np.float64)
        w = np.clip(w, 1e-6, None)
        w_sum = float(w.sum())
        self.mean = (z * w[:, None]).sum(axis=0) / max(w_sum, 1e-6)
        centered = z - self.mean[None, :]
        cov = (centered * w[:, None]).T @ centered / max(w_sum, 1e-6)
        cov = np.atleast_2d(cov)
        if cov.shape != (self.dim, self.dim):
            cov = np.eye(self.dim)

        shrink = float(np.clip(cfg.covariance_shrinkage, 0.0, 0.95))
        diag = np.diag(np.maximum(np.diag(cov), 1e-5))
        cov = (1.0 - shrink) * cov + shrink * diag
        cov = 0.5 * (cov + cov.T)
        jitter_used = 0.0
        try:
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = float(np.min(eigvals))
        except Exception:
            min_eig = -1.0
        if min_eig <= 1e-8:
            jitter_used = float(abs(min_eig) + 1e-5)
            cov = cov + jitter_used * np.eye(self.dim)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-5, None)
        self.cov = (vecs * vals[None, :]) @ vecs.T
        self.cov_sqrt = vecs * np.sqrt(vals)[None, :]
        cond_number = float(np.max(vals) / max(np.min(vals), 1e-12))

        self.channel_min = np.quantile(x_group, 0.001, axis=(0, 2))
        self.channel_max = np.quantile(x_group, 0.999, axis=(0, 2))
        self.ramp_abs_q = self._estimate_ramp_quantile(x_group, cfg.ramp_clip_quantile)
        self.mean_profile = np.mean(x_group, axis=0)
        self.fit_diagnostics = {
            "weight_min": float(np.min(w)),
            "weight_mean": float(np.mean(w)),
            "weight_max": float(np.max(w)),
            "condition_number": cond_number,
            "jitter_used": float(jitter_used),
            "is_positive_definite": bool(np.min(vals) > 0.0),
        }

    def _estimate_ramp_quantile(self, x: np.ndarray, q: float) -> np.ndarray:
        out = np.ones(self.n_channels, dtype=np.float64)
        for c in range(self.n_channels):
            r = np.abs(np.diff(x[:, c, :], axis=1)).reshape(-1)
            out[c] = float(np.quantile(r, np.clip(q, 0.5, 0.9999))) if r.size else np.inf
            out[c] = max(out[c], 1e-6)
        return out

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((int(n), self.dim)) @ self.cov_sqrt.T + self.mean[None, :]
        u = stats.norm.cdf(z)
        x_flat = np.zeros_like(u, dtype=np.float64)
        for j, marginal in enumerate(self.marginals):
            x_flat[:, j] = marginal.ppf(u[:, j])
        return unflatten_nct(x_flat, self.n_channels, self.seq_len)


def _rank_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.nanmax(arr) - np.nanmin(arr) <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    order = np.argsort(np.argsort(arr))
    denom = max(len(arr) - 1, 1)
    return order.astype(float) / float(denom)


def _compute_tail_scores(train_risk: pd.DataFrame, cond_train: pd.DataFrame, cfg: TailWeightedConfig, dataset: str, out_dir: Path) -> pd.DataFrame:
    cum_r = _rank_norm(train_risk["cum_deficit"].to_numpy(float))
    core_r = _rank_norm(train_risk["core_cum_deficit"].to_numpy(float))
    ramp_r = _rank_norm(train_risk["netload_ramp_max"].to_numpy(float))
    dur_r = _rank_norm(train_risk["imbalance_duration"].to_numpy(float))
    tail = (
        float(cfg.tail_w_cum) * cum_r
        + float(cfg.tail_w_core) * core_r
        + float(cfg.tail_w_ramp) * ramp_r
        + float(cfg.tail_w_duration) * dur_r
    )
    sample_weight = 1.0 + float(cfg.alpha_tail) * tail
    out = pd.DataFrame(
        {
            "dataset": dataset,
            "sample_id": np.arange(len(train_risk), dtype=int),
            "month": cond_train["month"].astype(int).to_numpy(),
            "season": cond_train["au_season"].astype(str).to_numpy(),
            "cum_deficit": train_risk["cum_deficit"].to_numpy(float),
            "core_cum_deficit": train_risk["core_cum_deficit"].to_numpy(float),
            "netload_ramp_max": train_risk["netload_ramp_max"].to_numpy(float),
            "imbalance_duration": train_risk["imbalance_duration"].to_numpy(float),
            "tail_score": tail,
            "sample_weight": sample_weight,
        }
    )
    out.to_csv(out_dir / "tail_score_summary.csv", index=False, encoding="utf-8-sig")
    return out


def _fit_tailweighted_group_copulas(
    x_train: np.ndarray,
    cond_train: pd.DataFrame,
    tail_df: pd.DataFrame,
    cfg: TailWeightedConfig,
    out_dir: Path,
    dataset_name: str,
) -> tuple[dict[tuple[str, str], TailWeightedGaussianCopulaGroupModel], pd.DataFrame, pd.DataFrame]:
    cop_cfg = _copula_cfg(cfg, int(x_train.shape[2]), out_dir / "tailweighted_copula_groups")
    weights = tail_df["sample_weight"].to_numpy(float)
    models: dict[tuple[str, str], TailWeightedGaussianCopulaGroupModel] = {}
    group_rows: list[dict] = []
    fit_rows: list[dict] = []

    def fit_one(group_type: str, group_name: str, idx: np.ndarray, used: bool, fallback_to: str, notes: str) -> None:
        group_rows.append(
            {
                "dataset": dataset_name,
                "method": "TailWeighted",
                "group_type": group_type,
                "group_name": group_name,
                "alpha_tail": float(cfg.alpha_tail),
                "n_train_samples": int(len(idx)),
                "used_for_fit": bool(used),
                "fallback_to": fallback_to,
                "notes": notes,
            }
        )
        if not used:
            return
        model = TailWeightedGaussianCopulaGroupModel(f"{group_type}:{group_name}", x_train[idx], weights[idx], cop_cfg)
        models[(group_type, group_name)] = model
        fit_rows.append(
            {
                "dataset": dataset_name,
                "group_type": group_type,
                "group_name": group_name,
                "alpha_tail": float(cfg.alpha_tail),
                "n_train_samples": int(len(idx)),
                "fallback_to": fallback_to,
                **model.fit_diagnostics,
            }
        )

    all_idx = np.arange(len(x_train), dtype=np.int64)
    fit_one("global", "all", all_idx, True, "", "global train-only tail-weighted Copula")
    for season in ["summer", "autumn", "winter", "spring"]:
        idx = np.where(cond_train["au_season"].astype(str).to_numpy() == season)[0]
        used = len(idx) >= int(cfg.min_season_samples)
        fit_one("season", season, idx, used, "global" if not used else "", "season fallback if too few samples")
    for month in range(1, 13):
        idx = np.where(cond_train["month"].astype(int).to_numpy() == month)[0]
        season = get_au_season(month)
        season_idx = np.where(cond_train["au_season"].astype(str).to_numpy() == season)[0]
        used = len(idx) >= int(cfg.min_month_samples)
        fallback = "" if used else (f"season:{season}" if len(season_idx) >= int(cfg.min_season_samples) else "global")
        fit_one("month", str(month), idx, used, fallback, "month fallback if too few samples")

    group_df = pd.DataFrame(group_rows)
    fit_df = pd.DataFrame(fit_rows)
    group_df.to_csv(out_dir / "copula_condition_group_summary.csv", index=False, encoding="utf-8-sig")
    fit_df.to_csv(out_dir / "tailweighted_copula_fit_summary.csv", index=False, encoding="utf-8-sig")
    return models, group_df, fit_df


def _copy_existing_month_outputs(spec, out_dir: Path) -> list[dict]:
    source_dir = BASE_DIR / "results" / "month_evt_copula_risk_selection" / spec.out_name
    rows: list[dict] = []
    mapping = {
        "traditional_gaussian_copula": source_dir / "generated_samples_traditional_gaussian_copula.npy",
        MONTH_FIXED_METHOD: source_dir / "generated_samples_fixed.npy",
        MONTH_ADAPTIVE_METHOD: source_dir / "generated_samples_adaptive.npy",
    }
    for method, source in mapping.items():
        if source.exists():
            target = out_dir / f"generated_samples_{method}.npy"
            shutil.copy2(source, target)
            rows.append(_evaluate_method(method, target, spec.data_dir, out_dir))
    return rows


def run_one_dataset(spec, cfg: TailWeightedConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = cfg.out_dir / spec.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.seed))

    x_train, cond_train_raw, meta_train, mask_train = _load_split(spec.data_dir, "train")
    _, cond_val_raw, meta_val, mask_val = _load_split(spec.data_dir, "val")
    _, cond_test_raw, meta_test, mask_test = _load_split(spec.data_dir, "test")
    cond_train = _add_month_season(cond_train_raw, meta_train)
    cond_val = _add_month_season(cond_val_raw, meta_val)
    cond_test = _add_month_season(cond_test_raw, meta_test)

    cop_cfg = _copula_cfg(cfg, int(x_train.shape[2]), out_dir / "tailweighted_copula_groups")
    tau_by_month, tau_df = _compute_monthly_tau(x_train, cond_train, cfg, spec.out_name, out_dir)
    train_risk = _build_train_risk_table(x_train, cond_train, mask_train, tau_by_month, cfg)
    tail_df = _compute_tail_scores(train_risk, cond_train, cfg, spec.out_name, out_dir)
    models, _, _ = _fit_tailweighted_group_copulas(x_train, cond_train, tail_df, cfg, out_dir, spec.out_name)

    val_candidates, val_metrics, val_targets, val_base = _generate_candidate_pool(
        "val", cond_val, mask_val, models, train_risk, tau_by_month, cfg, cop_cfg, rng, out_dir, spec.out_name
    )
    adaptive_weights, adaptive_df = _select_adaptive_weights(
        spec.data_dir, out_dir, val_candidates, val_metrics, val_targets, val_base, train_risk, cond_val, cfg
    )
    adaptive_df.insert(0, "alpha_tail", float(cfg.alpha_tail))
    adaptive_df.to_csv(out_dir / "adaptive_weight_selection.csv", index=False, encoding="utf-8-sig")

    test_candidates, test_metrics, test_targets, test_base = _generate_candidate_pool(
        "test", cond_test, mask_test, models, train_risk, tau_by_month, cfg, cop_cfg, rng, out_dir, spec.out_name
    )
    pd.DataFrame(test_targets).to_csv(out_dir / "risk_target_summary.csv", index=False, encoding="utf-8-sig")
    fixed_weights = _parse_weights(cfg.fixed_weights)
    fixed_gen, fixed_log = _select_candidates(test_candidates, test_metrics, test_targets, test_base, train_risk, cond_test, fixed_weights, cfg)
    adaptive_gen, adaptive_log = _select_candidates(test_candidates, test_metrics, test_targets, test_base, train_risk, cond_test, adaptive_weights, cfg)
    fixed_log.insert(0, "alpha_tail", float(cfg.alpha_tail))
    adaptive_log.insert(0, "alpha_tail", float(cfg.alpha_tail))
    fixed_log.to_csv(out_dir / "candidate_selection_log_fixed.csv", index=False, encoding="utf-8-sig")
    adaptive_log.to_csv(out_dir / "candidate_selection_log_adaptive.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "generated_samples_tailweighted_fixed.npy", fixed_gen.astype(np.float32))
    np.save(out_dir / "generated_samples_tailweighted_adaptive.npy", adaptive_gen.astype(np.float32))

    rows = _copy_existing_month_outputs(spec, out_dir)
    rows.append(_evaluate_method(TAIL_FIXED_METHOD, out_dir / "generated_samples_tailweighted_fixed.npy", spec.data_dir, out_dir))
    rows.append(_evaluate_method(TAIL_ADAPTIVE_METHOD, out_dir / "generated_samples_tailweighted_adaptive.npy", spec.data_dir, out_dir))
    for method, source in _existing_sources_for_dataset(spec).items():
        if not source.exists() or method in {r["method"] for r in rows}:
            continue
        target = out_dir / f"generated_samples_{method}.npy"
        shutil.copy2(source, target)
        rows.append(_evaluate_method(method, target, spec.data_dir, out_dir))

    df = add_risk_score(pd.DataFrame(rows))
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir, method_col="method")
    _write_method_report(spec, out_dir, risk_main, aux, cfg, adaptive_weights)
    return risk_main, aux


def _write_method_report(spec, out_dir: Path, risk_main: pd.DataFrame, aux: pd.DataFrame, cfg: TailWeightedConfig, adaptive_weights) -> None:
    lines = [
        f"# {spec.name} - TailWeighted Month EVT-Copula Risk Selection",
        "",
        "## Method",
        "",
        "This method upgrades the month-conditioned EVT-Copula selection pipeline with train-only tail weighting. Tail score is a rank-normalized joint-risk score over cumulative deficit, core cumulative deficit, 3h ramp, and imbalance duration. The Gaussian-score covariance of each month/season/global Copula is estimated with sample weights `1 + alpha_tail * tail_score`.",
        "",
        "## Tail Score",
        "",
        f"- weights `(cum, core, ramp, duration)`: ({cfg.tail_w_cum}, {cfg.tail_w_core}, {cfg.tail_w_ramp}, {cfg.tail_w_duration})",
        f"- alpha_tail: {cfg.alpha_tail}",
        "",
        "## Adaptive Candidate-Selection Weights",
        "",
        f"- selected `(cum, core, ramp, duration)`: {adaptive_weights}",
        "",
        "## Main Risk Table",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Auxiliary Realism Table",
        "",
        aux[[c for c in ["method", *AUXILIARY_REALISM_METRICS, "realism_check_pass"] if c in aux.columns]].to_markdown(index=False),
    ]
    (out_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_global_summaries(root: Path, dataset_tables: dict[str, pd.DataFrame], aux_tables: dict[str, pd.DataFrame]) -> None:
    risk_rows = []
    for dataset, table in dataset_tables.items():
        tmp = table.copy()
        tmp.insert(0, "dataset", dataset)
        risk_rows.append(tmp)
    risk_summary = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    risk_summary.to_csv(root / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")

    rank_rows = []
    methods = sorted(risk_summary["method"].astype(str).unique()) if len(risk_summary) else []
    for method in methods:
        row = {"method": method}
        ranks, scores = [], []
        for key in ["singleton", "muswellbrook", "cessnock_or_newarea"]:
            sub = risk_summary[(risk_summary["dataset"] == key) & (risk_summary["method"] == method)]
            rank = float(sub["risk_rank"].iloc[0]) if len(sub) and "risk_rank" in sub.columns else np.nan
            score = float(sub["risk_score"].iloc[0]) if len(sub) and "risk_score" in sub.columns else np.nan
            row[f"{key}_risk_rank"] = rank
            row[f"{key}_risk_score"] = score
            if np.isfinite(rank):
                ranks.append(rank)
            if np.isfinite(score):
                scores.append(score)
        row["mean_risk_rank"] = float(np.mean(ranks)) if ranks else np.nan
        row["mean_risk_score"] = float(np.mean(scores)) if scores else np.nan
        row["wins_count"] = int(sum(1 for r in ranks if int(r) == 1))
        row["top3_count"] = int(sum(1 for r in ranks if r <= 3))
        rank_rows.append(row)
    rank_df = pd.DataFrame(rank_rows).sort_values(["mean_risk_rank", "mean_risk_score"], na_position="last")
    rank_df.to_csv(root / "all_datasets_rank_summary.csv", index=False, encoding="utf-8-sig")

    aux_rows = []
    for dataset, table in aux_tables.items():
        tmp = table.copy()
        tmp.insert(0, "dataset", dataset)
        aux_rows.append(tmp)
    aux_summary = pd.concat(aux_rows, ignore_index=True) if aux_rows else pd.DataFrame()
    aux_summary.to_csv(root / "all_datasets_auxiliary_summary.csv", index=False, encoding="utf-8-sig")
    _write_final_report(root, risk_summary, rank_df)


def _write_final_report(root: Path, risk_summary: pd.DataFrame, rank_df: pd.DataFrame) -> None:
    lines = [
        "# Final TailWeighted Month EVT-Copula Report",
        "",
        "## Method Summary",
        "",
        "TailWeighted_Month_EVT_Copula_Risk_Selection fits train-only month/season/global Copulas with joint-risk tail-weighted Gaussian-score covariance, then uses EVT extreme probability and validation-selected risk-selection weights to choose the best candidate scenario.",
        "",
        "## All Dataset Risk Summary",
        "",
        risk_summary.to_markdown(index=False) if len(risk_summary) else "No rows.",
        "",
        "## Cross-Dataset Rank Summary",
        "",
        rank_df.to_markdown(index=False) if len(rank_df) else "No rows.",
        "",
        "## Notes",
        "",
        "- Tail weighting uses train split only.",
        "- Validation split selects candidate-selection weights only.",
        "- Test split is used only for final evaluation.",
    ]
    (root / "final_tailweighted_month_evt_copula_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(cfg: TailWeightedConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_tables: dict[str, pd.DataFrame] = {}
    aux_tables: dict[str, pd.DataFrame] = {}
    for spec in [s for s in _dataset_specs() if s.data_dir.exists()]:
        print(f"\n=== Dataset: {spec.out_name} ===")
        risk_main, aux = run_one_dataset(spec, cfg)
        dataset_tables[spec.out_name] = risk_main
        aux_tables[spec.out_name] = aux
    _write_global_summaries(cfg.out_dir, dataset_tables, aux_tables)


def parse_args() -> TailWeightedConfig:
    parser = argparse.ArgumentParser(description="Run tail-weighted month-conditioned EVT-Copula risk selection.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "tailweighted_month_evt_copula")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-candidates", type=int, default=20)
    parser.add_argument("--min-month-samples", type=int, default=10)
    parser.add_argument("--min-season-samples", type=int, default=25)
    parser.add_argument("--tau-quantile", type=float, default=0.75)
    parser.add_argument("--alpha-tail", type=float, default=1.0)
    parser.add_argument("--tail-w-cum", type=float, default=0.50)
    parser.add_argument("--tail-w-core", type=float, default=0.20)
    parser.add_argument("--tail-w-ramp", type=float, default=0.20)
    parser.add_argument("--tail-w-duration", type=float, default=0.10)
    parser.add_argument("--fixed-weights", type=str, default="0.35,0.25,0.20,0.20")
    return TailWeightedConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_all(parse_args())
