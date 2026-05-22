from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from risk_ranking_utils import (
    AUXILIARY_REALISM_METRICS,
    RISK_EVALUATION_EXPLANATION,
    RISK_RANKING_EXPLANATION,
    add_risk_score,
    write_risk_tables,
)
from run_copula_guided_residual_diffusion import DatasetSpec, _dataset_specs
from run_simple_evt_risk_diffusion import FULL_COMPARE_METRICS, evaluate_method
from traditional_statistical_extreme_baseline.traditional_copula_baseline import (
    CopulaConfig,
    GaussianCopulaGroupModel,
    physical_projection_np,
    temporal_correction,
)


BASE_DIR = Path(__file__).resolve().parent
FIXED_METHOD = "Month_EVT_Copula_Risk_Selection_Fixed"
ADAPTIVE_METHOD = "Month_EVT_Copula_Risk_Selection_Adaptive"


@dataclass
class MonthEvtCopulaConfig:
    out_dir: Path = BASE_DIR / "results" / "month_evt_copula_risk_selection"
    seed: int = 42
    k_candidates: int = 20
    min_month_samples: int = 10
    min_season_samples: int = 25
    tau_quantile: float = 0.75
    covariance_shrinkage: float = 0.08
    temporal_smooth_strength: float = 0.15
    quantile_grid_size: int = 401
    delta_t_hours: float = 1.0
    fixed_weights: str = "0.35,0.25,0.20,0.20"
    adaptive_weight_grid: str = (
        "0.35,0.25,0.20,0.20;"
        "0.40,0.30,0.15,0.15;"
        "0.50,0.25,0.15,0.10;"
        "0.30,0.30,0.25,0.15;"
        "0.25,0.25,0.25,0.25;"
        "0.45,0.20,0.20,0.15"
    )


def get_au_season(month: int) -> str:
    """Return Australian NSW season for a calendar month."""

    m = int(month)
    if m in {12, 1, 2}:
        return "summer"
    if m in {3, 4, 5}:
        return "autumn"
    if m in {6, 7, 8}:
        return "winter"
    return "spring"


def _parse_weights(text: str) -> tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("weights must contain four comma-separated values")
    total = sum(vals)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return tuple(float(v / total) for v in vals)  # type: ignore[return-value]


def _parse_weight_grid(text: str) -> list[tuple[float, float, float, float]]:
    return [_parse_weights(part) for part in str(text).split(";") if part.strip()]


def _load_split(data_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
    cond = pd.read_csv(data_dir / f"cond_{split}.csv")
    meta = pd.read_csv(data_dir / f"meta_{split}.csv")
    mask_path = data_dir / f"event_mask_{split}.npy"
    mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else None
    return x, cond, meta, mask


def _ensure_month(cond: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.Series:
    if "month" in cond.columns:
        return pd.to_numeric(cond["month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
    if meta is not None and "window_start_time" in meta.columns:
        return pd.to_datetime(meta["window_start_time"], errors="coerce").dt.month.fillna(1).astype(int)
    return pd.Series(np.ones(len(cond), dtype=int), index=cond.index)


def _add_month_season(cond: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    out = cond.copy().reset_index(drop=True)
    out["month"] = _ensure_month(out, meta).to_numpy(dtype=int)
    out["au_season"] = out["month"].map(get_au_season)
    return out


def _copula_cfg(cfg: MonthEvtCopulaConfig, seq_len: int, out_dir: Path) -> CopulaConfig:
    return CopulaConfig(
        data_dir="",
        output_dir=str(out_dir),
        seed=int(cfg.seed),
        group_cols="global",
        fallback_group_cols="global",
        min_group_size=1,
        covariance_shrinkage=float(cfg.covariance_shrinkage),
        quantile_grid_size=int(cfg.quantile_grid_size),
        n_per_condition=1,
        seq_len=int(seq_len),
        temporal_smooth_strength=float(cfg.temporal_smooth_strength),
        make_plots=False,
    )


def _fit_group_copulas(
    x_train: np.ndarray,
    cond_train: pd.DataFrame,
    cfg: MonthEvtCopulaConfig,
    out_dir: Path,
    dataset_name: str,
) -> tuple[dict[tuple[str, str], GaussianCopulaGroupModel], pd.DataFrame]:
    cop_cfg = _copula_cfg(cfg, int(x_train.shape[2]), out_dir / "copula_groups")
    models: dict[tuple[str, str], GaussianCopulaGroupModel] = {}
    rows: list[dict] = []

    def fit_one(group_type: str, group_name: str, idx: np.ndarray, used: bool, fallback_to: str, notes: str) -> None:
        rows.append(
            {
                "dataset": dataset_name,
                "group_type": group_type,
                "group_name": group_name,
                "n_train_samples": int(len(idx)),
                "used_for_fit": bool(used),
                "fallback_to": fallback_to,
                "notes": notes,
            }
        )
        if used:
            models[(group_type, group_name)] = GaussianCopulaGroupModel(f"{group_type}:{group_name}", x_train[idx], cop_cfg)

    all_idx = np.arange(len(x_train), dtype=np.int64)
    fit_one("global", "all", all_idx, True, "", "global train-only Copula")

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

    group_df = pd.DataFrame(rows)
    group_df.to_csv(out_dir / "copula_condition_group_summary.csv", index=False, encoding="utf-8-sig")
    return models, group_df


def _choose_copula_model(
    models: dict[tuple[str, str], GaussianCopulaGroupModel],
    month: int,
) -> tuple[GaussianCopulaGroupModel, str, str]:
    month_key = ("month", str(int(month)))
    if month_key in models:
        return models[month_key], "month", str(int(month))
    season = get_au_season(int(month))
    season_key = ("season", season)
    if season_key in models:
        return models[season_key], "season", season
    return models[("global", "all")], "global", "all"


def _net_load(x: np.ndarray) -> np.ndarray:
    return x[:, 0, :] - x[:, 1, :] - x[:, 2, :]


def _compute_monthly_tau(
    x_train: np.ndarray,
    cond_train: pd.DataFrame,
    cfg: MonthEvtCopulaConfig,
    dataset_name: str,
    out_dir: Path,
) -> tuple[dict[int, float], pd.DataFrame]:
    net = _net_load(x_train)
    global_values = net.reshape(-1)
    global_tau = float(np.quantile(global_values, float(cfg.tau_quantile))) if len(global_values) else 0.0
    tau_by_month: dict[int, float] = {}
    rows: list[dict] = []
    months = cond_train["month"].astype(int).to_numpy()
    seasons = cond_train["au_season"].astype(str).to_numpy()
    for month in range(1, 13):
        season = get_au_season(month)
        month_idx = np.where(months == month)[0]
        season_idx = np.where(seasons == season)[0]
        source = "month"
        reason = ""
        if len(month_idx) >= int(cfg.min_month_samples):
            values = net[month_idx].reshape(-1)
        elif len(season_idx) >= int(cfg.min_season_samples):
            values = net[season_idx].reshape(-1)
            source = "season"
            reason = f"month samples {len(month_idx)} < {cfg.min_month_samples}"
        else:
            values = global_values
            source = "global"
            reason = f"month samples {len(month_idx)} and season samples {len(season_idx)} are insufficient"
        tau = float(np.quantile(values, float(cfg.tau_quantile))) if len(values) else global_tau
        tau_by_month[month] = tau
        rows.append(
            {
                "dataset": dataset_name,
                "month": month,
                "season": season,
                "n_samples": int(len(month_idx)),
                "imbalance_tau": tau,
                "source_used": source,
                "fallback_reason": reason,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "monthly_tau_summary.csv", index=False, encoding="utf-8-sig")
    return tau_by_month, df


def _risk_metrics_for_samples(
    x: np.ndarray,
    months: np.ndarray,
    tau_by_month: dict[int, float],
    event_mask: np.ndarray | None,
    delta_t: float = 1.0,
) -> pd.DataFrame:
    net = _net_load(x)
    rows = []
    for i in range(x.shape[0]):
        month = int(months[i])
        tau = float(tau_by_month.get(month, np.nan))
        if not np.isfinite(tau):
            tau = float(np.nanmean(list(tau_by_month.values())))
        excess = np.maximum(net[i] - tau, 0.0)
        if event_mask is not None:
            mask = np.asarray(event_mask[i], dtype=float)
            if mask.shape[0] != excess.shape[0] or not np.any(mask > 0.5):
                mask = np.ones_like(excess)
        else:
            mask = np.ones_like(excess)
        k = max(1, int(round(3.0 / max(delta_t, 1e-6))))
        ramp = net[i, k:] - net[i, :-k] if k < net.shape[1] else np.array([0.0])
        ramp = np.maximum(ramp, 0.0)
        rows.append(
            {
                "cum_deficit": float(excess.sum() * delta_t),
                "core_cum_deficit": float((excess * mask).sum() * delta_t),
                "netload_ramp_max": float(np.max(ramp) if ramp.size else 0.0),
                "imbalance_duration": float((net[i] > tau).sum() * delta_t),
            }
        )
    return pd.DataFrame(rows)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, np.clip(float(q), 0.0, 1.0)))


def _safe_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float(max(fallback, 1.0))
    q75, q25 = np.quantile(arr, [0.75, 0.25])
    iqr = float(q75 - q25)
    std = float(np.std(arr))
    scale = iqr if iqr > 1e-6 else std
    return float(scale if scale > 1e-6 else max(fallback, 1.0))


def _build_train_risk_table(
    x_train: np.ndarray,
    cond_train: pd.DataFrame,
    mask_train: np.ndarray | None,
    tau_by_month: dict[int, float],
    cfg: MonthEvtCopulaConfig,
) -> pd.DataFrame:
    risk = _risk_metrics_for_samples(
        x_train,
        cond_train["month"].astype(int).to_numpy(),
        tau_by_month,
        mask_train,
        float(cfg.delta_t_hours),
    )
    out = pd.concat([cond_train[["month", "au_season"]].reset_index(drop=True), risk], axis=1)
    return out


def _group_values(
    train_risk: pd.DataFrame,
    month: int,
    cfg: MonthEvtCopulaConfig,
) -> tuple[pd.DataFrame, str, str]:
    month_df = train_risk[train_risk["month"].astype(int) == int(month)]
    if len(month_df) >= int(cfg.min_month_samples):
        return month_df, "month", str(int(month))
    season = get_au_season(int(month))
    season_df = train_risk[train_risk["au_season"].astype(str) == season]
    if len(season_df) >= int(cfg.min_season_samples):
        return season_df, "season", season
    return train_risk, "global", "all"


def _target_for_condition(
    sample_id: int,
    row: pd.Series,
    train_risk: pd.DataFrame,
    cfg: MonthEvtCopulaConfig,
) -> tuple[dict[str, float | int | str], dict[str, float]]:
    month = int(row.get("month", 1))
    season = get_au_season(month)
    p = pd.to_numeric(pd.Series([row.get("extreme_prob", np.nan)]), errors="coerce").iloc[0]
    if not np.isfinite(p):
        p = 0.9
    q = float(np.clip(p, 0.50, 0.99))
    group_df, group_type, group_name = _group_values(train_risk, month, cfg)
    metrics = ["cum_deficit", "core_cum_deficit", "netload_ramp_max", "imbalance_duration"]
    target = {m: _safe_quantile(group_df[m].to_numpy(dtype=float), q) for m in metrics}
    scales = {m: _safe_scale(group_df[m].to_numpy(dtype=float), fallback=_safe_scale(train_risk[m].to_numpy(dtype=float))) for m in metrics}
    info = {
        "sample_id": int(sample_id),
        "month": month,
        "season": season,
        "extreme_prob": float(p),
        "target_cum_deficit": target["cum_deficit"],
        "target_core_cum_deficit": target["core_cum_deficit"],
        "target_ramp": target["netload_ramp_max"],
        "target_duration": target["imbalance_duration"],
        "target_group_type": group_type,
        "target_group_name": group_name,
        "quantile_level_used": q,
    }
    return info, scales


def _score_candidates(candidate_metrics: pd.DataFrame, target_info: dict, scales: dict[str, float], weights: tuple[float, float, float, float]) -> np.ndarray:
    w_cum, w_core, w_ramp, w_dur = weights
    err_cum = np.abs(candidate_metrics["cum_deficit"].to_numpy(float) - float(target_info["target_cum_deficit"])) / max(scales["cum_deficit"], 1e-6)
    err_core = np.abs(candidate_metrics["core_cum_deficit"].to_numpy(float) - float(target_info["target_core_cum_deficit"])) / max(scales["core_cum_deficit"], 1e-6)
    err_ramp = np.abs(candidate_metrics["netload_ramp_max"].to_numpy(float) - float(target_info["target_ramp"])) / max(scales["netload_ramp_max"], 1e-6)
    err_dur = np.abs(candidate_metrics["imbalance_duration"].to_numpy(float) - float(target_info["target_duration"])) / max(scales["imbalance_duration"], 1e-6)
    return w_cum * err_cum + w_core * err_core + w_ramp * err_ramp + w_dur * err_dur


def _generate_candidate_pool(
    split: str,
    cond: pd.DataFrame,
    event_mask: np.ndarray | None,
    models: dict[tuple[str, str], GaussianCopulaGroupModel],
    train_risk: pd.DataFrame,
    tau_by_month: dict[int, float],
    cfg: MonthEvtCopulaConfig,
    cop_cfg: CopulaConfig,
    rng: np.random.Generator,
    out_dir: Path,
    dataset_name: str,
) -> tuple[np.ndarray, pd.DataFrame, list[dict], list[dict]]:
    all_candidates = []
    metric_frames = []
    target_rows = []
    selection_base_rows = []
    k = int(cfg.k_candidates)
    for i, row in cond.reset_index(drop=True).iterrows():
        month = int(row.get("month", 1))
        model, group_type, group_name = _choose_copula_model(models, month)
        raw = model.sample_raw(k, rng)
        corrected = temporal_correction(raw, model, cop_cfg)
        cond_rows = pd.DataFrame([row] * k)
        candidates = physical_projection_np(corrected, cond_rows=cond_rows, cfg=cop_cfg).astype(np.float32)
        mask_i = None
        if event_mask is not None:
            mask_i = np.repeat(event_mask[i : i + 1], k, axis=0)
        cand_metrics = _risk_metrics_for_samples(
            candidates,
            np.full((k,), month, dtype=int),
            tau_by_month,
            mask_i,
            float(cfg.delta_t_hours),
        )
        cand_metrics.insert(0, "candidate_index", np.arange(k, dtype=int))
        cand_metrics.insert(0, "sample_id", int(i))
        metric_frames.append(cand_metrics)
        all_candidates.append(candidates)
        target_info, _ = _target_for_condition(i, row, train_risk, cfg)
        target_info["dataset"] = dataset_name
        target_rows.append(target_info)
        selection_base_rows.append(
            {
                "dataset": dataset_name,
                "sample_id": int(i),
                "month": month,
                "season": get_au_season(month),
                "copula_group_used": f"{group_type}:{group_name}",
                "target_group_used": f"{target_info['target_group_type']}:{target_info['target_group_name']}",
                "extreme_prob": target_info["extreme_prob"],
            }
        )
    candidates_arr = np.stack(all_candidates, axis=0) if all_candidates else np.empty((0, k, 3, 0), dtype=np.float32)
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return candidates_arr, metrics_df, target_rows, selection_base_rows


def _select_candidates(
    candidates: np.ndarray,
    candidate_metrics: pd.DataFrame,
    target_rows: list[dict],
    selection_base_rows: list[dict],
    train_risk: pd.DataFrame,
    cond: pd.DataFrame,
    weights: tuple[float, float, float, float],
    cfg: MonthEvtCopulaConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = []
    logs = []
    for i, row in cond.reset_index(drop=True).iterrows():
        target_info, scales = _target_for_condition(i, row, train_risk, cfg)
        sub = candidate_metrics[candidate_metrics["sample_id"].astype(int) == int(i)].reset_index(drop=True)
        scores = _score_candidates(sub, target_info, scales, weights)
        j = int(np.nanargmin(scores)) if len(scores) else 0
        selected.append(candidates[i, j])
        base = dict(selection_base_rows[i])
        base.update(
            {
                "selected_candidate_index": j,
                "selected_score": float(scores[j]),
                "selected_cum": float(sub.loc[j, "cum_deficit"]),
                "selected_core_cum": float(sub.loc[j, "core_cum_deficit"]),
                "selected_ramp": float(sub.loc[j, "netload_ramp_max"]),
                "selected_duration": float(sub.loc[j, "imbalance_duration"]),
                "target_cum": float(target_info["target_cum_deficit"]),
                "target_core_cum": float(target_info["target_core_cum_deficit"]),
                "target_ramp": float(target_info["target_ramp"]),
                "target_duration": float(target_info["target_duration"]),
                "w_cum": float(weights[0]),
                "w_core": float(weights[1]),
                "w_ramp": float(weights[2]),
                "w_duration": float(weights[3]),
            }
        )
        logs.append(base)
    return np.asarray(selected, dtype=np.float32), pd.DataFrame(logs)


def _evaluate_split_row(method: str, generated: Path, data_dir: Path, out_dir: Path, split: str) -> dict:
    eval_dir = out_dir / "evaluations" / f"{method}_{split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    mask_path = data_dir / f"event_mask_{split}.npy"
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / f"X_{split}.npy"),
            generated=str(generated),
            cond=str(data_dir / f"cond_{split}.csv"),
            meta=str(data_dir / f"meta_{split}.csv"),
            event_mask=str(mask_path) if mask_path.exists() else None,
            out_dir=str(eval_dir),
            model_name=method,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = {"method": method}
    row.update({metric: summary["metrics"].get(metric, np.nan) for metric in FULL_COMPARE_METRICS})
    return row


def _select_adaptive_weights(
    data_dir: Path,
    out_dir: Path,
    candidates_val: np.ndarray,
    metrics_val: pd.DataFrame,
    target_rows_val: list[dict],
    selection_base_val: list[dict],
    train_risk: pd.DataFrame,
    cond_val: pd.DataFrame,
    cfg: MonthEvtCopulaConfig,
) -> tuple[tuple[float, float, float, float], pd.DataFrame]:
    weight_grid = _parse_weight_grid(cfg.adaptive_weight_grid)
    rows = []
    generated_dir = out_dir / "val_weight_candidates"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for weights in weight_grid:
        gen, _ = _select_candidates(candidates_val, metrics_val, target_rows_val, selection_base_val, train_risk, cond_val, weights, cfg)
        name = f"val_w_{weights[0]:.2f}_{weights[1]:.2f}_{weights[2]:.2f}_{weights[3]:.2f}".replace(".", "p")
        path = generated_dir / f"{name}.npy"
        np.save(path, gen)
        row = _evaluate_split_row(name, path, data_dir, out_dir, split="val")
        row.update({"w_cum": weights[0], "w_core": weights[1], "w_ramp": weights[2], "w_duration": weights[3]})
        rows.append(row)
    df = add_risk_score(pd.DataFrame(rows))
    df = df.rename(
        columns={
            "q99_cum_deficit_error": "val_q99_cum_deficit_error",
            "core_q99_cum_deficit_error": "val_core_q99_cum_deficit_error",
            "netload_ramp_max_mae": "val_netload_ramp_max_mae",
            "imbalance_duration_mae": "val_imbalance_duration_mae",
            "risk_score": "val_risk_score",
            "risk_rank": "val_risk_rank",
        }
    )
    best_idx = df.sort_values(["val_risk_score", "val_risk_rank"], na_position="last").index[0]
    df["selected"] = False
    df.loc[best_idx, "selected"] = True
    df.to_csv(out_dir / "adaptive_weight_selection.csv", index=False, encoding="utf-8-sig")
    best = df.loc[best_idx]
    weights = (float(best["w_cum"]), float(best["w_core"]), float(best["w_ramp"]), float(best["w_duration"]))
    return weights, df


def _existing_sources_for_dataset(spec: DatasetSpec) -> dict[str, Path]:
    simple_sources = {
        "singleton": BASE_DIR / "results" / "copula_guided_residual_diffusion_valprotected" / "singleton" / "generated_samples_Simple_EVT_Risk_Diffusion.npy",
        "muswellbrook": BASE_DIR / "results" / "muswellbrook_all_methods_compare" / "generated_samples_Simple_EVT_Risk_Diffusion.npy",
        "cessnock_or_newarea": BASE_DIR / "results" / "cessnock_south_all_methods_compare" / "generated_samples_Simple_EVT_Risk_Diffusion.npy",
    }
    if spec.existing_sources:
        out = dict(spec.existing_sources)
        out.setdefault("Simple_EVT_Risk_Diffusion", simple_sources.get(spec.out_name, Path("__missing__")))
        return out
    if spec.existing_results_dir and spec.existing_results_dir.exists():
        names = {
            "Simple_EVT_Risk_Diffusion": "generated_samples_Simple_EVT_Risk_Diffusion.npy",
            "enhanced_gan": "generated_samples_enhanced_gan.npy",
            "plain_diffusion_baseline": "generated_samples_plain_diffusion_baseline.npy",
            "improved_diffusion": "generated_samples_improved_diffusion.npy",
            "proposed_E0": "generated_samples_proposed_E0.npy",
            "JRPD_best_3h": "generated_samples_JRPD_best_3h.npy",
        }
        return {name: spec.existing_results_dir / filename for name, filename in names.items()}
    return {}


def _evaluate_method(name: str, generated: Path, data_dir: Path, out_dir: Path) -> dict:
    return evaluate_method(name, generated, data_dir, out_dir)


def run_one_dataset(spec: DatasetSpec, cfg: MonthEvtCopulaConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = cfg.out_dir / spec.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = spec.data_dir
    rng = np.random.default_rng(int(cfg.seed))

    x_train, cond_train_raw, meta_train, mask_train = _load_split(data_dir, "train")
    x_val, cond_val_raw, meta_val, mask_val = _load_split(data_dir, "val")
    x_test, cond_test_raw, meta_test, mask_test = _load_split(data_dir, "test")
    cond_train = _add_month_season(cond_train_raw, meta_train)
    cond_val = _add_month_season(cond_val_raw, meta_val)
    cond_test = _add_month_season(cond_test_raw, meta_test)

    cop_cfg = _copula_cfg(cfg, int(x_train.shape[2]), out_dir / "copula_groups")
    models, _ = _fit_group_copulas(x_train, cond_train, cfg, out_dir, spec.out_name)
    tau_by_month, _ = _compute_monthly_tau(x_train, cond_train, cfg, spec.out_name, out_dir)
    train_risk = _build_train_risk_table(x_train, cond_train, mask_train, tau_by_month, cfg)

    val_candidates, val_metrics, val_targets, val_base = _generate_candidate_pool(
        "val", cond_val, mask_val, models, train_risk, tau_by_month, cfg, cop_cfg, rng, out_dir, spec.out_name
    )
    fixed_weights = _parse_weights(cfg.fixed_weights)
    adaptive_weights, _ = _select_adaptive_weights(
        data_dir, out_dir, val_candidates, val_metrics, val_targets, val_base, train_risk, cond_val, cfg
    )

    test_candidates, test_metrics, test_targets, test_base = _generate_candidate_pool(
        "test", cond_test, mask_test, models, train_risk, tau_by_month, cfg, cop_cfg, rng, out_dir, spec.out_name
    )
    pd.DataFrame(test_targets).to_csv(out_dir / "risk_target_summary.csv", index=False, encoding="utf-8-sig")
    fixed_gen, fixed_log = _select_candidates(test_candidates, test_metrics, test_targets, test_base, train_risk, cond_test, fixed_weights, cfg)
    adaptive_gen, adaptive_log = _select_candidates(test_candidates, test_metrics, test_targets, test_base, train_risk, cond_test, adaptive_weights, cfg)
    fixed_log.to_csv(out_dir / "candidate_selection_log_fixed.csv", index=False, encoding="utf-8-sig")
    adaptive_log.to_csv(out_dir / "candidate_selection_log_adaptive.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "generated_samples_fixed.npy", fixed_gen.astype(np.float32))
    np.save(out_dir / "generated_samples_adaptive.npy", adaptive_gen.astype(np.float32))

    # Save compact candidate metrics instead of the full [N,K,3,T] array to avoid large artifacts.
    test_metrics.to_csv(out_dir / "candidate_risk_metrics_test.csv", index=False, encoding="utf-8-sig")

    rows: list[dict] = []
    # Re-create the plain train-only Copula baseline in the same directory for fair comparison.
    plain_model = GaussianCopulaGroupModel("global_plain_copula", x_train, cop_cfg)
    plain_samples = []
    for _, row in cond_test.reset_index(drop=True).iterrows():
        raw = plain_model.sample_raw(1, rng)
        corrected = temporal_correction(raw, plain_model, cop_cfg)
        plain_samples.append(physical_projection_np(corrected, pd.DataFrame([row]), cop_cfg)[0])
    plain_path = out_dir / "generated_samples_traditional_gaussian_copula.npy"
    np.save(plain_path, np.asarray(plain_samples, dtype=np.float32))
    rows.append(_evaluate_method("traditional_gaussian_copula", plain_path, data_dir, out_dir))
    rows.append(_evaluate_method(FIXED_METHOD, out_dir / "generated_samples_fixed.npy", data_dir, out_dir))
    rows.append(_evaluate_method(ADAPTIVE_METHOD, out_dir / "generated_samples_adaptive.npy", data_dir, out_dir))

    for method, source in _existing_sources_for_dataset(spec).items():
        if not source.exists() or method in {r["method"] for r in rows}:
            continue
        target = out_dir / f"generated_samples_{method}.npy"
        shutil.copy2(source, target)
        rows.append(_evaluate_method(method, target, data_dir, out_dir))

    df = add_risk_score(pd.DataFrame(rows))
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir, method_col="method")
    _write_method_report(spec, out_dir, risk_main, aux, cfg, adaptive_weights)
    return risk_main, aux


def _write_method_report(
    spec: DatasetSpec,
    out_dir: Path,
    risk_main: pd.DataFrame,
    aux: pd.DataFrame,
    cfg: MonthEvtCopulaConfig,
    adaptive_weights: tuple[float, float, float, float],
) -> None:
    lines = [
        f"# {spec.name} - Month-conditioned EVT-Copula Risk Selection",
        "",
        "## Method",
        "",
        "This method fits train-only monthly, seasonal, and global Gaussian Copula priors. For each condition, it generates K candidate scenarios from the most specific available month/season/global prior, maps `extreme_prob` to train-only empirical risk targets, and selects the candidate with the lowest weighted joint-risk mismatch.",
        "",
        "## Month Conditioning",
        "",
        "- Australian NSW seasons are used: summer=Dec/Jan/Feb, autumn=Mar/Apr/May, winter=Jun/Jul/Aug, spring=Sep/Oct/Nov.",
        "- Copula fallback order: month -> Australian season -> global.",
        "- Monthly imbalance tau fallback order: month -> Australian season -> global.",
        "",
        "## Adaptive Weights",
        "",
        f"- selected weights `(cum, core, ramp, duration)`: {adaptive_weights}",
        f"- K candidates per condition: {cfg.k_candidates}",
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
        "# Final Month-conditioned EVT-Copula Report",
        "",
        "## Method Summary",
        "",
        "Month_EVT_Copula_Risk_Selection uses train-only month/season/global Copula priors, monthly imbalance thresholds, train-only empirical risk targets, and validation-selected risk-selection weights. Test curves are used only for final evaluation.",
        "",
        "## Evaluation Logic",
        "",
        RISK_RANKING_EXPLANATION,
        RISK_EVALUATION_EXPLANATION,
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
        "- `extreme_prob` is interpreted as a monotone extremeness probability and clipped to `[0.50, 0.99]` for empirical risk targets.",
        "- Candidate selection never uses the target test curve; it uses only month, event condition, extreme probability, event mask if available, and train-derived risk distributions.",
    ]
    (root / "final_month_evt_copula_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(cfg: MonthEvtCopulaConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_tables: dict[str, pd.DataFrame] = {}
    aux_tables: dict[str, pd.DataFrame] = {}
    for spec in [s for s in _dataset_specs() if s.data_dir.exists()]:
        print(f"\n=== Dataset: {spec.out_name} ===")
        risk_main, aux = run_one_dataset(spec, cfg)
        dataset_tables[spec.out_name] = risk_main
        aux_tables[spec.out_name] = aux
    _write_global_summaries(cfg.out_dir, dataset_tables, aux_tables)


def parse_args() -> MonthEvtCopulaConfig:
    parser = argparse.ArgumentParser(description="Run month-conditioned EVT-Copula risk-controllable generation.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "month_evt_copula_risk_selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-candidates", type=int, default=20)
    parser.add_argument("--min-month-samples", type=int, default=10)
    parser.add_argument("--min-season-samples", type=int, default=25)
    parser.add_argument("--tau-quantile", type=float, default=0.75)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.08)
    parser.add_argument("--temporal-smooth-strength", type=float, default=0.15)
    parser.add_argument("--fixed-weights", type=str, default="0.35,0.25,0.20,0.20")
    return MonthEvtCopulaConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_all(parse_args())
