from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from risk_ranking_utils import AUXILIARY_REALISM_METRICS, add_risk_score, write_risk_tables
from risk_evt_transfer_augmentation import (
    RiskEVTTransferConfig,
    generate_candidates,
    physical_filter_and_clip,
    train_conditional_wgan_gp,
)
from run_copula_guided_residual_diffusion import DatasetSpec, _dataset_specs
from run_month_evt_copula_risk_selection import (
    FIXED_METHOD as MONTH_FIXED_METHOD,
    _add_month_season,
    _build_train_risk_table,
    _compute_monthly_tau,
    _copula_cfg,
    _evaluate_method,
    _generate_candidate_pool,
    _load_split,
    _parse_weights,
    _select_candidates,
)
from run_tailweighted_month_evt_copula import (
    TAIL_FIXED_METHOD,
    TailWeightedConfig,
    _compute_tail_scores,
    _fit_tailweighted_group_copulas,
)


BASE_DIR = Path(__file__).resolve().parent
METHOD_NAME = "GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed"
CAISO_OUT_NAME = "openenergyhub_caiso_balanced_relaxed"


@dataclass
class GanAugmentedTailWeightedConfig(TailWeightedConfig):
    out_dir: Path = BASE_DIR / "results" / "gan_augmented_tailweighted_copula"
    augmented_data_root: Path = BASE_DIR / "outputs" / "gan_augmented_datasets"
    tail_sample_threshold: float = 0.70
    tail_sample_threshold_relaxed: float = 0.60
    min_tail_samples_after_relax: int = 20
    min_tail_samples_before_relax: int = 30
    gan_candidate_ratio: float = 3.0
    gan_keep_ratio: float = 0.02
    corr_error_threshold: float = 0.20
    corr_error_threshold_relaxed: float = 0.30
    risk_filter_low_quantile: float = 0.70
    risk_filter_low_quantile_relaxed: float = 0.60
    risk_filter_high_quantile: float = 0.995
    ramp_upper_factor: float = 1.20
    risk_filter_min_keep: int = 2
    risk_fallback_priority_weight: float = 0.75
    gan_target_cum_quantile: float = 0.85
    gan_target_core_quantile: float = 0.85
    gan_target_duration_quantile: float = 0.75
    gan_ramp_soft_max_quantile: float = 0.75
    gan_duration_soft_max_quantile: float = 0.95
    gan_epochs: int = 120
    gan_epochs_small: int = 80
    gan_batch_size: int = 32
    gan_noise_dim: int = 64
    gan_base_channels: int = 32
    gan_n_critic: int = 5
    gan_gp_lambda: float = 10.0
    gan_lr: float = 1.0e-4
    gan_keep_min: int = 1
    corr_filter_min_keep: int = 1
    corr_error_threshold_adaptive_factor: float = 1.35


def _build_dataset_specs_four() -> list[DatasetSpec]:
    specs = list(_dataset_specs())
    caiso_sources = {
        "traditional_gaussian_copula": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_traditional_gaussian_copula.npy",
        MONTH_FIXED_METHOD: BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_Month_EVT_Copula_Risk_Selection_Fixed.npy",
        "Month_EVT_Copula_Risk_Selection_Adaptive": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_Month_EVT_Copula_Risk_Selection_Adaptive.npy",
        TAIL_FIXED_METHOD: BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_tailweighted_fixed.npy",
        "TailWeighted_Month_EVT_Copula_Risk_Selection_Adaptive": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_tailweighted_adaptive.npy",
        "Simple_EVT_Risk_Diffusion": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_Simple_EVT_Risk_Diffusion.npy",
        "enhanced_gan": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_enhanced_gan.npy",
        "improved_diffusion": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_improved_diffusion.npy",
        "proposed_E0": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_proposed_E0.npy",
        "plain_diffusion_baseline": BASE_DIR
        / "results"
        / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y"
        / CAISO_OUT_NAME
        / "generated_samples_plain_diffusion_baseline.npy",
    }
    specs.append(
        DatasetSpec(
            name="OpenEnergyHub CAISO balanced_relaxed",
            out_name=CAISO_OUT_NAME,
            data_dir=BASE_DIR / "outputs" / "openenergyhub_caiso_threshold_sweep" / "balanced_relaxed" / "dataset",
            existing_sources=caiso_sources,
        )
    )
    return [spec for spec in specs if spec.data_dir.exists()]


def _baseline_results_dir(spec: DatasetSpec) -> Path:
    if spec.out_name == CAISO_OUT_NAME:
        return BASE_DIR / "results" / "openenergyhub_caiso_balanced_relaxed_tailweighted_month_evt_copula_3y" / CAISO_OUT_NAME
    return BASE_DIR / "results" / "tailweighted_month_evt_copula" / spec.out_name


def _existing_baseline_paths(spec: DatasetSpec) -> dict[str, Path]:
    base = _baseline_results_dir(spec)
    return {
        "traditional_gaussian_copula": base / "generated_samples_traditional_gaussian_copula.npy",
        MONTH_FIXED_METHOD: base / "generated_samples_Month_EVT_Copula_Risk_Selection_Fixed.npy",
        "Month_EVT_Copula_Risk_Selection_Adaptive": base / "generated_samples_Month_EVT_Copula_Risk_Selection_Adaptive.npy",
        TAIL_FIXED_METHOD: base / "generated_samples_tailweighted_fixed.npy",
        "TailWeighted_Month_EVT_Copula_Risk_Selection_Adaptive": base / "generated_samples_tailweighted_adaptive.npy",
        "Simple_EVT_Risk_Diffusion": base / "generated_samples_Simple_EVT_Risk_Diffusion.npy",
        "enhanced_gan": base / "generated_samples_enhanced_gan.npy",
        "improved_diffusion": base / "generated_samples_improved_diffusion.npy",
        "proposed_E0": base / "generated_samples_proposed_E0.npy",
        "JRPD_best_3h": base / "generated_samples_JRPD_best_3h.npy",
        "plain_diffusion_baseline": base / "generated_samples_plain_diffusion_baseline.npy",
    }


def _merged_meta(cond: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merged = cond.reset_index(drop=True).copy()
    meta = meta.reset_index(drop=True).copy()
    for col in meta.columns:
        if col not in merged.columns:
            merged[col] = meta[col].to_numpy()
    merged["_source_index"] = np.arange(len(merged), dtype=int)
    if "sample_id" in merged.columns:
        merged["source_sample_id"] = merged["sample_id"].astype(str)
    else:
        merged["source_sample_id"] = merged["_source_index"].astype(str)
    return merged


def _to_channel_time(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[2] == 3:
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)
    raise ValueError(f"Cannot infer channel axis from {arr.shape}")


def _to_time_channel(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    if arr.shape[2] == 3:
        return arr
    if arr.shape[1] == 3:
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)
    raise ValueError(f"Cannot infer channel axis from {arr.shape}")


def _match_reference_orientation(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference)
    arr = np.asarray(x, dtype=np.float32)
    if ref.ndim != 3:
        raise ValueError("Reference must be 3D.")
    if ref.shape[1] == 3:
        return _to_channel_time(arr)
    if ref.shape[2] == 3:
        return _to_time_channel(arr)
    raise ValueError(f"Cannot infer reference orientation from {ref.shape}")


def _rank_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or (np.nanmax(arr) - np.nanmin(arr) <= 1e-12):
        return np.zeros_like(arr, dtype=float)
    order = np.argsort(np.argsort(arr))
    denom = max(len(arr) - 1, 1)
    return order.astype(float) / float(denom)


def _tail_threshold_and_index(tail_score: np.ndarray, cfg: GanAugmentedTailWeightedConfig) -> tuple[float, np.ndarray, str]:
    threshold = float(cfg.tail_sample_threshold)
    idx = np.where(tail_score >= threshold)[0]
    note = f"initial threshold={threshold:.2f}"
    if len(idx) < int(cfg.min_tail_samples_before_relax):
        threshold = float(cfg.tail_sample_threshold_relaxed)
        idx = np.where(tail_score >= threshold)[0]
        note = f"relaxed threshold={threshold:.2f} because initial tail count was {len(np.where(tail_score >= cfg.tail_sample_threshold)[0])}"
    return threshold, idx, note


def _json_counts(series: pd.Series) -> str:
    counts = series.astype(str).value_counts(dropna=False).to_dict()
    return json.dumps(counts, ensure_ascii=False)


def _build_wgan_cfg(
    cfg: GanAugmentedTailWeightedConfig,
    data_dir: Path,
    out_dir: Path,
    x_train: np.ndarray,
    n_tail: int,
) -> RiskEVTTransferConfig:
    epochs = int(cfg.gan_epochs_small if n_tail < 60 else cfg.gan_epochs)
    x_ct = _to_channel_time(x_train)
    channel_max = np.max(x_ct, axis=(0, 2)).astype(float).tolist()
    return RiskEVTTransferConfig(
        data_dir=str(data_dir),
        out_dir=str(out_dir / "gan_tail_augmentor"),
        seq_len=int(x_ct.shape[2]),
        seed=int(cfg.seed),
        device="cpu",
        batch_size=int(min(cfg.gan_batch_size, max(4, n_tail))),
        noise_dim=int(cfg.gan_noise_dim),
        gan_base_channels=int(cfg.gan_base_channels),
        n_critic=int(cfg.gan_n_critic),
        gp_lambda=float(cfg.gan_gp_lambda),
        lr_g=float(cfg.gan_lr),
        lr_d=float(cfg.gan_lr),
        wgan_epochs=epochs,
        candidate_multiplier=int(np.ceil(cfg.gan_candidate_ratio)),
        channel_upper_factor=1.05,
        solar_night_zero=False,
        channel_max_train=channel_max,
    )


def _compute_corr_matrix(x: np.ndarray) -> np.ndarray:
    x_ct = _to_channel_time(x)
    flat = np.transpose(x_ct, (0, 2, 1)).reshape(-1, x_ct.shape[1])
    if flat.shape[0] <= 1:
        return np.eye(x_ct.shape[1], dtype=float)
    corr = np.corrcoef(flat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr.astype(float)


def _corr_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    corr_ref = _compute_corr_matrix(reference)
    corr_cand = _compute_corr_matrix(candidate)
    return float(np.linalg.norm(corr_cand - corr_ref, ord="fro"))


def _risk_filter_candidates(
    candidate_x: np.ndarray,
    candidate_meta: pd.DataFrame,
    candidate_mask: np.ndarray | None,
    train_risk: pd.DataFrame,
    tau_by_month: dict[int, float],
    cfg: GanAugmentedTailWeightedConfig,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None, pd.DataFrame, str]:
    months = pd.to_numeric(candidate_meta.get("month", 1), errors="coerce").fillna(1).astype(int).to_numpy()
    # Use the true month-based tau from the upstream pipeline logic if available in cond/meta.
    if "imbalance_tau" in candidate_meta.columns:
        tau_override = pd.to_numeric(candidate_meta["imbalance_tau"], errors="coerce").fillna(np.nan).to_numpy()
    else:
        tau_override = None
    risk = _build_candidate_risk_table(candidate_x, months, candidate_mask, train_risk, tau_by_month, cfg, tau_override)

    def bounds_for_low_q(low_q: float) -> dict[str, float]:
        cum_low = float(np.quantile(train_risk["cum_deficit"], low_q))
        core_low = float(np.quantile(train_risk["core_cum_deficit"], low_q))
        cum_high = float(np.quantile(train_risk["cum_deficit"], float(cfg.risk_filter_high_quantile)))
        core_high = float(np.quantile(train_risk["core_cum_deficit"], float(cfg.risk_filter_high_quantile)))
        ramp_high = float(np.quantile(train_risk["netload_ramp_max"], float(cfg.risk_filter_high_quantile))) * float(cfg.ramp_upper_factor)
        dur_high = float(np.max(train_risk["imbalance_duration"]))
        return {
            "cum_low": cum_low,
            "core_low": core_low,
            "cum_high": cum_high,
            "core_high": core_high,
            "ramp_high": ramp_high,
            "dur_high": max(dur_high, float(cfg.delta_t_hours)),
        }

    def filter_with_bounds(bounds: dict[str, float]) -> np.ndarray:
        keep = (
            (risk["cum_deficit"].to_numpy(float) >= bounds["cum_low"])
            & (risk["cum_deficit"].to_numpy(float) <= bounds["cum_high"])
            & (risk["core_cum_deficit"].to_numpy(float) >= bounds["core_low"])
            & (risk["core_cum_deficit"].to_numpy(float) <= bounds["core_high"])
            & (risk["netload_ramp_max"].to_numpy(float) <= bounds["ramp_high"])
            & (risk["imbalance_duration"].to_numpy(float) <= bounds["dur_high"])
        )
        return keep

    def soft_violation_score(bounds: dict[str, float]) -> np.ndarray:
        eps = 1e-6
        cum = risk["cum_deficit"].to_numpy(float)
        core = risk["core_cum_deficit"].to_numpy(float)
        ramp = risk["netload_ramp_max"].to_numpy(float)
        dur = risk["imbalance_duration"].to_numpy(float)

        def lower_violation(values: np.ndarray, lo: float) -> np.ndarray:
            scale = max(abs(lo), 1.0)
            return np.maximum(lo - values, 0.0) / (scale + eps)

        def upper_violation(values: np.ndarray, hi: float) -> np.ndarray:
            scale = max(abs(hi), 1.0)
            return np.maximum(values - hi, 0.0) / (scale + eps)

        return (
            lower_violation(cum, bounds["cum_low"])
            + upper_violation(cum, bounds["cum_high"])
            + lower_violation(core, bounds["core_low"])
            + upper_violation(core, bounds["core_high"])
            + upper_violation(ramp, bounds["ramp_high"])
            + upper_violation(dur, bounds["dur_high"])
        ).astype(float)

    def tail_candidate_priority_score() -> np.ndarray:
        cum = risk["cum_deficit"].to_numpy(float)
        core = risk["core_cum_deficit"].to_numpy(float)
        ramp = risk["netload_ramp_max"].to_numpy(float)
        dur = risk["imbalance_duration"].to_numpy(float)

        def scaled_abs_to_quantile(values: np.ndarray, train_col: str, q: float) -> np.ndarray:
            train = train_risk[train_col].to_numpy(float)
            target = float(np.quantile(train, q))
            q75, q25 = np.quantile(train, [0.75, 0.25])
            scale = float(q75 - q25)
            if scale <= 1e-6:
                scale = float(np.std(train))
            scale = scale if scale > 1e-6 else 1.0
            return np.abs(values - target) / scale

        def upper_quantile_penalty(values: np.ndarray, train_col: str, q: float) -> np.ndarray:
            train = train_risk[train_col].to_numpy(float)
            limit = float(np.quantile(train, q))
            scale = max(abs(limit), 1.0)
            return np.maximum(values - limit, 0.0) / scale

        return (
            0.35 * scaled_abs_to_quantile(cum, "cum_deficit", float(cfg.gan_target_cum_quantile))
            + 0.30 * scaled_abs_to_quantile(core, "core_cum_deficit", float(cfg.gan_target_core_quantile))
            + 0.15 * scaled_abs_to_quantile(dur, "imbalance_duration", float(cfg.gan_target_duration_quantile))
            + 0.10 * upper_quantile_penalty(ramp, "netload_ramp_max", float(cfg.gan_ramp_soft_max_quantile))
            + 0.10 * upper_quantile_penalty(dur, "imbalance_duration", float(cfg.gan_duration_soft_max_quantile))
        ).astype(float)

    keep = filter_with_bounds(bounds_for_low_q(float(cfg.risk_filter_low_quantile)))
    note = f"risk filter low quantile={cfg.risk_filter_low_quantile:.2f}"
    if int(keep.sum()) == 0:
        relaxed_bounds = bounds_for_low_q(float(cfg.risk_filter_low_quantile_relaxed))
        keep = filter_with_bounds(relaxed_bounds)
        note = f"risk filter relaxed to low quantile={cfg.risk_filter_low_quantile_relaxed:.2f}"
        min_keep = min(int(cfg.risk_filter_min_keep), len(candidate_x))
        if int(keep.sum()) < min_keep and len(candidate_x) > 0:
            violation = soft_violation_score(relaxed_bounds)
            priority = tail_candidate_priority_score()
            combined = violation + float(cfg.risk_fallback_priority_weight) * priority
            top_idx = np.argsort(combined)[:min_keep]
            keep = np.zeros(len(candidate_x), dtype=bool)
            keep[top_idx] = True
            note = (
                f"risk filter adaptive fallback kept top {min_keep} candidates "
                f"by relaxed-bound violation plus tail-priority score after low quantile={cfg.risk_filter_low_quantile_relaxed:.2f}"
            )
    return candidate_x[keep], candidate_meta.loc[keep].reset_index(drop=True), (candidate_mask[keep] if candidate_mask is not None else None), risk.loc[keep].reset_index(drop=True), note


def _build_candidate_risk_table(
    x: np.ndarray,
    months: np.ndarray,
    event_mask: np.ndarray | None,
    train_risk: pd.DataFrame,
    tau_by_month: dict[int, float],
    cfg: GanAugmentedTailWeightedConfig,
    tau_override: np.ndarray | None = None,
) -> pd.DataFrame:
    x_ct = _to_channel_time(x)
    net = x_ct[:, 0, :] - x_ct[:, 1, :] - x_ct[:, 2, :]
    k = max(1, int(round(3.0 / max(float(cfg.delta_t_hours), 1e-6))))
    rows: list[dict] = []
    month_tau_fallback = {int(k): float(v) for k, v in tau_by_month.items()}
    global_tau = float(np.median(list(month_tau_fallback.values()))) if month_tau_fallback else 0.0
    for i in range(x_ct.shape[0]):
        month = int(months[i]) if len(months) else 1
        tau = float(tau_override[i]) if tau_override is not None and np.isfinite(tau_override[i]) else float(month_tau_fallback.get(month, global_tau))
        excess = np.maximum(net[i] - tau, 0.0)
        if event_mask is not None and i < len(event_mask):
            mask = np.asarray(event_mask[i], dtype=float)
            if mask.shape[0] != excess.shape[0]:
                mask = np.ones_like(excess)
        else:
            mask = np.ones_like(excess)
        ramp = np.maximum(net[i, k:] - net[i, :-k], 0.0) if k < net.shape[1] else np.zeros((1,), dtype=float)
        rows.append(
            {
                "cum_deficit": float(excess.sum() * float(cfg.delta_t_hours)),
                "core_cum_deficit": float((excess * mask).sum() * float(cfg.delta_t_hours)),
                "netload_ramp_max": float(np.max(ramp) if ramp.size else 0.0),
                "imbalance_duration": float((net[i] > tau).sum() * float(cfg.delta_t_hours)),
            }
        )
    return pd.DataFrame(rows)


def _risk_match_scores(candidate_risk: pd.DataFrame, tail_real_risk: pd.DataFrame, full_train_risk: pd.DataFrame) -> np.ndarray:
    def safe_scale(col: str) -> float:
        arr = full_train_risk[col].to_numpy(dtype=float)
        if len(arr) <= 1:
            return 1.0
        q75, q25 = np.quantile(arr, [0.75, 0.25])
        scale = float(q75 - q25)
        if scale <= 1e-6:
            scale = float(np.std(arr))
        return scale if scale > 1e-6 else 1.0

    scales = {col: safe_scale(col) for col in ["cum_deficit", "core_cum_deficit", "netload_ramp_max", "imbalance_duration"]}
    targets = {
        "cum_deficit": float(np.quantile(full_train_risk["cum_deficit"].to_numpy(float), 0.85)),
        "core_cum_deficit": float(np.quantile(full_train_risk["core_cum_deficit"].to_numpy(float), 0.85)),
        "netload_ramp_max": float(np.quantile(full_train_risk["netload_ramp_max"].to_numpy(float), 0.75)),
        "imbalance_duration": float(np.quantile(full_train_risk["imbalance_duration"].to_numpy(float), 0.75)),
    }
    ramp_limit = float(np.quantile(full_train_risk["netload_ramp_max"].to_numpy(float), 0.90))
    duration_limit = float(np.quantile(full_train_risk["imbalance_duration"].to_numpy(float), 0.95))
    scores = (
        0.35 * np.abs(candidate_risk["cum_deficit"].to_numpy(float) - targets["cum_deficit"]) / scales["cum_deficit"]
        + 0.30 * np.abs(candidate_risk["core_cum_deficit"].to_numpy(float) - targets["core_cum_deficit"]) / scales["core_cum_deficit"]
        + 0.15 * np.abs(candidate_risk["imbalance_duration"].to_numpy(float) - targets["imbalance_duration"]) / scales["imbalance_duration"]
        + 0.10 * np.maximum(candidate_risk["netload_ramp_max"].to_numpy(float) - ramp_limit, 0.0) / scales["netload_ramp_max"]
        + 0.10 * np.maximum(candidate_risk["imbalance_duration"].to_numpy(float) - duration_limit, 0.0) / scales["imbalance_duration"]
    )
    return scores.astype(float)


def _correlation_filter_candidates(
    x_candidates: np.ndarray,
    meta_candidates: pd.DataFrame,
    mask_candidates: np.ndarray | None,
    risk_candidates: pd.DataFrame,
    x_tail_real: np.ndarray,
    tail_real_risk: pd.DataFrame,
    full_train_risk: pd.DataFrame,
    n_keep: int,
    cfg: GanAugmentedTailWeightedConfig,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None, pd.DataFrame, dict]:
    if len(x_candidates) == 0 or n_keep <= 0:
        empty = x_candidates[:0]
        empty_meta = meta_candidates.iloc[:0].copy()
        empty_mask = mask_candidates[:0] if mask_candidates is not None else None
        empty_risk = risk_candidates.iloc[:0].copy()
        summary = {
            "corr_threshold_used": float(cfg.corr_error_threshold),
            "corr_error_before": np.nan,
            "corr_error_after": np.nan,
            "corr_filter_note": "no candidates after risk filter",
        }
        return empty, empty_meta, empty_mask, empty_risk, summary

    scores = _risk_match_scores(risk_candidates, tail_real_risk, full_train_risk)
    order = np.argsort(scores)
    risk_sorted_x = x_candidates[order]
    risk_sorted_meta = meta_candidates.iloc[order].reset_index(drop=True)
    risk_sorted_risk = risk_candidates.iloc[order].reset_index(drop=True)
    risk_sorted_mask = mask_candidates[order] if mask_candidates is not None else None
    trial_top = min(len(risk_sorted_x), n_keep)
    corr_error_before = _corr_error(x_tail_real, np.concatenate([x_tail_real, _to_channel_time(risk_sorted_x[:trial_top])], axis=0))

    def greedy(threshold: float):
        kept_idx: list[int] = []
        current = _to_channel_time(x_tail_real).copy()
        for idx in range(len(risk_sorted_x)):
            if len(kept_idx) >= n_keep:
                break
            candidate_ct = _to_channel_time(risk_sorted_x[idx : idx + 1])
            trial = np.concatenate([current, candidate_ct], axis=0)
            err = _corr_error(x_tail_real, trial)
            if err <= threshold:
                kept_idx.append(idx)
                current = trial
        return kept_idx, current

    def greedy_min_error(target_keep: int):
        kept_idx: list[int] = []
        current = _to_channel_time(x_tail_real).copy()
        available = list(range(len(risk_sorted_x)))
        while available and len(kept_idx) < target_keep:
            best_idx = None
            best_trial = None
            best_err = None
            for idx in available:
                candidate_ct = _to_channel_time(risk_sorted_x[idx : idx + 1])
                trial = np.concatenate([current, candidate_ct], axis=0)
                err = _corr_error(x_tail_real, trial)
                if best_err is None or err < best_err:
                    best_idx = idx
                    best_trial = trial
                    best_err = err
            if best_idx is None or best_trial is None:
                break
            kept_idx.append(best_idx)
            current = best_trial
            available.remove(best_idx)
        return kept_idx, current

    kept_idx, current = greedy(float(cfg.corr_error_threshold))
    threshold_used = float(cfg.corr_error_threshold)
    note = "strict corr threshold"
    if len(kept_idx) == 0 and len(risk_sorted_x) > 0:
        kept_idx, current = greedy(float(cfg.corr_error_threshold_relaxed))
        threshold_used = float(cfg.corr_error_threshold_relaxed)
        note = "relaxed corr threshold"
    min_keep = min(int(cfg.corr_filter_min_keep), int(n_keep), len(risk_sorted_x))
    if len(kept_idx) < min_keep and len(risk_sorted_x) > 0:
        adaptive_idx, adaptive_current = greedy_min_error(min_keep)
        if len(adaptive_idx) > len(kept_idx):
            kept_idx = adaptive_idx
            current = adaptive_current
            adaptive_after = _corr_error(x_tail_real, current)
            threshold_used = max(float(cfg.corr_error_threshold_relaxed), float(adaptive_after) * float(cfg.corr_error_threshold_adaptive_factor))
            note = f"adaptive corr fallback kept {len(kept_idx)} lowest-error candidates"

    kept_x = risk_sorted_x[kept_idx] if kept_idx else risk_sorted_x[:0]
    kept_meta = risk_sorted_meta.iloc[kept_idx].reset_index(drop=True) if kept_idx else risk_sorted_meta.iloc[:0].copy()
    kept_mask = risk_sorted_mask[kept_idx] if (risk_sorted_mask is not None and kept_idx) else (risk_sorted_mask[:0] if risk_sorted_mask is not None else None)
    kept_risk = risk_sorted_risk.iloc[kept_idx].reset_index(drop=True) if kept_idx else risk_sorted_risk.iloc[:0].copy()
    corr_error_after = _corr_error(x_tail_real, current) if len(kept_idx) else np.nan
    summary = {
        "corr_threshold_used": threshold_used,
        "corr_error_before": float(corr_error_before),
        "corr_error_after": float(corr_error_after) if np.isfinite(corr_error_after) else np.nan,
        "corr_filter_note": note,
    }
    return kept_x, kept_meta, kept_mask, kept_risk, summary


def _save_augmented_dataset(
    dataset_name: str,
    x_train_ref: np.ndarray,
    x_train_aug: np.ndarray,
    cond_train_aug_raw: pd.DataFrame,
    meta_train_aug: pd.DataFrame,
    mask_train_aug: np.ndarray | None,
    cfg: GanAugmentedTailWeightedConfig,
) -> Path:
    out_dir = cfg.augmented_data_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train_aug.npy", _match_reference_orientation(x_train_aug, x_train_ref).astype(np.float32))
    cond_train_aug_raw.to_csv(out_dir / "cond_train_aug.csv", index=False, encoding="utf-8-sig")
    meta_train_aug.to_csv(out_dir / "meta_train_aug.csv", index=False, encoding="utf-8-sig")
    if mask_train_aug is not None:
        np.save(out_dir / "event_mask_train_aug.npy", mask_train_aug.astype(np.float32))
    return out_dir


def _copy_baseline_fit_summary_if_available(spec: DatasetSpec, out_dir: Path) -> None:
    baseline_fit = _baseline_results_dir(spec) / "tailweighted_copula_fit_summary.csv"
    if baseline_fit.exists():
        shutil.copy2(baseline_fit, out_dir / "tailweighted_copula_fit_summary_augmented.csv")


def _write_method_report(
    spec: DatasetSpec,
    out_dir: Path,
    risk_main: pd.DataFrame,
    aux: pd.DataFrame,
    tail_summary: pd.DataFrame,
    filter_summary: pd.DataFrame,
    note: str,
) -> None:
    base_row = risk_main[risk_main["method"] == TAIL_FIXED_METHOD].head(1)
    gan_row = risk_main[risk_main["method"] == METHOD_NAME].head(1)
    lines = [
        f"# {spec.name} - GAN Augmented TailWeighted Month EVT-Copula",
        "",
        "## Method",
        "",
        "GAN is used only as a tail sample augmenter on the train split. Final test scenario generation is still performed by TailWeighted Month EVT-Copula with fixed risk-selection weights.",
        "",
        "## Tail Sample Summary",
        "",
        tail_summary.to_markdown(index=False) if len(tail_summary) else "No tail summary rows.",
        "",
        "## GAN Filtering Summary",
        "",
        filter_summary.to_markdown(index=False) if len(filter_summary) else "No filtering summary rows.",
        "",
        "## Main Risk Table",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Auxiliary Realism Table",
        "",
        aux[[c for c in ["method", *AUXILIARY_REALISM_METRICS, "realism_check_pass"] if c in aux.columns]].to_markdown(index=False),
        "",
        "## Notes",
        "",
        f"- {note}",
    ]
    if len(base_row) and len(gan_row):
        lines.extend(
            [
                "",
                "## GAN vs TailWeighted Fixed",
                "",
                f"- baseline risk_rank: {float(base_row['risk_rank'].iloc[0]):.0f}",
                f"- GAN-aug risk_rank: {float(gan_row['risk_rank'].iloc[0]):.0f}",
                f"- baseline risk_score: {float(base_row['risk_score'].iloc[0]):.6f}",
                f"- GAN-aug risk_score: {float(gan_row['risk_score'].iloc[0]):.6f}",
            ]
        )
    (out_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_global_summaries(root: Path, dataset_tables: dict[str, pd.DataFrame], aux_tables: dict[str, pd.DataFrame], dataset_notes: dict[str, dict]) -> None:
    risk_rows = []
    for dataset, table in dataset_tables.items():
        tmp = table.copy()
        tmp.insert(0, "dataset", dataset)
        risk_rows.append(tmp)
    risk_summary = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    risk_summary.to_csv(root / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")

    dataset_keys = list(dataset_tables.keys())
    rank_rows = []
    methods = sorted(risk_summary["method"].astype(str).unique()) if len(risk_summary) else []
    for method in methods:
        row = {"method": method}
        ranks, scores = [], []
        for key in dataset_keys:
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

    lines = [
        "# Final GAN Augmented TailWeighted Copula Report",
        "",
        "## Method Summary",
        "",
        "GAN is used only for train-split tail augmentation. Each dataset then re-fits a TailWeighted Month EVT-Copula on the augmented train set, and final test generation still comes from Copula candidate generation plus fixed risk selection.",
        "",
        "## All Dataset Risk Summary",
        "",
        risk_summary.to_markdown(index=False) if len(risk_summary) else "No rows.",
        "",
        "## Cross-Dataset Rank Summary",
        "",
        rank_df.to_markdown(index=False) if len(rank_df) else "No rows.",
        "",
        "## Dataset Notes",
        "",
    ]
    for key, note in dataset_notes.items():
        lines.append(f"### {key}")
        lines.append("")
        for item_key, value in note.items():
            lines.append(f"- {item_key}: {value}")
        lines.append("")
    (root / "final_gan_augmented_tailweighted_copula_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_augmented_tailweighted(
    spec: DatasetSpec,
    cfg: GanAugmentedTailWeightedConfig,
    out_dir: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, str]:
    x_train, cond_train_raw, meta_train, mask_train = _load_split(spec.data_dir, "train")
    _, cond_test_raw, meta_test, mask_test = _load_split(spec.data_dir, "test")
    cond_train = _add_month_season(cond_train_raw, meta_train)
    cond_test = _add_month_season(cond_test_raw, meta_test)
    train_meta_full = _merged_meta(cond_train_raw, meta_train)

    tau_by_month, _ = _compute_monthly_tau(x_train, cond_train, cfg, spec.out_name, out_dir)
    train_risk = _build_train_risk_table(x_train, cond_train, mask_train, tau_by_month, cfg)
    tail_df = _compute_tail_scores(train_risk, cond_train, cfg, spec.out_name, out_dir)
    threshold_used, tail_idx, threshold_note = _tail_threshold_and_index(tail_df["tail_score"].to_numpy(float), cfg)

    tail_summary = pd.DataFrame(
        [
            {
                "dataset": spec.out_name,
                "tail_threshold": float(threshold_used),
                "n_train_samples": int(len(x_train)),
                "n_tail_samples": int(len(tail_idx)),
                "event_type_distribution": _json_counts(cond_train_raw.get("event_type", pd.Series(["unknown"] * len(cond_train_raw)))),
                "month_distribution": _json_counts(cond_train["month"]),
                "severity_distribution_if_available": _json_counts(cond_train_raw["severity_level"]) if "severity_level" in cond_train_raw.columns else "",
                "notes": threshold_note,
            }
        ]
    )
    tail_summary.to_csv(out_dir / "gan_tail_training_samples_summary.csv", index=False, encoding="utf-8-sig")

    gan_note = threshold_note
    gan_generated_path = out_dir / "gan_generated_candidates.npy"
    gan_kept_path = out_dir / "gan_kept_samples.npy"
    generated_final = out_dir / "generated_samples_gan_aug_tailweighted.npy"

    if len(tail_idx) < int(cfg.min_tail_samples_after_relax):
        gan_note = f"GAN skipped because tail samples after relaxation = {len(tail_idx)} < {cfg.min_tail_samples_after_relax}."
        np.save(gan_generated_path, np.empty((0, 3, int(_to_channel_time(x_train).shape[2])), dtype=np.float32))
        np.save(gan_kept_path, np.empty((0, 3, int(_to_channel_time(x_train).shape[2])), dtype=np.float32))
        baseline_path = _existing_baseline_paths(spec)[TAIL_FIXED_METHOD]
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline TailWeighted fixed samples missing: {baseline_path}")
        shutil.copy2(baseline_path, generated_final)
        filter_summary = pd.DataFrame(
            [
                {
                    "dataset": spec.out_name,
                    "n_gan_candidates": 0,
                    "n_after_physics_filter": 0,
                    "n_after_risk_filter": 0,
                    "n_after_corr_filter": 0,
                    "n_gan_kept": 0,
                    "corr_error_before": np.nan,
                    "corr_error_after": np.nan,
                    "notes": gan_note,
                }
            ]
        )
        filter_summary.to_csv(out_dir / "gan_sample_filtering_summary.csv", index=False, encoding="utf-8-sig")
        _save_augmented_dataset(spec.out_name, x_train, x_train, cond_train_raw, meta_train, mask_train, cfg)
        tail_df.to_csv(out_dir / "tail_score_summary_augmented.csv", index=False, encoding="utf-8-sig")
        _copy_baseline_fit_summary_if_available(spec, out_dir)
        return generated_final, tail_summary, filter_summary, gan_note

    x_tail = x_train[tail_idx]
    tail_meta = train_meta_full.iloc[tail_idx].reset_index(drop=True).copy()
    tail_mask = mask_train[tail_idx] if mask_train is not None else None
    gan_cfg = _build_wgan_cfg(cfg, spec.data_dir, out_dir, x_train, len(tail_idx))
    state = train_conditional_wgan_gp(x_tail, tail_meta, None, gan_cfg)

    n_gan_candidates = max(1, int(round(float(cfg.gan_candidate_ratio) * len(x_train))))
    x_gan_candidates, meta_gan_candidates, sampled_idx = generate_candidates(state, tail_meta, n_gan_candidates, gan_cfg)
    mask_gan_candidates = tail_mask[sampled_idx] if tail_mask is not None else None
    x_phys, meta_phys, mask_phys, phys_summary = physical_filter_and_clip(x_gan_candidates, meta_gan_candidates, gan_cfg, mask_gan_candidates)
    np.save(gan_generated_path, _match_reference_orientation(x_phys, x_train).astype(np.float32))

    x_risk, meta_risk, mask_risk, risk_candidates, risk_note = _risk_filter_candidates(
        x_phys, meta_phys, mask_phys, train_risk, tau_by_month, cfg
    )
    n_keep = min(max(int(round(float(cfg.gan_keep_ratio) * len(x_train))), int(cfg.gan_keep_min)), len(tail_idx))
    kept_x, kept_meta, kept_mask, kept_risk, corr_summary = _correlation_filter_candidates(
        x_risk,
        meta_risk,
        mask_risk,
        risk_candidates,
        x_tail,
        train_risk.iloc[tail_idx].reset_index(drop=True),
        train_risk,
        n_keep,
        cfg,
    )
    np.save(gan_kept_path, _match_reference_orientation(kept_x, x_train).astype(np.float32))

    filter_summary = pd.DataFrame(
        [
            {
                "dataset": spec.out_name,
                "n_gan_candidates": int(n_gan_candidates),
                "n_after_physics_filter": int(len(x_phys)),
                "n_after_risk_filter": int(len(x_risk)),
                "n_after_corr_filter": int(len(kept_x)),
                "n_gan_kept": int(len(kept_x)),
                "corr_error_before": corr_summary["corr_error_before"],
                "corr_error_after": corr_summary["corr_error_after"],
                "notes": f"{risk_note}; {corr_summary['corr_filter_note']}",
            }
        ]
    )
    filter_summary.to_csv(out_dir / "gan_sample_filtering_summary.csv", index=False, encoding="utf-8-sig")

    if len(kept_x) == 0:
        gan_note = f"GAN trained but no samples survived filters. {risk_note}; {corr_summary['corr_filter_note']}"
        baseline_path = _existing_baseline_paths(spec)[TAIL_FIXED_METHOD]
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline TailWeighted fixed samples missing: {baseline_path}")
        shutil.copy2(baseline_path, generated_final)
        _save_augmented_dataset(spec.out_name, x_train, x_train, cond_train_raw, meta_train, mask_train, cfg)
        tail_df.to_csv(out_dir / "tail_score_summary_augmented.csv", index=False, encoding="utf-8-sig")
        _copy_baseline_fit_summary_if_available(spec, out_dir)
        return generated_final, tail_summary, filter_summary, gan_note

    cond_cols = list(cond_train_raw.columns)
    meta_cols = list(meta_train.columns)
    cond_gan = kept_meta.reindex(columns=cond_cols).reset_index(drop=True)
    meta_gan = kept_meta.reindex(columns=meta_cols).reset_index(drop=True)
    x_train_aug = np.concatenate([_to_channel_time(x_train), _to_channel_time(kept_x)], axis=0).astype(np.float32)
    cond_train_aug_raw = pd.concat([cond_train_raw.reset_index(drop=True), cond_gan], ignore_index=True)
    meta_train_aug = pd.concat([meta_train.reset_index(drop=True), meta_gan], ignore_index=True)
    if mask_train is not None and kept_mask is not None:
        mask_train_aug = np.concatenate([mask_train.astype(np.float32), kept_mask.astype(np.float32)], axis=0)
    else:
        mask_train_aug = mask_train
    _save_augmented_dataset(spec.out_name, x_train, x_train_aug, cond_train_aug_raw, meta_train_aug, mask_train_aug, cfg)

    cond_train_aug = _add_month_season(cond_train_aug_raw, meta_train_aug)
    tau_by_month_aug, _ = _compute_monthly_tau(x_train_aug, cond_train_aug, cfg, spec.out_name, out_dir)
    train_risk_aug = _build_train_risk_table(x_train_aug, cond_train_aug, mask_train_aug, tau_by_month_aug, cfg)
    tail_df_aug = _compute_tail_scores(train_risk_aug, cond_train_aug, cfg, spec.out_name, out_dir)
    tail_df_aug.to_csv(out_dir / "tail_score_summary_augmented.csv", index=False, encoding="utf-8-sig")
    models_aug, _, fit_df_aug = _fit_tailweighted_group_copulas(x_train_aug, cond_train_aug, tail_df_aug, cfg, out_dir, spec.out_name)
    fit_df_aug.to_csv(out_dir / "tailweighted_copula_fit_summary_augmented.csv", index=False, encoding="utf-8-sig")

    cop_cfg = _copula_cfg(cfg, int(_to_channel_time(x_train_aug).shape[2]), out_dir / "tailweighted_copula_groups")
    rng = np.random.default_rng(int(cfg.seed))
    test_candidates, test_metrics, test_targets, test_base = _generate_candidate_pool(
        "test", cond_test, mask_test, models_aug, train_risk_aug, tau_by_month_aug, cfg, cop_cfg, rng, out_dir, spec.out_name
    )
    pd.DataFrame(test_targets).to_csv(out_dir / "risk_target_summary.csv", index=False, encoding="utf-8-sig")
    fixed_weights = _parse_weights(cfg.fixed_weights)
    generated, selection_log = _select_candidates(test_candidates, test_metrics, test_targets, test_base, train_risk_aug, cond_test, fixed_weights, cfg)
    selection_log.to_csv(out_dir / "candidate_selection_log_fixed.csv", index=False, encoding="utf-8-sig")
    np.save(generated_final, generated.astype(np.float32))
    gan_note = f"GAN kept {len(kept_x)} / {n_gan_candidates} candidates after filters."
    return generated_final, tail_summary, filter_summary, gan_note


def run_one_dataset(spec: DatasetSpec, cfg: GanAugmentedTailWeightedConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out_dir = cfg.out_dir / spec.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_final, tail_summary, filter_summary, gan_note = _run_augmented_tailweighted(spec, cfg, out_dir)

    baseline_compare_path = _baseline_results_dir(spec) / "compare_all_methods.csv"
    if not baseline_compare_path.exists():
        raise FileNotFoundError(f"Baseline compare table not found: {baseline_compare_path}")
    baseline_df = pd.read_csv(baseline_compare_path)
    baseline_df = baseline_df[baseline_df["method"].astype(str) != METHOD_NAME].copy()

    gan_row = _evaluate_method(METHOD_NAME, generated_final, spec.data_dir, out_dir)
    df = pd.concat([baseline_df, pd.DataFrame([gan_row])], ignore_index=True)
    df = add_risk_score(df)
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir, method_col="method")
    _write_method_report(spec, out_dir, risk_main, aux, tail_summary, filter_summary, gan_note)
    dataset_note = {
        "gan_note": gan_note,
        "tail_threshold": float(tail_summary["tail_threshold"].iloc[0]) if len(tail_summary) else np.nan,
        "n_tail_samples": int(tail_summary["n_tail_samples"].iloc[0]) if len(tail_summary) else 0,
        "n_gan_kept": int(filter_summary["n_gan_kept"].iloc[0]) if len(filter_summary) else 0,
        "corr_error_after": float(filter_summary["corr_error_after"].iloc[0]) if len(filter_summary) and pd.notna(filter_summary["corr_error_after"].iloc[0]) else np.nan,
    }
    return risk_main, aux, dataset_note


def run_all(cfg: GanAugmentedTailWeightedConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.augmented_data_root.mkdir(parents=True, exist_ok=True)
    dataset_tables: dict[str, pd.DataFrame] = {}
    aux_tables: dict[str, pd.DataFrame] = {}
    dataset_notes: dict[str, dict] = {}
    for spec in _build_dataset_specs_four():
        print(f"\n=== Dataset: {spec.out_name} ===")
        risk_main, aux, note = run_one_dataset(spec, cfg)
        dataset_tables[spec.out_name] = risk_main
        aux_tables[spec.out_name] = aux
        dataset_notes[spec.out_name] = note
    _write_global_summaries(cfg.out_dir, dataset_tables, aux_tables, dataset_notes)


def parse_args() -> GanAugmentedTailWeightedConfig:
    parser = argparse.ArgumentParser(description="Run GAN-augmented TailWeighted Month EVT-Copula risk selection.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "gan_augmented_tailweighted_copula")
    parser.add_argument("--augmented-data-root", type=Path, default=BASE_DIR / "outputs" / "gan_augmented_datasets")
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
    parser.add_argument("--tail-sample-threshold", type=float, default=0.70)
    parser.add_argument("--tail-sample-threshold-relaxed", type=float, default=0.60)
    parser.add_argument("--min-tail-samples-after-relax", type=int, default=20)
    parser.add_argument("--min-tail-samples-before-relax", type=int, default=30)
    parser.add_argument("--gan-candidate-ratio", type=float, default=3.0)
    parser.add_argument("--gan-keep-ratio", type=float, default=0.02)
    parser.add_argument("--corr-error-threshold", type=float, default=0.20)
    parser.add_argument("--corr-error-threshold-relaxed", type=float, default=0.30)
    parser.add_argument("--risk-filter-low-quantile", type=float, default=0.70)
    parser.add_argument("--risk-filter-low-quantile-relaxed", type=float, default=0.60)
    parser.add_argument("--risk-filter-high-quantile", type=float, default=0.995)
    parser.add_argument("--ramp-upper-factor", type=float, default=1.20)
    parser.add_argument("--risk-filter-min-keep", type=int, default=2)
    parser.add_argument("--risk-fallback-priority-weight", type=float, default=0.75)
    parser.add_argument("--gan-target-cum-quantile", type=float, default=0.85)
    parser.add_argument("--gan-target-core-quantile", type=float, default=0.85)
    parser.add_argument("--gan-target-duration-quantile", type=float, default=0.75)
    parser.add_argument("--gan-ramp-soft-max-quantile", type=float, default=0.75)
    parser.add_argument("--gan-duration-soft-max-quantile", type=float, default=0.95)
    parser.add_argument("--gan-epochs", type=int, default=120)
    parser.add_argument("--gan-epochs-small", type=int, default=80)
    parser.add_argument("--gan-batch-size", type=int, default=32)
    parser.add_argument("--gan-noise-dim", type=int, default=64)
    parser.add_argument("--gan-base-channels", type=int, default=32)
    parser.add_argument("--gan-n-critic", type=int, default=5)
    parser.add_argument("--gan-gp-lambda", type=float, default=10.0)
    parser.add_argument("--gan-lr", type=float, default=1e-4)
    parser.add_argument("--corr-filter-min-keep", type=int, default=1)
    parser.add_argument("--corr-error-threshold-adaptive-factor", type=float, default=1.35)
    return GanAugmentedTailWeightedConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_all(parse_args())
