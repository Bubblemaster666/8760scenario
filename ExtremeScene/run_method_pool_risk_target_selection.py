from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, compute_metrics, evaluate_generation
from risk_ranking_utils import RISK_MAIN_METRICS, RISK_SCORE_WEIGHTS, add_risk_score
from run_month_evt_copula_risk_selection import MonthEvtCopulaConfig, _compute_monthly_tau, _load_split
from run_riskfirst_tcn_quantile import (
    DATASET_PATHS,
    BASE_DIR,
    DURATION_AWARE_RISKFIRST,
    RISKFIRST_EMPIRICAL,
    EmpiricalRiskTargetBuilder,
    RiskFirstConfig,
    _build_train_risk_table_with_event,
    _ensure_condition,
    _hard_risk_frame_for_x,
    _select_calibration,
)


DATASETS = ["singleton", "muswellbrook", "cessnock_or_newarea"]


@dataclass(frozen=True)
class SourceMethod:
    name: str
    path_template: str

    def path(self, dataset: str) -> Path:
        return BASE_DIR / self.path_template.format(dataset=dataset)


CORE_SOURCES = [
    SourceMethod(
        DURATION_AWARE_RISKFIRST,
        "results/duration_aware_riskfirst/formal/{dataset}/generated_samples_DurationAware_RiskFirst_TCN_Empirical.npy",
    ),
    SourceMethod(
        RISKFIRST_EMPIRICAL,
        "results/duration_aware_riskfirst/formal/{dataset}/generated_samples_RiskFirst_TCN_Empirical.npy",
    ),
    SourceMethod(
        "extreme_conditioned_gaussian_copula",
        "results/evt_copula/{dataset}/generated_samples_evt_copula.npy",
    ),
    SourceMethod(
        "traditional_gaussian_copula",
        "results/tailweighted_month_evt_copula/{dataset}/generated_samples_traditional_gaussian_copula.npy",
    ),
    SourceMethod(
        "enhanced_gan",
        "results/tailweighted_month_evt_copula/{dataset}/generated_samples_enhanced_gan.npy",
    ),
    SourceMethod(
        "ordinary_conditional_tcn",
        "results/expanded_model_pool_formal50/{dataset}/generated_samples_Conditional_TCN_Risk_Generator.npy",
    ),
]

GANFIX_MODEL_POOL_METHODS = [
    "Conditional_NormalizingFlow_Risk_Generator",
    "Conditional_Transformer_Risk_Generator",
    "GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
    "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
    "TransformerVAE_Augmented_TailWeighted_Copula",
    "ValSelected_Model_Ensemble",
    "ValSelected_Model_Ensemble_Margin_0.00",
    "ValSelected_Model_Ensemble_Margin_0.05",
    "ValSelected_Model_Ensemble_Margin_0.10",
]

TARGETED_EXPANDED_METHODS = [
    "Conditional_NormalizingFlow_Risk_Generator",
    "Conditional_TCN_Risk_Generator",
    "Conditional_Transformer_Risk_Generator",
    "Expanded_ValSelected_Model_Ensemble",
    "Expanded_ValSelected_Model_Ensemble_Margin_0.05",
    "GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
    "RiskKNN_Bootstrap_Generator",
    "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
    "TransformerVAE_Augmented_TailWeighted_Copula",
]

GANFIX_SOURCES = [
    SourceMethod(
        f"ganfix_{method}",
        f"results/model_pool_experiments_new_machine_margin_formal50_ganfix/{{dataset}}/generated_samples_{method}.npy",
    )
    for method in GANFIX_MODEL_POOL_METHODS
]

TARGETED_EXPANDED_SOURCES = [
    SourceMethod(
        f"targeted_seed{seed}_{method}",
        f"results/targeted_expanded_pool_multiseed/seed{seed}/expanded_pool/{{dataset}}/generated_samples_{method}.npy",
    )
    for seed in [45, 46, 47]
    for method in TARGETED_EXPANDED_METHODS
]

SOURCES = CORE_SOURCES + GANFIX_SOURCES + TARGETED_EXPANDED_SOURCES

CORE_SOURCE_NAMES = [source.name for source in CORE_SOURCES]
GANFIX_SOURCE_NAMES = [source.name for source in GANFIX_SOURCES]
TARGETED_EXPANDED_SOURCE_NAMES = [source.name for source in TARGETED_EXPANDED_SOURCES]


VARIANT_POOLS = {
    "rf_duration_pool": [DURATION_AWARE_RISKFIRST, RISKFIRST_EMPIRICAL],
    "riskfirst_evt_pool": [DURATION_AWARE_RISKFIRST, RISKFIRST_EMPIRICAL, "extreme_conditioned_gaussian_copula"],
    "ganfix_model_pool": CORE_SOURCE_NAMES + GANFIX_SOURCE_NAMES,
    "targeted_expanded_pool": CORE_SOURCE_NAMES + TARGETED_EXPANDED_SOURCE_NAMES,
    "all_method_pool": [source.name for source in SOURCES],
}

VARIANT_WEIGHTS = {
    "paper_weights": {"cum_deficit": 0.30, "core_cum_deficit": 0.30, "netload_ramp_max": 0.25, "imbalance_duration": 0.15},
    "duration_guard": {"cum_deficit": 0.25, "core_cum_deficit": 0.25, "netload_ramp_max": 0.20, "imbalance_duration": 0.25, "max_imbalance_run": 0.05},
    "core_ramp_guard": {"cum_deficit": 0.25, "core_cum_deficit": 0.35, "netload_ramp_max": 0.25, "imbalance_duration": 0.15},
    "q99_guard": {"cum_deficit": 0.40, "core_cum_deficit": 0.25, "netload_ramp_max": 0.20, "imbalance_duration": 0.15},
}


def _load_cond(data_dir: Path, split: str) -> pd.DataFrame:
    cond = pd.read_csv(data_dir / f"cond_{split}.csv")
    meta_path = data_dir / f"meta_{split}.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame(index=np.arange(len(cond)))
    return _ensure_condition(cond, meta)


def _load_mask(data_dir: Path, split: str) -> np.ndarray | None:
    path = data_dir / f"event_mask_{split}.npy"
    return np.load(path).astype(np.float32) if path.exists() else None


def _target_context(dataset: str, out_dir: Path, target_mode: str) -> tuple[pd.DataFrame, list[dict[str, float]], dict[int, float]]:
    data_dir = DATASET_PATHS[dataset]
    x_train, cond_train_raw, meta_train, mask_train = _load_split(data_dir, "train")
    x_val, cond_val_raw, meta_val, mask_val = _load_split(data_dir, "val")
    cond_train = _ensure_condition(cond_train_raw, meta_train)
    cond_val = _ensure_condition(cond_val_raw, meta_val)
    cfg = RiskFirstConfig(out_dir=out_dir, dataset=dataset)
    tau_cfg = MonthEvtCopulaConfig(out_dir=out_dir, tau_quantile=0.75, delta_t_hours=1.0, seed=42)
    tau_by_month, tau_df = _compute_monthly_tau(x_train, cond_train, tau_cfg, dataset, out_dir)
    tau_df.to_csv(out_dir / f"monthly_tau_summary_{target_mode}.csv", index=False, encoding="utf-8-sig")
    train_risk = _build_train_risk_table_with_event(x_train, cond_train, mask_train, tau_by_month, cfg)
    cond_test = _load_cond(data_dir, "test")
    if target_mode == "empirical":
        builder = EmpiricalRiskTargetBuilder(train_risk, cfg, dataset)
    elif target_mode == "val_calibrated":
        calibration_dir = out_dir / "val_calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        builder, calibration_grid, calibration_diag = _select_calibration(
            dataset,
            train_risk,
            cond_val,
            x_val,
            mask_val,
            tau_by_month,
            cfg,
            calibration_dir,
        )
        calibration_grid.to_csv(out_dir / "val_calibrated_quantile_grid.csv", index=False, encoding="utf-8-sig")
        calibration_diag.to_csv(out_dir / "val_calibrated_target_diag.csv", index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    target_df, scales = builder.build_frame(cond_test)
    target_df.to_csv(out_dir / f"risk_target_{target_mode}_test.csv", index=False, encoding="utf-8-sig")
    return target_df, scales, tau_by_month


def _available_sources(dataset: str, cond_len: int) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    arrays: dict[str, np.ndarray] = {}
    status = []
    for source in SOURCES:
        path = source.path(dataset)
        if not path.exists():
            status.append({"method": source.name, "status": "missing", "path": str(path)})
            continue
        arr = np.load(path).astype(np.float32)
        if len(arr) < cond_len:
            status.append({"method": source.name, "status": "too_short", "path": str(path)})
            continue
        arrays[source.name] = arr[:cond_len]
        status.append({"method": source.name, "status": "loaded", "path": str(path)})
    return arrays, status


def _metric_error(
    value: float,
    target: dict,
    scale: dict[str, float],
    metric: str,
) -> float:
    target_key = {
        "cum_deficit": "target_cum",
        "core_cum_deficit": "target_core",
        "netload_ramp_max": "target_ramp",
        "imbalance_duration": "target_duration",
        "max_imbalance_run": "target_max_imbalance_run",
    }[metric]
    return abs(float(value) - float(target[target_key])) / max(float(scale.get(metric, 1.0)), 1e-6)


def _local_scores(
    metrics_by_source: dict[str, pd.DataFrame],
    source_names: list[str],
    target_row: dict,
    scale: dict[str, float],
    weights: dict[str, float],
    i: int,
) -> np.ndarray:
    scores = []
    total_weight = sum(float(v) for v in weights.values())
    for name in source_names:
        row = metrics_by_source[name].iloc[i]
        score = 0.0
        for metric, weight in weights.items():
            if metric not in row:
                continue
            score += float(weight) * _metric_error(float(row[metric]), target_row, scale, metric)
        scores.append(score / max(total_weight, 1e-12))
    return np.asarray(scores, dtype=float)


def _select_local(
    arrays: dict[str, np.ndarray],
    metrics_by_source: dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    scales: list[dict[str, float]],
    source_names: list[str],
    weights: dict[str, float],
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = []
    logs = []
    for i in range(len(target_df)):
        target_row = target_df.iloc[i].to_dict()
        score = _local_scores(metrics_by_source, source_names, target_row, scales[i], weights, i)
        j = int(np.nanargmin(score))
        name = source_names[j]
        selected.append(arrays[name][i])
        log = {
            "sample_index": i,
            "selected_method": name,
            "selected_score": float(score[j]),
            "target_cum": float(target_row["target_cum"]),
            "target_core": float(target_row["target_core"]),
            "target_ramp": float(target_row["target_ramp"]),
            "target_duration": float(target_row["target_duration"]),
        }
        for src, src_score in zip(source_names, score):
            log[f"score_{src}"] = float(src_score)
        logs.append(log)
    return np.stack(selected, axis=0).astype(np.float32), pd.DataFrame(logs)


def _q99(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, 0.99)) if arr.size else 0.0


def _select_global_q99(
    arrays: dict[str, np.ndarray],
    metrics_by_source: dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    scales: list[dict[str, float]],
    source_names: list[str],
    weights: dict[str, float],
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = []
    local_log = []
    local_scores = []
    for i in range(len(target_df)):
        score = _local_scores(metrics_by_source, source_names, target_df.iloc[i].to_dict(), scales[i], weights, i)
        j = int(np.nanargmin(score))
        selected.append(j)
        local_scores.append(score)
        local_log.append(float(score[j]))

    target_q99_cum = _q99(target_df["target_cum"].to_numpy(float))
    target_q99_core = _q99(target_df["target_core"].to_numpy(float))
    scale_cum = max(float(np.nanmean([s.get("cum_deficit", 1.0) for s in scales])), 1e-6)
    scale_core = max(float(np.nanmean([s.get("core_cum_deficit", 1.0) for s in scales])), 1e-6)

    def objective(indices: list[int]) -> float:
        cum = np.asarray([metrics_by_source[source_names[j]].iloc[i]["cum_deficit"] for i, j in enumerate(indices)], dtype=float)
        core = np.asarray([metrics_by_source[source_names[j]].iloc[i]["core_cum_deficit"] for i, j in enumerate(indices)], dtype=float)
        score_mean = float(np.mean([local_scores[i][j] for i, j in enumerate(indices)]))
        return (
            RISK_SCORE_WEIGHTS["q99_cum_deficit_error"] * abs(_q99(cum) - target_q99_cum) / scale_cum
            + RISK_SCORE_WEIGHTS["core_q99_cum_deficit_error"] * abs(_q99(core) - target_q99_core) / scale_core
            + 0.10 * score_mean
        )

    best = objective(selected)
    for _ in range(8):
        changed = False
        for i in range(len(selected)):
            local_best = selected[i]
            local_obj = best
            for j in range(len(source_names)):
                trial = list(selected)
                trial[i] = j
                obj = objective(trial)
                if obj + 1e-9 < local_obj:
                    local_best = j
                    local_obj = obj
            if local_best != selected[i]:
                selected[i] = local_best
                best = local_obj
                changed = True
        if not changed:
            break

    out = np.stack([arrays[source_names[j]][i] for i, j in enumerate(selected)], axis=0).astype(np.float32)
    logs = []
    for i, j in enumerate(selected):
        logs.append(
            {
                "sample_index": i,
                "selected_method": source_names[j],
                "selected_score": float(local_scores[i][j]),
                "initial_local_score": local_log[i],
                "global_objective": float(best),
            }
        )
    return out, pd.DataFrame(logs)


def _select_global_main_metrics(
    arrays: dict[str, np.ndarray],
    metrics_by_source: dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    scales: list[dict[str, float]],
    source_names: list[str],
    weights: dict[str, float],
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = []
    local_scores = []
    for i in range(len(target_df)):
        score = _local_scores(metrics_by_source, source_names, target_df.iloc[i].to_dict(), scales[i], weights, i)
        j = int(np.nanargmin(score))
        selected.append(j)
        local_scores.append(score)

    target_q99_cum = _q99(target_df["target_cum"].to_numpy(float))
    target_q99_core = _q99(target_df["target_core"].to_numpy(float))
    scale_cum = max(float(np.nanmean([s.get("cum_deficit", 1.0) for s in scales])), 1e-6)
    scale_core = max(float(np.nanmean([s.get("core_cum_deficit", 1.0) for s in scales])), 1e-6)
    scale_ramp = max(float(np.nanmean([s.get("netload_ramp_max", 1.0) for s in scales])), 1e-6)
    scale_dur = max(float(np.nanmean([s.get("imbalance_duration", 1.0) for s in scales])), 1e-6)
    target_ramp = target_df["target_ramp"].to_numpy(float)
    target_dur = target_df["target_duration"].to_numpy(float)

    def objective(indices: list[int]) -> float:
        cum = np.asarray([metrics_by_source[source_names[j]].iloc[i]["cum_deficit"] for i, j in enumerate(indices)], dtype=float)
        core = np.asarray([metrics_by_source[source_names[j]].iloc[i]["core_cum_deficit"] for i, j in enumerate(indices)], dtype=float)
        ramp = np.asarray([metrics_by_source[source_names[j]].iloc[i]["netload_ramp_max"] for i, j in enumerate(indices)], dtype=float)
        dur = np.asarray([metrics_by_source[source_names[j]].iloc[i]["imbalance_duration"] for i, j in enumerate(indices)], dtype=float)
        score_mean = float(np.mean([local_scores[i][j] for i, j in enumerate(indices)]))
        return (
            RISK_SCORE_WEIGHTS["q99_cum_deficit_error"] * abs(_q99(cum) - target_q99_cum) / scale_cum
            + RISK_SCORE_WEIGHTS["core_q99_cum_deficit_error"] * abs(_q99(core) - target_q99_core) / scale_core
            + RISK_SCORE_WEIGHTS["netload_ramp_max_mae"] * float(np.mean(np.abs(ramp - target_ramp))) / scale_ramp
            + RISK_SCORE_WEIGHTS["imbalance_duration_mae"] * float(np.mean(np.abs(dur - target_dur))) / scale_dur
            + 0.05 * score_mean
        )

    best = objective(selected)
    for _ in range(10):
        changed = False
        for i in range(len(selected)):
            local_best = selected[i]
            local_obj = best
            for j in range(len(source_names)):
                trial = list(selected)
                trial[i] = j
                obj = objective(trial)
                if obj + 1e-9 < local_obj:
                    local_best = j
                    local_obj = obj
            if local_best != selected[i]:
                selected[i] = local_best
                best = local_obj
                changed = True
        if not changed:
            break

    out = np.stack([arrays[source_names[j]][i] for i, j in enumerate(selected)], axis=0).astype(np.float32)
    logs = []
    for i, j in enumerate(selected):
        logs.append(
            {
                "sample_index": i,
                "selected_method": source_names[j],
                "selected_score": float(local_scores[i][j]),
                "global_objective": float(best),
            }
        )
    return out, pd.DataFrame(logs)


def _evaluate_variant_fast(dataset: str, variant_name: str, gen_path: Path, data_dir: Path, out_dir: Path) -> dict:
    real = np.load(data_dir / "X_test.npy").astype(np.float32)
    gen = np.load(gen_path).astype(np.float32)
    cond = pd.read_csv(data_dir / "cond_test.csv")
    meta = pd.read_csv(data_dir / "meta_test.csv")
    n = min(len(real), len(gen), len(cond), len(meta))
    event_mask_path = data_dir / "event_mask_test.npy"
    event_mask = np.load(event_mask_path).astype(np.float32)[:n] if event_mask_path.exists() else None
    metrics = compute_metrics(
        real[:n],
        gen[:n],
        cond.iloc[:n].reset_index(drop=True),
        max_lag=12,
        event_mask=event_mask,
        ramp_metric_mode="window_3h",
        ramp_window_hours=3.0,
    )
    row = dict(metrics)
    row["dataset"] = dataset
    row["method"] = variant_name
    row["model_name"] = variant_name
    row["generated_path"] = str(gen_path)
    eval_dir = out_dir / "evaluations_fast" / variant_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(eval_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    return row


def _evaluate_variant_full(dataset: str, variant_name: str, gen_path: Path, data_dir: Path, out_dir: Path) -> dict:
    event_mask = data_dir / "event_mask_test.npy"
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(gen_path),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(out_dir / "evaluations" / variant_name),
            model_name=variant_name,
            event_mask=str(event_mask) if event_mask.exists() else None,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = dict(summary["metrics"])
    row["dataset"] = dataset
    row["method"] = variant_name
    row["generated_path"] = str(gen_path)
    return row


def _existing_compare(dataset: str) -> pd.DataFrame:
    path = BASE_DIR / "results" / "duration_aware_riskfirst_compare" / f"main_compare_{dataset}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_dataset(dataset: str, out_root: Path) -> pd.DataFrame:
    data_dir = DATASET_PATHS[dataset]
    ds_out = out_root / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    cond = _load_cond(data_dir, "test")
    mask = _load_mask(data_dir, "test")
    arrays, status = _available_sources(dataset, len(cond))
    pd.DataFrame(status).to_csv(ds_out / "source_status.csv", index=False, encoding="utf-8-sig")
    if not arrays:
        raise RuntimeError(f"No generated source arrays found for {dataset}.")

    target_contexts = {
        mode: _target_context(dataset, ds_out, mode)
        for mode in ["empirical", "val_calibrated"]
    }
    tau_by_month = target_contexts["empirical"][2]
    metrics_by_source = {
        name: _hard_risk_frame_for_x(arr, cond, mask, tau_by_month, 1.0)
        for name, arr in arrays.items()
    }
    for name, metrics in metrics_by_source.items():
        metrics.to_csv(ds_out / f"candidate_risk_metrics_{name}.csv", index=False, encoding="utf-8-sig")

    existing_df = _existing_compare(dataset)
    existing_methods = set(existing_df["method"].dropna().astype(str)) if "method" in existing_df.columns else set()
    source_rows = []
    for source in SOURCES:
        if source.name not in arrays or source.name in existing_methods:
            continue
        source_rows.append(_evaluate_variant_fast(dataset, source.name, source.path(dataset), data_dir, ds_out))

    rows = []
    for target_mode, (target_df, scales, _) in target_contexts.items():
        for pool_name, desired_sources in VARIANT_POOLS.items():
            source_names = [name for name in desired_sources if name in arrays]
            if len(source_names) < 2:
                continue
            for weight_name, weights in VARIANT_WEIGHTS.items():
                variant = f"TargetSelected_{target_mode}_{pool_name}_{weight_name}"
                gen, log = _select_local(arrays, metrics_by_source, target_df, scales, source_names, weights)
                gen_path = ds_out / f"generated_samples_{variant}.npy"
                np.save(gen_path, gen)
                log.to_csv(ds_out / f"selection_log_{variant}.csv", index=False, encoding="utf-8-sig")
                rows.append(_evaluate_variant_fast(dataset, variant, gen_path, data_dir, ds_out))

            variant = f"TargetSelected_{target_mode}_{pool_name}_global_q99"
            gen, log = _select_global_q99(
                arrays,
                metrics_by_source,
                target_df,
                scales,
                source_names,
                VARIANT_WEIGHTS["paper_weights"],
            )
            gen_path = ds_out / f"generated_samples_{variant}.npy"
            np.save(gen_path, gen)
            log.to_csv(ds_out / f"selection_log_{variant}.csv", index=False, encoding="utf-8-sig")
            rows.append(_evaluate_variant_fast(dataset, variant, gen_path, data_dir, ds_out))

            variant = f"TargetSelected_{target_mode}_{pool_name}_global_main"
            gen, log = _select_global_main_metrics(
                arrays,
                metrics_by_source,
                target_df,
                scales,
                source_names,
                VARIANT_WEIGHTS["paper_weights"],
            )
            gen_path = ds_out / f"generated_samples_{variant}.npy"
            np.save(gen_path, gen)
            log.to_csv(ds_out / f"selection_log_{variant}.csv", index=False, encoding="utf-8-sig")
            rows.append(_evaluate_variant_fast(dataset, variant, gen_path, data_dir, ds_out))

    new_df = pd.DataFrame(rows)
    new_df.to_csv(ds_out / "new_variant_raw_metrics.csv", index=False, encoding="utf-8-sig")
    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(ds_out / "source_method_raw_metrics.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([existing_df, source_df, new_df], ignore_index=True, sort=False)
    combined = add_risk_score(combined)
    combined.to_csv(ds_out / "compare_with_existing_methods.csv", index=False, encoding="utf-8-sig")
    return combined


def summarize(all_rows: pd.DataFrame, out_root: Path) -> None:
    summary_rows = []
    for method, g in all_rows.groupby("method"):
        ranks = pd.to_numeric(g["risk_rank"], errors="coerce")
        row = {
            "method": method,
            "mean_risk_score": float(pd.to_numeric(g["risk_score"], errors="coerce").mean()),
            "mean_risk_rank": float(ranks.mean()),
            "wins_count": int((ranks == 1).sum()),
            "top3_count": int((ranks <= 3).sum()),
            "available_dataset_count": int(g["dataset"].nunique()),
        }
        for metric in RISK_MAIN_METRICS:
            row[f"avg_{metric}"] = float(pd.to_numeric(g[metric], errors="coerce").mean())
        if "extreme_degree_match_rate" in g.columns:
            row["avg_extreme_degree_match_rate"] = float(pd.to_numeric(g["extreme_degree_match_rate"], errors="coerce").mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["mean_risk_score", "mean_risk_rank"])
    summary.to_csv(out_root / "method_pool_selection_summary.csv", index=False, encoding="utf-8-sig")
    all_rows.to_csv(out_root / "method_pool_selection_all_datasets_long.csv", index=False, encoding="utf-8-sig")
    best_by_score = summary.sort_values(["mean_risk_score", "mean_risk_rank"]).iloc[0].to_dict()
    best_by_rank = summary.sort_values(["mean_risk_rank", "mean_risk_score"]).iloc[0].to_dict()

    lines = [
        "# Method Pool Risk-Target Selection",
        "",
        "Evaluation metrics are unchanged: q99 cumulative deficit, core q99 cumulative deficit, 3h net-load ramp MAE, and imbalance duration MAE.",
        "",
        "## Best candidates",
        "",
        f"- Best by mean risk_score: {best_by_score['method']} ({best_by_score['mean_risk_score']:.6f})",
        f"- Best by mean risk_rank: {best_by_rank['method']} ({best_by_rank['mean_risk_rank']:.3f})",
        "",
        "## Summary",
        summary.to_markdown(index=False),
    ]
    (out_root / "method_pool_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_root = BASE_DIR / "results" / "method_pool_risk_target_selection_expanded_fast"
    out_root.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset in DATASETS:
        print(f"[method-pool] dataset={dataset}")
        compare = run_dataset(dataset, out_root)
        frames.append(compare)
    summarize(pd.concat(frames, ignore_index=True, sort=False), out_root)
    print(f"[method-pool] wrote {out_root}")


if __name__ == "__main__":
    main()
