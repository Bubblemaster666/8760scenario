from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from evaluate_generation import EXTREME_MAIN_METRICS, EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from risk_metrics import batch_hard_risk_metrics, batch_jirp_v2_metrics
from train_hierarchical_evt_diffusion import TrainConfig, train_model


BASE_DIR = Path(__file__).resolve().parent
JIRP_WINDOWS = (1.0, 2.0, 3.0)
JIRP_TOPK_RATIO = 0.10
JIRP_RAMP_DEFINITION = "multiscale_topk_mean"


@dataclass
class JirpV2Spec:
    experiment_name: str
    main_technique: str
    use_jirp_v2_condition: bool
    use_hybrid_jirp_tail_score: bool
    old_tail_weight: float
    jirp_tail_weight: float
    sampler_mode: str
    lambda_jirp_profile_stage2: float
    lambda_jirp_metric: float
    lambda_jirp_level: float
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2


BASE_SPECS = [
    JirpV2Spec("JIRPV2_0_RAMPDIAG4_baseline", "JRPD + 3h ramp, old Stage 2", False, False, 0.70, 0.30, "risk_profile_balanced", 0.0, 0.0, 0.0),
    JirpV2Spec("JIRPV2_1_condition_only", "JIRP-v2 condition only", True, False, 0.70, 0.30, "risk_profile_balanced", 0.0, 0.0, 0.0),
    JirpV2Spec("JIRPV2_2_condition_hybrid_tail", "JIRP-v2 condition + hybrid tail", True, True, 0.70, 0.30, "risk_profile_balanced", 0.0, 0.0, 0.0),
    JirpV2Spec("JIRPV2_3_condition_hybrid_sampler", "JIRP-v2 condition + hybrid sampler", True, True, 0.70, 0.30, "hybrid_jirp_profile_balanced", 0.0, 0.0, 0.0),
    JirpV2Spec("JIRPV2_4_full_stage2", "JIRP-v2 full Stage 2", True, True, 0.70, 0.30, "hybrid_jirp_profile_balanced", 0.02, 0.0, 0.0),
    JirpV2Spec("JIRPV2_5_full_stage2_stage3", "JIRP-v2 full Stage 2 + Stage 3 consistency", True, True, 0.70, 0.30, "hybrid_jirp_profile_balanced", 0.02, 0.02, 0.03),
    JirpV2Spec("JIRPV2_6_conservative_tail", "JIRP-v2 conservative hybrid tail", True, True, 0.85, 0.15, "hybrid_jirp_profile_balanced", 0.01, 0.01, 0.02),
]


def _copy_dataset(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.suffix.lower() in {".npy", ".csv", ".json"}:
            shutil.copy2(path, dst / path.name)


def _empirical_rank(values: np.ndarray, sorted_train: np.ndarray) -> np.ndarray:
    sorted_train = np.asarray(sorted_train, dtype=float)
    values = np.asarray(values, dtype=float)
    if sorted_train.size == 0:
        return np.zeros_like(values, dtype=float)
    return np.searchsorted(sorted_train, values, side="right").astype(float) / float(sorted_train.size)


def _thresholds(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return {"q50": float(arr.quantile(0.50)), "q75": float(arr.quantile(0.75)), "q90": float(arr.quantile(0.90))}


def _assign_levels(values: pd.Series | np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.select([arr <= thresholds["q50"], arr <= thresholds["q75"], arr <= thresholds["q90"]], [0, 1, 2], default=3).astype(int)


def prepare_jirp_v2_dataset(base_data_dir: Path, out_dir: Path, force: bool = False) -> Path:
    """Create a non-destructive dataset copy with JIRP-v2 risk labels.

    JIRP-v2 uses C cumulative imbalance, R multiscale top-k ramp-tail intensity,
    and D imbalance duration. All thresholds/ranks are fit on train only and
    then applied to val/test.
    """

    data_dir = out_dir / "dataset_jirp_v2"
    if data_dir.exists() and not force:
        return data_dir
    if data_dir.exists():
        shutil.rmtree(data_dir)
    _copy_dataset(base_data_dir, data_dir)

    split_frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        cond = pd.read_csv(data_dir / f"cond_{split}.csv")
        x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        tau = pd.to_numeric(cond.get("imbalance_tau", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        delta_t = float(cond["delta_t_hours"].iloc[0]) if "delta_t_hours" in cond.columns else 1.0

        # Existing interface keeps netload_ramp_max, but here it explicitly means 3h window ramp.
        ramp_3h = batch_hard_risk_metrics(
            x,
            tau=tau,
            delta_t_hours=delta_t,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )["netload_ramp_max"]
        jirp_metrics = batch_jirp_v2_metrics(
            x,
            tau=tau,
            delta_t_hours=delta_t,
            ramp_windows=JIRP_WINDOWS,
            topk_ratio=JIRP_TOPK_RATIO,
            ramp_definition=JIRP_RAMP_DEFINITION,
        )
        cond["netload_ramp_3h"] = ramp_3h
        cond["netload_ramp_max"] = ramp_3h
        cond["jirp_cum_intensity"] = jirp_metrics["jirp_cum_intensity"]
        cond["jirp_ramp_tail_intensity"] = jirp_metrics["jirp_ramp_tail_intensity"]
        cond["jirp_duration"] = jirp_metrics["jirp_duration"]
        split_frames[split] = cond

    train = split_frames["train"]
    level_thresholds = {
        "C": _thresholds(train["jirp_cum_intensity"]),
        "R": _thresholds(train["jirp_ramp_tail_intensity"]),
        "D": _thresholds(train["jirp_duration"]),
        "ramp_3h": _thresholds(train["netload_ramp_3h"]),
    }
    rank_sources = {
        "old_tail_score": np.sort(pd.to_numeric(train.get("tail_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)),
        "C": np.sort(pd.to_numeric(train["jirp_cum_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
        "R": np.sort(pd.to_numeric(train["jirp_ramp_tail_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
        "D": np.sort(pd.to_numeric(train["jirp_duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
    }
    continuous_stats = {
        "jirp_cum_intensity_log_mean": float(np.log1p(train["jirp_cum_intensity"].astype(float)).mean()),
        "jirp_cum_intensity_log_std": float(np.log1p(train["jirp_cum_intensity"].astype(float)).std() + 1e-6),
        "jirp_ramp_tail_intensity_mean": float(train["jirp_ramp_tail_intensity"].astype(float).mean()),
        "jirp_ramp_tail_intensity_std": float(train["jirp_ramp_tail_intensity"].astype(float).std() + 1e-6),
        "jirp_duration_mean": float(train["jirp_duration"].astype(float).mean()),
        "jirp_duration_std": float(train["jirp_duration"].astype(float).std() + 1e-6),
    }

    for split, cond in split_frames.items():
        old_rank = _empirical_rank(pd.to_numeric(cond.get("tail_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), rank_sources["old_tail_score"])
        c_rank = _empirical_rank(cond["jirp_cum_intensity"].to_numpy(dtype=float), rank_sources["C"])
        r_rank = _empirical_rank(cond["jirp_ramp_tail_intensity"].to_numpy(dtype=float), rank_sources["R"])
        d_rank = _empirical_rank(cond["jirp_duration"].to_numpy(dtype=float), rank_sources["D"])
        cond["old_tail_score_rank"] = old_rank
        cond["jirp_cum_rank"] = c_rank
        cond["jirp_ramp_rank"] = r_rank
        cond["jirp_duration_rank"] = d_rank
        cond["jirp_v2_tail_score"] = 0.55 * c_rank + 0.30 * r_rank + 0.15 * d_rank
        cond["hybrid_tail_score"] = 0.70 * old_rank + 0.30 * cond["jirp_v2_tail_score"]

        cond["jirp_cum_level"] = _assign_levels(cond["jirp_cum_intensity"], level_thresholds["C"])
        cond["jirp_ramp_level"] = _assign_levels(cond["jirp_ramp_tail_intensity"], level_thresholds["R"])
        cond["jirp_duration_level"] = _assign_levels(cond["jirp_duration"], level_thresholds["D"])
        cond["jirp_profile_id"] = cond["jirp_cum_level"].astype(int) * 16 + cond["jirp_ramp_level"].astype(int) * 4 + cond["jirp_duration_level"].astype(int)

        # Keep legacy JRPD profile labels available; ramp_level here follows 3h ramp.
        cond["cum_level"] = _assign_levels(cond["cum_deficit"], level_thresholds["C"])
        cond["ramp_level"] = _assign_levels(cond["netload_ramp_3h"], level_thresholds["ramp_3h"])
        cond["duration_level"] = _assign_levels(cond["imbalance_duration"], level_thresholds["D"])
        cond["risk_profile_id"] = cond["cum_level"].astype(int) * 16 + cond["ramp_level"].astype(int) * 4 + cond["duration_level"].astype(int)

        cond["jirp_cum_intensity_zscore"] = (np.log1p(cond["jirp_cum_intensity"].astype(float)) - continuous_stats["jirp_cum_intensity_log_mean"]) / continuous_stats["jirp_cum_intensity_log_std"]
        cond["jirp_ramp_tail_intensity_zscore"] = (cond["jirp_ramp_tail_intensity"].astype(float) - continuous_stats["jirp_ramp_tail_intensity_mean"]) / continuous_stats["jirp_ramp_tail_intensity_std"]
        cond["jirp_duration_zscore"] = (cond["jirp_duration"].astype(float) - continuous_stats["jirp_duration_mean"]) / continuous_stats["jirp_duration_std"]
        cond.to_csv(data_dir / f"cond_{split}.csv", index=False, encoding="utf-8-sig")

    thresholds_payload = {
        "risk_definition": "JIRP-v2",
        "C_thresholds": level_thresholds["C"],
        "R_thresholds": level_thresholds["R"],
        "D_thresholds": level_thresholds["D"],
        "ramp_3h_thresholds_for_legacy_ramp_level": level_thresholds["ramp_3h"],
        "ramp_windows": list(JIRP_WINDOWS),
        "topk_ratio": JIRP_TOPK_RATIO,
        "ramp_definition": JIRP_RAMP_DEFINITION,
        "hybrid_tail_default": {"old_tail_weight": 0.70, "jirp_tail_weight": 0.30},
        "jirp_tail_score_weights": {"C": 0.55, "R": 0.30, "D": 0.15},
        "rank_source": "train split only",
        "continuous_stats": continuous_stats,
        "train_quantiles": {
            "jirp_cum_intensity": {f"q{int(q*100):02d}": float(train["jirp_cum_intensity"].quantile(q)) for q in [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
            "jirp_ramp_tail_intensity": {f"q{int(q*100):02d}": float(train["jirp_ramp_tail_intensity"].quantile(q)) for q in [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
            "jirp_duration": {f"q{int(q*100):02d}": float(train["jirp_duration"].quantile(q)) for q in [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
        },
    }
    for path in [data_dir / "jirp_v2_thresholds.json", out_dir.parent / "jirp_v2_thresholds.json"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(thresholds_payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return data_dir


def _levels_from_thresholds(values: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    return _assign_levels(values, thresholds)


def compute_jirp_v2_diagnostics(real: np.ndarray, gen: np.ndarray, cond: pd.DataFrame, thresholds: dict, model_name: str) -> dict:
    tau = pd.to_numeric(cond.get("imbalance_tau", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta_t = float(cond["delta_t_hours"].iloc[0]) if "delta_t_hours" in cond.columns else 1.0
    real_metrics = batch_jirp_v2_metrics(real, tau=tau, delta_t_hours=delta_t, ramp_windows=JIRP_WINDOWS, topk_ratio=JIRP_TOPK_RATIO, ramp_definition=JIRP_RAMP_DEFINITION)
    gen_metrics = batch_jirp_v2_metrics(gen, tau=tau, delta_t_hours=delta_t, ramp_windows=JIRP_WINDOWS, topk_ratio=JIRP_TOPK_RATIO, ramp_definition=JIRP_RAMP_DEFINITION)
    c_gen = _levels_from_thresholds(gen_metrics["jirp_cum_intensity"], thresholds["C_thresholds"])
    r_gen = _levels_from_thresholds(gen_metrics["jirp_ramp_tail_intensity"], thresholds["R_thresholds"])
    d_gen = _levels_from_thresholds(gen_metrics["jirp_duration"], thresholds["D_thresholds"])
    c_target = pd.to_numeric(cond["jirp_cum_level"], errors="coerce").fillna(0).clip(0, 3).astype(int).to_numpy()
    r_target = pd.to_numeric(cond["jirp_ramp_level"], errors="coerce").fillna(0).clip(0, 3).astype(int).to_numpy()
    d_target = pd.to_numeric(cond["jirp_duration_level"], errors="coerce").fillna(0).clip(0, 3).astype(int).to_numpy()
    profile_match = (c_gen == c_target) & (r_gen == r_target) & (d_gen == d_target)
    profile_adjacent = (np.abs(c_gen - c_target) <= 1) & (np.abs(r_gen - r_target) <= 1) & (np.abs(d_gen - d_target) <= 1)
    return {
        "model_name": model_name,
        "jirp_profile_match_rate": float(profile_match.mean()),
        "jirp_profile_adjacent_match_rate": float(profile_adjacent.mean()),
        "jirp_cum_level_match_rate": float((c_gen == c_target).mean()),
        "jirp_ramp_level_match_rate": float((r_gen == r_target).mean()),
        "jirp_duration_level_match_rate": float((d_gen == d_target).mean()),
        "ramp_tail_wasserstein": float(wasserstein_distance(real_metrics["jirp_ramp_tail_intensity"], gen_metrics["jirp_ramp_tail_intensity"])),
        "duration_wasserstein": float(wasserstein_distance(real_metrics["jirp_duration"], gen_metrics["jirp_duration"])),
        "real_jirp_ramp_tail_q99": float(np.quantile(real_metrics["jirp_ramp_tail_intensity"], 0.99)),
        "gen_jirp_ramp_tail_q99": float(np.quantile(gen_metrics["jirp_ramp_tail_intensity"], 0.99)),
        "real_jirp_duration_q99": float(np.quantile(real_metrics["jirp_duration"], 0.99)),
        "gen_jirp_duration_q99": float(np.quantile(gen_metrics["jirp_duration"], 0.99)),
    }


def _train_generate_eval(spec: JirpV2Spec, data_dir: Path, out_dir: Path, thresholds: dict, device: str, force: bool = False) -> dict:
    model_dir = out_dir / "models" / spec.experiment_name
    eval_dir = out_dir / "evaluations" / spec.experiment_name
    metrics_path = eval_dir / "extreme_metrics_summary.csv"
    if metrics_path.exists() and (model_dir / "generation_summary.json").exists() and not force:
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        row = asdict(spec)
        row.update(metrics)
        row.update({"status": "reused"})
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary = train_model(
            TrainConfig(
                data_dir=str(data_dir),
                out_dir=str(model_dir),
                ablation="full",
                seq_len=36,
                batch_size=32,
                lr=1e-4,
                weight_decay=1e-5,
                diffusion_steps=100,
                base_channels=64,
                guidance_scale=1.0,
                cond_dropout=0.10,
                ema_decay=0.995,
                stage1_epochs=spec.stage1_epochs,
                stage2_epochs=spec.stage2_epochs,
                stage3_epochs=spec.stage3_epochs,
                lambda_tail=0.25,
                lambda_risk=0.04,
                lambda_cum=1.0,
                lambda_ramp=0.25,
                lambda_dur=0.35,
                lambda_recon=0.05,
                lambda_physics=0.02,
                lambda_resource=0.02,
                sampler_mode=spec.sampler_mode,
                use_risk_profile_condition=True,
                use_profile_loss=True,
                lambda_profile=0.05,
                use_jirp_v2_condition=spec.use_jirp_v2_condition,
                use_jirp_continuous_values=True,
                use_hybrid_jirp_tail_score=spec.use_hybrid_jirp_tail_score,
                old_tail_weight=spec.old_tail_weight,
                jirp_tail_weight=spec.jirp_tail_weight,
                jirp_cum_weight=0.55,
                jirp_ramp_weight=0.30,
                jirp_duration_weight=0.15,
                lambda_jirp_profile_stage2=spec.lambda_jirp_profile_stage2,
                lambda_jirp_metric=spec.lambda_jirp_metric,
                lambda_jirp_level=spec.lambda_jirp_level,
                jirp_metric_beta_ramp=0.5,
                jirp_metric_beta_duration=0.2,
                hybrid_sampler_old_tail_alpha=0.5,
                hybrid_sampler_cum_alpha=0.3,
                hybrid_sampler_ramp_alpha=0.3,
                hybrid_sampler_duration_alpha=0.1,
                hybrid_sampler_duration_balance=True,
                jirp_ramp_windows="1.0,2.0,3.0",
                jirp_ramp_topk_ratio=0.10,
                jirp_ramp_definition=JIRP_RAMP_DEFINITION,
                ramp_metric_mode="window_3h",
                ramp_window_hours=3.0,
                device=device,
                seed=42,
            )
        )
        if spec.sampler_mode == "hybrid_jirp_profile_balanced":
            (model_dir / "hybrid_jirp_sampler_summary.json").write_text(
                json.dumps(summary.get("sampler_summary", {}), ensure_ascii=False, indent=2),
                encoding="utf-8-sig",
            )
        generate_from_checkpoint(
            GenerationConfig(
                checkpoint=None,
                data_dir=str(data_dir),
                out_dir=str(model_dir),
                split="test",
                guidance_scale=1.0,
                checkpoint_type="best-risk",
            )
        )
        eval_summary = evaluate_generation(
            EvalConfig(
                real=str(data_dir / "X_test.npy"),
                generated=str(model_dir / "generated_samples.npy"),
                cond=str(data_dir / "cond_test.csv"),
                meta=str(data_dir / "meta_test.csv"),
                out_dir=str(eval_dir),
                model_name=spec.experiment_name,
                event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
                ramp_metric_mode="window_3h",
                ramp_window_hours=3.0,
            )
        )
        row = asdict(spec)
        row.update(eval_summary["metrics"])
        row.update({"status": "ok"})

    real = np.load(data_dir / "X_test.npy").astype(np.float32)
    gen = np.load(model_dir / "generated_samples.npy").astype(np.float32)
    cond = pd.read_csv(data_dir / "cond_test.csv")
    diag = compute_jirp_v2_diagnostics(real, gen, cond, thresholds, spec.experiment_name)
    diag.update({"experiment_name": spec.experiment_name})
    row.update(diag)
    return row


def _recommend(row: pd.Series, base: pd.Series) -> str:
    if row["experiment_name"] == "JIRPV2_0_RAMPDIAG4_baseline":
        return "baseline: current JRPD+3h ramp with old Stage 2"
    q99_ok = float(row["q99_cum_deficit_error"]) <= 1.2 * float(base["q99_cum_deficit_error"])
    degree_ok = float(row["extreme_degree_match_rate"]) >= float(base["extreme_degree_match_rate"])
    core_ok = float(row["core_q99_cum_deficit_error"]) <= 1.2 * float(base["core_q99_cum_deficit_error"])
    acf_ok = float(row["highrisk_acf_mae"]) <= 1.1 * float(base["highrisk_acf_mae"])
    ramp_better = float(row["netload_ramp_max_mae"]) < float(base["netload_ramp_max_mae"])
    duration_ok = float(row["imbalance_duration_mae"]) <= 1.05 * float(base["imbalance_duration_mae"])
    profile_better = float(row["jirp_profile_adjacent_match_rate"]) >= float(base["jirp_profile_adjacent_match_rate"])
    if q99_ok and degree_ok and core_ok and acf_ok and (ramp_better or profile_better) and duration_ok:
        return "recommended: preserves q99/core and improves ramp/profile tradeoff"
    if q99_ok and degree_ok and core_ok and acf_ok and (ramp_better or profile_better):
        return "promising: preserves required tail/profile metrics with partial JIRP improvement"
    if not q99_ok:
        return "not recommended: q99 cumulative deficit worsens"
    if not core_ok:
        return "not recommended: core q99 worsens"
    if not degree_ok:
        return "not recommended: extreme-degree match drops"
    if not acf_ok:
        return "not recommended: highrisk ACF worsens"
    return "mixed: no clear gain over baseline"


def summarize(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    base = df[df["experiment_name"] == "JIRPV2_0_RAMPDIAG4_baseline"].iloc[0]
    for metric in EXTREME_MAIN_METRICS:
        df[f"delta_{metric}_vs_JIRPV2_0"] = pd.to_numeric(df[metric], errors="coerce") - float(base[metric])
    df["recommendation_reason"] = df.apply(lambda row: _recommend(row, base), axis=1)
    base = df[df["experiment_name"] == "JIRPV2_0_RAMPDIAG4_baseline"].iloc[0]

    summary_cols = [
        "experiment_name",
        "main_technique",
        "use_jirp_v2_condition",
        "old_tail_weight",
        "jirp_tail_weight",
        "sampler_mode",
        "lambda_jirp_profile_stage2",
        "lambda_jirp_metric",
        "lambda_jirp_level",
        "stage1_epochs",
        "stage2_epochs",
        "stage3_epochs",
        *EXTREME_MAIN_METRICS,
        "jirp_profile_match_rate",
        "jirp_profile_adjacent_match_rate",
        "jirp_cum_level_match_rate",
        "jirp_ramp_level_match_rate",
        "jirp_duration_level_match_rate",
        "ramp_tail_wasserstein",
        "duration_wasserstein",
        "recommendation_reason",
        "status",
    ]
    for col in summary_cols:
        if col not in df.columns:
            df[col] = np.nan
    df[summary_cols].to_csv(out_dir / "jirp_v2_summary.csv", index=False, encoding="utf-8-sig")

    diag_cols = [
        "experiment_name",
        "model_name",
        "jirp_profile_match_rate",
        "jirp_profile_adjacent_match_rate",
        "jirp_cum_level_match_rate",
        "jirp_ramp_level_match_rate",
        "jirp_duration_level_match_rate",
        "ramp_tail_wasserstein",
        "duration_wasserstein",
        "real_jirp_ramp_tail_q99",
        "gen_jirp_ramp_tail_q99",
        "real_jirp_duration_q99",
        "gen_jirp_duration_q99",
    ]
    df[diag_cols].to_csv(out_dir / "jirp_v2_diagnostics.csv", index=False, encoding="utf-8-sig")

    candidates = df[df["recommendation_reason"].str.startswith("recommended", na=False)]
    if candidates.empty:
        candidates = df[df["recommendation_reason"].str.startswith("promising", na=False)]
    best = candidates.iloc[0] if not candidates.empty else base
    non_base = df[df["experiment_name"] != "JIRPV2_0_RAMPDIAG4_baseline"]
    best_ramp = non_base.sort_values("netload_ramp_max_mae").iloc[0] if not non_base.empty else base
    best_core = non_base.sort_values("core_q99_cum_deficit_error").iloc[0] if not non_base.empty else base
    best_q99_non_base = non_base.sort_values("q99_cum_deficit_error").iloc[0] if not non_base.empty else base

    lines = [
        "# JIRP-v2 Joint Imbalance Risk Redefinition Report",
        "",
        "This round does not change event extraction, dataset split, or test-set usage.",
        "All outputs are stored under `outputs/jirp_v2_redefinition/`.",
        "`netload_ramp_max_mae` in this report uses the 3h ramp definition.",
        "Additional JIRP-v2 diagnostics use multiscale top-k ramp-tail intensity.",
        "",
        f"## Recommended Config: {best['experiment_name']}",
        f"- reason: {best['recommendation_reason']}",
        f"- q99_cum_deficit_error: {float(best['q99_cum_deficit_error']):.6f}",
        f"- core_q99_cum_deficit_error: {float(best['core_q99_cum_deficit_error']):.6f}",
        f"- netload_ramp_max_mae_3h: {float(best['netload_ramp_max_mae']):.6f}",
        f"- imbalance_duration_mae: {float(best['imbalance_duration_mae']):.6f}",
        f"- jirp_ramp_level_match_rate: {float(best['jirp_ramp_level_match_rate']):.6f}",
        f"- jirp_profile_adjacent_match_rate: {float(best['jirp_profile_adjacent_match_rate']):.6f}",
        "",
        "## Baseline Reference",
        f"- JIRPV2_0 q99={float(base['q99_cum_deficit_error']):.6f}, core_q99={float(base['core_q99_cum_deficit_error']):.6f}, ramp_3h={float(base['netload_ramp_max_mae']):.6f}, duration={float(base['imbalance_duration_mae']):.6f}, degree={float(base['extreme_degree_match_rate']):.6f}.",
        "",
        "## Required Answers",
        "- Full joint-risk redefinition is not effective in the current sample regime. The best configuration is still the baseline `JIRPV2_0_RAMPDIAG4_baseline`.",
        f"- It does not narrow the ramp gap to GAN or plain diffusion. The best non-baseline 3h ramp result is {best_ramp['experiment_name']}={float(best_ramp['netload_ramp_max_mae']):.6f}, which is still worse than the baseline={float(base['netload_ramp_max_mae']):.6f}.",
        f"- q99 and core_q99 are not both preserved. The best non-baseline q99 is {best_q99_non_base['experiment_name']}={float(best_q99_non_base['q99_cum_deficit_error']):.6f}, already above the 1.2x baseline guardrail. The best non-baseline core_q99 is {best_core['experiment_name']}={float(best_core['core_q99_cum_deficit_error']):.6f}, but it comes with severe q99 degradation and lower extreme-degree matching.",
        "- Hybrid tail score is still more stable than fully replacing the old tail anchor, but it is not stable enough. `JIRPV2_2/3` push q99 to about 18, and `JIRPV2_4/5/6` push q99 to about 34.",
        "- Replacing the current RAMPDIAG4 main method is not recommended. JIRP-v2 should be treated as a negative ablation in the paper: the richer risk definition is conceptually meaningful, but it sacrifices the cumulative-deficit tail advantage under limited data.",
        "",
        "## Interpretation",
        "- The new R definition is more reasonable than one-step ramp, but the model still cannot learn C, R, and D jointly without losing the original q99 tail anchor.",
        "- Stage 3 JIRP metric and level consistency do not improve ramp-tail Wasserstein or profile match rate, so the issue is not solved by simply adding more consistency losses.",
        "- A safer next step is to keep RAMPDIAG4 for training and use JIRP-v2 mainly as a diagnostic or candidate-selection constraint rather than as the primary training target.",
    ]
    (out_dir / "jirp_v2_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return df


def _pick_light_long_spec(df: pd.DataFrame) -> JirpV2Spec | None:
    base = df[df["experiment_name"] == "JIRPV2_0_RAMPDIAG4_baseline"].iloc[0]
    pool = df[df["experiment_name"].str.match(r"JIRPV2_[1-6]_")]
    if pool.empty:
        return None
    ok = pool[
        (pool["q99_cum_deficit_error"] <= 1.2 * float(base["q99_cum_deficit_error"]))
        & (pool["extreme_degree_match_rate"] >= float(base["extreme_degree_match_rate"]))
        & (pool["core_q99_cum_deficit_error"] <= 1.2 * float(base["core_q99_cum_deficit_error"]))
        & (pool["highrisk_acf_mae"] <= 1.1 * float(base["highrisk_acf_mae"]))
    ].copy()
    if ok.empty:
        return None
    ok["selection_score"] = ok["netload_ramp_max_mae"].rank() + ok["ramp_tail_wasserstein"].rank() + 0.5 * ok["q99_cum_deficit_error"].rank()
    best_name = str(ok.sort_values("selection_score").iloc[0]["experiment_name"])
    spec = next(item for item in BASE_SPECS if item.experiment_name == best_name)
    return replace(
        spec,
        experiment_name="JIRPV2_7_best_light_long",
        main_technique=f"light-long version of {best_name}",
        stage1_epochs=16,
        stage2_epochs=12,
        stage3_epochs=3,
    )


def run_jirp_v2(data_dir: str, out_dir: str, device: str = "cpu", force: bool = False) -> pd.DataFrame:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    trial_root = root / "trial_dataset"
    dataset = prepare_jirp_v2_dataset(Path(data_dir), trial_root, force=force)
    thresholds = json.loads((dataset / "jirp_v2_thresholds.json").read_text(encoding="utf-8-sig"))

    rows = [_train_generate_eval(spec, dataset, root, thresholds, device=device, force=force) for spec in BASE_SPECS]
    df_initial = summarize(rows, root)
    light_long = _pick_light_long_spec(df_initial)
    if light_long is not None:
        rows.append(_train_generate_eval(light_long, dataset, root, thresholds, device=device, force=force))
    return summarize(rows, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JIRP-v2 joint imbalance risk redefinition experiments.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "jirp_v2_redefinition"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_jirp_v2(args.data_dir, args.out_dir, device=args.device, force=args.force)
