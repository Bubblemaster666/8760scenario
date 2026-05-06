from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from check_proposed_risk import build_risk_check
from run_experiments import ExperimentConfig, run_experiments


STAT_LIMITS = {
    "mean_wasserstein": 1.25,
    "mean_js": 1.25,
    "acf_mae": 1.30,
    "corr_matrix_error": 1.30,
}

LOWER_IS_BETTER = [
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
    "cum_deficit_mae",
    "q95_cum_deficit_error",
    "q99_cum_deficit_error",
    "core_cum_deficit_mae",
    "core_q95_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
    "core_netload_ramp_max_mae",
    "core_imbalance_duration_mae",
]

SUMMARY_COLUMNS = [
    "experiment_name",
    "method_name",
    "checkpoint_type_used_for_generation",
    "checkpoint_stage_used_for_generation",
    "stage1_epochs",
    "stage2_epochs",
    "stage3_epochs",
    "lambda_tail",
    "lambda_risk",
    "lambda_cum",
    "lambda_ramp",
    "lambda_dur",
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
    "cum_deficit_mae",
    "q95_cum_deficit_error",
    "q99_cum_deficit_error",
    "core_cum_deficit_mae",
    "core_q95_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
    "core_netload_ramp_max_mae",
    "core_imbalance_duration_mae",
    "extreme_degree_match_rate",
    "extreme_degree_adjacent_match_rate",
    "cum_deficit_improvement",
    "q95_cum_deficit_improvement",
    "q99_cum_deficit_improvement",
    "core_q99_cum_deficit_improvement",
    "netload_ramp_improvement",
    "duration_improvement",
    "wasserstein_degradation",
    "acf_degradation",
    "risk_priority_score",
    "statistical_degradation_ok",
    "verdict",
]


@dataclass(frozen=True)
class EpochExperiment:
    name: str
    stage1_epochs: int
    stage2_epochs: int
    stage3_epochs: int
    lambda_tail: float
    lambda_risk: float
    lambda_cum: float
    lambda_ramp: float
    lambda_dur: float
    checkpoint_type: str


@dataclass
class EpochTuningConfig:
    data_dir: str
    out_dir: str
    device: str = "cpu"
    experiments: list[str] | None = None
    summarize_only: bool = False
    seq_len: int = 36
    diffusion_steps: int = 100
    base_channels: int = 64
    batch_size: int = 32
    guidance_scale: float = 1.0
    cond_dropout: float = 0.10
    ema_decay: float = 0.995
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lambda_recon: float = 0.05
    lambda_physics: float = 0.02
    lambda_resource: float = 0.02
    seed: int = 42


EXPERIMENTS = [
    EpochExperiment("E0_current_baseline", 12, 10, 2, 0.25, 0.04, 1.0, 0.25, 0.35, "best-risk"),
    EpochExperiment("E1_medium_epochs", 20, 15, 4, 0.25, 0.04, 1.0, 0.25, 0.35, "best-risk"),
    EpochExperiment("E2_more_stage3", 20, 15, 6, 0.25, 0.04, 1.0, 0.25, 0.35, "best-risk"),
    EpochExperiment("E3_long_distribution", 30, 20, 4, 0.25, 0.04, 1.0, 0.25, 0.35, "best-risk"),
    EpochExperiment("E4_cum_focus", 20, 15, 4, 0.25, 0.04, 1.5, 0.15, 0.25, "best-risk"),
    EpochExperiment("E5_cum_focus_stage3", 20, 15, 6, 0.25, 0.04, 1.5, 0.15, 0.25, "best-risk"),
    EpochExperiment("E6_risk_006", 20, 15, 4, 0.25, 0.06, 1.0, 0.25, 0.35, "best-risk"),
    EpochExperiment("E7_final_checkpoint", 20, 15, 4, 0.25, 0.04, 1.0, 0.25, 0.35, "final"),
]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _improvement(proposed: float, baseline: float) -> float:
    if not np.isfinite(proposed) or not np.isfinite(baseline):
        return float("nan")
    denom = abs(baseline) + 1e-8
    return float((baseline - proposed) / denom)


def _ratio(proposed: float, baseline: float) -> float:
    if not np.isfinite(proposed) or not np.isfinite(baseline):
        return float("nan")
    return float(proposed / (abs(baseline) + 1e-8))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stage3_risk_loss_summary(model_dir: Path) -> dict[str, Any]:
    history_path = model_dir / "training_history.csv"
    if not history_path.exists():
        return {"stage3_val_risk_loss_decreased": None}
    history = pd.read_csv(history_path)
    stage3 = history.loc[history["stage"].astype(str).eq("stage3_risk")].copy()
    if stage3.empty or "val_risk_loss" not in stage3.columns:
        return {"stage3_val_risk_loss_decreased": None}
    start = _safe_float(stage3["val_risk_loss"].iloc[0])
    end = _safe_float(stage3["val_risk_loss"].iloc[-1])
    min_value = _safe_float(stage3["val_risk_loss"].min())
    decreased = bool(np.isfinite(start) and np.isfinite(min_value) and min_value < start * 0.98)
    return {
        "stage3_val_risk_loss_start": start,
        "stage3_val_risk_loss_end": end,
        "stage3_val_risk_loss_min": min_value,
        "stage3_val_risk_loss_decreased": decreased,
    }


def _run_one_experiment(cfg: EpochTuningConfig, exp: EpochExperiment) -> None:
    exp_dir = Path(cfg.out_dir) / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = ExperimentConfig(
        data_dir=cfg.data_dir,
        out_dir=str(exp_dir),
        methods=["proposed", "no_risk_loss"],
        seq_len=cfg.seq_len,
        stage1_epochs=exp.stage1_epochs,
        stage2_epochs=exp.stage2_epochs,
        stage3_epochs=exp.stage3_epochs,
        batch_size=cfg.batch_size,
        diffusion_steps=cfg.diffusion_steps,
        base_channels=cfg.base_channels,
        device=cfg.device,
        guidance_scale=cfg.guidance_scale,
        lambda_tail=exp.lambda_tail,
        lambda_risk=exp.lambda_risk,
        lambda_cum=exp.lambda_cum,
        lambda_ramp=exp.lambda_ramp,
        lambda_dur=exp.lambda_dur,
        lambda_recon=cfg.lambda_recon,
        lambda_physics=cfg.lambda_physics,
        lambda_resource=cfg.lambda_resource,
        cond_dropout=cfg.cond_dropout,
        ema_decay=cfg.ema_decay,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        checkpoint_type=exp.checkpoint_type,
        use_augmented_train=False,
        seed=cfg.seed,
    )
    run_experiments(run_cfg)
    build_risk_check(cfg.data_dir, exp_dir)


def _collect_experiment_rows(out_dir: Path, exp: EpochExperiment) -> list[dict[str, Any]]:
    exp_dir = out_dir / exp.name
    metrics_path = exp_dir / "evaluations" / "all_model_metrics.csv"
    if not metrics_path.exists():
        return [
            {
                "experiment_name": exp.name,
                "method_name": "proposed",
                "status": "missing_metrics",
            }
        ]
    metrics = pd.read_csv(metrics_path)
    metrics.to_csv(exp_dir / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for _, metric_row in metrics.iterrows():
        method = str(metric_row.get("model_name"))
        model_dir = exp_dir / "models" / method
        gen_summary = _read_json(model_dir / "generation_summary.json")
        train_summary = _read_json(model_dir / "summary.json")
        row: dict[str, Any] = {
            "experiment_name": exp.name,
            "method_name": method,
            "status": metric_row.get("status", "unknown"),
            "checkpoint_type_used_for_generation": gen_summary.get("resolved_checkpoint_type"),
            "checkpoint_stage_used_for_generation": gen_summary.get("checkpoint_stage"),
            "checkpoint_path_used_for_generation": gen_summary.get("checkpoint_path"),
            "best_model_epoch": train_summary.get("best_model_epoch"),
            "best_model_stage": train_summary.get("best_model_stage"),
            "best_risk_model_epoch": train_summary.get("best_risk_model_epoch"),
            "best_risk_model_stage": train_summary.get("best_risk_model_stage"),
            "final_epoch": train_summary.get("final_epoch"),
            "final_stage": train_summary.get("final_stage"),
            "stage1_epochs": exp.stage1_epochs,
            "stage2_epochs": exp.stage2_epochs,
            "stage3_epochs": exp.stage3_epochs,
            "lambda_tail": exp.lambda_tail,
            "lambda_risk": train_summary.get("lambda_risk_used", exp.lambda_risk if method == "proposed" else 0.0),
            "lambda_cum": exp.lambda_cum,
            "lambda_ramp": exp.lambda_ramp,
            "lambda_dur": exp.lambda_dur,
        }
        for metric in LOWER_IS_BETTER + ["extreme_degree_match_rate", "extreme_degree_adjacent_match_rate"]:
            if metric in metric_row:
                row[metric] = _safe_float(metric_row[metric])
        if method == "proposed":
            row.update(_stage3_risk_loss_summary(model_dir))
        rows.append(row)
    return rows


def _add_relative_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    improvement_pairs = {
        "cum_deficit_improvement": "cum_deficit_mae",
        "q95_cum_deficit_improvement": "q95_cum_deficit_error",
        "q99_cum_deficit_improvement": "q99_cum_deficit_error",
        "core_q99_cum_deficit_improvement": "core_q99_cum_deficit_error",
        "netload_ramp_improvement": "netload_ramp_max_mae",
        "duration_improvement": "imbalance_duration_mae",
    }
    out["wasserstein_degradation"] = np.nan
    out["acf_degradation"] = np.nan
    for new_col in improvement_pairs:
        out[new_col] = np.nan
    out["statistical_degradation_ok"] = False
    out["risk_priority_score"] = np.nan
    out["verdict"] = ""

    for exp_name, group in out.groupby("experiment_name"):
        baseline_rows = group.loc[group["method_name"].eq("no_risk_loss")]
        proposed_rows = group.loc[group["method_name"].eq("proposed")]
        if baseline_rows.empty or proposed_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for idx in proposed_rows.index:
            proposed = out.loc[idx]
            for new_col, metric in improvement_pairs.items():
                out.loc[idx, new_col] = _improvement(_safe_float(proposed.get(metric)), _safe_float(baseline.get(metric)))
            out.loc[idx, "wasserstein_degradation"] = _ratio(
                _safe_float(proposed.get("mean_wasserstein")), _safe_float(baseline.get("mean_wasserstein"))
            )
            out.loc[idx, "acf_degradation"] = _ratio(
                _safe_float(proposed.get("acf_mae")), _safe_float(baseline.get("acf_mae"))
            )
            stat_ok = all(
                _ratio(_safe_float(proposed.get(metric)), _safe_float(baseline.get(metric))) <= limit
                for metric, limit in STAT_LIMITS.items()
            )
            out.loc[idx, "statistical_degradation_ok"] = bool(stat_ok)

            score = (
                0.35 * _safe_float(out.loc[idx, "q99_cum_deficit_improvement"])
                + 0.30 * _safe_float(out.loc[idx, "cum_deficit_improvement"])
                + 0.20 * _safe_float(out.loc[idx, "core_q99_cum_deficit_improvement"])
                + 0.10 * _safe_float(out.loc[idx, "netload_ramp_improvement"])
                + 0.05 * _safe_float(out.loc[idx, "duration_improvement"])
            )
            if not stat_ok:
                score -= 0.25
            out.loc[idx, "risk_priority_score"] = score

            same_like = all(abs(_safe_float(out.loc[idx, col])) < 1e-6 for col in improvement_pairs)
            if same_like:
                verdict = "possible checkpoint or risk loss ineffective"
            elif _safe_float(out.loc[idx, "q99_cum_deficit_improvement"]) > 0 and _safe_float(out.loc[idx, "cum_deficit_improvement"]) >= 0:
                if _ratio(_safe_float(proposed.get("acf_mae")), _safe_float(baseline.get("acf_mae"))) > STAT_LIMITS["acf_mae"]:
                    verdict = "risk improved but temporal continuity degraded"
                elif stat_ok:
                    verdict = "risk improved with acceptable statistical degradation"
                else:
                    verdict = "risk improved but statistical degradation needs review"
            elif stat_ok:
                verdict = "statistically good but risk weak"
            else:
                verdict = "risk weak with statistical degradation"
            out.loc[idx, "verdict"] = verdict
        no_risk_idx = baseline_rows.index
        out.loc[no_risk_idx, "statistical_degradation_ok"] = True
        out.loc[no_risk_idx, "verdict"] = "baseline"
    return out


def _pick_recommended(summary: pd.DataFrame) -> pd.Series | None:
    proposed = summary.loc[summary["method_name"].eq("proposed")].copy()
    if proposed.empty:
        return None
    candidates = proposed.loc[
        proposed["statistical_degradation_ok"].astype(bool)
        & (pd.to_numeric(proposed["q99_cum_deficit_improvement"], errors="coerce") > 0)
        & (pd.to_numeric(proposed["cum_deficit_improvement"], errors="coerce") >= 0)
    ].copy()
    if candidates.empty:
        candidates = proposed.loc[proposed["statistical_degradation_ok"].astype(bool)].copy()
    if candidates.empty:
        candidates = proposed
    candidates["_sort_score"] = pd.to_numeric(candidates["risk_priority_score"], errors="coerce").fillna(-999.0)
    candidates["_q99"] = pd.to_numeric(candidates["q99_cum_deficit_error"], errors="coerce")
    return candidates.sort_values(["_sort_score", "_q99"], ascending=[False, True]).iloc[0]


def _primary_risk_better(summary: pd.DataFrame, candidate: str, reference: str) -> str:
    proposed = summary.loc[summary["method_name"].eq("proposed")].set_index("experiment_name")
    if candidate not in proposed.index or reference not in proposed.index:
        return "not evaluated"
    cand = proposed.loc[candidate]
    ref = proposed.loc[reference]
    cand_q99 = _safe_float(cand.get("q99_cum_deficit_error"))
    ref_q99 = _safe_float(ref.get("q99_cum_deficit_error"))
    cand_cum = _safe_float(cand.get("cum_deficit_mae"))
    ref_cum = _safe_float(ref.get("cum_deficit_mae"))
    cand_core = _safe_float(cand.get("core_q99_cum_deficit_error"))
    ref_core = _safe_float(ref.get("core_q99_cum_deficit_error"))
    if not all(np.isfinite(x) for x in [cand_q99, ref_q99, cand_cum, ref_cum]):
        return "inconclusive"
    primary_ok = cand_q99 < ref_q99 and cand_cum <= ref_cum
    core_ok = (not np.isfinite(cand_core)) or (not np.isfinite(ref_core)) or cand_core <= ref_core
    if primary_ok and core_ok:
        return "yes"
    if primary_ok:
        return "marginal; core_q99 did not improve"
    return "no"


def _best_longer_epochs_help(summary: pd.DataFrame) -> str:
    proposed = summary.loc[summary["method_name"].eq("proposed")].set_index("experiment_name")
    if "E0_current_baseline" not in proposed.index:
        return "not evaluated"
    candidates = ["E1_medium_epochs", "E2_more_stage3", "E3_long_distribution"]
    verdicts = [_primary_risk_better(summary, candidate, "E0_current_baseline") for candidate in candidates]
    if "yes" in verdicts:
        return "yes"
    if any(str(v).startswith("marginal") for v in verdicts):
        return "marginal"
    if all(v == "not evaluated" for v in verdicts):
        return "not evaluated"
    return "no"


def _write_report(summary: pd.DataFrame, cfg: EpochTuningConfig, out_dir: Path) -> None:
    rec = _pick_recommended(summary)
    proposed = summary.loc[summary["method_name"].eq("proposed")].copy()
    stage3_warnings = []
    for _, row in proposed.iterrows():
        if row.get("checkpoint_type_used_for_generation") == "best-risk" and row.get("checkpoint_stage_used_for_generation") != "stage3_risk":
            stage3_warnings.append(f"{row['experiment_name']}: proposed generation did not use stage3_risk.")
        if row.get("stage3_val_risk_loss_decreased") is False:
            stage3_warnings.append(
                f"{row['experiment_name']}: Stage 3 risk loss did not decrease clearly; consider increasing stage3_epochs or adjusting lambda_cum/lambda_dur."
            )

    if rec is None:
        recommended_config = "none"
        recommendation_reason = "No proposed rows were available."
    else:
        recommended_config = str(rec["experiment_name"])
        recommendation_reason = (
            f"q99 improvement={_safe_float(rec.get('q99_cum_deficit_improvement')):.3f}, "
            f"cum improvement={_safe_float(rec.get('cum_deficit_improvement')):.3f}, "
            f"core_q99 improvement={_safe_float(rec.get('core_q99_cum_deficit_improvement')):.3f}, "
            f"stat_ok={bool(rec.get('statistical_degradation_ok'))}."
        )

    more_epochs_help = _best_longer_epochs_help(summary)
    more_stage3_help = _primary_risk_better(summary, "E2_more_stage3", "E1_medium_epochs")
    long_distribution_help = _primary_risk_better(summary, "E3_long_distribution", "E1_medium_epochs")
    cum_focus_pair_1 = _primary_risk_better(summary, "E4_cum_focus", "E1_medium_epochs")
    cum_focus_pair_2 = _primary_risk_better(summary, "E5_cum_focus_stage3", "E2_more_stage3")
    cum_focus_help = "yes" if "yes" in {cum_focus_pair_1, cum_focus_pair_2} else (
        "marginal" if any(str(v).startswith("marginal") for v in [cum_focus_pair_1, cum_focus_pair_2]) else "no"
    )
    lambda_risk_006_help = _primary_risk_better(summary, "E6_risk_006", "E1_medium_epochs")
    final_checkpoint_help = _primary_risk_better(summary, "E7_final_checkpoint", "E1_medium_epochs")
    stage3_effective = bool((pd.to_numeric(proposed.get("q99_cum_deficit_improvement"), errors="coerce") > 0).any())

    top_table = proposed.sort_values("risk_priority_score", ascending=False).head(5)
    top_lines = [
        f"- {row.experiment_name}: score={row.risk_priority_score:.3f}, q99_imp={row.q99_cum_deficit_improvement:.3f}, "
        f"cum_imp={row.cum_deficit_improvement:.3f}, core_q99_imp={row.core_q99_cum_deficit_improvement:.3f}, verdict={row.verdict}"
        for row in top_table.itertuples(index=False)
    ]

    lines = [
        "# Epoch Tuning Report",
        "",
        f"recommended_config: {recommended_config}",
        "",
        f"recommendation_reason: {recommendation_reason}",
        "",
        "relative_to_no_risk_loss_summary:",
        *top_lines,
        "",
        "statistical_degradation_check:",
        f"- limits: mean_wasserstein<=1.25x, mean_js<=1.25x, acf_mae<=1.30x, corr_matrix_error<=1.30x",
        f"- recommended_config_stat_ok: {bool(rec.get('statistical_degradation_ok')) if rec is not None else False}",
        "",
        f"whether_stage3_is_effective: {stage3_effective}",
        f"whether_more_epochs_help: {more_epochs_help}",
        f"whether_more_stage3_help: {more_stage3_help}",
        f"whether_long_distribution_help: {long_distribution_help}",
        f"whether_cum_focus_help: {cum_focus_help}",
        f"whether_lambda_risk_006_help: {lambda_risk_006_help}",
        f"whether_final_checkpoint_help: {final_checkpoint_help}",
        "",
        "notes:",
        "- Model selection is risk-first and does not average all metrics.",
        "- Ramp and duration are treated as secondary risk indicators; q99/cum/core_q99 drive the recommendation.",
        "- The fixed dataset artifacts were read from the data_dir and were not regenerated.",
        "",
        "warnings:",
    ]
    lines.extend([f"- {warning}" for warning in stage3_warnings] or ["- none"])
    lines.append("")
    (out_dir / "epoch_tuning_report.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "config": asdict(cfg),
        "recommended_config": recommended_config,
        "recommendation_reason": recommendation_reason,
        "whether_stage3_is_effective": stage3_effective,
        "whether_more_epochs_help": more_epochs_help,
        "whether_cum_focus_help": cum_focus_help,
        "whether_lambda_risk_006_help": lambda_risk_006_help,
        "whether_final_checkpoint_help": final_checkpoint_help,
        "stage3_warnings": stage3_warnings,
    }
    (out_dir / "epoch_tuning_report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_epoch_tuning(cfg: EpochTuningConfig) -> pd.DataFrame:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [exp for exp in EXPERIMENTS if cfg.experiments is None or exp.name in set(cfg.experiments)]
    (out_dir / "epoch_tuning_config.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "experiments": [asdict(exp) for exp in selected],
                "statistical_limits": STAT_LIMITS,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    for exp in selected:
        if not cfg.summarize_only:
            print(f"[epoch-tuning] running {exp.name}", flush=True)
            _run_one_experiment(cfg, exp)

    rows: list[dict[str, Any]] = []
    for exp in selected:
        rows.extend(_collect_experiment_rows(out_dir, exp))
    summary = _add_relative_columns(pd.DataFrame(rows))
    ordered = [col for col in SUMMARY_COLUMNS if col in summary.columns]
    extras = [col for col in summary.columns if col not in ordered]
    summary = summary[ordered + extras]
    summary.to_csv(out_dir / "epoch_tuning_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(summary, cfg, out_dir)
    return summary


def parse_args() -> EpochTuningConfig:
    parser = argparse.ArgumentParser(description="Run fixed-dataset epoch and Stage 3 risk tuning.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--experiments", nargs="+", choices=[exp.name for exp in EXPERIMENTS], default=None)
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild summary/report from existing experiment outputs.")
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--cond-dropout", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-recon", type=float, default=0.05)
    parser.add_argument("--lambda-physics", type=float, default=0.02)
    parser.add_argument("--lambda-resource", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return EpochTuningConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        device=args.device,
        experiments=args.experiments,
        summarize_only=args.summarize_only,
        seq_len=args.seq_len,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        cond_dropout=args.cond_dropout,
        ema_decay=args.ema_decay,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_recon=args.lambda_recon,
        lambda_physics=args.lambda_physics,
        lambda_resource=args.lambda_resource,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_epoch_tuning(parse_args())
