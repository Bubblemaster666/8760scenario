from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from train_hierarchical_evt_diffusion import TrainConfig, train_model


STAT_LIMITS = {
    "mean_wasserstein": 1.25,
    "mean_js": 1.25,
    "acf_mae": 1.30,
    "corr_matrix_error": 1.30,
}

METRIC_COLS = [
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
]

SUMMARY_COLS = [
    "experiment_name",
    "method_name",
    "enhancement_type",
    "sampler_mode",
    "tail_weight_mode",
    "tail_weight_alpha",
    "tail_weight_max",
    "lambda_tail_dist",
    "tail_dist_topk_ratio",
    "lambda_core_risk",
    "core_risk_mode",
    "num_candidates_per_condition",
    "candidate_selection_mode",
    "checkpoint_type_used_for_generation",
    "checkpoint_stage_used_for_generation",
    *METRIC_COLS,
    "delta_cum_deficit_mae",
    "delta_q95_cum_deficit_error",
    "delta_q99_cum_deficit_error",
    "delta_core_q99_cum_deficit_error",
    "delta_imbalance_duration_mae",
    "delta_mean_wasserstein",
    "delta_acf_mae",
    "improve_cum_deficit_mae_pct",
    "improve_q99_cum_deficit_error_pct",
    "improve_core_q99_cum_deficit_error_pct",
    "improve_duration_pct",
    "statistical_degradation_ok",
    "risk_priority_score",
    "verdict",
]


@dataclass(frozen=True)
class TailExperiment:
    name: str
    sampler_mode: str = "none"
    tail_sampler_alpha: float = 0.5
    tail_weight_mode: str = "relu"
    tail_weight_alpha: float = 0.5
    tail_weight_max: float = 3.0
    lambda_tail_dist: float = 0.0
    tail_dist_topk_ratio: float = 0.10
    lambda_core_risk: float = 0.0
    core_risk_mode: str = "off"
    num_candidates_per_condition: int = 1
    candidate_selection_mode: str = "none"
    inference_only: bool = False


@dataclass
class TailEnhancementConfig:
    data_dir: str
    out_dir: str
    device: str = "cpu"
    experiments: list[str] | None = None
    seq_len: int = 36
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2
    diffusion_steps: int = 100
    base_channels: int = 64
    batch_size: int = 32
    guidance_scale: float = 1.0
    lambda_tail: float = 0.25
    lambda_risk: float = 0.04
    lambda_cum: float = 1.0
    lambda_ramp: float = 0.25
    lambda_dur: float = 0.35
    lambda_recon: float = 0.05
    lambda_physics: float = 0.02
    lambda_resource: float = 0.02
    cond_dropout: float = 0.10
    ema_decay: float = 0.995
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42


EXPERIMENTS = [
    TailExperiment("A0_E0_base"),
    TailExperiment("A1_severity_sampler", sampler_mode="severity"),
    TailExperiment("A2_tail_score_sampler", sampler_mode="tail_score"),
    TailExperiment("A3_clipped_tail_weight", tail_weight_mode="clipped_relu"),
    TailExperiment("A4_sigmoid_tail_weight", tail_weight_mode="sigmoid"),
    TailExperiment("A5_topk_tail_dist_002", lambda_tail_dist=0.02),
    TailExperiment("A6_topk_tail_dist_003", lambda_tail_dist=0.03),
    TailExperiment("A7_core_risk_002", lambda_core_risk=0.02, core_risk_mode="event_mask"),
    TailExperiment("A8_core_risk_003", lambda_core_risk=0.03, core_risk_mode="event_mask"),
    TailExperiment("A9_sampler_topk", sampler_mode="severity", lambda_tail_dist=0.02),
    TailExperiment("A10_topk_core", lambda_tail_dist=0.02, lambda_core_risk=0.02, core_risk_mode="event_mask"),
    TailExperiment("A11_sampler_topk_core", sampler_mode="severity", lambda_tail_dist=0.02, lambda_core_risk=0.02, core_risk_mode="event_mask"),
    TailExperiment("A12_inference_candidates_K5", num_candidates_per_condition=5, candidate_selection_mode="risk_target", inference_only=True),
    TailExperiment("A13_inference_candidates_K10", num_candidates_per_condition=10, candidate_selection_mode="risk_target", inference_only=True),
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _improve(base: float, value: float) -> float:
    if not np.isfinite(base) or not np.isfinite(value):
        return float("nan")
    return float((base - value) / (abs(base) + 1e-8))


def _stat_ok(base: pd.Series, row: pd.Series) -> bool:
    return all(
        _safe_float(row.get(metric)) <= limit * (_safe_float(base.get(metric)) + 1e-8)
        for metric, limit in STAT_LIMITS.items()
    )


def _train_generate_eval(cfg: TailEnhancementConfig, exp: TailExperiment, method: str) -> dict[str, Any]:
    exp_dir = Path(cfg.out_dir) / exp.name
    model_dir = exp_dir / "models" / method
    eval_dir = exp_dir / "evaluations" / method
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)

    if not exp.inference_only:
        train_cfg = TrainConfig(
            data_dir=cfg.data_dir,
            out_dir=str(model_dir),
            ablation="no_risk_loss" if method == "no_risk_loss" else "full",
            seed=cfg.seed,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            diffusion_steps=cfg.diffusion_steps,
            guidance_scale=cfg.guidance_scale,
            cond_dropout=cfg.cond_dropout,
            base_channels=cfg.base_channels,
            ema_decay=cfg.ema_decay,
            stage1_epochs=cfg.stage1_epochs,
            stage2_epochs=cfg.stage2_epochs,
            stage3_epochs=cfg.stage3_epochs,
            lambda_tail=cfg.lambda_tail,
            lambda_risk=0.0 if method == "no_risk_loss" else cfg.lambda_risk,
            lambda_cum=cfg.lambda_cum,
            lambda_ramp=cfg.lambda_ramp,
            lambda_dur=cfg.lambda_dur,
            lambda_recon=cfg.lambda_recon,
            lambda_physics=cfg.lambda_physics,
            lambda_resource=cfg.lambda_resource,
            sampler_mode="none" if method == "no_risk_loss" else exp.sampler_mode,
            tail_sampler_alpha=exp.tail_sampler_alpha,
            tail_weight_mode=exp.tail_weight_mode,
            tail_weight_alpha=exp.tail_weight_alpha,
            tail_weight_max=exp.tail_weight_max,
            lambda_tail_dist=0.0 if method == "no_risk_loss" else exp.lambda_tail_dist,
            tail_dist_topk_ratio=exp.tail_dist_topk_ratio,
            lambda_core_risk=0.0 if method == "no_risk_loss" else exp.lambda_core_risk,
            core_risk_mode="off" if method == "no_risk_loss" else exp.core_risk_mode,
            device=cfg.device,
        )
        train_model(train_cfg)
        checkpoint = None
    else:
        source_dir = Path(cfg.out_dir) / "A0_E0_base" / "models" / "proposed"
        checkpoint = str(source_dir / "best_risk_model.pt")
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"A0 checkpoint is required for inference-only enhancement: {checkpoint}")
        for name in ["summary.json", "condition_meta.json", "normalization_stats.npz"]:
            src = source_dir / name
            if src.exists():
                shutil.copy2(src, model_dir / name)

    generate_from_checkpoint(
        GenerationConfig(
            checkpoint=checkpoint,
            data_dir=cfg.data_dir,
            out_dir=str(model_dir),
            split="test",
            guidance_scale=cfg.guidance_scale,
            checkpoint_type="best-risk",
            num_candidates_per_condition=exp.num_candidates_per_condition,
            candidate_selection_mode=exp.candidate_selection_mode,
        )
    )
    eval_summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(model_dir / "generated_samples.npy"),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=method,
            event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
        )
    )
    row = {"model_name": method, "status": "ok"}
    row.update(eval_summary["metrics"])
    return row


def _run_one(cfg: TailEnhancementConfig, exp: TailExperiment) -> None:
    exp_dir = Path(cfg.out_dir) / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(_train_generate_eval(cfg, exp, "proposed"))
    if exp.name == "A0_E0_base":
        rows.append(_train_generate_eval(cfg, exp, "no_risk_loss"))
    metrics_df = pd.DataFrame(rows)
    evals_dir = exp_dir / "evaluations"
    evals_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(evals_dir / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(exp_dir / "all_model_metrics.csv", index=False, encoding="utf-8-sig")


def _collect_rows(out_dir: Path, exp: TailExperiment) -> list[dict[str, Any]]:
    exp_dir = out_dir / exp.name
    metrics_path = exp_dir / "all_model_metrics.csv"
    if not metrics_path.exists():
        return [{"experiment_name": exp.name, "method_name": "proposed", "status": "missing_metrics"}]
    metrics = pd.read_csv(metrics_path)
    rows = []
    for _, metric_row in metrics.iterrows():
        method = str(metric_row["model_name"])
        model_dir = exp_dir / "models" / method
        gen_summary = _read_json(model_dir / "generation_summary.json")
        train_summary = _read_json(model_dir / "summary.json")
        row: dict[str, Any] = {
            "experiment_name": exp.name,
            "method_name": method,
            "enhancement_type": "inference" if exp.inference_only else "training",
            "status": metric_row.get("status", "ok"),
            "sampler_mode": exp.sampler_mode if method == "proposed" else "none",
            "tail_weight_mode": exp.tail_weight_mode,
            "tail_weight_alpha": exp.tail_weight_alpha,
            "tail_weight_max": exp.tail_weight_max,
            "lambda_tail_dist": train_summary.get("lambda_tail_dist", 0.0),
            "tail_dist_topk_ratio": exp.tail_dist_topk_ratio,
            "lambda_core_risk": train_summary.get("lambda_core_risk", 0.0),
            "core_risk_mode": train_summary.get("core_risk_mode", exp.core_risk_mode),
            "num_candidates_per_condition": gen_summary.get("num_candidates_per_condition", exp.num_candidates_per_condition),
            "candidate_selection_mode": gen_summary.get("candidate_selection_mode", exp.candidate_selection_mode),
            "checkpoint_type_used_for_generation": gen_summary.get("resolved_checkpoint_type"),
            "checkpoint_stage_used_for_generation": gen_summary.get("checkpoint_stage"),
            "checkpoint_path_used_for_generation": gen_summary.get("checkpoint_path"),
            "sampler_summary": json.dumps(train_summary.get("sampler_summary", {}), ensure_ascii=False),
        }
        for col in METRIC_COLS:
            if col in metric_row:
                row[col] = _safe_float(metric_row[col])
        rows.append(row)
    return rows


def _add_delta_and_verdict(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    base_rows = out.loc[(out["experiment_name"] == "A0_E0_base") & (out["method_name"] == "proposed")]
    if base_rows.empty:
        return out
    base = base_rows.iloc[0]
    deltas = {
        "delta_cum_deficit_mae": "cum_deficit_mae",
        "delta_q95_cum_deficit_error": "q95_cum_deficit_error",
        "delta_q99_cum_deficit_error": "q99_cum_deficit_error",
        "delta_core_q99_cum_deficit_error": "core_q99_cum_deficit_error",
        "delta_imbalance_duration_mae": "imbalance_duration_mae",
        "delta_mean_wasserstein": "mean_wasserstein",
        "delta_acf_mae": "acf_mae",
    }
    improves = {
        "improve_cum_deficit_mae_pct": "cum_deficit_mae",
        "improve_q99_cum_deficit_error_pct": "q99_cum_deficit_error",
        "improve_core_q99_cum_deficit_error_pct": "core_q99_cum_deficit_error",
        "improve_duration_pct": "imbalance_duration_mae",
    }
    for new_col, metric in deltas.items():
        out[new_col] = pd.to_numeric(out.get(metric), errors="coerce") - _safe_float(base.get(metric))
    for new_col, metric in improves.items():
        out[new_col] = out[metric].map(lambda value: _improve(_safe_float(base.get(metric)), _safe_float(value)))

    out["statistical_degradation_ok"] = out.apply(lambda row: _stat_ok(base, row), axis=1)
    out["risk_priority_score"] = (
        0.40 * out["improve_q99_cum_deficit_error_pct"].astype(float)
        + 0.30 * out["improve_cum_deficit_mae_pct"].astype(float)
        + 0.20 * out["improve_core_q99_cum_deficit_error_pct"].astype(float)
        + 0.10 * out["improve_duration_pct"].astype(float)
    )
    verdicts = []
    for _, row in out.iterrows():
        if row["experiment_name"] == "A0_E0_base" and row["method_name"] == "proposed":
            verdicts.append("baseline")
            continue
        if not bool(row.get("statistical_degradation_ok")):
            verdicts.append("not recommended due to statistical degradation")
        elif row.get("enhancement_type") == "inference":
            verdicts.append("inference-only enhancement")
        elif _safe_float(row.get("improve_q99_cum_deficit_error_pct")) > 0 and _safe_float(row.get("improve_cum_deficit_mae_pct")) > 0:
            verdicts.append("recommended_tail_enhanced_training")
        elif _safe_float(row.get("improve_core_q99_cum_deficit_error_pct")) > 0 and _safe_float(row.get("improve_q99_cum_deficit_error_pct")) <= 0:
            verdicts.append("core-risk improved but full-window tail degraded")
        else:
            verdicts.append("not recommended")
    out["verdict"] = verdicts
    return out


def _best_row(summary: pd.DataFrame, enhancement_type: str | None = None) -> pd.Series | None:
    rows = summary.loc[summary["method_name"].eq("proposed")].copy()
    if enhancement_type:
        rows = rows.loc[rows["enhancement_type"].eq(enhancement_type)]
    if rows.empty:
        return None
    preferred = rows.loc[rows["verdict"].eq("recommended_tail_enhanced_training")].copy()
    if preferred.empty and enhancement_type == "inference":
        preferred = rows.loc[rows["verdict"].eq("inference-only enhancement") & rows["statistical_degradation_ok"].astype(bool)].copy()
    if preferred.empty and enhancement_type == "training":
        baseline = rows.loc[rows["experiment_name"].eq("A0_E0_base")]
        if not baseline.empty:
            return baseline.iloc[0]
    if preferred.empty:
        preferred = rows.loc[rows["statistical_degradation_ok"].astype(bool)].copy()
    if preferred.empty:
        preferred = rows
    preferred["_score"] = pd.to_numeric(preferred["risk_priority_score"], errors="coerce").fillna(-999.0)
    return preferred.sort_values("_score", ascending=False).iloc[0]


def _answer_pair(summary: pd.DataFrame, candidate: str, reference: str, metric: str = "risk_priority_score", threshold: float = 0.01) -> str:
    rows = summary.loc[summary["method_name"].eq("proposed")].set_index("experiment_name")
    if candidate not in rows.index or reference not in rows.index:
        return "not evaluated"
    c = _safe_float(rows.loc[candidate, metric])
    r = _safe_float(rows.loc[reference, metric])
    if not np.isfinite(c) or not np.isfinite(r):
        return "inconclusive"
    return "yes" if c > r + threshold else "no"


def _write_report(summary: pd.DataFrame, out_dir: Path) -> None:
    train_rec = _best_row(summary, "training")
    infer_rec = _best_row(summary, "inference")
    if train_rec is None:
        rec_name = "none"
        rec_reason = "No training rows were available."
    elif train_rec["experiment_name"] == "A0_E0_base":
        rec_name = "A0_E0_base"
        rec_reason = "No training enhancement beat A0 on the q99/cum risk-first criteria under statistical constraints."
    else:
        rec_name = str(train_rec["experiment_name"])
        rec_reason = (
            f"q99 improve={train_rec['improve_q99_cum_deficit_error_pct']:.3f}, "
            f"cum improve={train_rec['improve_cum_deficit_mae_pct']:.3f}, "
            f"core_q99 improve={train_rec['improve_core_q99_cum_deficit_error_pct']:.3f}."
        )

    top_training = summary.loc[(summary["method_name"] == "proposed") & (summary["enhancement_type"] == "training")].sort_values("risk_priority_score", ascending=False).head(6)
    top_inference = summary.loc[(summary["method_name"] == "proposed") & (summary["enhancement_type"] == "inference")].sort_values("risk_priority_score", ascending=False)
    inference_line = "none recommended"
    if infer_rec is not None:
        inference_line = f"{infer_rec['experiment_name']} ({infer_rec['verdict']})"

    lines = [
        "# Tail Enhancement Report",
        "",
        f"recommended_training_config: {rec_name}",
        f"recommendation_reason: {rec_reason}",
        "",
        "top_training_results:",
    ]
    for row in top_training.itertuples(index=False):
        lines.append(
            f"- {row.experiment_name}: score={row.risk_priority_score:.3f}, "
            f"q99_imp={row.improve_q99_cum_deficit_error_pct:.3f}, "
            f"cum_imp={row.improve_cum_deficit_mae_pct:.3f}, "
            f"core_q99_imp={row.improve_core_q99_cum_deficit_error_pct:.3f}, verdict={row.verdict}"
        )
    lines.extend(
        [
            "",
            "inference_enhancement:",
            f"- best inference config: {inference_line}",
            "- risk-guided candidate selection is an inference-stage controllability enhancement.",
            "- Main comparison should keep K=1 unless every method is also allowed multi-candidate selection.",
            "",
            "questions:",
            f"1. Weighted sampler effective? {_answer_pair(summary, 'A1_severity_sampler', 'A0_E0_base')}",
            f"2. tail_score sampler better than severity sampler? {_answer_pair(summary, 'A2_tail_score_sampler', 'A1_severity_sampler')} (not recommended if still below A0)",
            f"3. clipped tail weight more stable than relu? {_answer_pair(summary, 'A3_clipped_tail_weight', 'A0_E0_base', threshold=0.005)}",
            f"4. sigmoid tail weight improves stability? {_answer_pair(summary, 'A4_sigmoid_tail_weight', 'A0_E0_base', threshold=0.005)}",
            f"5. top-k tail distribution improves q95/q99? {_answer_pair(summary, 'A5_topk_tail_dist_002', 'A0_E0_base')}",
            f"6. core risk improves core_q95/core_q99? {_answer_pair(summary, 'A7_core_risk_002', 'A0_E0_base', metric='improve_core_q99_cum_deficit_error_pct', threshold=0.01)}",
            f"7. sampler + top-k has additive benefit? {_answer_pair(summary, 'A9_sampler_topk', 'A5_topk_tail_dist_002')}",
            f"8. top-k + core risk has additive benefit? {_answer_pair(summary, 'A10_topk_core', 'A5_topk_tail_dist_002')}",
            f"9. sampler + top-k + core too strong? {_answer_pair(summary, 'A11_sampler_topk_core', 'A10_topk_core')}",
            f"10. K=5/K=10 candidate selection improves risk? {', '.join(row.experiment_name + ':' + row.verdict for row in top_inference.itertuples(index=False))}",
            f"11. Replace E0 or keep E0? {'replace with ' + rec_name if rec_name != 'A0_E0_base' else 'keep E0 as main method; no tail enhancement met the replacement criteria.'}",
            "",
            "positioning:",
            "- Keep the paper claim focused on risk-oriented improvement, especially cumulative deficit and q99 cumulative deficit.",
            "- Treat core_q99, ramp, duration, and extreme degree as supporting evidence.",
            "- Do not claim all metrics are globally best.",
        ]
    )
    lines.append("")
    (out_dir / "tail_enhancement_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_tail_enhancement(cfg: TailEnhancementConfig) -> pd.DataFrame:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [exp for exp in EXPERIMENTS if cfg.experiments is None or exp.name in set(cfg.experiments)]
    (out_dir / "tail_enhancement_config.json").write_text(
        json.dumps({"config": asdict(cfg), "experiments": [asdict(exp) for exp in selected]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for exp in selected:
        print(f"[tail-enhancement] running {exp.name}", flush=True)
        _run_one(cfg, exp)

    rows: list[dict[str, Any]] = []
    for exp in selected:
        rows.extend(_collect_rows(out_dir, exp))
    summary = _add_delta_and_verdict(pd.DataFrame(rows))
    ordered = [col for col in SUMMARY_COLS if col in summary.columns]
    extras = [col for col in summary.columns if col not in ordered]
    summary = summary[ordered + extras]
    summary.to_csv(out_dir / "tail_enhancement_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(summary, out_dir)
    return summary


def parse_args() -> TailEnhancementConfig:
    parser = argparse.ArgumentParser(description="Run fixed-dataset tail risk enhancement experiments.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--experiments", nargs="+", choices=[exp.name for exp in EXPERIMENTS], default=None)
    args = parser.parse_args()
    return TailEnhancementConfig(data_dir=args.data_dir, out_dir=args.out_dir, device=args.device, experiments=args.experiments)


if __name__ == "__main__":
    run_tail_enhancement(parse_args())
