from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from train_hierarchical_evt_diffusion import TrainConfig, train_model


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"

PAPER_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "extreme_degree_match_rate",
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

EXPERIMENTS: dict[str, dict[str, float]] = {
    "R0_current_proposed": {
        "lambda_delta_net_stage2": 0.0,
        "lambda_shape_stage2": 0.0,
        "lambda_delta_net_stage3": 0.0,
        "lambda_ramp_topk": 0.0,
        "lambda_shape_stage3": 0.0,
    },
    "R1_delta_stage3": {
        "lambda_delta_net_stage2": 0.0,
        "lambda_shape_stage2": 0.0,
        "lambda_delta_net_stage3": 0.20,
        "lambda_ramp_topk": 0.0,
        "lambda_shape_stage3": 0.0,
    },
    "R2_delta_topk_stage3": {
        "lambda_delta_net_stage2": 0.0,
        "lambda_shape_stage2": 0.0,
        "lambda_delta_net_stage3": 0.20,
        "lambda_ramp_topk": 0.20,
        "lambda_shape_stage3": 0.0,
    },
    "R3_stage2_shape_stage3_ramp": {
        "lambda_delta_net_stage2": 0.10,
        "lambda_shape_stage2": 0.02,
        "lambda_delta_net_stage3": 0.20,
        "lambda_ramp_topk": 0.20,
        "lambda_shape_stage3": 0.03,
    },
    "R4_conservative_shape": {
        "lambda_delta_net_stage2": 0.05,
        "lambda_shape_stage2": 0.01,
        "lambda_delta_net_stage3": 0.15,
        "lambda_ramp_topk": 0.15,
        "lambda_shape_stage3": 0.02,
    },
    "R5_stronger_ramp": {
        "lambda_delta_net_stage2": 0.10,
        "lambda_shape_stage2": 0.02,
        "lambda_delta_net_stage3": 0.30,
        "lambda_ramp_topk": 0.30,
        "lambda_shape_stage3": 0.03,
    },
}


@dataclass
class ShapeRampTuningConfig:
    data_dir: str = str(DEFAULT_DATA_DIR)
    out_dir: str = str(BASE_DIR / "outputs" / "shape_ramp_tuning")
    device: str = "cpu"
    seed: int = 42
    seq_len: int = 36
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2
    diffusion_steps: int = 100
    base_channels: int = 64
    batch_size: int = 32
    checkpoint_type: str = "best-risk"
    ramp_topk_ratio: float = 0.10
    reuse_existing: bool = False


def _base_train_kwargs(cfg: ShapeRampTuningConfig, model_dir: Path) -> dict[str, Any]:
    return {
        "data_dir": cfg.data_dir,
        "out_dir": str(model_dir),
        "ablation": "full",
        "seed": cfg.seed,
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "diffusion_steps": cfg.diffusion_steps,
        "base_channels": cfg.base_channels,
        "guidance_scale": 1.0,
        "cond_dropout": 0.10,
        "ema_decay": 0.995,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "stage1_epochs": cfg.stage1_epochs,
        "stage2_epochs": cfg.stage2_epochs,
        "stage3_epochs": cfg.stage3_epochs,
        "lambda_tail": 0.25,
        "lambda_risk": 0.04,
        "lambda_cum": 1.0,
        "lambda_ramp": 0.25,
        "lambda_dur": 0.35,
        "lambda_recon": 0.05,
        "lambda_physics": 0.02,
        "lambda_resource": 0.02,
        "ramp_topk_ratio": cfg.ramp_topk_ratio,
        "shape_highrisk_only": True,
        "device": cfg.device,
    }


def _evaluate_model(data_dir: Path, model_dir: Path, eval_dir: Path, model_name: str) -> dict[str, float]:
    event_mask_path = data_dir / "event_mask_test.npy"
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(model_dir / "generated_samples.npy"),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=model_name,
            event_mask=str(event_mask_path) if event_mask_path.exists() else None,
        )
    )
    return summary["metrics"]


def _generation_meta(model_dir: Path) -> dict:
    path = model_dir / "generation_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _recommend(row: pd.Series, base: pd.Series) -> str:
    ramp_base = float(base["netload_ramp_max_mae"])
    acf_base = float(base["highrisk_acf_mae"])
    q99_base = float(base["q99_cum_deficit_error"])
    core_base = float(base["core_q99_cum_deficit_error"])
    ramp_improve = (ramp_base - float(row["netload_ramp_max_mae"])) / max(abs(ramp_base), 1e-12)
    acf_improve = (acf_base - float(row["highrisk_acf_mae"])) / max(abs(acf_base), 1e-12)
    q99_worse = (float(row["q99_cum_deficit_error"]) - q99_base) / max(abs(q99_base), 1e-12)
    core_worse = (float(row["core_q99_cum_deficit_error"]) - core_base) / max(abs(core_base), 1e-12)
    if row["experiment_name"] == "R0_current_proposed":
        return "baseline current proposed"
    if ramp_improve >= 0.20 and acf_improve >= 0.10 and q99_worse <= 0.20 and core_worse <= 0.20:
        return "recommended: ramp and high-risk shape improved while q99/core-q99 are preserved"
    if ramp_improve >= 0.20 and q99_worse > 0.20:
        return "ramp improved but q99 cumulative deficit degraded"
    if acf_improve >= 0.10 and core_worse > 0.20:
        return "high-risk shape improved but core tail risk degraded"
    if q99_worse <= 0.20 and core_worse <= 0.20:
        return "tail risk preserved but ramp/shape improvement is limited"
    return "not recommended: no stable improvement over R0"


def _write_report(summary_df: pd.DataFrame, out_dir: Path) -> None:
    base = summary_df.loc[summary_df["experiment_name"] == "R0_current_proposed"]
    lines = ["# Shape/Ramp Tuning Report", ""]
    if base.empty:
        lines.append("R0_current_proposed is missing; cannot judge improvements.")
        (out_dir / "shape_ramp_tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    base_row = base.iloc[0]
    candidates = summary_df[summary_df["experiment_name"] != "R0_current_proposed"].copy()
    valid = candidates[
        (candidates["delta_q99_cum_deficit_error_vs_R0"] <= 0.20 * max(abs(float(base_row["q99_cum_deficit_error"])), 1e-12))
        & (candidates["delta_core_q99_cum_deficit_error_vs_R0"] <= 0.20 * max(abs(float(base_row["core_q99_cum_deficit_error"])), 1e-12))
    ].copy()
    if valid.empty:
        recommended = "R0_current_proposed"
        reason = "No new-loss experiment preserved both q99 and core-q99 within the 20% tolerance."
    else:
        valid["joint_shape_ramp_gain"] = (
            -valid["delta_netload_ramp_max_mae_vs_R0"] / max(abs(float(base_row["netload_ramp_max_mae"])), 1e-12)
            -valid["delta_highrisk_acf_mae_vs_R0"] / max(abs(float(base_row["highrisk_acf_mae"])), 1e-12)
        )
        best = valid.sort_values("joint_shape_ramp_gain", ascending=False).iloc[0]
        recommended = str(best["experiment_name"])
        reason = str(best["recommendation_reason"])
    lines.extend(
        [
            f"- recommended_config: `{recommended}`",
            f"- recommendation_reason: {reason}",
            "",
            "## Baseline R0",
            f"- highrisk_acf_mae: {float(base_row['highrisk_acf_mae']):.6f}",
            f"- netload_ramp_max_mae: {float(base_row['netload_ramp_max_mae']):.6f}",
            f"- q99_cum_deficit_error: {float(base_row['q99_cum_deficit_error']):.6f}",
            f"- core_q99_cum_deficit_error: {float(base_row['core_q99_cum_deficit_error']):.6f}",
            "",
            "## Judgement Rules",
            "- netload_ramp_max_mae decreases by at least 20%: ramp improvement is effective.",
            "- highrisk_acf_mae decreases by at least 10%: temporal shape improvement is effective.",
            "- q99/core_q99 degradation within 20%: tail-risk advantage is preserved.",
            "",
            "## Summary Table",
            summary_df.round(6).to_markdown(index=False),
        ]
    )
    (out_dir / "shape_ramp_tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_shape_ramp_tuning(cfg: ShapeRampTuningConfig) -> pd.DataFrame:
    out_dir = Path(cfg.out_dir)
    data_dir = Path(cfg.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, loss_cfg in EXPERIMENTS.items():
        print(f"\n=== Running {name} ===")
        exp_dir = out_dir / name
        model_dir = exp_dir / "models" / "proposed"
        eval_dir = exp_dir / "evaluations" / "proposed"
        model_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        if not (cfg.reuse_existing and (model_dir / "generated_samples.npy").exists()):
            train_cfg = TrainConfig(**{**_base_train_kwargs(cfg, model_dir), **loss_cfg})
            train_model(train_cfg)
            generate_from_checkpoint(
                GenerationConfig(
                    checkpoint=None,
                    data_dir=str(data_dir),
                    out_dir=str(model_dir),
                    split="test",
                    guidance_scale=1.0,
                    checkpoint_type=cfg.checkpoint_type,
                )
            )
        metrics = _evaluate_model(data_dir, model_dir, eval_dir, name)
        gen_meta = _generation_meta(model_dir)
        row = {"experiment_name": name, **loss_cfg}
        for metric in PAPER_METRICS:
            row[metric] = metrics.get(metric, np.nan)
        row["checkpoint_stage_used_for_generation"] = gen_meta.get("checkpoint_stage")
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    base = summary_df.loc[summary_df["experiment_name"] == "R0_current_proposed"].iloc[0]
    for metric in ["highrisk_acf_mae", "netload_ramp_max_mae", "q99_cum_deficit_error", "core_q99_cum_deficit_error"]:
        summary_df[f"delta_{metric}_vs_R0"] = pd.to_numeric(summary_df[metric], errors="coerce") - float(base[metric])
    summary_df["recommendation_reason"] = summary_df.apply(lambda row: _recommend(row, base), axis=1)
    ordered_cols = [
        "experiment_name",
        *PAPER_METRICS,
        "delta_highrisk_acf_mae_vs_R0",
        "delta_netload_ramp_max_mae_vs_R0",
        "delta_q99_cum_deficit_error_vs_R0",
        "delta_core_q99_cum_deficit_error_vs_R0",
        "recommendation_reason",
        "lambda_delta_net_stage2",
        "lambda_shape_stage2",
        "lambda_delta_net_stage3",
        "lambda_ramp_topk",
        "lambda_shape_stage3",
        "checkpoint_stage_used_for_generation",
    ]
    summary_df = summary_df[ordered_cols]
    summary_df.to_csv(out_dir / "shape_ramp_tuning_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(summary_df, out_dir)
    return summary_df


def parse_args() -> ShapeRampTuningConfig:
    parser = argparse.ArgumentParser(description="Tune optional net-load shape and ramp losses for proposed diffusion.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "shape_ramp_tuning"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage3-epochs", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-type", type=str, default="best-risk", choices=["best", "best-risk", "final"])
    parser.add_argument("--ramp-topk-ratio", type=float, default=0.10)
    parser.add_argument("--reuse-existing", action="store_true")
    return ShapeRampTuningConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_shape_ramp_tuning(parse_args())
