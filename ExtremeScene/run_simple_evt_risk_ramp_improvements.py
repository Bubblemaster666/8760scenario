from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from risk_ranking_utils import (
    AUXILIARY_REALISM_METRICS,
    RISK_EVALUATION_EXPLANATION,
    RISK_MAIN_METRICS,
    RISK_RANKING_EXPLANATION,
    add_risk_score,
    write_risk_tables,
)
from run_simple_evt_risk_diffusion import evaluate_method, generate_simple, train_simple_model


BASE_DIR = Path(__file__).resolve().parent


BASELINE_SOURCES = {
    "Plain_Diffusion": BASE_DIR / "outputs" / "main_compare_e0_fixed" / "models" / "plain_diffusion_baseline" / "generation" / "generated_samples.npy",
    "A0_JRPD_best_3h": BASE_DIR / "outputs" / "ramp_window_retrain" / "models" / "RAMPDIAG4_JRPD_ramp3h" / "generated_samples.npy",
    "Simple_EVT_Risk_Diffusion": BASE_DIR / "results" / "simple_evt_risk_diffusion" / "generated_samples_simple.npy",
    "Simple_EVT_Risk_Diffusion_RampCurve": BASE_DIR / "results" / "simple_evt_risk_diffusion_rampcurve" / "generated_samples_simple_rampcurve.npy",
}


METHOD_CONFIGS = [
    {
        "method_name": "Simple_EVT_Risk_Diffusion_RampLevel",
        "use_ramp_level_condition": True,
        "use_ramp_sampler": False,
        "lambda_ramp": 0.4,
        "lambda_ramp_curve": 0.0,
    },
    {
        "method_name": "Simple_EVT_Risk_Diffusion_RampSampler",
        "use_ramp_level_condition": False,
        "use_ramp_sampler": True,
        "lambda_ramp": 0.4,
        "lambda_ramp_curve": 0.0,
    },
    {
        "method_name": "Simple_EVT_Risk_Diffusion_RampLevelSampler",
        "use_ramp_level_condition": True,
        "use_ramp_sampler": True,
        "lambda_ramp": 0.4,
        "lambda_ramp_curve": 0.0,
    },
    {
        "method_name": "Simple_EVT_Risk_Diffusion_RampLevelSampler_LightCurve",
        "use_ramp_level_condition": True,
        "use_ramp_sampler": True,
        "lambda_ramp": 0.35,
        "lambda_ramp_curve": 0.05,
    },
]


FULL_COMPARE_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "extreme_degree_match_rate",
    *RISK_MAIN_METRICS,
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
]


def _method_args(args: argparse.Namespace, cfg: dict[str, object], out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=str(args.data_dir),
        out_dir=str(out_dir),
        method_name=str(cfg["method_name"]),
        device=str(args.device),
        seed=int(args.seed),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        batch_size=int(args.batch_size),
        diffusion_steps=int(args.diffusion_steps),
        base_channels=int(args.base_channels),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        ema_decay=float(args.ema_decay),
        cond_dropout=float(args.cond_dropout),
        duration_temp=float(args.duration_temp),
        lambda_cum=1.0,
        lambda_ramp=float(cfg["lambda_ramp"]),
        lambda_ramp_curve=float(cfg["lambda_ramp_curve"]),
        use_core_ramp_curve_loss=False,
        lambda_core_ramp_curve=0.10,
        lambda_dur=0.30,
        lambda_phy=0.02,
        use_ramp_level_condition=bool(cfg["use_ramp_level_condition"]),
        use_ramp_sampler=bool(cfg["use_ramp_sampler"]),
        sampler_alpha_cum=float(args.sampler_alpha_cum),
        sampler_alpha_ramp=float(args.sampler_alpha_ramp),
        sampler_alpha_dur=float(args.sampler_alpha_dur),
    )


def _copy_and_eval_baselines(args: argparse.Namespace, out_dir: Path) -> list[dict[str, object]]:
    rows = []
    data_dir = Path(args.data_dir)
    for name, source in BASELINE_SOURCES.items():
        if not source.exists():
            print(f"warning: missing baseline {name}: {source}")
            continue
        target = out_dir / f"generated_samples_{name}.npy"
        shutil.copy2(source, target)
        rows.append(evaluate_method(name, target, data_dir, out_dir))
    return rows


def _run_new_methods(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame]]:
    rows = []
    dist_frames = []
    sampler_frames = []
    data_dir = Path(args.data_dir)
    for cfg in METHOD_CONFIGS:
        method_name = str(cfg["method_name"])
        method_out = out_dir / "method_runs" / method_name
        method_args = _method_args(args, cfg, method_out)
        trained = train_simple_model(method_args)
        generated = generate_simple(trained, method_args)
        target = out_dir / f"generated_samples_{method_name}.npy"
        shutil.copy2(generated, target)
        rows.append(evaluate_method(method_name, target, data_dir, out_dir))

        model_dir = method_out / "models" / method_name
        dist_path = model_dir / "ramp_level_distribution.csv"
        if dist_path.exists():
            frame = pd.read_csv(dist_path)
            frame.insert(0, "method", method_name)
            dist_frames.append(frame)
        sampler_path = model_dir / "sampler_weight_summary.csv"
        if sampler_path.exists():
            frame = pd.read_csv(sampler_path)
            frame.insert(0, "method", method_name)
            frame["alpha_cum"] = float(args.sampler_alpha_cum)
            frame["alpha_ramp"] = float(args.sampler_alpha_ramp) if bool(cfg["use_ramp_sampler"]) else 0.0
            frame["alpha_dur"] = float(args.sampler_alpha_dur)
            frame["use_ramp_sampler"] = bool(cfg["use_ramp_sampler"])
            sampler_frames.append(frame)
    return rows, dist_frames, sampler_frames


def _write_report(out_dir: Path, risk_main: pd.DataFrame, aux: pd.DataFrame, sampler_ablation: pd.DataFrame) -> None:
    best = risk_main.iloc[0].to_dict() if len(risk_main) else {}
    simple = risk_main.loc[risk_main["method"].eq("Simple_EVT_Risk_Diffusion")]
    best_method = str(best.get("method", ""))
    simple_rank = int(simple["risk_rank"].iloc[0]) if len(simple) else None
    lines = [
        "# Simple EVT Risk Ramp Improvements",
        "",
        "## Evaluation Logic",
        "",
        RISK_EVALUATION_EXPLANATION,
        RISK_RANKING_EXPLANATION,
        "",
        "## Methods",
        "",
        "- B1 RampLevel: c_simple + ramp_level one-hot condition.",
        "- B2 RampSampler: original c_simple + ramp-focused WeightedRandomSampler.",
        "- B3 RampLevelSampler: ramp_level condition + ramp-focused sampler.",
        "- B4 LightCurve: B3 + very light ramp_3h_curve_loss.",
        "",
        "## ramp_level",
        "",
        "ramp_level is split by train-set netload_ramp_max quantiles: q50/q75/q90 -> levels 0/1/2/3. Val/test reuse train thresholds only.",
        "",
        "## ramp sampler",
        "",
        "sample_weight = 1 + alpha_cum * rank(cum_deficit) + alpha_ramp * rank(netload_ramp_max) + alpha_dur * rank(imbalance_duration).",
        "",
        "## Main Risk Results",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Sampler Ablation",
        "",
        sampler_ablation.to_markdown(index=False) if len(sampler_ablation) else "No sampler rows.",
        "",
        "## Auxiliary Realism",
        "",
        aux[[col for col in ["method", *AUXILIARY_REALISM_METRICS, "realism_check_pass"] if col in aux.columns]].to_markdown(index=False),
        "",
        "## Conclusion",
        "",
        f"- Best method by risk_score: {best_method}",
        f"- Simple_EVT_Risk_Diffusion risk_rank: {simple_rank}",
        "- If a ramp-focused method improves ramp but worsens q99/core_q99, the result indicates a cumulative-tail vs ramp-shape tradeoff.",
    ]
    (out_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _copy_and_eval_baselines(args, out_dir)
    new_rows, dist_frames, sampler_frames = _run_new_methods(args, out_dir)
    rows.extend(new_rows)
    df = add_risk_score(pd.DataFrame(rows))
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir, method_col="method")
    if dist_frames:
        pd.concat(dist_frames, ignore_index=True).to_csv(out_dir / "ramp_level_distribution.csv", index=False, encoding="utf-8-sig")
    if sampler_frames:
        sampler_summary = pd.concat(sampler_frames, ignore_index=True)
        sampler_summary.to_csv(out_dir / "sampler_weight_summary.csv", index=False, encoding="utf-8-sig")
    sampler_methods = df[df["method"].astype(str).str.contains("RampSampler|RampLevelSampler", regex=True)].copy()
    sampler_methods["alpha_cum"] = float(args.sampler_alpha_cum)
    sampler_methods["alpha_ramp"] = float(args.sampler_alpha_ramp)
    sampler_methods["alpha_dur"] = float(args.sampler_alpha_dur)
    sampler_ablation_cols = [
        "method",
        "alpha_cum",
        "alpha_ramp",
        "alpha_dur",
        "q99_cum_deficit_error",
        "core_q99_cum_deficit_error",
        "netload_ramp_max_mae",
        "imbalance_duration_mae",
        "risk_score",
        "risk_rank",
    ]
    sampler_ablation = sampler_methods[[col for col in sampler_ablation_cols if col in sampler_methods.columns]]
    sampler_ablation.to_csv(out_dir / "ramp_sampler_ablation.csv", index=False, encoding="utf-8-sig")
    _write_report(out_dir, risk_main, aux, sampler_ablation)
    print(RISK_RANKING_EXPLANATION)
    print(risk_main.to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ramp-level and ramp-sampler improvements for Simple EVT risk diffusion.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "outputs" / "ramp_window_retrain" / "datasets" / "dataset_window_3h")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "simple_evt_risk_ramp_improvements")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--cond-dropout", type=float, default=0.10)
    parser.add_argument("--duration-temp", type=float, default=12.0)
    parser.add_argument("--sampler-alpha-cum", type=float, default=0.3)
    parser.add_argument("--sampler-alpha-ramp", type=float, default=0.8)
    parser.add_argument("--sampler-alpha-dur", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
