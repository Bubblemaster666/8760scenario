from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from augment_risk_shapelet_trainset import RiskShapeletAugConfig, augment_risk_shapelet_trainset
from build_pretrain_windows import PretrainWindowConfig, build_pretrain_windows
from evaluate_generation import EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from risk_metrics import batch_hard_risk_metrics
from train_hierarchical_evt_diffusion import TrainConfig, train_model


@dataclass
class PretrainAugRunConfig:
    data_dir: str
    out_dir: str
    timeseries_csv: str | None = None
    device: str = "cpu"
    seed: int = 42
    seq_len: int = 36
    stage0_epochs: int = 10
    pretrain_max_windows: int = 500
    pretrain_stride_hours: float = 6.0
    batch_size: int = 32
    diffusion_steps: int = 100
    base_channels: int = 64
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2
    checkpoint_type: str = "best-risk"
    reuse_existing: bool = False


def _resolve_timeseries_csv(data_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = data_dir / "timeseries_input.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Normal-window pretraining requires a long timeseries CSV. Not found: {path}. "
            "Pass --timeseries-csv explicitly."
        )
    return path


def _base_train_config(cfg: PretrainAugRunConfig, model_dir: Path) -> dict[str, Any]:
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
        "device": cfg.device,
    }


def _load_generation_checkpoint_meta(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "generation_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _highrisk_metrics(data_dir: Path, generated_path: Path) -> dict[str, float]:
    real = np.load(data_dir / "X_test.npy").astype(np.float32)
    generated = np.load(generated_path).astype(np.float32)
    cond = pd.read_csv(data_dir / "cond_test.csv")
    tau = pd.to_numeric(cond["imbalance_tau"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta_t = float(pd.to_numeric(cond.get("delta_t_hours", pd.Series([1.0])), errors="coerce").fillna(1.0).iloc[0])
    real_risk = batch_hard_risk_metrics(real, tau=tau, delta_t_hours=delta_t)
    gen_risk = batch_hard_risk_metrics(generated, tau=tau, delta_t_hours=delta_t)

    def subset(prefix: str, mask: np.ndarray) -> dict[str, float]:
        if not mask.any():
            return {
                f"{prefix}_cum_deficit_mae": float("nan"),
                f"{prefix}_q99_cum_deficit_error": float("nan"),
                f"{prefix}_duration_mae": float("nan"),
            }
        real_cum = real_risk["cum_deficit"][mask]
        gen_cum = gen_risk["cum_deficit"][mask]
        return {
            f"{prefix}_cum_deficit_mae": float(np.mean(np.abs(gen_cum - real_cum))),
            f"{prefix}_q99_cum_deficit_error": float(abs(np.quantile(gen_cum, 0.99) - np.quantile(real_cum, 0.99))),
            f"{prefix}_duration_mae": float(np.mean(np.abs(gen_risk["imbalance_duration"][mask] - real_risk["imbalance_duration"][mask]))),
        }

    severity = pd.to_numeric(cond.get("severity_level", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
    event_type = cond.get("event_type", pd.Series([""] * len(cond))).astype(str)
    highrisk_mask = severity >= 2
    heat_mask = event_type.str.contains("高温", na=False).to_numpy() & (severity >= 1)
    out = subset("highrisk", highrisk_mask)
    out.update(
        {
            "heat_highrisk_cum_deficit_mae": subset("heat_highrisk", heat_mask)["heat_highrisk_cum_deficit_mae"],
            "heat_highrisk_q99_cum_deficit_error": subset("heat_highrisk", heat_mask)["heat_highrisk_q99_cum_deficit_error"],
        }
    )
    return out


def _run_one(
    cfg: PretrainAugRunConfig,
    experiment_name: str,
    method_name: str,
    train_kwargs: dict[str, Any],
    checkpoint_type: str,
) -> dict[str, Any]:
    data_dir = Path(cfg.data_dir)
    exp_dir = Path(cfg.out_dir) / experiment_name
    model_dir = exp_dir / "models" / method_name
    eval_dir = exp_dir / "evaluations" / method_name
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    generated_path = model_dir / "generated_samples.npy"
    if not (cfg.reuse_existing and generated_path.exists()):
        train_cfg = TrainConfig(**{**_base_train_config(cfg, model_dir), **train_kwargs})
        train_model(train_cfg)
        generate_from_checkpoint(
            GenerationConfig(
                checkpoint=None,
                data_dir=str(data_dir),
                out_dir=str(model_dir),
                split="test",
                guidance_scale=1.0,
                checkpoint_type=checkpoint_type,
            )
        )

    event_mask_path = data_dir / "event_mask_test.npy"
    eval_summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(generated_path),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=method_name,
            event_mask=str(event_mask_path) if event_mask_path.exists() else None,
        )
    )
    gen_meta = _load_generation_checkpoint_meta(model_dir)
    row = {
        "experiment_name": experiment_name,
        "method_name": method_name,
        "checkpoint_type_used_for_generation": gen_meta.get("resolved_checkpoint_type"),
        "checkpoint_stage_used_for_generation": gen_meta.get("checkpoint_stage"),
        "stage0_epochs": int(train_kwargs.get("stage0_epochs", 0)),
        "use_pretrain": bool(train_kwargs.get("use_pretrain", False)),
        "use_risk_shapelet_aug": bool(train_kwargs.get("use_risk_shapelet_aug", False)),
        "risk_shapelet_aug_dir": train_kwargs.get("risk_shapelet_aug_dir"),
    }
    row.update(eval_summary["metrics"])
    row.update(_highrisk_metrics(data_dir, generated_path))
    pd.DataFrame([row]).to_csv(exp_dir / "evaluations" / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    return row


def _safe_ratio(value: float, base: float) -> float:
    if not np.isfinite(value) or not np.isfinite(base) or abs(base) < 1e-12:
        return float("nan")
    return float(value / base)


def _write_report(summary_df: pd.DataFrame, out_dir: Path) -> None:
    base = summary_df.loc[summary_df["experiment_name"] == "B0_proposed_E0"]
    if base.empty:
        report = "# Pretrain/Augmentation Report\n\nB0 baseline is missing, so no recommendation can be made.\n"
        (out_dir / "pretrain_aug_report.md").write_text(report, encoding="utf-8")
        return
    base_row = base.iloc[0]
    candidates = []
    for _, row in summary_df.iterrows():
        if row["experiment_name"] == "B0_proposed_E0":
            continue
        stat_ok = (
            _safe_ratio(float(row.get("mean_wasserstein", np.nan)), float(base_row.get("mean_wasserstein", np.nan))) <= 1.25
            and _safe_ratio(float(row.get("mean_js", np.nan)), float(base_row.get("mean_js", np.nan))) <= 1.25
            and _safe_ratio(float(row.get("acf_mae", np.nan)), float(base_row.get("acf_mae", np.nan))) <= 1.30
            and _safe_ratio(float(row.get("corr_matrix_error", np.nan)), float(base_row.get("corr_matrix_error", np.nan))) <= 1.30
        )
        risk_score = (
            float(row.get("cum_deficit_mae", np.inf))
            + float(row.get("q99_cum_deficit_error", np.inf))
            + float(row.get("highrisk_q99_cum_deficit_error", np.inf))
        )
        base_risk_score = (
            float(base_row.get("cum_deficit_mae", np.inf))
            + float(base_row.get("q99_cum_deficit_error", np.inf))
            + float(base_row.get("highrisk_q99_cum_deficit_error", np.inf))
        )
        label = "candidate"
        if stat_ok and risk_score < base_risk_score:
            label = "recommended_candidate"
        elif not stat_ok:
            label = "not recommended due to statistical degradation"
        elif risk_score >= base_risk_score:
            label = "risk not improved over B0"
        candidates.append((label, risk_score, row["experiment_name"], stat_ok))
    recommended = sorted([item for item in candidates if item[0] == "recommended_candidate"], key=lambda x: x[1])
    recommended_name = recommended[0][2] if recommended else "B0_proposed_E0"

    def improves(exp: str, metric: str) -> str:
        row = summary_df.loc[summary_df["experiment_name"] == exp]
        if row.empty:
            return "not run"
        value = float(row.iloc[0].get(metric, np.nan))
        base_value = float(base_row.get(metric, np.nan))
        if not np.isfinite(value) or not np.isfinite(base_value):
            return "insufficient data"
        return "yes" if value < base_value else "no"

    lines = [
        "# Pretrain And Risk-Shapelet Augmentation Report",
        "",
        f"recommended_config: `{recommended_name}`",
        "",
        "## Key Judgement",
        f"- B3 pretraining improves Wasserstein: {improves('B3_proposed_pretrain', 'mean_wasserstein')}; improves q99: {improves('B3_proposed_pretrain', 'q99_cum_deficit_error')}.",
        f"- B1 light risk-shapelet augmentation improves q99/highrisk_q99: {improves('B1_proposed_riskshape_aug_light', 'q99_cum_deficit_error')} / {improves('B1_proposed_riskshape_aug_light', 'highrisk_q99_cum_deficit_error')}.",
        f"- B2 strict risk-shapelet augmentation improves q99/highrisk_q99: {improves('B2_proposed_riskshape_aug_strict', 'q99_cum_deficit_error')} / {improves('B2_proposed_riskshape_aug_strict', 'highrisk_q99_cum_deficit_error')}.",
        f"- B4 pretrain + risk-shapelet augmentation improves q99/highrisk_q99: {improves('B4_proposed_pretrain_riskshape_aug', 'q99_cum_deficit_error')} / {improves('B4_proposed_pretrain_riskshape_aug', 'highrisk_q99_cum_deficit_error')}.",
        "",
        "## Interpretation",
        "- Normal-window pretraining is effective when it improves Wasserstein/ACF/correlation without weakening q99 cumulative deficit.",
        "- Risk-shapelet augmentation is effective when q99 or highrisk_q99 improves and the statistical metrics stay within the B0 degradation thresholds.",
        "- If only high-risk subsets improve, keep the method as an auxiliary experiment rather than replacing the main method.",
        "",
        "## Candidate Labels",
    ]
    for label, risk_score, name, stat_ok in candidates:
        lines.append(f"- `{name}`: {label}; risk_score={risk_score:.6g}; statistical_ok={stat_ok}")
    (out_dir / "pretrain_aug_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pretrain_finetune_artifacts(summary_df: pd.DataFrame, out_dir: Path) -> None:
    subset = summary_df[summary_df["experiment_name"].isin(["B0_proposed_E0", "B3_proposed_pretrain"])].copy()
    if subset.empty:
        return
    subset.to_csv(out_dir / "pretrain_finetune_summary.csv", index=False, encoding="utf-8-sig")
    base = subset.loc[subset["experiment_name"] == "B0_proposed_E0"]
    pre = subset.loc[subset["experiment_name"] == "B3_proposed_pretrain"]
    lines = ["# Pretrain Fine-Tune Report", ""]
    if base.empty or pre.empty:
        lines.append("B0 or B3 is missing; only the filtered summary CSV was written.")
    else:
        b = base.iloc[0]
        p = pre.iloc[0]
        lines.extend(
            [
                f"- Wasserstein improved: {float(p['mean_wasserstein']) < float(b['mean_wasserstein'])}",
                f"- ACF improved: {float(p['acf_mae']) < float(b['acf_mae'])}",
                f"- Correlation error improved: {float(p['corr_matrix_error']) < float(b['corr_matrix_error'])}",
                f"- cum_deficit_mae improved: {float(p['cum_deficit_mae']) < float(b['cum_deficit_mae'])}",
                f"- q99_cum_deficit_error improved: {float(p['q99_cum_deficit_error']) < float(b['q99_cum_deficit_error'])}",
                "",
                "If statistical realism improves but high-risk q99 worsens, keep pretraining as an auxiliary result or tune Stage 3 before replacing E0.",
            ]
        )
    (out_dir / "pretrain_finetune_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pretrain_aug_experiments(cfg: PretrainAugRunConfig) -> pd.DataFrame:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)
    timeseries_csv = _resolve_timeseries_csv(data_dir, cfg.timeseries_csv)

    pretrain_dir = out_dir / "_pretrain_windows"
    if not (cfg.reuse_existing and (pretrain_dir / "pretrain_X_train.npy").exists()):
        build_pretrain_windows(
            PretrainWindowConfig(
                timeseries_csv=str(timeseries_csv),
                out_dir=str(pretrain_dir),
                seq_len=cfg.seq_len,
                stride_hours=cfg.pretrain_stride_hours,
                max_windows=cfg.pretrain_max_windows,
                seed=cfg.seed,
            )
        )

    aug_light_dir = out_dir / "_riskshape_aug_light"
    if not (cfg.reuse_existing and (aug_light_dir / "X_train_riskshape_aug.npy").exists()):
        augment_risk_shapelet_trainset(
            RiskShapeletAugConfig(
                data_dir=cfg.data_dir,
                out_dir=str(aug_light_dir),
                risk_tolerance_cum=0.20,
                risk_tolerance_ramp=0.30,
                risk_tolerance_dur=0.30,
                seed=cfg.seed,
            )
        )
    aug_strict_dir = out_dir / "_riskshape_aug_strict"
    if not (cfg.reuse_existing and (aug_strict_dir / "X_train_riskshape_aug.npy").exists()):
        augment_risk_shapelet_trainset(
            RiskShapeletAugConfig(
                data_dir=cfg.data_dir,
                out_dir=str(aug_strict_dir),
                risk_tolerance_cum=0.10,
                risk_tolerance_ramp=0.20,
                risk_tolerance_dur=0.20,
                seed=cfg.seed,
            )
        )

    experiments = [
        ("B0_proposed_E0", "proposed", {}),
        (
            "B1_proposed_riskshape_aug_light",
            "proposed_riskshape_aug",
            {"use_risk_shapelet_aug": True, "risk_shapelet_aug_dir": str(aug_light_dir)},
        ),
        (
            "B2_proposed_riskshape_aug_strict",
            "proposed_riskshape_aug",
            {"use_risk_shapelet_aug": True, "risk_shapelet_aug_dir": str(aug_strict_dir)},
        ),
        (
            "B3_proposed_pretrain",
            "proposed_pretrain",
            {"use_pretrain": True, "pretrain_data_dir": str(pretrain_dir), "stage0_epochs": cfg.stage0_epochs},
        ),
        (
            "B4_proposed_pretrain_riskshape_aug",
            "proposed_pretrain_riskshape_aug",
            {
                "use_pretrain": True,
                "pretrain_data_dir": str(pretrain_dir),
                "stage0_epochs": cfg.stage0_epochs,
                "use_risk_shapelet_aug": True,
                "risk_shapelet_aug_dir": str(aug_light_dir),
            },
        ),
    ]

    rows = []
    for exp_name, method_name, train_kwargs in experiments:
        print(f"\n=== Running {exp_name} ({method_name}) ===")
        rows.append(_run_one(cfg, exp_name, method_name, train_kwargs, cfg.checkpoint_type))

    summary_df = pd.DataFrame(rows)
    base = summary_df.loc[summary_df["experiment_name"] == "B0_proposed_E0"].iloc[0]
    for metric in [
        "cum_deficit_mae",
        "q99_cum_deficit_error",
        "highrisk_q99_cum_deficit_error",
        "mean_wasserstein",
        "acf_mae",
        "corr_matrix_error",
    ]:
        if metric in summary_df.columns:
            summary_df[f"delta_{metric}"] = summary_df[metric].astype(float) - float(base[metric])
            summary_df[f"improve_{metric}_pct"] = (float(base[metric]) - summary_df[metric].astype(float)) / max(abs(float(base[metric])), 1e-12) * 100.0

    summary_path = out_dir / "pretrain_aug_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    _write_pretrain_finetune_artifacts(summary_df, out_dir)
    _write_report(summary_df, out_dir)
    return summary_df


def parse_args() -> PretrainAugRunConfig:
    parser = argparse.ArgumentParser(description="Run proposed E0 vs pretraining and risk-shapelet augmentation experiments.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--timeseries-csv", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--stage0-epochs", type=int, default=10)
    parser.add_argument("--pretrain-max-windows", type=int, default=500)
    parser.add_argument("--pretrain-stride-hours", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage3-epochs", type=int, default=2)
    parser.add_argument("--checkpoint-type", type=str, default="best-risk", choices=["best", "best-risk", "final"])
    parser.add_argument("--reuse-existing", action="store_true")
    return PretrainAugRunConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_pretrain_aug_experiments(parse_args())
