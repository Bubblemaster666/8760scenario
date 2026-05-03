from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from train_hierarchical_evt_diffusion import TrainConfig, train_model

BASE_DIR = Path(__file__).resolve().parent

MAIN_COMPARE_METHODS = [
    "traditional_gaussian_copula",
    "plain_diffusion_baseline",
    "improved_diffusion",
    "enhanced_gan",
    "proposed",
]

ABLATION_METHODS = [
    "proposed",
    "no_evt_strict",
    "no_evt",
    "no_risk_loss",
    "no_month",
    "flat_condition",
]


@dataclass
class ExperimentConfig:
    data_dir: str
    out_dir: str
    methods: list[str]
    seq_len: int = 24
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2
    batch_size: int = 32
    diffusion_steps: int = 100
    base_channels: int = 64
    plain_epochs: int = 40
    gan_epochs: int = 40
    device: str = "cpu"
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
    checkpoint_type: str = "best-risk"
    seed: int = 42


def _historical_resampling(data_dir: Path, out_dir: Path) -> Path:
    x_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    x_test = np.load(data_dir / "X_test.npy").astype(np.float32)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(x_train), size=len(x_test))
    generated = x_train[idx]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "generated_samples.npy", generated)
    summary = {"method": "traditional_baseline", "strategy": "historical_resampling", "num_generated": int(len(generated))}
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir / "generated_samples.npy"


def _run_subprocess(args: list[str]) -> None:
    subprocess.run(args, cwd=BASE_DIR, check=True)


def _run_external_pipeline(script_path: Path, extra_args: list[str]) -> None:
    _run_subprocess([sys.executable, str(script_path), *extra_args])


def _run_plain_diffusion_baseline(data_dir: Path, model_dir: Path, cfg: ExperimentConfig) -> Path:
    script = BASE_DIR / "plain_diffusion_baseline" / "plain_ddpm_baseline.py"
    _run_external_pipeline(
        script,
        [
            "pipeline",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(model_dir),
            "--epochs",
            str(cfg.plain_epochs),
            "--batch-size",
            str(cfg.batch_size),
            "--diffusion-steps",
            str(max(cfg.diffusion_steps, 64)),
            "--base-channels",
            str(max(cfg.base_channels, 48)),
            "--seed",
            str(cfg.seed),
            "--device",
            cfg.device,
        ],
    )
    return model_dir / "generation" / "generated_samples.npy"


def _run_enhanced_gan_baseline(data_dir: Path, model_dir: Path, cfg: ExperimentConfig) -> Path:
    script = BASE_DIR / "enhanced_gan_extreme_repro" / "enhanced_gan_extreme.py"
    _run_external_pipeline(
        script,
        [
            "pipeline",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(model_dir),
            "--epochs",
            str(cfg.gan_epochs),
            "--batch-size",
            str(cfg.batch_size),
            "--n-per-condition",
            "1",
            "--device",
            cfg.device,
            "--seed",
            str(cfg.seed),
        ],
    )
    return model_dir / "generation" / "generated_samples.npy"


def _run_traditional_copula_baseline(data_dir: Path, model_dir: Path, cfg: ExperimentConfig) -> Path:
    script = BASE_DIR / "traditional_statistical_extreme_baseline" / "traditional_copula_baseline.py"
    _run_external_pipeline(
        script,
        [
            "pipeline",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(model_dir),
            "--n-per-condition",
            "1",
            "--seed",
            str(cfg.seed),
            "--seq-len",
            str(cfg.seq_len),
        ],
    )
    return model_dir / "generation" / "generated_samples.npy"


def _train_and_generate(data_dir: Path, model_dir: Path, method_name: str, cfg: ExperimentConfig) -> Path:
    if method_name == "proposed":
        ablation = "full"
        s1, s2, s3 = cfg.stage1_epochs, cfg.stage2_epochs, cfg.stage3_epochs
        lambda_tail = cfg.lambda_tail
        lambda_risk = cfg.lambda_risk
    elif method_name in {"no_evt", "no_evt_continuous", "no_evt_strict", "no_risk_loss", "no_month", "flat_condition"}:
        ablation = "no_evt" if method_name == "no_evt_continuous" else method_name
        s1, s2, s3 = cfg.stage1_epochs, cfg.stage2_epochs, cfg.stage3_epochs
        lambda_tail = cfg.lambda_tail
        lambda_risk = 0.0 if method_name == "no_risk_loss" else cfg.lambda_risk
    elif method_name == "conditional_ddpm":
        ablation = "flat_condition"
        s1, s2, s3 = cfg.stage1_epochs + cfg.stage2_epochs + cfg.stage3_epochs, 0, 0
        lambda_tail = 0.0
        lambda_risk = 0.0
    elif method_name == "improved_diffusion":
        ablation = "no_risk_loss"
        s1, s2, s3 = cfg.stage1_epochs, cfg.stage2_epochs, cfg.stage3_epochs
        lambda_tail = cfg.lambda_tail
        lambda_risk = 0.0
    else:
        raise ValueError(f"Unsupported trainable method: {method_name}")

    train_cfg = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(model_dir),
        ablation=ablation,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        diffusion_steps=cfg.diffusion_steps,
        base_channels=cfg.base_channels,
        guidance_scale=cfg.guidance_scale,
        cond_dropout=cfg.cond_dropout,
        ema_decay=cfg.ema_decay,
        stage1_epochs=s1,
        stage2_epochs=s2,
        stage3_epochs=s3,
        lambda_tail=lambda_tail,
        lambda_risk=lambda_risk,
        lambda_cum=cfg.lambda_cum,
        lambda_ramp=cfg.lambda_ramp,
        lambda_dur=cfg.lambda_dur,
        lambda_recon=cfg.lambda_recon,
        lambda_physics=cfg.lambda_physics,
        lambda_resource=cfg.lambda_resource,
        device=cfg.device,
        seed=cfg.seed,
    )
    train_model(train_cfg)
    risk_trained_method = method_name in {"proposed", "no_evt", "no_evt_continuous", "no_evt_strict", "no_month", "flat_condition"}
    checkpoint_type = cfg.checkpoint_type if risk_trained_method else "best"
    guidance_scale = cfg.guidance_scale if method_name in {"proposed", "no_evt", "no_evt_continuous", "no_evt_strict", "no_risk_loss", "no_month", "flat_condition"} else None
    generate_from_checkpoint(
        GenerationConfig(
            checkpoint=None,
            data_dir=str(data_dir),
            out_dir=str(model_dir),
            split="test",
            guidance_scale=guidance_scale,
            checkpoint_type=checkpoint_type,
        )
    )
    return model_dir / "generated_samples.npy"


def run_experiments(cfg: ExperimentConfig) -> pd.DataFrame:
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    models_dir = out_dir / "models"
    evals_dir = out_dir / "evaluations"
    models_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for method in cfg.methods:
        model_dir = models_dir / method
        eval_dir = evals_dir / method
        status = "ok"
        try:
            if method == "traditional_baseline":
                generated_path = _historical_resampling(data_dir, model_dir)
            elif method == "traditional_gaussian_copula":
                generated_path = _run_traditional_copula_baseline(data_dir, model_dir, cfg)
            elif method == "plain_diffusion_baseline":
                generated_path = _run_plain_diffusion_baseline(data_dir, model_dir, cfg)
            elif method == "enhanced_gan":
                generated_path = _run_enhanced_gan_baseline(data_dir, model_dir, cfg)
            else:
                generated_path = _train_and_generate(data_dir, model_dir, method, cfg)

            eval_summary = evaluate_generation(
                EvalConfig(
                    real=str(data_dir / "X_test.npy"),
                    generated=str(generated_path),
                    cond=str(data_dir / "cond_test.csv"),
                    meta=str(data_dir / "meta_test.csv"),
                    out_dir=str(eval_dir),
                    model_name=method,
                )
            )
            row = {"model_name": method, "status": status}
            row.update(eval_summary["metrics"])
            metric_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            metric_rows.append({"model_name": method, "status": "failed", "error": str(exc)})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(evals_dir / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics_df


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run unified scenario-generation experiments.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--preset", type=str, default="custom", choices=["custom", "main", "ablation", "all"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
    )
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage3-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--plain-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--guidance-scale", "--proposed-guidance", dest="guidance_scale", type=float, default=1.0)
    parser.add_argument("--lambda-tail", type=float, default=0.25)
    parser.add_argument("--lambda-risk", type=float, default=0.04)
    parser.add_argument("--lambda-cum", type=float, default=1.0)
    parser.add_argument("--lambda-ramp", type=float, default=0.25)
    parser.add_argument("--lambda-dur", type=float, default=0.35)
    parser.add_argument("--lambda-recon", type=float, default=0.05)
    parser.add_argument("--lambda-physics", type=float, default=0.02)
    parser.add_argument("--lambda-resource", type=float, default=0.02)
    parser.add_argument("--cond-dropout", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--checkpoint-type", type=str, default="best-risk", choices=["best", "best-risk", "final"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.methods is not None:
        methods = list(args.methods)
    elif args.preset == "main":
        methods = MAIN_COMPARE_METHODS
    elif args.preset == "ablation":
        methods = ABLATION_METHODS
    elif args.preset == "all":
        methods = [*MAIN_COMPARE_METHODS, *[m for m in ABLATION_METHODS if m not in MAIN_COMPARE_METHODS]]
    else:
        methods = [*MAIN_COMPARE_METHODS, *[m for m in ABLATION_METHODS if m not in MAIN_COMPARE_METHODS]]
    return ExperimentConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        methods=methods,
        seq_len=args.seq_len,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        plain_epochs=args.plain_epochs,
        gan_epochs=args.gan_epochs,
        device=args.device,
        guidance_scale=args.guidance_scale,
        lambda_tail=args.lambda_tail,
        lambda_risk=args.lambda_risk,
        lambda_cum=args.lambda_cum,
        lambda_ramp=args.lambda_ramp,
        lambda_dur=args.lambda_dur,
        lambda_recon=args.lambda_recon,
        lambda_physics=args.lambda_physics,
        lambda_resource=args.lambda_resource,
        cond_dropout=args.cond_dropout,
        ema_decay=args.ema_decay,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_type=args.checkpoint_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_experiments(parse_args())
