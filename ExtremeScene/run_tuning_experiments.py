from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from train_hierarchical_evt_diffusion import TrainConfig, train_model


RANK_WEIGHTS = {
    "cum_deficit_mae": 2.0,
    "q99_cum_deficit_error": 2.0,
    "mean_wasserstein": 1.0,
    "mean_js": 1.0,
    "acf_mae": 1.0,
    "corr_matrix_error": 1.0,
    "netload_ramp_max_mae": 0.8,
    "imbalance_duration_mae": 0.8,
}


@dataclass(frozen=True)
class TuningPreset:
    name: str
    stage1_epochs: int
    stage2_epochs: int
    stage3_epochs: int
    lambda_tail: float
    lambda_risk: float
    guidance_scale: float


@dataclass
class TuningConfig:
    data_dir: str
    out_dir: str
    variants: list[str]
    seq_len: int = 24
    diffusion_steps: int = 100
    base_channels: int = 64
    batch_size: int = 32
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
    device: str = "cpu"
    seed: int = 42


PRESETS = {
    "P1_tail_stable": TuningPreset("P1_tail_stable", 12, 10, 0, 0.25, 0.00, 1.0),
    "P2_risk_light": TuningPreset("P2_risk_light", 12, 10, 2, 0.25, 0.04, 1.0),
    "P3_risk_mid": TuningPreset("P3_risk_mid", 16, 10, 4, 0.25, 0.05, 1.0),
    "P4_tail_strong": TuningPreset("P4_tail_strong", 12, 10, 2, 0.35, 0.04, 1.0),
    "P5_low_guidance": TuningPreset("P5_low_guidance", 12, 10, 2, 0.25, 0.04, 0.8),
    "P6_high_guidance": TuningPreset("P6_high_guidance", 12, 10, 2, 0.25, 0.04, 1.2),
}


def parse_args() -> TuningConfig:
    parser = argparse.ArgumentParser(description="Run P1-P6 tuning experiments for the proposed method.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--variants", nargs="+", choices=sorted(PRESETS), default=list(PRESETS))
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TuningConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        variants=list(args.variants),
        seq_len=args.seq_len,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
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
        device=args.device,
        seed=args.seed,
    )


def add_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ok_mask = out["status"].eq("ok") if "status" in out.columns else pd.Series(True, index=out.index)
    rank_cols: list[str] = []
    for metric, weight in RANK_WEIGHTS.items():
        if metric not in out.columns:
            continue
        values = pd.to_numeric(out[metric], errors="coerce")
        rank_col = f"{metric}_rank"
        out[rank_col] = values[ok_mask].rank(method="min", ascending=True)
        out.loc[~ok_mask, rank_col] = pd.NA
        rank_cols.append(rank_col)
        out[f"{metric}_weighted_rank"] = out[rank_col] * weight

    weighted_cols = [f"{metric}_weighted_rank" for metric in RANK_WEIGHTS if f"{metric}_weighted_rank" in out.columns]
    total_weight = sum(weight for metric, weight in RANK_WEIGHTS.items() if f"{metric}_weighted_rank" in out.columns)
    if weighted_cols and total_weight > 0:
        out["rank_score"] = out[weighted_cols].sum(axis=1, min_count=1) / total_weight
        out.loc[~ok_mask, "rank_score"] = pd.NA
        out["rank_score_rank"] = pd.to_numeric(out["rank_score"], errors="coerce").rank(method="min", ascending=True)
    else:
        out["rank_score"] = pd.NA
        out["rank_score_rank"] = pd.NA
    return out


def run_one_variant(cfg: TuningConfig, preset: TuningPreset) -> dict[str, object]:
    data_dir = Path(cfg.data_dir)
    variant_dir = Path(cfg.out_dir) / preset.name
    eval_dir = variant_dir / "evaluation"
    variant_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(variant_dir),
        ablation="full",
        seed=cfg.seed,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        diffusion_steps=cfg.diffusion_steps,
        guidance_scale=preset.guidance_scale,
        cond_dropout=cfg.cond_dropout,
        base_channels=cfg.base_channels,
        ema_decay=cfg.ema_decay,
        stage1_epochs=preset.stage1_epochs,
        stage2_epochs=preset.stage2_epochs,
        stage3_epochs=preset.stage3_epochs,
        lambda_tail=preset.lambda_tail,
        lambda_risk=preset.lambda_risk,
        lambda_cum=cfg.lambda_cum,
        lambda_ramp=cfg.lambda_ramp,
        lambda_dur=cfg.lambda_dur,
        lambda_recon=cfg.lambda_recon,
        lambda_physics=cfg.lambda_physics,
        lambda_resource=cfg.lambda_resource,
        device=cfg.device,
    )

    train_summary = train_model(train_cfg)
    generate_from_checkpoint(
        GenerationConfig(
            checkpoint=str(variant_dir / "best_model.pt"),
            data_dir=str(data_dir),
            out_dir=str(variant_dir),
            split="test",
            guidance_scale=preset.guidance_scale,
        )
    )
    eval_summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(variant_dir / "generated_samples.npy"),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=preset.name,
        )
    )

    row: dict[str, object] = {
        "variant": preset.name,
        "model_name": preset.name,
        "status": "ok",
        "stage1_epochs": preset.stage1_epochs,
        "stage2_epochs": preset.stage2_epochs,
        "stage3_epochs": preset.stage3_epochs,
        "lambda_tail": preset.lambda_tail,
        "lambda_risk": preset.lambda_risk,
        "guidance_scale": preset.guidance_scale,
        "diffusion_steps": cfg.diffusion_steps,
        "base_channels": cfg.base_channels,
        "batch_size": cfg.batch_size,
        "lambda_cum": cfg.lambda_cum,
        "lambda_ramp": cfg.lambda_ramp,
        "lambda_dur": cfg.lambda_dur,
        "lambda_recon": cfg.lambda_recon,
        "lambda_physics": cfg.lambda_physics,
        "lambda_resource": cfg.lambda_resource,
        "cond_dropout": cfg.cond_dropout,
        "ema_decay": cfg.ema_decay,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "best_val_loss": train_summary.get("best_val_loss"),
        "variant_dir": str(variant_dir),
    }
    row.update(eval_summary["metrics"])
    return row


def run_tuning(cfg: TuningConfig) -> pd.DataFrame:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tuning_config.json").write_text(
        json.dumps({"config": asdict(cfg), "presets": {name: asdict(preset) for name, preset in PRESETS.items()}, "rank_weights": RANK_WEIGHTS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for name in cfg.variants:
        preset = PRESETS[name]
        try:
            rows.append(run_one_variant(cfg, preset))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "variant": name,
                    "model_name": name,
                    "status": "failed",
                    "error": str(exc),
                    "stage1_epochs": preset.stage1_epochs,
                    "stage2_epochs": preset.stage2_epochs,
                    "stage3_epochs": preset.stage3_epochs,
                    "lambda_tail": preset.lambda_tail,
                    "lambda_risk": preset.lambda_risk,
                    "guidance_scale": preset.guidance_scale,
                }
            )

    summary_df = add_rank_score(pd.DataFrame(rows))
    summary_df.to_csv(out_dir / "tuning_metrics_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


if __name__ == "__main__":
    run_tuning(parse_args())
