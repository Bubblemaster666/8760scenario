from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from risk_evt_transfer_augmentation import RiskEVTTransferConfig, run_risk_evt_transfer_augmentation
from risk_metrics import batch_hard_risk_metrics
from train_hierarchical_evt_diffusion import TrainConfig, train_model


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"
DEFAULT_OUT_DIR = BASE_DIR / "outputs" / "evt_transfer_experiments"


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "C0_proposed_E0": {
        "use_transfer": False,
        "risk_mode": "composite",
        "epsilon": 0.70,
        "num_transfer_iters": 0,
        "candidate_multiplier": 0,
        "keep_original_ratio": 1.0,
        "resample_strategy": "mixed",
    },
    "C1_evttransfer_cum": {
        "use_transfer": True,
        "risk_mode": "cum",
        "epsilon": 0.70,
        "num_transfer_iters": 3,
        "candidate_multiplier": 3,
        "keep_original_ratio": 0.30,
        "resample_strategy": "mixed",
    },
    "C2_evttransfer_composite": {
        "use_transfer": True,
        "risk_mode": "composite",
        "epsilon": 0.70,
        "num_transfer_iters": 3,
        "candidate_multiplier": 3,
        "keep_original_ratio": 0.30,
        "resample_strategy": "mixed",
    },
    "C3_evttransfer_strict": {
        "use_transfer": True,
        "risk_mode": "strict",
        "epsilon": 0.75,
        "num_transfer_iters": 3,
        "candidate_multiplier": 5,
        "keep_original_ratio": 0.20,
        "resample_strategy": "mixed",
    },
    "C4_evttransfer_more": {
        "use_transfer": True,
        "risk_mode": "composite",
        "epsilon": 0.75,
        "num_transfer_iters": 5,
        "candidate_multiplier": 5,
        "keep_original_ratio": 0.20,
        "resample_strategy": "mixed",
    },
}


@dataclass
class EVTTransferRunConfig:
    exp: str
    data_dir: str = str(DEFAULT_DATA_DIR)
    out_dir: str = str(DEFAULT_OUT_DIR)
    device: str = "cpu"
    seed: int = 42
    seq_len: int = 36
    checkpoint_type: str = "best-risk"
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2
    diffusion_steps: int = 100
    base_channels: int = 64
    batch_size: int = 32
    wgan_epochs: int = 120
    gan_base_channels: int = 48
    n_critic: int = 3
    reuse_existing: bool = False
    override_num_transfer_iters: int | None = None
    override_candidate_multiplier: int | None = None


def _train_config(cfg: EVTTransferRunConfig, model_dir: Path, augmented_train_dir: Path | None) -> TrainConfig:
    return TrainConfig(
        data_dir=cfg.data_dir,
        out_dir=str(model_dir),
        ablation="full",
        seed=cfg.seed,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        diffusion_steps=cfg.diffusion_steps,
        base_channels=cfg.base_channels,
        guidance_scale=1.0,
        cond_dropout=0.10,
        ema_decay=0.995,
        lr=1e-4,
        weight_decay=1e-5,
        stage1_epochs=cfg.stage1_epochs,
        stage2_epochs=cfg.stage2_epochs,
        stage3_epochs=cfg.stage3_epochs,
        lambda_tail=0.25,
        lambda_risk=0.04,
        lambda_cum=1.0,
        lambda_ramp=0.25,
        lambda_dur=0.35,
        lambda_recon=0.05,
        lambda_physics=0.02,
        lambda_resource=0.02,
        use_augmented_train=augmented_train_dir is not None,
        augmented_train_dir=str(augmented_train_dir) if augmented_train_dir is not None else None,
        device=cfg.device,
    )


def _generation_meta(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "generation_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _highrisk_metrics(data_dir: Path, generated_path: Path) -> dict[str, float]:
    real = np.load(data_dir / "X_test.npy").astype(np.float32)
    generated = np.load(generated_path).astype(np.float32)
    cond = pd.read_csv(data_dir / "cond_test.csv")
    tau = pd.to_numeric(cond["imbalance_tau"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta_t = float(pd.to_numeric(cond.get("delta_t_hours", pd.Series([1.0])), errors="coerce").fillna(1.0).iloc[0])
    real_risk = batch_hard_risk_metrics(real, tau=tau, delta_t_hours=delta_t)
    gen_risk = batch_hard_risk_metrics(generated, tau=tau, delta_t_hours=delta_t)
    severity = pd.to_numeric(cond.get("severity_level", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
    mask = severity >= 2
    if not mask.any():
        return {"highrisk_q99_cum_deficit_error": float("nan"), "highrisk_cum_deficit_mae": float("nan")}
    real_cum = real_risk["cum_deficit"][mask]
    gen_cum = gen_risk["cum_deficit"][mask]
    return {
        "highrisk_q99_cum_deficit_error": float(abs(np.quantile(gen_cum, 0.99) - np.quantile(real_cum, 0.99))),
        "highrisk_cum_deficit_mae": float(np.mean(np.abs(gen_cum - real_cum))),
    }


def _write_comparison_summary(root_out: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in root_out.glob("C*/metrics_row.json"):
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("experiment_name")
    c0 = df.loc[df["experiment_name"] == "C0_proposed_E0"]
    if not c0.empty:
        base = c0.iloc[0]
        for metric in [
            "mean_wasserstein",
            "mean_js",
            "acf_mae",
            "corr_matrix_error",
            "cum_deficit_mae",
            "q95_cum_deficit_error",
            "q99_cum_deficit_error",
            "highrisk_q99_cum_deficit_error",
        ]:
            if metric in df.columns and metric in base:
                df[f"delta_vs_C0_{metric}"] = pd.to_numeric(df[metric], errors="coerce") - float(base[metric])
    df.to_csv(root_out / "evt_transfer_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# EVT Transfer Experiment Summary", ""]
    if c0.empty:
        lines.append("C0 baseline is not available yet. Run `python run_evt_transfer_experiment.py --exp C0_proposed_E0` for direct deltas.")
    else:
        lines.append("Lower is better for the listed metrics. Deltas are relative to `C0_proposed_E0`.")
    key_cols = [
        "experiment_name",
        "mean_wasserstein",
        "acf_mae",
        "cum_deficit_mae",
        "q99_cum_deficit_error",
        "highrisk_q99_cum_deficit_error",
    ]
    lines.append("")
    lines.append(df[[c for c in key_cols if c in df.columns]].to_markdown(index=False))
    (root_out / "evt_transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def run_evt_transfer_experiment(cfg: EVTTransferRunConfig) -> dict[str, Any]:
    if cfg.exp not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {cfg.exp}. Available: {list(EXPERIMENTS)}")
    exp_cfg = dict(EXPERIMENTS[cfg.exp])
    if cfg.override_num_transfer_iters is not None:
        exp_cfg["num_transfer_iters"] = int(cfg.override_num_transfer_iters)
    if cfg.override_candidate_multiplier is not None:
        exp_cfg["candidate_multiplier"] = int(cfg.override_candidate_multiplier)

    data_dir = Path(cfg.data_dir)
    root_out = Path(cfg.out_dir)
    exp_dir = root_out / cfg.exp
    aug_dir = exp_dir / "augmentation"
    model_dir = exp_dir / "models" / "proposed"
    eval_dir = exp_dir / "evaluations" / "proposed"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    augmented_train_dir: Path | None = None
    if exp_cfg["use_transfer"]:
        augmented_train_dir = aug_dir
        if not (cfg.reuse_existing and (aug_dir / "X_train_aug.npy").exists()):
            transfer_cfg = RiskEVTTransferConfig(
                data_dir=str(data_dir),
                out_dir=str(aug_dir),
                seq_len=cfg.seq_len,
                seed=cfg.seed,
                device=cfg.device,
                risk_mode=exp_cfg["risk_mode"],
                epsilon=float(exp_cfg["epsilon"]),
                num_transfer_iters=int(exp_cfg["num_transfer_iters"]),
                candidate_multiplier=int(exp_cfg["candidate_multiplier"]),
                keep_original_ratio=float(exp_cfg["keep_original_ratio"]),
                resample_strategy=str(exp_cfg["resample_strategy"]),
                wgan_epochs=cfg.wgan_epochs,
                batch_size=cfg.batch_size,
                gan_base_channels=cfg.gan_base_channels,
                n_critic=cfg.n_critic,
            )
            run_risk_evt_transfer_augmentation(transfer_cfg)

    generated_path = model_dir / "generated_samples.npy"
    if not (cfg.reuse_existing and generated_path.exists()):
        train_model(_train_config(cfg, model_dir, augmented_train_dir))
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

    event_mask_path = data_dir / "event_mask_test.npy"
    eval_summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(generated_path),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=cfg.exp,
            event_mask=str(event_mask_path) if event_mask_path.exists() else None,
        )
    )
    gen_meta = _generation_meta(model_dir)
    row = {
        "experiment_name": cfg.exp,
        "use_transfer": bool(exp_cfg["use_transfer"]),
        "risk_mode": exp_cfg["risk_mode"],
        "epsilon": exp_cfg["epsilon"],
        "num_transfer_iters": exp_cfg["num_transfer_iters"],
        "candidate_multiplier": exp_cfg["candidate_multiplier"],
        "keep_original_ratio": exp_cfg["keep_original_ratio"],
        "checkpoint_type_used_for_generation": gen_meta.get("resolved_checkpoint_type"),
        "checkpoint_stage_used_for_generation": gen_meta.get("checkpoint_stage"),
    }
    row.update(eval_summary["metrics"])
    row.update(_highrisk_metrics(data_dir, generated_path))
    (exp_dir / "metrics_row.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([row]).to_csv(exp_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    _write_comparison_summary(root_out)
    return row


def parse_args() -> EVTTransferRunConfig:
    parser = argparse.ArgumentParser(description="Run risk EVT-transfer augmentation experiments for the proposed diffusion model.")
    parser.add_argument("--exp", type=str, required=True, choices=list(EXPERIMENTS))
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--checkpoint-type", type=str, default="best-risk", choices=["best", "best-risk", "final"])
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage3-epochs", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wgan-epochs", type=int, default=120)
    parser.add_argument("--gan-base-channels", type=int, default=48)
    parser.add_argument("--n-critic", type=int, default=3)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--num-transfer-iters", dest="override_num_transfer_iters", type=int, default=None)
    parser.add_argument("--candidate-multiplier", dest="override_candidate_multiplier", type=int, default=None)
    return EVTTransferRunConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_evt_transfer_experiment(parse_args())
