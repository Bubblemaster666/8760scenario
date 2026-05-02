from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hierarchical_diffusion import (
    ConditionedWindowDataset,
    DiffusionScheduler,
    EMA,
    HierarchicalConditionalUNet1D,
    apply_physical_projection,
    compute_condition_normalizers,
    condition_dropout,
    denormalize_x_torch,
    load_split_arrays,
    physics_penalty,
    resource_consistency_loss,
    risk_consistency_loss,
)

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    data_dir: str = str(BASE_DIR / "mock_dataset_outputs")
    out_dir: str = str(BASE_DIR / "outputs" / "proposed")
    ablation: str = "full"
    seed: int = 42

    seq_len: int = 24
    in_channels: int = 3
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    guidance_scale: float = 1.0
    cond_dropout: float = 0.10

    base_channels: int = 64
    time_emb_dim: int = 128
    cond_emb_dim: int = 128
    ema_decay: float = 0.995

    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2

    lambda_tail: float = 0.25
    lambda_risk: float = 0.04
    lambda_cum: float = 1.0
    lambda_ramp: float = 0.25
    lambda_dur: float = 0.35
    lambda_recon: float = 0.05
    lambda_physics: float = 0.02
    lambda_resource: float = 0.02

    duration_temp: float = 12.0
    delta_t_hours: float = 1.0
    daylight_start_hour: int = 6
    daylight_end_hour: int = 18
    device: str = "auto"
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stage_name(epoch: int, cfg: TrainConfig) -> str:
    if epoch <= cfg.stage1_epochs:
        return "stage1_distribution"
    if epoch <= cfg.stage1_epochs + cfg.stage2_epochs:
        return "stage2_tail"
    return "stage3_risk"


def innovation_flags(cfg: TrainConfig) -> dict[str, bool]:
    flags = {
        "hierarchical_condition": cfg.ablation != "flat_condition",
        "evt_continuous_risk": cfg.ablation != "no_evt",
        "tail_sensitive_loss": True,
        "risk_consistency_loss": cfg.ablation != "no_risk_loss",
        "month_feature": cfg.ablation != "no_month",
        "resource_state_flags": True,
    }
    if cfg.ablation == "flat_condition":
        flags["hierarchical_condition"] = False
    return flags


def ablation_notes(cfg: TrainConfig) -> str:
    if cfg.ablation == "no_evt":
        return "EVT continuous extreme_prob and tail_score are removed by the condition builder; tail weights fall back to severity_level only."
    if cfg.ablation == "no_risk_loss":
        return "Risk targets remain in the condition path, but lambda_risk is forced to zero in stage 3."
    if cfg.ablation == "no_month":
        return "month_sin and month_cos are zeroed in the background condition while event type is retained."
    if cfg.ablation == "flat_condition":
        return "Background, process, and risk fields are encoded through one flat concat MLP instead of separate hierarchical encoders."
    return "Full proposed model: hierarchical conditions, EVT continuous risk, tail-sensitive loss, light risk consistency, month features, and resource flags."


def is_proposed_final_config(cfg: TrainConfig) -> bool:
    expected = {
        "ablation": "full",
        "stage1_epochs": 12,
        "stage2_epochs": 10,
        "stage3_epochs": 2,
        "batch_size": 32,
        "diffusion_steps": 100,
        "base_channels": 64,
        "guidance_scale": 1.0,
        "lambda_tail": 0.25,
        "lambda_risk": 0.04,
        "lambda_cum": 1.0,
        "lambda_ramp": 0.25,
        "lambda_dur": 0.35,
        "lambda_recon": 0.05,
        "lambda_physics": 0.02,
        "lambda_resource": 0.02,
        "cond_dropout": 0.10,
        "ema_decay": 0.995,
        "lr": 1e-4,
        "weight_decay": 1e-5,
    }
    cfg_dict = asdict(cfg)
    for key, expected_value in expected.items():
        actual = cfg_dict[key]
        if isinstance(expected_value, float):
            if abs(float(actual) - expected_value) > 1e-12:
                return False
        elif actual != expected_value:
            return False
    return True


def stage_weights(stage: str, cfg: TrainConfig) -> dict[str, float]:
    weights = {
        "lambda_tail": 0.0,
        "lambda_risk": 0.0,
        "lambda_recon": cfg.lambda_recon,
        "lambda_physics": cfg.lambda_physics,
        "lambda_resource": 0.0,
    }
    if stage == "stage2_tail":
        weights["lambda_tail"] = cfg.lambda_tail
        weights["lambda_resource"] = cfg.lambda_resource
    if stage == "stage3_risk":
        weights["lambda_tail"] = cfg.lambda_tail
        weights["lambda_risk"] = 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_risk
        weights["lambda_resource"] = cfg.lambda_resource
    return weights


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return requested


def tail_weight(proc_risk_cond: torch.Tensor, ablation: str) -> torch.Tensor:
    if ablation == "no_evt":
        severity_signal = proc_risk_cond[:, 2]
        return 1.0 + severity_signal.clamp(min=0.0)
    tail_signal = proc_risk_cond[:, 1]
    return 1.0 + F.relu(tail_signal)


def plot_history(history_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11.0, 6.0))
    for col in [
        "train_total_loss",
        "train_eps_loss",
        "train_tail_loss",
        "train_risk_loss",
        "val_total_loss",
        "val_eps_loss",
        "val_risk_loss",
    ]:
        if col in history_df.columns:
            plt.plot(history_df["epoch"], history_df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Hierarchical EVT Diffusion Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _forward_loss(
    model: HierarchicalConditionalUNet1D,
    scheduler: DiffusionScheduler,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    x_mean_t: torch.Tensor,
    x_std_t: torch.Tensor,
    risk_norm: dict[str, torch.Tensor],
    cfg: TrainConfig,
    stage: str,
    train_mode: bool,
) -> dict[str, torch.Tensor]:
    x, bg_cond, proc_cond, risk_cond, risk_targets, day_mask = batch
    x = x.to(device)
    bg_cond = bg_cond.to(device)
    proc_cond = proc_cond.to(device)
    risk_cond = risk_cond.to(device)
    risk_targets = risk_targets.to(device)
    day_mask = day_mask.to(device)

    if train_mode:
        bg_in, proc_in, risk_in = condition_dropout(bg_cond, proc_cond, risk_cond, cfg.cond_dropout)
    else:
        bg_in, proc_in, risk_in = bg_cond, proc_cond, risk_cond

    t = torch.randint(0, cfg.diffusion_steps, (x.size(0),), device=device)
    noise = torch.randn_like(x)
    x_t = scheduler.q_sample(x, t, noise)
    pred_noise = model(x_t, t, bg_in, proc_in, risk_in)

    per_sample_mse = ((pred_noise - noise) ** 2).mean(dim=(1, 2))
    eps_loss = per_sample_mse.mean()
    weights = tail_weight(risk_cond, cfg.ablation)
    tail_loss = ((weights - 1.0) * per_sample_mse).mean()
    tail_weight_mean = weights.mean()

    x0_pred_norm = scheduler.predict_x0(x_t, t, pred_noise).clamp(-5.0, 5.0)
    x0_pred = denormalize_x_torch(x0_pred_norm, x_mean_t, x_std_t)
    x_true = denormalize_x_torch(x, x_mean_t, x_std_t)
    x_proj = apply_physical_projection(x0_pred, day_mask)

    recon_loss = F.smooth_l1_loss(x_proj, x_true)
    physics_loss = physics_penalty(x0_pred, day_mask)
    resource_loss = resource_consistency_loss(x_proj, x_true, day_mask, proc_cond)
    raw_risk_loss, cum_loss, ramp_loss, dur_loss = risk_consistency_loss(
        x_proj=x_proj,
        risk_targets=risk_targets,
        delta_t_hours=cfg.delta_t_hours,
        duration_temp=cfg.duration_temp,
        risk_norm=risk_norm,
    )
    weighted_risk_loss = cfg.lambda_cum * cum_loss + cfg.lambda_ramp * ramp_loss + cfg.lambda_dur * dur_loss

    sw = stage_weights(stage, cfg)
    total_loss = (
        eps_loss
        + sw["lambda_tail"] * tail_loss
        + sw["lambda_risk"] * weighted_risk_loss
        + sw["lambda_recon"] * recon_loss
        + sw["lambda_physics"] * physics_loss
        + sw["lambda_resource"] * resource_loss
    )
    return {
        "total_loss": total_loss,
        "eps_loss": eps_loss,
        "tail_loss": tail_loss,
        "tail_weight_mean": tail_weight_mean,
        "risk_loss": weighted_risk_loss,
        "raw_risk_loss": raw_risk_loss,
        "cum_loss": cum_loss,
        "ramp_loss": ramp_loss,
        "dur_loss": dur_loss,
        "recon_loss": recon_loss,
        "physics_loss": physics_loss,
        "resource_loss": resource_loss,
    }


def evaluate(
    model: HierarchicalConditionalUNet1D,
    scheduler: DiffusionScheduler,
    loader: DataLoader,
    device: torch.device,
    x_mean_t: torch.Tensor,
    x_std_t: torch.Tensor,
    risk_norm: dict[str, torch.Tensor],
    cfg: TrainConfig,
    stage: str,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = _forward_loss(model, scheduler, batch, device, x_mean_t, x_std_t, risk_norm, cfg, stage, train_mode=False)
            bs = batch[0].size(0)
            count += bs
            for key, value in out.items():
                stats[key] = stats.get(key, 0.0) + float(value.item()) * bs
    return {key: value / max(count, 1) for key, value in stats.items()}


def train_model(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    torch.set_num_threads(1)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, cond_train, meta_train = load_split_arrays(cfg.data_dir, "train")
    x_val, cond_val, meta_val = load_split_arrays(cfg.data_dir, "val")
    x_mean = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    x_train_norm = ((x_train - x_mean) / x_std).astype(np.float32)
    x_val_norm = ((x_val - x_mean) / x_std).astype(np.float32)
    cond_normalizers = compute_condition_normalizers(cond_train)

    train_ds = ConditionedWindowDataset(
        x_train_norm,
        cond_train,
        meta_train,
        seq_len=cfg.seq_len,
        ablation=cfg.ablation,
        normalizers=cond_normalizers,
        daylight_start_hour=cfg.daylight_start_hour,
        daylight_end_hour=cfg.daylight_end_hour,
    )
    val_ds = ConditionedWindowDataset(
        x_val_norm,
        cond_val,
        meta_val,
        seq_len=cfg.seq_len,
        ablation=cfg.ablation,
        normalizers=cond_normalizers,
        daylight_start_hour=cfg.daylight_start_hour,
        daylight_end_hour=cfg.daylight_end_hour,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    risk_norm_np = {
        "log_cum_mean": np.float32(np.log1p(cond_train["cum_deficit"].astype(float).to_numpy()).mean()),
        "log_cum_std": np.float32(np.log1p(cond_train["cum_deficit"].astype(float).to_numpy()).std() + 1e-6),
        "ramp_mean": np.float32(cond_train["netload_ramp_max"].astype(float).to_numpy().mean()),
        "ramp_std": np.float32(cond_train["netload_ramp_max"].astype(float).to_numpy().std() + 1e-6),
        "dur_mean": np.float32(cond_train["imbalance_duration"].astype(float).to_numpy().mean()),
        "dur_std": np.float32(cond_train["imbalance_duration"].astype(float).to_numpy().std() + 1e-6),
    }
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)
    risk_norm_t = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in risk_norm_np.items()}

    flat_condition = cfg.ablation == "flat_condition"
    bg_dim = train_ds.background.shape[1]
    proc_dim = train_ds.process.shape[1]
    risk_dim = train_ds.risk.shape[1]
    model = HierarchicalConditionalUNet1D(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        time_dim=cfg.time_emb_dim,
        cond_dim=cfg.cond_emb_dim,
        bg_dim=bg_dim,
        proc_dim=proc_dim,
        risk_dim=risk_dim,
        flat_condition=flat_condition,
    ).to(device)
    ema_model = HierarchicalConditionalUNet1D(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        time_dim=cfg.time_emb_dim,
        cond_dim=cfg.cond_emb_dim,
        bg_dim=bg_dim,
        proc_dim=proc_dim,
        risk_dim=risk_dim,
        flat_condition=flat_condition,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema = EMA(model, cfg.ema_decay)
    scheduler = DiffusionScheduler(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    history_rows: list[dict[str, float | str | int]] = []
    total_epochs = cfg.stage1_epochs + cfg.stage2_epochs + cfg.stage3_epochs
    if total_epochs <= 0:
        raise ValueError("At least one stage epoch must be positive.")

    for epoch in range(1, total_epochs + 1):
        stage = stage_name(epoch, cfg)
        model.train()
        epoch_stats: dict[str, float] = {}
        count = 0

        for batch in train_loader:
            out = _forward_loss(model, scheduler, batch, device, x_mean_t, x_std_t, risk_norm_t, cfg, stage, train_mode=True)
            optimizer.zero_grad()
            out["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            bs = batch[0].size(0)
            count += bs
            for key, value in out.items():
                epoch_stats[key] = epoch_stats.get(key, 0.0) + float(value.item()) * bs

        train_stats = {f"train_{key}": value / max(count, 1) for key, value in epoch_stats.items()}
        ema.copy_to(ema_model)
        val_stats_raw = evaluate(ema_model, scheduler, val_loader, device, x_mean_t, x_std_t, risk_norm_t, cfg, stage)
        val_stats = {f"val_{key}": value for key, value in val_stats_raw.items()}

        row = {"epoch": epoch, "stage": stage}
        row.update(train_stats)
        row.update(val_stats)
        history_rows.append(row)

        if val_stats["val_total_loss"] < best_val:
            best_val = val_stats["val_total_loss"]
            checkpoint = {
                "model_state": ema_model.state_dict(),
                "train_config": asdict(cfg),
                "condition_meta": train_ds.condition_meta,
                "condition_normalizers": asdict(cond_normalizers),
                "risk_normalization": {k: float(v) for k, v in risk_norm_np.items()},
                "x_mean": x_mean,
                "x_std": x_std,
                "seq_len": cfg.seq_len,
                "cond_dims": {"background": bg_dim, "process": proc_dim, "risk": risk_dim},
                "flat_condition": flat_condition,
            }
            torch.save(checkpoint, out_dir / "best_model.pt")

        if epoch == 1 or epoch % 5 == 0 or epoch == total_epochs:
            print(
                f"Epoch {epoch:03d}/{total_epochs} | {stage} | "
                f"train_total={row['train_total_loss']:.4f} | "
                f"train_eps={row['train_eps_loss']:.4f} | "
                f"val_total={row['val_total_loss']:.4f}"
            )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_history(history_df, out_dir / "loss_curve.png")

    condition_meta = dict(train_ds.condition_meta)
    condition_meta["cond_dims"] = {"background": bg_dim, "process": proc_dim, "risk": risk_dim}
    condition_meta["flat_condition"] = flat_condition
    (out_dir / "condition_meta.json").write_text(json.dumps(condition_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(out_dir / "normalization_stats.npz", x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32))

    summary = {
        "ablation_name": cfg.ablation,
        "ablation_note": ablation_notes(cfg),
        "formal_config_name": "proposed_final" if is_proposed_final_config(cfg) else f"custom_{cfg.ablation}",
        "innovation_flags": innovation_flags(cfg),
        "device": str(device),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "seq_len": cfg.seq_len,
        "best_val_loss": float(best_val),
        "stage1_epochs": cfg.stage1_epochs,
        "stage2_epochs": cfg.stage2_epochs,
        "stage3_epochs": cfg.stage3_epochs,
        "flat_condition": flat_condition,
        "condition_dims": {"background": bg_dim, "process": proc_dim, "risk": risk_dim},
        "train_config": asdict(cfg),
        "stage_loss_policy": {
            "stage1_distribution": "eps_loss + lambda_recon * recon_loss + lambda_physics * physics_loss",
            "stage2_tail": "eps_loss + lambda_tail * tail_loss + light recon/physics/resource losses",
            "stage3_risk": "eps_loss + lambda_tail * tail_loss + lambda_risk * weighted_risk_loss + light recon/physics/resource losses",
            "weighted_risk_loss": "lambda_cum * cum_loss + lambda_ramp * ramp_loss + lambda_dur * dur_loss",
        },
        "loss_weights": {
            "lambda_tail": cfg.lambda_tail,
            "lambda_risk": 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_risk,
            "lambda_cum": cfg.lambda_cum,
            "lambda_ramp": cfg.lambda_ramp,
            "lambda_dur": cfg.lambda_dur,
            "lambda_recon": cfg.lambda_recon,
            "lambda_physics": cfg.lambda_physics,
            "lambda_resource": cfg.lambda_resource,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the hierarchical EVT-risk diffusion model.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "no_evt", "no_risk_loss", "no_month", "flat_condition"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4)
    parser.add_argument("--steps", "--diffusion-steps", dest="steps", type=int, default=100)
    parser.add_argument("--guidance", "--guidance-scale", dest="guidance", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage3-epochs", type=int, default=2)
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
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        ablation=args.ablation,
        seed=args.seed,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        diffusion_steps=args.steps,
        guidance_scale=args.guidance,
        base_channels=args.base_channels,
        cond_dropout=args.cond_dropout,
        ema_decay=args.ema_decay,
        weight_decay=args.weight_decay,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        lambda_tail=args.lambda_tail,
        lambda_risk=args.lambda_risk,
        lambda_cum=args.lambda_cum,
        lambda_ramp=args.lambda_ramp,
        lambda_dur=args.lambda_dur,
        lambda_recon=args.lambda_recon,
        lambda_physics=args.lambda_physics,
        lambda_resource=args.lambda_resource,
        device=args.device,
    )


if __name__ == "__main__":
    train_model(parse_args())
