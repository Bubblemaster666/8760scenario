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
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from risk_metrics import (
    highrisk_shape_moment_loss_torch,
    net_load_delta_loss_torch,
    ramp_topk_loss_torch,
    soft_core_risk_metrics_torch,
    soft_risk_metrics_torch,
    topk_tail_distribution_loss_torch,
)

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    data_dir: str = str(BASE_DIR / "mock_dataset_outputs")
    out_dir: str = str(BASE_DIR / "outputs" / "proposed")
    ablation: str = "full"
    seed: int = 42

    seq_len: int = 36
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
    sampler_mode: str = "none"
    severity_sample_weights: str = "0:1.0,1:1.5,2:2.5,3:3.5"
    tail_sampler_alpha: float = 0.5
    tail_weight_mode: str = "relu"
    tail_weight_alpha: float = 0.5
    tail_weight_max: float = 3.0
    lambda_tail_dist: float = 0.0
    tail_dist_topk_ratio: float = 0.10
    tail_dist_metric: str = "cum_deficit"
    lambda_core_risk: float = 0.0
    core_risk_mode: str = "off"
    lambda_core_cum: float = 1.0
    lambda_core_ramp: float = 0.15
    lambda_core_dur: float = 0.25
    # Stage 2 净负荷差分形态损失权重，用于防止尾部强化阶段破坏爬坡过程。
    lambda_delta_net_stage2: float = 0.0
    # Stage 3 净负荷差分形态损失权重，用于改善生成场景的净负荷变化率过程。
    lambda_delta_net_stage3: float = 0.0
    # top-k 正向爬坡尾部损失权重，用于改善最大净负荷爬坡误差。
    lambda_ramp_topk: float = 0.0
    # 选取净负荷正向爬坡序列中前多少比例作为尾部爬坡样本。
    ramp_topk_ratio: float = 0.10
    # Stage 2 高风险样本形态统计锚定损失权重，用于改善 highrisk_acf_mae。
    lambda_shape_stage2: float = 0.0
    # Stage 3 高风险样本形态统计锚定损失权重，用于改善 highrisk_acf_mae。
    lambda_shape_stage3: float = 0.0
    # 是否仅在 highrisk 样本上计算形态锚定损失。
    shape_highrisk_only: bool = True
    # Stage 3 持续失衡时长过高的单边惩罚权重，用于压低生成样本系统性 duration 偏长。
    lambda_duration_over_stage3: float = 0.0
    # Stage 3 逐时刻超阈值掩码损失权重，用于学习每个小时是否超过失衡阈值 tau。
    lambda_exceed_mask_stage3: float = 0.0

    duration_temp: float = 12.0
    delta_t_hours: float = 1.0
    daylight_start_hour: int = 6
    daylight_end_hour: int = 18
    use_augmented_train: bool = False
    use_pretrain: bool = False
    pretrain_data_dir: Optional[str] = None
    stage0_epochs: int = 0
    pretrain_lr: float = 1e-4
    pretrain_checkpoint: Optional[str] = None
    freeze_after_pretrain: bool = False
    pretrain_loss_mode: str = "distribution_only"
    augmented_train_dir: Optional[str] = None
    use_risk_shapelet_aug: bool = False
    risk_shapelet_aug_dir: Optional[str] = None
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
        "evt_continuous_risk": cfg.ablation not in {"no_evt", "no_evt_continuous", "no_evt_strict"},
        "tail_sensitive_loss": True,
        "risk_consistency_loss": cfg.ablation != "no_risk_loss",
        "month_feature": cfg.ablation != "no_month",
        "resource_state_flags": True,
    }
    if cfg.ablation == "no_evt_strict":
        flags["tail_sensitive_loss"] = False
    if cfg.ablation == "flat_condition":
        flags["hierarchical_condition"] = False
    return flags


def ablation_notes(cfg: TrainConfig) -> str:
    if cfg.ablation in {"no_evt", "no_evt_continuous"}:
        return "EVT continuous extreme_prob and tail_score are removed by the condition builder; tail weights fall back to severity_level only."
    if cfg.ablation == "no_evt_strict":
        return "The whole EVT risk layer is zeroed: extreme_prob, tail_score, and severity_level are all unavailable to the condition encoder."
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
        "lambda_tail_dist": 0.0,
        "lambda_core_risk": 0.0,
        "lambda_delta_net": 0.0,
        "lambda_ramp_topk": 0.0,
        "lambda_shape_moment": 0.0,
        "lambda_duration_over": 0.0,
        "lambda_exceed_mask": 0.0,
        "lambda_recon": cfg.lambda_recon,
        "lambda_physics": cfg.lambda_physics,
        "lambda_resource": 0.0,
    }
    if stage == "stage2_tail":
        weights["lambda_tail"] = cfg.lambda_tail
        weights["lambda_resource"] = cfg.lambda_resource
        weights["lambda_delta_net"] = cfg.lambda_delta_net_stage2
        weights["lambda_shape_moment"] = cfg.lambda_shape_stage2
    if stage == "stage3_risk":
        weights["lambda_tail"] = cfg.lambda_tail
        weights["lambda_risk"] = 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_risk
        weights["lambda_tail_dist"] = 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_tail_dist
        weights["lambda_core_risk"] = 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_core_risk
        weights["lambda_resource"] = cfg.lambda_resource
        weights["lambda_delta_net"] = cfg.lambda_delta_net_stage3
        weights["lambda_ramp_topk"] = cfg.lambda_ramp_topk
        weights["lambda_shape_moment"] = cfg.lambda_shape_stage3
        weights["lambda_duration_over"] = cfg.lambda_duration_over_stage3
        weights["lambda_exceed_mask"] = cfg.lambda_exceed_mask_stage3
    return weights


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return requested


def tail_weight(proc_risk_cond: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    ablation = cfg.ablation
    if ablation == "no_evt_strict":
        return torch.ones((proc_risk_cond.size(0),), dtype=proc_risk_cond.dtype, device=proc_risk_cond.device)
    if ablation in {"no_evt", "no_evt_continuous"}:
        severity_signal = proc_risk_cond[:, 2]
        return 1.0 + severity_signal.clamp(min=0.0)
    tail_signal = proc_risk_cond[:, 1]
    mode = cfg.tail_weight_mode.strip().lower()
    if mode == "relu":
        return 1.0 + F.relu(tail_signal)
    if mode == "sigmoid":
        return 1.0 + float(cfg.tail_weight_alpha) * torch.sigmoid(tail_signal)
    if mode == "clipped_relu":
        return torch.clamp(
            1.0 + float(cfg.tail_weight_alpha) * F.relu(tail_signal),
            min=1.0,
            max=float(cfg.tail_weight_max),
        )
    raise ValueError("tail_weight_mode must be one of {'relu', 'sigmoid', 'clipped_relu'}.")


def _parse_severity_sample_weights(text: str) -> dict[int, float]:
    weights = {0: 1.0, 1: 1.5, 2: 2.5, 3: 3.5}
    if not text:
        return weights
    for item in str(text).split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        weights[int(key.strip())] = float(value)
    return weights


def _build_train_sampler(cond_train: pd.DataFrame, cfg: TrainConfig) -> tuple[WeightedRandomSampler | None, dict]:
    mode = cfg.sampler_mode.strip().lower()
    if mode in {"none", ""}:
        return None, {"sampler_mode": "none", "enabled": False}
    if mode == "severity":
        severity_weights = _parse_severity_sample_weights(cfg.severity_sample_weights)
        severity = pd.to_numeric(cond_train.get("severity_level", 0), errors="coerce").fillna(0).astype(int)
        weights_np = severity.map(lambda level: severity_weights.get(int(level), 1.0)).to_numpy(dtype=np.float32)
        config = {"severity_sample_weights": {str(k): float(v) for k, v in severity_weights.items()}}
    elif mode == "tail_score":
        if "tail_score_zscore" in cond_train.columns:
            tail = pd.to_numeric(cond_train["tail_score_zscore"], errors="coerce").fillna(0.0).clip(0.0, 3.0)
            norm_tail = tail.to_numpy(dtype=np.float32)
        else:
            tail = pd.to_numeric(cond_train.get("tail_score", 0.0), errors="coerce").fillna(0.0)
            min_v = float(tail.min())
            max_v = float(tail.max())
            norm_tail = ((tail - min_v) / (max_v - min_v + 1e-6)).to_numpy(dtype=np.float32)
        weights_np = 1.0 + float(cfg.tail_sampler_alpha) * norm_tail
        config = {"tail_sampler_alpha": float(cfg.tail_sampler_alpha)}
    else:
        raise ValueError("sampler_mode must be one of {'none', 'severity', 'tail_score'}.")

    weights_np = np.asarray(weights_np, dtype=np.float32)
    weights_np = np.clip(weights_np, 1e-6, None)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights_np, dtype=torch.double),
        num_samples=len(weights_np),
        replacement=True,
        generator=generator,
    )
    summary = {
        "sampler_mode": mode,
        "enabled": True,
        "weight_min": float(weights_np.min()),
        "weight_max": float(weights_np.max()),
        "weight_mean": float(weights_np.mean()),
        "weight_sum": float(weights_np.sum()),
        **config,
    }
    return sampler, summary


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
    if len(batch) == 7:
        x, bg_cond, proc_cond, risk_cond, risk_targets, day_mask, event_mask = batch
    else:
        x, bg_cond, proc_cond, risk_cond, risk_targets, day_mask = batch
        event_mask = None
    x = x.to(device)
    bg_cond = bg_cond.to(device)
    proc_cond = proc_cond.to(device)
    risk_cond = risk_cond.to(device)
    risk_targets = risk_targets.to(device)
    day_mask = day_mask.to(device)
    event_mask = event_mask.to(device) if event_mask is not None else torch.ones((x.size(0), x.size(2)), device=device)

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
    weights = tail_weight(risk_cond, cfg)
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
    cum_pred, _, dur_pred = soft_risk_metrics_torch(
        x_proj,
        tau=risk_targets[:, 3],
        delta_t_hours=cfg.delta_t_hours,
        duration_temp=cfg.duration_temp,
    )
    # 单边压 duration：只惩罚生成持续失衡时长超过目标时长的部分。
    dur_over = F.relu((dur_pred - risk_targets[:, 2]) / risk_norm["dur_std"])
    duration_over_loss = F.smooth_l1_loss(dur_over, torch.zeros_like(dur_over))
    # 逐时刻超阈值掩码：直接约束哪些小时的 net_load 应超过 tau，用于改善 duration 形态。
    net_pred = x_proj[:, 0, :] - x_proj[:, 1, :] - x_proj[:, 2, :]
    net_true = x_true[:, 0, :] - x_true[:, 1, :] - x_true[:, 2, :]
    tau_t = risk_targets[:, 3].view(-1, 1)
    exceed_prob_pred = torch.sigmoid((net_pred - tau_t) * cfg.duration_temp)
    exceed_true = (net_true > tau_t).to(dtype=x_proj.dtype)
    exceed_mask_loss = F.binary_cross_entropy(exceed_prob_pred.clamp(1e-5, 1.0 - 1e-5), exceed_true)
    tail_dist_loss = topk_tail_distribution_loss_torch(
        generated_cum=cum_pred,
        target_cum=risk_targets[:, 0],
        topk_ratio=cfg.tail_dist_topk_ratio,
    )
    if cfg.tail_dist_metric != "cum_deficit":
        tail_dist_loss = tail_dist_loss.new_tensor(0.0)

    core_cum_pred, core_ramp_pred, core_dur_pred = soft_core_risk_metrics_torch(
        x_proj,
        tau=risk_targets[:, 3],
        event_mask=event_mask,
        delta_t_hours=cfg.delta_t_hours,
        duration_temp=cfg.duration_temp,
    )
    with torch.no_grad():
        core_cum_true, core_ramp_true, core_dur_true = soft_core_risk_metrics_torch(
            x_true,
            tau=risk_targets[:, 3],
            event_mask=event_mask,
            delta_t_hours=cfg.delta_t_hours,
            duration_temp=cfg.duration_temp,
        )
    core_cum_loss = F.l1_loss(
        (torch.log1p(core_cum_pred) - risk_norm["log_cum_mean"]) / risk_norm["log_cum_std"],
        (torch.log1p(core_cum_true) - risk_norm["log_cum_mean"]) / risk_norm["log_cum_std"],
    )
    core_ramp_loss = F.l1_loss(
        (core_ramp_pred - risk_norm["ramp_mean"]) / risk_norm["ramp_std"],
        (core_ramp_true - risk_norm["ramp_mean"]) / risk_norm["ramp_std"],
    )
    core_dur_loss = F.l1_loss(
        (core_dur_pred - risk_norm["dur_mean"]) / risk_norm["dur_std"],
        (core_dur_true - risk_norm["dur_mean"]) / risk_norm["dur_std"],
    )
    core_risk_loss = (
        cfg.lambda_core_cum * core_cum_loss
        + cfg.lambda_core_ramp * core_ramp_loss
        + cfg.lambda_core_dur * core_dur_loss
    )

    sw = stage_weights(stage, cfg)
    if cfg.core_risk_mode.strip().lower() != "event_mask":
        core_risk_loss = core_risk_loss.new_tensor(0.0)
    highrisk_mask = risk_cond[:, 2] >= (2.0 / 3.0)
    delta_net_loss = net_load_delta_loss_torch(x_proj, x_true)
    ramp_topk_loss = ramp_topk_loss_torch(x_proj, x_true, topk_ratio=cfg.ramp_topk_ratio)
    shape_moment_loss = highrisk_shape_moment_loss_torch(
        x_proj,
        x_true,
        highrisk_mask=highrisk_mask,
        highrisk_only=bool(cfg.shape_highrisk_only),
    )
    total_loss = (
        eps_loss
        + sw["lambda_tail"] * tail_loss
        + sw["lambda_risk"] * weighted_risk_loss
        + sw["lambda_tail_dist"] * tail_dist_loss
        + sw["lambda_core_risk"] * core_risk_loss
        + sw["lambda_delta_net"] * delta_net_loss
        + sw["lambda_ramp_topk"] * ramp_topk_loss
        + sw["lambda_shape_moment"] * shape_moment_loss
        + sw["lambda_duration_over"] * duration_over_loss
        + sw["lambda_exceed_mask"] * exceed_mask_loss
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
        "tail_dist_loss": tail_dist_loss,
        "core_risk_loss": core_risk_loss,
        "core_cum_loss": core_cum_loss,
        "core_ramp_loss": core_ramp_loss,
        "core_dur_loss": core_dur_loss,
        "delta_net_loss": delta_net_loss,
        "ramp_topk_loss": ramp_topk_loss,
        "shape_moment_loss": shape_moment_loss,
        "duration_over_loss": duration_over_loss,
        "exceed_mask_loss": exceed_mask_loss,
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


def _risk_aware_score(val_stats: dict[str, float]) -> float:
    return (
        0.5 * float(val_stats.get("val_eps_loss", 0.0))
        + 0.5 * float(val_stats.get("val_risk_loss", 0.0))
        + 0.2 * float(val_stats.get("val_recon_loss", 0.0))
    )


def _build_checkpoint(
    model: HierarchicalConditionalUNet1D,
    cfg: TrainConfig,
    train_ds: ConditionedWindowDataset,
    cond_normalizers,
    risk_norm_np: dict[str, np.float32],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    bg_dim: int,
    proc_dim: int,
    risk_dim: int,
    flat_condition: bool,
    epoch: int,
    stage: str,
    checkpoint_type: str,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "train_config": asdict(cfg),
        "condition_meta": train_ds.condition_meta,
        "condition_normalizers": asdict(cond_normalizers),
        "risk_normalization": {k: float(v) for k, v in risk_norm_np.items()},
        "x_mean": x_mean,
        "x_std": x_std,
        "seq_len": cfg.seq_len,
        "cond_dims": {"background": bg_dim, "process": proc_dim, "risk": risk_dim},
        "flat_condition": flat_condition,
        "checkpoint_type": checkpoint_type,
        "checkpoint_epoch": int(epoch),
        "checkpoint_stage": stage,
    }


def _max_event_type_code(*frames: pd.DataFrame | None) -> int:
    max_code = 0
    for frame in frames:
        if frame is None or "event_type_code" not in frame.columns or frame.empty:
            continue
        value = pd.to_numeric(frame["event_type_code"], errors="coerce").fillna(0).max()
        max_code = max(max_code, int(value))
    return max_code


def _load_training_arrays(cfg: TrainConfig, warnings: list[str]) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray | None, dict]:
    data_dir = Path(cfg.data_dir)
    aug_summary: dict = {
        "use_risk_shapelet_aug": bool(cfg.use_risk_shapelet_aug),
        "use_augmented_train": bool(cfg.use_augmented_train),
    }
    if cfg.use_risk_shapelet_aug:
        aug_dir = Path(cfg.risk_shapelet_aug_dir) if cfg.risk_shapelet_aug_dir else data_dir
        required = [
            aug_dir / "X_train_riskshape_aug.npy",
            aug_dir / "cond_train_riskshape_aug.csv",
            aug_dir / "meta_train_riskshape_aug.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Risk-shapelet augmentation was requested, but files are missing: " + ", ".join(missing))
        x_train = np.load(aug_dir / "X_train_riskshape_aug.npy").astype(np.float32)
        cond_train = pd.read_csv(aug_dir / "cond_train_riskshape_aug.csv")
        meta_train = pd.read_csv(aug_dir / "meta_train_riskshape_aug.csv")
        mask_path = aug_dir / "event_mask_train_riskshape_aug.npy"
        train_event_mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else None
        if train_event_mask is None:
            warnings.append("Risk-shapelet augmentation is enabled, but event_mask_train_riskshape_aug.npy is missing.")
        summary_path = aug_dir / "augmentation_summary.json"
        if summary_path.exists():
            try:
                aug_summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                warnings.append(f"Could not parse augmentation summary: {summary_path}")
        aug_summary.update(
            {
                "risk_shapelet_aug_dir": str(aug_dir),
                "augmented_train_size": int(len(x_train)),
                "original_train_size": int(np.load(data_dir / "X_train.npy", mmap_mode="r").shape[0]),
            }
        )
        original_size = max(int(aug_summary.get("original_train_size", len(x_train))), 1)
        aug_summary["augmentation_ratio"] = float(len(x_train) / original_size)
        return x_train, cond_train, meta_train, train_event_mask, aug_summary

    generic_aug_dir = Path(cfg.augmented_train_dir) if cfg.augmented_train_dir else data_dir
    if cfg.use_augmented_train and (generic_aug_dir / "X_train_aug.npy").exists():
        x_train = np.load(generic_aug_dir / "X_train_aug.npy").astype(np.float32)
        cond_train = pd.read_csv(generic_aug_dir / "cond_train_aug.csv")
        meta_train = pd.read_csv(generic_aug_dir / "meta_train_aug.csv")
        mask_path = generic_aug_dir / "event_mask_train_aug.npy"
        train_event_mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else None
        summary_path = generic_aug_dir / "augmentation_log.json"
        if summary_path.exists():
            try:
                aug_summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                warnings.append(f"Could not parse augmentation summary: {summary_path}")
        aug_summary.update(
            {
                "augmented_train_dir": str(generic_aug_dir),
                "augmented_train_size": int(len(x_train)),
                "original_train_size": int(np.load(data_dir / "X_train.npy", mmap_mode="r").shape[0]),
                "augmentation_ratio": float(len(x_train) / max(int(np.load(data_dir / "X_train.npy", mmap_mode="r").shape[0]), 1)),
            }
        )
        return x_train, cond_train, meta_train, train_event_mask, aug_summary

    x_train, cond_train, meta_train, train_event_mask = load_split_arrays(data_dir, "train", include_event_mask=True)
    aug_summary.update(
        {
            "augmented_train_size": int(len(x_train)),
            "original_train_size": int(len(x_train)),
            "augmentation_ratio": 1.0,
        }
    )
    return x_train, cond_train, meta_train, train_event_mask, aug_summary


def _load_pretrain_frames(pretrain_data_dir: str | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not pretrain_data_dir:
        return None, None
    root = Path(pretrain_data_dir)
    train_path = root / "pretrain_cond_train.csv"
    val_path = root / "pretrain_cond_val.csv"
    if not train_path.exists():
        return None, None
    cond_train = pd.read_csv(train_path)
    cond_val = pd.read_csv(val_path) if val_path.exists() else cond_train.iloc[:0].copy()
    return cond_train, cond_val


def _make_pretrain_dataset(
    pretrain_data_dir: str,
    split: str,
    cfg: TrainConfig,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    cond_normalizers,
    expected_event_types: int,
) -> ConditionedWindowDataset:
    root = Path(pretrain_data_dir)
    x = np.load(root / f"pretrain_X_{split}.npy").astype(np.float32)
    cond = pd.read_csv(root / f"pretrain_cond_{split}.csv")
    meta = pd.read_csv(root / f"pretrain_meta_{split}.csv")
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    return ConditionedWindowDataset(
        x_norm,
        cond,
        meta,
        seq_len=cfg.seq_len,
        ablation=cfg.ablation,
        normalizers=cond_normalizers,
        expected_event_types=expected_event_types,
        daylight_start_hour=cfg.daylight_start_hour,
        daylight_end_hour=cfg.daylight_end_hour,
        event_mask=None,
    )


def _freeze_distribution_blocks(model: HierarchicalConditionalUNet1D) -> list[str]:
    frozen: list[str] = []
    for name, module in model.named_children():
        if name in {"init_conv", "down1", "down2", "mid1", "mid2", "up1", "up2"}:
            for param in module.parameters():
                param.requires_grad = False
            frozen.append(name)
    return frozen


def train_model(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    torch.set_num_threads(1)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    x_train, cond_train, meta_train, train_event_mask, augmentation_summary = _load_training_arrays(cfg, warnings)
    x_val, cond_val, meta_val, val_event_mask = load_split_arrays(cfg.data_dir, "val", include_event_mask=True)
    pretrain_frame_dir = cfg.pretrain_data_dir if (cfg.use_pretrain or cfg.stage0_epochs > 0) else None
    pretrain_cond_train, pretrain_cond_val = _load_pretrain_frames(pretrain_frame_dir)
    if cfg.use_pretrain and cfg.stage0_epochs > 0 and pretrain_cond_train is None:
        warnings.append("Stage 0 pretraining was requested, but pretrain_cond_train.csv was not found; pretraining is skipped.")
        cfg = TrainConfig(**{**asdict(cfg), "use_pretrain": False, "stage0_epochs": 0})
    core_risk_requested = cfg.core_risk_mode.strip().lower() == "event_mask" and float(cfg.lambda_core_risk) > 0
    if core_risk_requested and (train_event_mask is None or val_event_mask is None):
        warnings.append("event_mask_train.npy or event_mask_val.npy is missing; core risk loss was disabled.")
        cfg = TrainConfig(**{**asdict(cfg), "lambda_core_risk": 0.0, "core_risk_mode": "off"})
    x_mean = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    x_train_norm = ((x_train - x_mean) / x_std).astype(np.float32)
    x_val_norm = ((x_val - x_mean) / x_std).astype(np.float32)
    cond_normalizers = compute_condition_normalizers(cond_train)
    expected_event_types = _max_event_type_code(cond_train, cond_val, pretrain_cond_train, pretrain_cond_val) + 1

    train_ds = ConditionedWindowDataset(
        x_train_norm,
        cond_train,
        meta_train,
        seq_len=cfg.seq_len,
        ablation=cfg.ablation,
        normalizers=cond_normalizers,
        expected_event_types=expected_event_types,
        daylight_start_hour=cfg.daylight_start_hour,
        daylight_end_hour=cfg.daylight_end_hour,
        event_mask=train_event_mask,
    )
    val_ds = ConditionedWindowDataset(
        x_val_norm,
        cond_val,
        meta_val,
        seq_len=cfg.seq_len,
        ablation=cfg.ablation,
        normalizers=cond_normalizers,
        expected_event_types=expected_event_types,
        daylight_start_hour=cfg.daylight_start_hour,
        daylight_end_hour=cfg.daylight_end_hour,
        event_mask=val_event_mask,
    )

    train_sampler, sampler_summary = _build_train_sampler(cond_train, cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
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

    pretrain_summary: dict | None = None
    frozen_blocks: list[str] = []
    if cfg.pretrain_checkpoint:
        ckpt_path = Path(cfg.pretrain_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema_model.load_state_dict(model.state_dict())
        ema = EMA(model, cfg.ema_decay)
        warnings.append(f"Loaded Stage 0 pretrain checkpoint before fine-tuning: {ckpt_path}")

    if cfg.use_pretrain and cfg.stage0_epochs > 0 and cfg.pretrain_data_dir:
        pretrain_train_ds = _make_pretrain_dataset(
            cfg.pretrain_data_dir,
            "train",
            cfg,
            x_mean,
            x_std,
            cond_normalizers,
            expected_event_types,
        )
        pretrain_val_ds = _make_pretrain_dataset(
            cfg.pretrain_data_dir,
            "val",
            cfg,
            x_mean,
            x_std,
            cond_normalizers,
            expected_event_types,
        )
        pretrain_train_loader = DataLoader(
            pretrain_train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        pretrain_val_loader = DataLoader(
            pretrain_val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay)
        pretrain_history_rows: list[dict[str, float | str | int]] = []
        pretrain_best_val = float("inf")
        pretrain_best_epoch: int | None = None
        for pre_epoch in range(1, cfg.stage0_epochs + 1):
            model.train()
            epoch_stats: dict[str, float] = {}
            count = 0
            for batch in pretrain_train_loader:
                out = _forward_loss(
                    model,
                    scheduler,
                    batch,
                    device,
                    x_mean_t,
                    x_std_t,
                    risk_norm_t,
                    cfg,
                    "stage0_pretrain",
                    train_mode=True,
                )
                pretrain_optimizer.zero_grad()
                out["total_loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                pretrain_optimizer.step()
                ema.update(model)
                bs = batch[0].size(0)
                count += bs
                for key, value in out.items():
                    epoch_stats[key] = epoch_stats.get(key, 0.0) + float(value.item()) * bs
            train_stats = {f"train_{key}": value / max(count, 1) for key, value in epoch_stats.items()}
            ema.copy_to(ema_model)
            val_stats_raw = evaluate(
                ema_model,
                scheduler,
                pretrain_val_loader,
                device,
                x_mean_t,
                x_std_t,
                risk_norm_t,
                cfg,
                "stage0_pretrain",
            )
            val_stats = {f"val_{key}": value for key, value in val_stats_raw.items()}
            row = {"epoch": pre_epoch, "stage": "stage0_pretrain"}
            row.update(train_stats)
            row.update(val_stats)
            pretrain_history_rows.append(row)
            if val_stats["val_total_loss"] < pretrain_best_val:
                pretrain_best_val = val_stats["val_total_loss"]
                pretrain_best_epoch = pre_epoch
                pretrain_checkpoint = _build_checkpoint(
                    ema_model,
                    cfg,
                    train_ds,
                    cond_normalizers,
                    risk_norm_np,
                    x_mean,
                    x_std,
                    bg_dim,
                    proc_dim,
                    risk_dim,
                    flat_condition,
                    pre_epoch,
                    "stage0_pretrain",
                    "pretrain-best",
                )
                torch.save(pretrain_checkpoint, out_dir / "pretrain_model.pt")
            if pre_epoch == 1 or pre_epoch % 5 == 0 or pre_epoch == cfg.stage0_epochs:
                print(
                    f"Stage0 {pre_epoch:03d}/{cfg.stage0_epochs} | "
                    f"train_total={row['train_total_loss']:.4f} | "
                    f"val_total={row['val_total_loss']:.4f}"
                )
        ema.copy_to(ema_model)
        model.load_state_dict(ema_model.state_dict())
        ema = EMA(model, cfg.ema_decay)
        pretrain_history_df = pd.DataFrame(pretrain_history_rows)
        pretrain_history_df.to_csv(out_dir / "pretrain_history.csv", index=False, encoding="utf-8-sig")
        pretrain_final_checkpoint = _build_checkpoint(
            ema_model,
            cfg,
            train_ds,
            cond_normalizers,
            risk_norm_np,
            x_mean,
            x_std,
            bg_dim,
            proc_dim,
            risk_dim,
            flat_condition,
            cfg.stage0_epochs,
            "stage0_pretrain",
            "pretrain-final",
        )
        torch.save(pretrain_final_checkpoint, out_dir / "pretrain_final_model.pt")
        pretrain_summary = {
            "enabled": True,
            "pretrain_data_dir": str(cfg.pretrain_data_dir),
            "stage0_epochs": int(cfg.stage0_epochs),
            "pretrain_lr": float(cfg.pretrain_lr),
            "pretrain_loss_mode": cfg.pretrain_loss_mode,
            "pretrain_train_size": int(len(pretrain_train_ds)),
            "pretrain_val_size": int(len(pretrain_val_ds)),
            "best_pretrain_val_loss": float(pretrain_best_val),
            "best_pretrain_epoch": pretrain_best_epoch,
        }
        (out_dir / "pretrain_summary.json").write_text(json.dumps(pretrain_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.freeze_after_pretrain:
        frozen_blocks = _freeze_distribution_blocks(model)
        if frozen_blocks:
            warnings.append("freeze_after_pretrain enabled; frozen blocks: " + ", ".join(frozen_blocks))

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema_model.load_state_dict(model.state_dict())
    ema = EMA(model, cfg.ema_decay)

    best_val = float("inf")
    best_model_epoch: int | None = None
    best_model_stage: str | None = None
    best_risk_score = float("inf")
    best_risk_model_epoch: int | None = None
    best_risk_model_stage: str | None = None
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
            best_model_epoch = epoch
            best_model_stage = stage
            checkpoint = _build_checkpoint(
                ema_model,
                cfg,
                train_ds,
                cond_normalizers,
                risk_norm_np,
                x_mean,
                x_std,
                bg_dim,
                proc_dim,
                risk_dim,
                flat_condition,
                epoch,
                stage,
                "best",
            )
            torch.save(checkpoint, out_dir / "best_model.pt")

        risk_score = _risk_aware_score(val_stats)
        risk_model_allowed = stage == "stage3_risk" and cfg.ablation != "no_risk_loss" and cfg.lambda_risk > 0
        if risk_model_allowed and risk_score < best_risk_score:
            best_risk_score = risk_score
            best_risk_model_epoch = epoch
            best_risk_model_stage = stage
            risk_checkpoint = _build_checkpoint(
                ema_model,
                cfg,
                train_ds,
                cond_normalizers,
                risk_norm_np,
                x_mean,
                x_std,
                bg_dim,
                proc_dim,
                risk_dim,
                flat_condition,
                epoch,
                stage,
                "best-risk",
            )
            torch.save(risk_checkpoint, out_dir / "best_risk_model.pt")

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

    final_checkpoint = _build_checkpoint(
        ema_model,
        cfg,
        train_ds,
        cond_normalizers,
        risk_norm_np,
        x_mean,
        x_std,
        bg_dim,
        proc_dim,
        risk_dim,
        flat_condition,
        total_epochs,
        stage_name(total_epochs, cfg),
        "final",
    )
    torch.save(final_checkpoint, out_dir / "final_model.pt")

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
        "warnings": warnings,
        "train_size": int(len(train_ds)),
        "use_augmented_train": bool(cfg.use_augmented_train),
        "use_risk_shapelet_aug": bool(cfg.use_risk_shapelet_aug),
        "augmented_train_dir": cfg.augmented_train_dir,
        "risk_shapelet_aug_dir": cfg.risk_shapelet_aug_dir,
        "augmentation_summary": augmentation_summary,
        "use_pretrain": bool(cfg.use_pretrain),
        "pretrain_data_dir": cfg.pretrain_data_dir,
        "stage0_epochs": int(cfg.stage0_epochs),
        "pretrain_checkpoint": cfg.pretrain_checkpoint,
        "pretrain_summary": pretrain_summary,
        "freeze_after_pretrain": bool(cfg.freeze_after_pretrain),
        "frozen_blocks_after_pretrain": frozen_blocks,
        "val_size": int(len(val_ds)),
        "seq_len": cfg.seq_len,
        "best_val_loss": float(best_val),
        "best_model_epoch": best_model_epoch,
        "best_model_stage": best_model_stage,
        "best_risk_model_score": None if best_risk_model_epoch is None else float(best_risk_score),
        "best_risk_model_epoch": best_risk_model_epoch,
        "best_risk_model_stage": best_risk_model_stage,
        "final_epoch": int(total_epochs),
        "final_stage": stage_name(total_epochs, cfg),
        "lambda_risk_used": 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_risk,
        "sampler_summary": sampler_summary,
        "sampler_mode": cfg.sampler_mode,
        "severity_sample_weights": cfg.severity_sample_weights,
        "tail_sampler_alpha": cfg.tail_sampler_alpha,
        "tail_weight_mode": cfg.tail_weight_mode,
        "tail_weight_alpha": cfg.tail_weight_alpha,
        "tail_weight_max": cfg.tail_weight_max,
        "lambda_tail_dist": 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_tail_dist,
        "tail_dist_topk_ratio": cfg.tail_dist_topk_ratio,
        "tail_dist_metric": cfg.tail_dist_metric,
        "lambda_core_risk": 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_core_risk,
        "core_risk_mode": cfg.core_risk_mode,
        "lambda_core_cum": cfg.lambda_core_cum,
        "lambda_core_ramp": cfg.lambda_core_ramp,
        "lambda_core_dur": cfg.lambda_core_dur,
        "lambda_delta_net_stage2": cfg.lambda_delta_net_stage2,
        "lambda_delta_net_stage3": cfg.lambda_delta_net_stage3,
        "lambda_ramp_topk": cfg.lambda_ramp_topk,
        "ramp_topk_ratio": cfg.ramp_topk_ratio,
        "lambda_shape_stage2": cfg.lambda_shape_stage2,
        "lambda_shape_stage3": cfg.lambda_shape_stage3,
        "shape_highrisk_only": cfg.shape_highrisk_only,
        "lambda_duration_over_stage3": cfg.lambda_duration_over_stage3,
        "lambda_exceed_mask_stage3": cfg.lambda_exceed_mask_stage3,
        "checkpoint_type_used_for_generation": None,
        "stage1_epochs": cfg.stage1_epochs,
        "stage2_epochs": cfg.stage2_epochs,
        "stage3_epochs": cfg.stage3_epochs,
        "flat_condition": flat_condition,
        "condition_dims": {"background": bg_dim, "process": proc_dim, "risk": risk_dim},
        "train_config": asdict(cfg),
        "stage_loss_policy": {
            "stage0_pretrain": "eps_loss + lambda_recon * recon_loss + lambda_physics * physics_loss on normal windows only",
            "stage1_distribution": "eps_loss + lambda_recon * recon_loss + lambda_physics * physics_loss",
            "stage2_tail": "eps_loss + lambda_tail * tail_loss + light recon/physics/resource losses",
            "stage3_risk": "eps_loss + lambda_tail * tail_loss + lambda_risk * weighted_risk_loss + light recon/physics/resource losses",
            "weighted_risk_loss": "lambda_cum * cum_loss + lambda_ramp * ramp_loss + lambda_dur * dur_loss",
            "tail_distribution_loss": "stage3 only: lambda_tail_dist * top-k SmoothL1(cum_deficit_pred, cum_deficit_target)",
            "core_risk_loss": "stage3 only: lambda_core_risk * event-mask risk loss when event_mask is available",
            "delta_net_loss": "stage2/stage3 optional: SmoothL1 of full net-load first differences",
            "ramp_topk_loss": "stage3 optional: SmoothL1 of top-k positive net-load ramps",
            "shape_moment_loss": "stage2/stage3 optional: high-risk channel/difference moment anchor",
            "duration_over_loss": "stage3 optional: one-sided penalty when generated imbalance duration exceeds target duration",
            "exceed_mask_loss": "stage3 optional: BCE on hourly net-load exceedance mask relative to tau",
        },
        "loss_weights": {
            "lambda_tail": cfg.lambda_tail,
            "lambda_risk": 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_risk,
            "lambda_tail_dist": 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_tail_dist,
            "lambda_core_risk": 0.0 if cfg.ablation == "no_risk_loss" else cfg.lambda_core_risk,
            "lambda_cum": cfg.lambda_cum,
            "lambda_ramp": cfg.lambda_ramp,
            "lambda_dur": cfg.lambda_dur,
            "lambda_core_cum": cfg.lambda_core_cum,
            "lambda_core_ramp": cfg.lambda_core_ramp,
            "lambda_core_dur": cfg.lambda_core_dur,
            "lambda_delta_net_stage2": cfg.lambda_delta_net_stage2,
            "lambda_delta_net_stage3": cfg.lambda_delta_net_stage3,
            "lambda_ramp_topk": cfg.lambda_ramp_topk,
            "ramp_topk_ratio": cfg.ramp_topk_ratio,
            "lambda_shape_stage2": cfg.lambda_shape_stage2,
            "lambda_shape_stage3": cfg.lambda_shape_stage3,
            "lambda_duration_over_stage3": cfg.lambda_duration_over_stage3,
            "lambda_exceed_mask_stage3": cfg.lambda_exceed_mask_stage3,
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
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_evt", "no_evt_continuous", "no_evt_strict", "no_risk_loss", "no_month", "flat_condition"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=36)
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
    parser.add_argument("--sampler-mode", type=str, default="none", choices=["none", "severity", "tail_score"])
    parser.add_argument("--severity-sample-weights", type=str, default="0:1.0,1:1.5,2:2.5,3:3.5")
    parser.add_argument("--tail-sampler-alpha", type=float, default=0.5)
    parser.add_argument("--tail-weight-mode", type=str, default="relu", choices=["relu", "sigmoid", "clipped_relu"])
    parser.add_argument("--tail-weight-alpha", type=float, default=0.5)
    parser.add_argument("--tail-weight-max", type=float, default=3.0)
    parser.add_argument("--lambda-tail-dist", type=float, default=0.0)
    parser.add_argument("--tail-dist-topk-ratio", type=float, default=0.10)
    parser.add_argument("--tail-dist-metric", type=str, default="cum_deficit", choices=["cum_deficit"])
    parser.add_argument("--lambda-core-risk", type=float, default=0.0)
    parser.add_argument("--core-risk-mode", type=str, default="off", choices=["off", "event_mask"])
    parser.add_argument("--lambda-core-cum", type=float, default=1.0)
    parser.add_argument("--lambda-core-ramp", type=float, default=0.15)
    parser.add_argument("--lambda-core-dur", type=float, default=0.25)
    parser.add_argument("--lambda-delta-net-stage2", type=float, default=0.0, help="Stage 2 净负荷差分形态损失权重，用于防止尾部强化阶段破坏爬坡过程。")
    parser.add_argument("--lambda-delta-net-stage3", type=float, default=0.0, help="Stage 3 净负荷差分形态损失权重，用于改善生成场景的净负荷变化率过程。")
    parser.add_argument("--lambda-ramp-topk", type=float, default=0.0, help="top-k 正向爬坡尾部损失权重，用于改善最大净负荷爬坡误差。")
    parser.add_argument("--ramp-topk-ratio", type=float, default=0.10, help="选取净负荷正向爬坡序列中前多少比例作为尾部爬坡样本。")
    parser.add_argument("--lambda-shape-stage2", type=float, default=0.0, help="Stage 2 高风险样本形态统计锚定损失权重，用于改善 highrisk_acf_mae。")
    parser.add_argument("--lambda-shape-stage3", type=float, default=0.0, help="Stage 3 高风险样本形态统计锚定损失权重，用于改善 highrisk_acf_mae。")
    parser.add_argument("--shape-highrisk-only", action=argparse.BooleanOptionalAction, default=True, help="是否仅在 highrisk 样本上计算形态锚定损失。")
    parser.add_argument("--lambda-duration-over-stage3", type=float, default=0.0, help="Stage 3 持续失衡时长过高的单边惩罚权重，用于压低生成样本系统性 duration 偏长。")
    parser.add_argument("--lambda-exceed-mask-stage3", type=float, default=0.0, help="Stage 3 逐时刻超阈值掩码损失权重，用于学习每个小时是否超过失衡阈值 tau。")
    parser.add_argument("--cond-dropout", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-augmented-train", action="store_true")
    parser.add_argument("--use-pretrain", action="store_true")
    parser.add_argument("--pretrain-data-dir", type=str, default=None)
    parser.add_argument("--stage0-epochs", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-checkpoint", type=str, default=None)
    parser.add_argument("--freeze-after-pretrain", action="store_true")
    parser.add_argument("--pretrain-loss-mode", type=str, default="distribution_only", choices=["distribution_only"])
    parser.add_argument("--augmented-train-dir", type=str, default=None)
    parser.add_argument("--use-risk-shapelet-aug", action="store_true")
    parser.add_argument("--risk-shapelet-aug-dir", type=str, default=None)
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
        sampler_mode=args.sampler_mode,
        severity_sample_weights=args.severity_sample_weights,
        tail_sampler_alpha=args.tail_sampler_alpha,
        tail_weight_mode=args.tail_weight_mode,
        tail_weight_alpha=args.tail_weight_alpha,
        tail_weight_max=args.tail_weight_max,
        lambda_tail_dist=args.lambda_tail_dist,
        tail_dist_topk_ratio=args.tail_dist_topk_ratio,
        tail_dist_metric=args.tail_dist_metric,
        lambda_core_risk=args.lambda_core_risk,
        core_risk_mode=args.core_risk_mode,
        lambda_core_cum=args.lambda_core_cum,
        lambda_core_ramp=args.lambda_core_ramp,
        lambda_core_dur=args.lambda_core_dur,
        lambda_delta_net_stage2=args.lambda_delta_net_stage2,
        lambda_delta_net_stage3=args.lambda_delta_net_stage3,
        lambda_ramp_topk=args.lambda_ramp_topk,
        ramp_topk_ratio=args.ramp_topk_ratio,
        lambda_shape_stage2=args.lambda_shape_stage2,
        lambda_shape_stage3=args.lambda_shape_stage3,
        shape_highrisk_only=args.shape_highrisk_only,
        lambda_duration_over_stage3=args.lambda_duration_over_stage3,
        lambda_exceed_mask_stage3=args.lambda_exceed_mask_stage3,
        device=args.device,
        use_augmented_train=args.use_augmented_train,
        use_pretrain=args.use_pretrain,
        pretrain_data_dir=args.pretrain_data_dir,
        stage0_epochs=args.stage0_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_checkpoint=args.pretrain_checkpoint,
        freeze_after_pretrain=args.freeze_after_pretrain,
        pretrain_loss_mode=args.pretrain_loss_mode,
        augmented_train_dir=args.augmented_train_dir,
        use_risk_shapelet_aug=args.use_risk_shapelet_aug,
        risk_shapelet_aug_dir=args.risk_shapelet_aug_dir,
    )


if __name__ == "__main__":
    train_model(parse_args())
