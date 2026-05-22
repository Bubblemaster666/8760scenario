from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from evaluate_generation import EvalConfig, evaluate_generation
from hierarchical_diffusion import (
    DiffusionScheduler,
    EMA,
    HierarchicalConditionalUNet1D,
    HierarchicalConditionEncoder,
    SinusoidalTimeEmbedding,
    apply_physical_projection,
    condition_dropout,
    denormalize_x_np,
    denormalize_x_torch,
    load_split_arrays,
    physics_penalty,
    risk_consistency_loss,
    sample_sequences,
)
from risk_ranking_utils import (
    AUXILIARY_REALISM_METRICS,
    RISK_EVALUATION_EXPLANATION,
    RISK_MAIN_METRICS,
    RISK_RANKING_EXPLANATION,
    add_risk_score,
    write_risk_tables,
)


BASE_DIR = Path(__file__).resolve().parent


COMPARE_SOURCES = {
    "Simple_EVT_Risk_Diffusion": BASE_DIR / "results" / "simple_evt_risk_diffusion" / "generated_samples_simple.npy",
    "A0_JRPD_best_3h": BASE_DIR / "outputs" / "ramp_window_retrain" / "models" / "RAMPDIAG4_JRPD_ramp3h" / "generated_samples.npy",
    "Plain_Diffusion": BASE_DIR / "outputs" / "main_compare_e0_fixed" / "models" / "plain_diffusion_baseline" / "generation" / "generated_samples.npy",
    "A2_Diffusion_with_risk_profile": BASE_DIR / "outputs" / "joint_profile_diffusion_on_ramp3h" / "models" / "JP1_profile_condition" / "generated_samples.npy",
    "Simple_EVT_Risk_Diffusion_RampLevelSampler": BASE_DIR / "results" / "simple_evt_risk_ramp_improvements" / "generated_samples_Simple_EVT_Risk_Diffusion_RampLevelSampler.npy",
}


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


@dataclass
class SimpleNormalizers:
    duration_scale: float
    event_type_count: int
    ramp_thresholds: tuple[float, float, float]


class SimpleEVTRiskDataset(Dataset):
    """简化模型数据集。

    x：风光荷三通道真实序列，形状 [样本数, 3, 36]。
    c_simple：简化条件，仅包含 event_type、month_sin、month_cos、duration_norm、extreme_prob。
    risk_targets：联合失衡风险目标 [累计缺额, 3h净负荷爬坡, 持续失衡时长, 失衡阈值]。
    """

    def __init__(
        self,
        x_norm: np.ndarray,
        x_raw: np.ndarray,
        cond: pd.DataFrame,
        meta: pd.DataFrame,
        normalizers: SimpleNormalizers,
        event_mask: np.ndarray | None = None,
        use_ramp_level_condition: bool = False,
    ) -> None:
        self.x = x_norm.astype(np.float32)
        self.cond = cond.reset_index(drop=True).copy()
        self.meta = meta.reset_index(drop=True).copy()
        self.seq_len = int(x_norm.shape[2])
        self.use_ramp_level_condition = bool(use_ramp_level_condition)
        self.ramp_level = self._assign_ramp_level(normalizers)
        self.event_mask = (
            event_mask.astype(np.float32)
            if event_mask is not None
            else np.ones((len(self.x), self.seq_len), dtype=np.float32)
        )
        self.bg, self.proc, self.risk = self._build_simple_conditions(normalizers)
        self.day_mask = self._build_day_mask(x_raw)
        self.risk_targets = self._build_risk_targets()

    def _assign_ramp_level(self, normalizers: SimpleNormalizers) -> np.ndarray:
        """按训练集阈值映射 ramp_level，0=低爬坡，3=极高爬坡。"""

        ramp = pd.to_numeric(self.cond.get("netload_ramp_max", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        q50, q75, q90 = normalizers.ramp_thresholds
        return np.digitize(ramp, bins=np.asarray([q50, q75, q90], dtype=np.float32), right=True).astype(np.int64)

    def _build_simple_conditions(self, normalizers: SimpleNormalizers) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        codes = pd.to_numeric(self.cond.get("event_type_code", 0), errors="coerce").fillna(0).astype(int).to_numpy()
        codes = np.clip(codes, 0, max(normalizers.event_type_count - 1, 0))
        event_onehot = np.eye(normalizers.event_type_count, dtype=np.float32)[codes]

        if "month_sin" in self.cond.columns and "month_cos" in self.cond.columns:
            month_sin = pd.to_numeric(self.cond["month_sin"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            month_cos = pd.to_numeric(self.cond["month_cos"], errors="coerce").fillna(1).to_numpy(dtype=np.float32)
        else:
            month = pd.to_numeric(self.cond.get("month", 1), errors="coerce").fillna(1).to_numpy(dtype=np.float32)
            month_sin = np.sin(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32)
            month_cos = np.cos(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32)

        # background 条件：事件类型 + 月份周期。season/low wind/low irradiance/start_hour 等在简化版中禁用。
        bg = np.concatenate([event_onehot, month_sin[:, None], month_cos[:, None]], axis=1).astype(np.float32)

        duration = pd.to_numeric(self.cond.get("duration_hours", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        duration_norm = np.clip(duration / max(float(normalizers.duration_scale), 1.0), 0.0, 2.0).astype(np.float32)
        proc = duration_norm[:, None]

        # risk 条件只保留 EVT 连续极端概率 extreme_prob，禁用 tail_score/severity/profile id。
        extreme_prob = pd.to_numeric(self.cond.get("extreme_prob", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float32)
        risk_parts = [np.clip(extreme_prob, 0.0, 1.0)[:, None].astype(np.float32)]
        if self.use_ramp_level_condition:
            # ramp_level：三小时净负荷爬坡等级，one-hot 注入简化风险条件。
            risk_parts.append(np.eye(4, dtype=np.float32)[np.clip(self.ramp_level, 0, 3)])
        risk = np.concatenate(risk_parts, axis=1).astype(np.float32)
        return bg, proc.astype(np.float32), risk

    def _build_day_mask(self, x_raw: np.ndarray) -> np.ndarray:
        if "window_start_time" in self.meta.columns:
            start = pd.to_datetime(self.meta["window_start_time"], errors="coerce")
            masks = []
            for ts in start:
                if pd.isna(ts):
                    masks.append((x_raw[len(masks), 2, :] > 1e-6).astype(np.float32))
                    continue
                hours = (int(ts.hour) + np.arange(self.seq_len)) % 24
                masks.append(((hours >= 6) & (hours < 18)).astype(np.float32))
            return np.asarray(masks, dtype=np.float32)
        return (x_raw[:, 2, :] > 1e-6).astype(np.float32)

    def _build_risk_targets(self) -> np.ndarray:
        def col(name: str, default: float = 0.0) -> np.ndarray:
            return pd.to_numeric(self.cond.get(name, default), errors="coerce").fillna(default).to_numpy(dtype=np.float32)

        # netload_ramp_max 在 dataset_window_3h 中已按 3h window ramp 重算。
        return np.stack(
            [
                col("cum_deficit"),
                col("netload_ramp_max"),
                col("imbalance_duration"),
                col("imbalance_tau"),
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "bg": torch.from_numpy(self.bg[idx]),
            "proc": torch.from_numpy(self.proc[idx]),
            "risk": torch.from_numpy(self.risk[idx]),
            "risk_targets": torch.from_numpy(self.risk_targets[idx]),
            "day_mask": torch.from_numpy(self.day_mask[idx]),
            "event_mask": torch.from_numpy(self.event_mask[idx]),
            "ramp_level": torch.tensor(self.ramp_level[idx], dtype=torch.long),
        }


def _load_split(data_dir: Path, split: str):
    loaded = load_split_arrays(data_dir, split, include_event_mask=True)
    x, cond, meta, event_mask = loaded
    return x.astype(np.float32), cond, meta, event_mask


def _normalizers(cond_train: pd.DataFrame) -> SimpleNormalizers:
    max_code = int(pd.to_numeric(cond_train.get("event_type_code", 0), errors="coerce").fillna(0).max())
    duration = pd.to_numeric(cond_train.get("duration_hours", 36.0), errors="coerce").fillna(36.0).to_numpy(dtype=np.float32)
    duration_scale = float(max(np.nanpercentile(duration, 95), 1.0))
    ramp = pd.to_numeric(cond_train.get("netload_ramp_max", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    thresholds = tuple(float(v) for v in np.quantile(ramp, [0.50, 0.75, 0.90]))
    return SimpleNormalizers(duration_scale=duration_scale, event_type_count=max_code + 1, ramp_thresholds=thresholds)


def _risk_norm(cond_train: pd.DataFrame, device: torch.device) -> dict[str, torch.Tensor]:
    cum = pd.to_numeric(cond_train.get("cum_deficit", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ramp = pd.to_numeric(cond_train.get("netload_ramp_max", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    dur = pd.to_numeric(cond_train.get("imbalance_duration", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    log_cum = np.log1p(np.maximum(cum, 0.0))
    return {
        "log_cum_mean": torch.tensor(float(log_cum.mean()), device=device),
        "log_cum_std": torch.tensor(float(log_cum.std() + 1e-6), device=device),
        "ramp_mean": torch.tensor(float(ramp.mean()), device=device),
        "ramp_std": torch.tensor(float(ramp.std() + 1e-6), device=device),
        "dur_mean": torch.tensor(float(dur.mean()), device=device),
        "dur_std": torch.tensor(float(dur.std() + 1e-6), device=device),
    }


def ramp_3h_curve_loss_torch(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    delta_t_hours: float = 1.0,
    ramp_window_hours: float = 3.0,
    event_mask: torch.Tensor | None = None,
    core_only: bool = False,
) -> torch.Tensor:
    """三小时净负荷爬坡过程曲线损失。

    net_load：净负荷，等于 load - wind_power - solar_power。
    ramp_3h_curve_pos：三小时正向净负荷爬坡曲线，只保留净负荷上升部分。
    L_ramp_curve：约束整段爬坡过程，而不是只约束最大爬坡点。
    """

    steps = max(1, int(round(float(ramp_window_hours) / max(float(delta_t_hours), 1e-6))))
    if x_pred.shape[-1] <= steps:
        return x_pred.new_tensor(0.0)
    net_pred = x_pred[:, 0, :] - x_pred[:, 1, :] - x_pred[:, 2, :]
    net_true = x_true[:, 0, :] - x_true[:, 1, :] - x_true[:, 2, :]
    ramp_pred = F.relu(net_pred[:, steps:] - net_pred[:, :-steps])
    ramp_true = F.relu(net_true[:, steps:] - net_true[:, :-steps])
    if core_only:
        if event_mask is None:
            return x_pred.new_tensor(0.0)
        mask = event_mask[:, steps:].to(device=x_pred.device, dtype=x_pred.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (torch.abs(ramp_pred - ramp_true) * mask).sum() / denom
    return F.l1_loss(ramp_pred, ramp_true)


def _rank_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1 or float(np.nanmax(values) - np.nanmin(values)) <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float32)
    return (ranks - 1.0) / max(float(values.size - 1), 1.0)


def _sampler_weights(cond_train: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    """ramp-focused sampler 权重：高三小时爬坡样本获得更高采样概率。"""

    cum = pd.to_numeric(cond_train.get("cum_deficit", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ramp = pd.to_numeric(cond_train.get("netload_ramp_max", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    dur = pd.to_numeric(cond_train.get("imbalance_duration", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    weights = (
        1.0
        + float(args.sampler_alpha_cum) * _rank_norm(cum)
        + float(args.sampler_alpha_ramp) * _rank_norm(ramp)
        + float(args.sampler_alpha_dur) * _rank_norm(dur)
    )
    return np.clip(weights, 1e-3, None).astype(np.float32)


def _distribution_rows(split: str, levels: np.ndarray) -> list[dict[str, object]]:
    unique, counts = np.unique(levels.astype(int), return_counts=True)
    count_map = {int(k): int(v) for k, v in zip(unique, counts)}
    return [{"split": split, "ramp_level": level, "count": count_map.get(level, 0)} for level in range(4)]


def _sampler_summary_rows(levels: np.ndarray, weights: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for level in range(4):
        idx = levels.astype(int) == level
        vals = weights[idx]
        rows.append(
            {
                "ramp_level": level,
                "sample_count": int(idx.sum()),
                "weight_mean": float(vals.mean()) if vals.size else 0.0,
                "weight_min": float(vals.min()) if vals.size else 0.0,
                "weight_max": float(vals.max()) if vals.size else 0.0,
            }
        )
    rows.append(
        {
            "ramp_level": "all",
            "sample_count": int(len(weights)),
            "weight_mean": float(weights.mean()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        }
    )
    return rows


class SimpleTemporalAttentionDenoiser(nn.Module):
    """简化 EVT 风险扩散的轻量时序注意力 denoiser。

    x：风光荷三通道序列，形状 [B, 3, T]。
    t：DDPM 扩散时间步。
    condition：仍然只使用 Simple 模型的 event_type、month_sin/month_cos、
    duration_norm 和 extreme_prob，不重新引入 JRPD/risk_profile/tail_score。

    Temporal Attention 用于捕捉 36h 窗口内跨时间步依赖和局部突变关系，
    尤其服务于三小时净负荷爬坡 ramp_3h(t)=net_load(t)-net_load(t-3) 的刻画。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        time_dim: int,
        cond_dim: int,
        bg_dim: int,
        proc_dim: int,
        risk_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        feedforward_dim: int = 128,
    ) -> None:
        super().__init__()
        self.bg_dim = bg_dim
        self.proc_dim = proc_dim
        self.risk_dim = risk_dim
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_encoder = HierarchicalConditionEncoder(bg_dim, proc_dim, risk_dim, cond_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=1),
        )

    def _zero_bundle(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.bg_dim, device=device),
            torch.zeros(batch_size, self.proc_dim, device=device),
            torch.zeros(batch_size, self.risk_dim, device=device),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        bg_cond: torch.Tensor | None,
        proc_cond: torch.Tensor | None,
        risk_cond: torch.Tensor | None,
        profile_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del profile_cond
        batch_size = x.size(0)
        if bg_cond is None or proc_cond is None or risk_cond is None:
            bg_cond, proc_cond, risk_cond = self._zero_bundle(batch_size, x.device)
        _, _, _, cond_emb = self.cond_encoder(bg_cond, proc_cond, risk_cond)
        time_emb = self.time_mlp(t)
        h = self.input_proj(x)
        h = h + self.time_proj(time_emb)[:, :, None] + self.cond_proj(cond_emb)[:, :, None]
        # Transformer 使用 [B, T, hidden_dim]，batch_first=True，输出再转回 [B, hidden_dim, T]。
        h = self.temporal_encoder(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def _use_temporal_attention(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_temporal_attention", False)) or "TemporalAttention" in str(args.method_name)


def _build_model(
    args: argparse.Namespace,
    bg_dim: int,
    proc_dim: int,
    risk_dim: int,
    device: torch.device,
) -> nn.Module:
    if _use_temporal_attention(args):
        return SimpleTemporalAttentionDenoiser(
            in_channels=3,
            hidden_dim=int(getattr(args, "temporal_hidden_dim", 64)),
            time_dim=128,
            cond_dim=128,
            bg_dim=bg_dim,
            proc_dim=proc_dim,
            risk_dim=risk_dim,
            num_layers=int(getattr(args, "temporal_layers", 1)),
            num_heads=int(getattr(args, "temporal_heads", 4)),
            dropout=float(getattr(args, "temporal_dropout", 0.10)),
            feedforward_dim=int(getattr(args, "temporal_feedforward_dim", 128)),
        ).to(device)
    return HierarchicalConditionalUNet1D(
        in_channels=3,
        base_channels=int(args.base_channels),
        time_dim=128,
        cond_dim=128,
        bg_dim=bg_dim,
        proc_dim=proc_dim,
        risk_dim=risk_dim,
        flat_condition=False,
        profile_channels=0,
    ).to(device)


def _make_loaders(data_dir: Path, batch_size: int, device: torch.device, args: argparse.Namespace):
    x_train, cond_train, meta_train, mask_train = _load_split(data_dir, "train")
    x_val, cond_val, meta_val, mask_val = _load_split(data_dir, "val")
    x_test, cond_test, meta_test, mask_test = _load_split(data_dir, "test")
    x_mean = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    norms = _normalizers(cond_train)
    use_ramp_level = bool(args.use_ramp_level_condition)
    train_ds = SimpleEVTRiskDataset((x_train - x_mean) / x_std, x_train, cond_train, meta_train, norms, mask_train, use_ramp_level_condition=use_ramp_level)
    val_ds = SimpleEVTRiskDataset((x_val - x_mean) / x_std, x_val, cond_val, meta_val, norms, mask_val, use_ramp_level_condition=use_ramp_level)
    test_ds = SimpleEVTRiskDataset((x_test - x_mean) / x_std, x_test, cond_test, meta_test, norms, mask_test, use_ramp_level_condition=use_ramp_level)
    weights = _sampler_weights(cond_train, args) if bool(args.use_ramp_sampler) else np.ones((len(train_ds),), dtype=np.float32)
    sampler = (
        WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        if bool(args.use_ramp_sampler)
        else None
    )
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=False),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        "test_ds": test_ds,
        "x_mean": x_mean,
        "x_std": x_std,
        "normalizers": norms,
        "risk_norm": _risk_norm(cond_train, device),
        "ramp_level_distribution": pd.DataFrame(
            _distribution_rows("train", train_ds.ramp_level)
            + _distribution_rows("val", val_ds.ramp_level)
            + _distribution_rows("test", test_ds.ramp_level)
        ),
        "sampler_weight_summary": pd.DataFrame(_sampler_summary_rows(train_ds.ramp_level, weights)),
    }


def _run_epoch(
    model: HierarchicalConditionalUNet1D,
    scheduler: DiffusionScheduler,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    stage: str,
    device: torch.device,
    x_mean_t: torch.Tensor,
    x_std_t: torch.Tensor,
    risk_norm: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, list[float]] = {k: [] for k in ["total_loss", "eps_loss", "cum_loss", "ramp_loss", "ramp_curve_loss", "core_ramp_curve_loss", "dur_loss", "physics_loss"]}
    for batch in loader:
        x = batch["x"].to(device)
        bg = batch["bg"].to(device)
        proc = batch["proc"].to(device)
        risk = batch["risk"].to(device)
        targets = batch["risk_targets"].to(device)
        day_mask = batch["day_mask"].to(device)
        event_mask = batch["event_mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.steps, (x.shape[0],), device=device)
        x_t = scheduler.q_sample(x, t, noise)
        bg_in, proc_in, risk_in = condition_dropout(bg, proc, risk, float(args.cond_dropout)) if train else (bg, proc, risk)
        pred = model(x_t, t, bg_in, proc_in, risk_in, None)
        eps_loss = F.mse_loss(pred, noise)
        total = eps_loss

        cum_loss = ramp_loss = ramp_curve_loss = core_ramp_curve_loss = dur_loss = physics_loss = x.new_tensor(0.0)
        if stage == "stage2_risk":
            x0_pred_norm = scheduler.predict_x0(x_t, t, pred).clamp(-5.0, 5.0)
            x0_pred = denormalize_x_torch(x0_pred_norm, x_mean_t, x_std_t)
            x_true = denormalize_x_torch(x, x_mean_t, x_std_t)
            x_proj = apply_physical_projection(x0_pred, day_mask)
            _, cum_loss, ramp_loss, dur_loss = risk_consistency_loss(
                x_proj=x_proj,
                risk_targets=targets,
                delta_t_hours=1.0,
                duration_temp=float(args.duration_temp),
                risk_norm=risk_norm,
                ramp_metric_mode="window_3h",
                ramp_window_hours=3.0,
            )
            ramp_curve_loss = ramp_3h_curve_loss_torch(
                x_proj,
                x_true,
                delta_t_hours=1.0,
                ramp_window_hours=3.0,
            )
            if bool(args.use_core_ramp_curve_loss):
                core_ramp_curve_loss = ramp_3h_curve_loss_torch(
                    x_proj,
                    x_true,
                    delta_t_hours=1.0,
                    ramp_window_hours=3.0,
                    event_mask=event_mask,
                    core_only=True,
                )
            physics_loss = physics_penalty(x0_pred, day_mask)
            total = (
                eps_loss
                + float(args.lambda_cum) * cum_loss
                + float(args.lambda_ramp) * ramp_loss
                + float(args.lambda_ramp_curve) * ramp_curve_loss
                + float(args.lambda_core_ramp_curve) * core_ramp_curve_loss
                + float(args.lambda_dur) * dur_loss
                + float(args.lambda_phy) * physics_loss
            )
        if train:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for key, value in [
            ("total_loss", total),
            ("eps_loss", eps_loss),
            ("cum_loss", cum_loss),
            ("ramp_loss", ramp_loss),
            ("ramp_curve_loss", ramp_curve_loss),
            ("core_ramp_curve_loss", core_ramp_curve_loss),
            ("dur_loss", dur_loss),
            ("physics_loss", physics_loss),
        ]:
            totals[key].append(float(value.detach().cpu()))
    return {key: float(np.mean(values)) if values else float("nan") for key, values in totals.items()}


def train_simple_model(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    model_name = str(args.method_name)
    model_dir = out_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(1)

    bundle = _make_loaders(data_dir, int(args.batch_size), device, args)
    x_mean_t = torch.from_numpy(bundle["x_mean"]).to(device)
    x_std_t = torch.from_numpy(bundle["x_std"]).to(device)
    bg_dim = bundle["normalizers"].event_type_count + 2
    proc_dim = 1
    risk_dim = int(bundle["train"].dataset.risk.shape[1])
    model = _build_model(args, bg_dim, proc_dim, risk_dim, device)
    scheduler = DiffusionScheduler(int(args.diffusion_steps), 1e-4, 0.02, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    ema = EMA(model, decay=float(args.ema_decay))
    history = []
    best_val = float("inf")
    best_state = None
    best_stage2_val = float("inf")
    best_stage2_state = None
    selected_checkpoint_stage = "stage1_diffusion"
    total_epochs = int(args.stage1_epochs) + int(args.stage2_epochs)
    for epoch in range(1, total_epochs + 1):
        stage = "stage1_diffusion" if epoch <= int(args.stage1_epochs) else "stage2_risk"
        train_metrics = _run_epoch(model, scheduler, bundle["train"], optimizer, stage, device, x_mean_t, x_std_t, bundle["risk_norm"], args)
        ema.update(model)
        ema_model = _build_model(args, bg_dim, proc_dim, risk_dim, device)
        ema.copy_to(ema_model)
        with torch.no_grad():
            val_metrics = _run_epoch(ema_model, scheduler, bundle["val"], None, stage, device, x_mean_t, x_std_t, bundle["risk_norm"], args)
        row = {"epoch": epoch, "stage": stage}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        if stage == "stage2_risk" and val_metrics["total_loss"] < best_stage2_val:
            best_stage2_val = val_metrics["total_loss"]
            best_stage2_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        print(f"{model_name} epoch {epoch:03d}/{total_epochs} | {stage} | train={train_metrics['total_loss']:.4f} | val={val_metrics['total_loss']:.4f}")

    if best_stage2_state is not None:
        model.load_state_dict(best_stage2_state)
        selected_checkpoint_stage = "stage2_risk"
    elif best_state is not None:
        model.load_state_dict(best_state)
    else:
        ema.copy_to(model)
    ckpt = {
        "model_state": model.state_dict(),
        "x_mean": bundle["x_mean"],
        "x_std": bundle["x_std"],
        "bg_dim": bg_dim,
        "proc_dim": proc_dim,
        "risk_dim": risk_dim,
        "base_channels": int(args.base_channels),
        "model_variant": "simple_temporal_attention" if _use_temporal_attention(args) else "simple_unet",
        "temporal_attention": {
            "enabled": _use_temporal_attention(args),
            "hidden_dim": int(getattr(args, "temporal_hidden_dim", 64)),
            "num_layers": int(getattr(args, "temporal_layers", 1)),
            "num_heads": int(getattr(args, "temporal_heads", 4)),
            "dropout": float(getattr(args, "temporal_dropout", 0.10)),
            "feedforward_dim": int(getattr(args, "temporal_feedforward_dim", 128)),
        },
        "diffusion_steps": int(args.diffusion_steps),
        "normalizers": bundle["normalizers"].__dict__,
        "train_config": vars(args),
        "selected_checkpoint_stage": selected_checkpoint_stage,
        "best_val_total_loss": best_val,
        "best_stage2_val_total_loss": best_stage2_val if np.isfinite(best_stage2_val) else None,
        "condition_policy": {
            "used": ["event_type", "month_sin", "month_cos", "duration_norm", "extreme_prob"] + (["ramp_level"] if bool(args.use_ramp_level_condition) else []),
            "disabled": ["season_onehot", "low_wind_flag", "low_irradiance_flag", "start_hour", "tail_score", "severity_level", "JRPD_profile_id", "risk_profile_36x4", "flow_matching"],
        },
    }
    torch.save(ckpt, model_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    bundle["ramp_level_distribution"].to_csv(model_dir / "ramp_level_distribution.csv", index=False, encoding="utf-8-sig")
    bundle["sampler_weight_summary"].to_csv(model_dir / "sampler_weight_summary.csv", index=False, encoding="utf-8-sig")
    (model_dir / "summary.json").write_text(json.dumps({k: v for k, v in ckpt.items() if k != "model_state"}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"model": model, "scheduler": scheduler, "bundle": bundle, "model_dir": model_dir, "device": device}


@torch.no_grad()
def generate_simple(trained: dict, args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    test_ds: SimpleEVTRiskDataset = trained["bundle"]["test_ds"]
    device = trained["device"]
    model = trained["model"].to(device).eval()
    scheduler = trained["scheduler"]
    bg = torch.from_numpy(test_ds.bg).to(device)
    proc = torch.from_numpy(test_ds.proc).to(device)
    risk = torch.from_numpy(test_ds.risk).to(device)
    day_mask = torch.from_numpy(test_ds.day_mask).to(device)
    x_mean_t = torch.from_numpy(trained["bundle"]["x_mean"]).to(device)
    x_std_t = torch.from_numpy(trained["bundle"]["x_std"]).to(device)
    generated_norm = sample_sequences(
        model,
        scheduler,
        bg,
        proc,
        risk,
        None,
        shape=(len(test_ds), 3, test_ds.seq_len),
        guidance_scale=1.0,
        device=device,
        day_mask=day_mask,
        x_mean=x_mean_t,
        x_std=x_std_t,
    )
    generated = denormalize_x_np(generated_norm.detach().cpu().numpy(), trained["bundle"]["x_mean"], trained["bundle"]["x_std"]).astype(np.float32)
    generated = np.maximum(generated, 0.0)
    generated[:, 2, :] *= test_ds.day_mask
    if str(args.method_name) == "Simple_EVT_Risk_Diffusion_RampCurve":
        path = out_dir / "generated_samples_simple_rampcurve.npy"
    elif str(args.method_name) == "Simple_EVT_Risk_Diffusion_TemporalAttention":
        path = out_dir / "generated_samples_temporal_attention.npy"
    elif str(args.method_name) == "Simple_EVT_Risk_Diffusion":
        path = out_dir / "generated_samples_simple.npy"
    else:
        path = out_dir / f"generated_samples_{str(args.method_name)}.npy"
    np.save(path, generated)
    shutil.copy2(path, trained["model_dir"] / "generated_samples.npy")
    return path


def evaluate_method(name: str, generated: Path, data_dir: Path, out_dir: Path) -> dict:
    eval_dir = out_dir / "evaluations" / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(generated),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=name,
            event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = {"method": name}
    row.update({metric: summary["metrics"].get(metric, np.nan) for metric in FULL_COMPARE_METRICS})
    return row


def run(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trained = train_simple_model(args)
    simple_path = generate_simple(trained, args)
    method_name = str(args.method_name)
    rows = [evaluate_method(method_name, simple_path, data_dir, out_dir)]
    if method_name == "Simple_EVT_Risk_Diffusion_RampCurve":
        metrics_name = "simple_rampcurve_metrics.csv"
    elif method_name == "Simple_EVT_Risk_Diffusion_TemporalAttention":
        metrics_name = "temporal_attention_metrics.csv"
    else:
        metrics_name = "simple_metrics.csv"
    pd.DataFrame([rows[0]]).to_csv(out_dir / metrics_name, index=False, encoding="utf-8-sig")

    for name, source in COMPARE_SOURCES.items():
        if name == method_name:
            continue
        if not source.exists():
            print(f"warning: skipped {name}, missing generated file: {source}")
            continue
        target = out_dir / f"generated_samples_{name}.npy"
        shutil.copy2(source, target)
        rows.append(evaluate_method(name, target, data_dir, out_dir))

    df = add_risk_score(pd.DataFrame(rows))
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir, method_col="method")
    rampcurve_cols = [
        "method",
        "q99_cum_deficit_error",
        "core_q99_cum_deficit_error",
        "netload_ramp_max_mae",
        "imbalance_duration_mae",
        "risk_score",
        "risk_rank",
    ]
    ablation = risk_main[[col for col in rampcurve_cols if col in risk_main.columns]].copy()
    ablation["lambda_ramp"] = np.where(ablation["method"].eq(method_name), float(args.lambda_ramp), np.nan)
    ablation["lambda_ramp_curve"] = np.where(ablation["method"].eq(method_name), float(args.lambda_ramp_curve), np.nan)
    ablation.to_csv(out_dir / "rampcurve_ablation.csv", index=False, encoding="utf-8-sig")
    if method_name == "Simple_EVT_Risk_Diffusion_TemporalAttention":
        temporal_ablation = ablation.copy()
        temporal_ablation["hidden_dim"] = np.where(temporal_ablation["method"].eq(method_name), int(getattr(args, "temporal_hidden_dim", 64)), np.nan)
        temporal_ablation["num_layers"] = np.where(temporal_ablation["method"].eq(method_name), int(getattr(args, "temporal_layers", 1)), np.nan)
        temporal_ablation["num_heads"] = np.where(temporal_ablation["method"].eq(method_name), int(getattr(args, "temporal_heads", 4)), np.nan)
        temporal_ablation["dropout"] = np.where(temporal_ablation["method"].eq(method_name), float(getattr(args, "temporal_dropout", 0.10)), np.nan)
        temporal_ablation.to_csv(out_dir / "temporal_attention_ablation.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Simple EVT Risk Diffusion Compare" if method_name != "Simple_EVT_Risk_Diffusion_TemporalAttention" else "# Temporal Attention Compare",
        "",
        RISK_EVALUATION_EXPLANATION,
        RISK_RANKING_EXPLANATION,
        "",
        "## Method Notes",
        "",
        (
            "Simple_EVT_Risk_Diffusion_TemporalAttention keeps the original simple conditions "
            "(event_type, month_sin, month_cos, duration_norm, extreme_prob), keeps the two-stage "
            "training objective, and only replaces the denoiser with a lightweight Conv1D + "
            "Transformer Encoder temporal module."
            if method_name == "Simple_EVT_Risk_Diffusion_TemporalAttention"
            else "This run keeps the simplified EVT risk diffusion comparison settings."
        ),
        "",
        "## Main Risk Table",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Auxiliary Realism Metrics",
        "",
        aux[[col for col in ["method", *AUXILIARY_REALISM_METRICS, "realism_check_pass"] if col in aux.columns]].to_markdown(index=False),
    ]
    report_name = "temporal_attention_report.md" if method_name == "Simple_EVT_Risk_Diffusion_TemporalAttention" else "simple_evt_risk_report.md"
    (out_dir / report_name).write_text("\n".join(report), encoding="utf-8")
    print(RISK_RANKING_EXPLANATION)
    print(RISK_EVALUATION_EXPLANATION)
    print(risk_main.to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simplified EVT probability conditioned risk diffusion.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "ramp_window_retrain" / "datasets" / "dataset_window_3h"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "results" / "simple_evt_risk_diffusion"))
    parser.add_argument("--method-name", type=str, default="Simple_EVT_Risk_Diffusion")
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
    parser.add_argument("--lambda-cum", type=float, default=1.0)
    parser.add_argument("--lambda-ramp", type=float, default=0.4)
    parser.add_argument("--lambda-ramp-curve", type=float, default=0.0)
    parser.add_argument("--use-core-ramp-curve-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda-core-ramp-curve", type=float, default=0.10)
    parser.add_argument("--lambda-dur", type=float, default=0.3)
    parser.add_argument("--lambda-phy", type=float, default=0.02)
    parser.add_argument("--use-ramp-level-condition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ramp-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sampler-alpha-cum", type=float, default=0.3)
    parser.add_argument("--sampler-alpha-ramp", type=float, default=0.8)
    parser.add_argument("--sampler-alpha-dur", type=float, default=0.2)
    parser.add_argument("--use-temporal-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--temporal-dropout", type=float, default=0.10)
    parser.add_argument("--temporal-feedforward-dim", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
