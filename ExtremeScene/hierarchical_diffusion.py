from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from risk_metrics import soft_risk_metrics_torch


@dataclass
class ConditionNormalizers:
    duration_mean: float
    duration_std: float
    tail_mean: float
    tail_std: float


def compute_condition_normalizers(cond_df: pd.DataFrame) -> ConditionNormalizers:
    duration = cond_df["duration_hours"].astype(float).to_numpy(dtype=np.float32)
    tail = cond_df["tail_score"].astype(float).to_numpy(dtype=np.float32)
    return ConditionNormalizers(
        duration_mean=float(duration.mean()),
        duration_std=float(duration.std() + 1e-6),
        tail_mean=float(tail.mean()),
        tail_std=float(tail.std() + 1e-6),
    )


def infer_day_mask(meta_df: pd.DataFrame, seq_len: int, daylight_start_hour: int, daylight_end_hour: int) -> np.ndarray:
    n = len(meta_df)
    day_masks = np.ones((n, seq_len), dtype=np.float32)
    if "window_start_time" not in meta_df.columns:
        return day_masks
    start_ts = pd.to_datetime(meta_df["window_start_time"])
    start_hours = start_ts.dt.hour.to_numpy(dtype=np.int64)
    for i, hour0 in enumerate(start_hours):
        hours = (hour0 + np.arange(seq_len)) % 24
        mask = ((hours >= daylight_start_hour) & (hours <= daylight_end_hour)).astype(np.float32)
        day_masks[i] = mask
    return day_masks


def _season_one_hot(season_code: np.ndarray) -> np.ndarray:
    season_code = season_code.astype(np.int64)
    return np.eye(4, dtype=np.float32)[season_code]


def build_condition_bundle(
    cond_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    seq_len: int,
    ablation: str = "full",
    normalizers: Optional[ConditionNormalizers] = None,
    expected_event_types: Optional[int] = None,
    daylight_start_hour: int = 6,
    daylight_end_hour: int = 18,
) -> tuple[dict[str, np.ndarray], dict]:
    cond_df = cond_df.reset_index(drop=True).copy()
    meta_df = meta_df.reset_index(drop=True).copy()
    normalizers = normalizers or compute_condition_normalizers(cond_df)

    event_codes = cond_df["event_type_code"].astype(int).to_numpy()
    inferred_events = int(np.max(event_codes)) + 1 if event_codes.size else 1
    n_events = int(max(expected_event_types or inferred_events, inferred_events, 1))
    event_onehot = np.eye(n_events, dtype=np.float32)[event_codes]

    month = cond_df["month"].astype(float).to_numpy(dtype=np.float32)
    month_sin = np.sin(2 * np.pi * month / 12.0)[:, None]
    month_cos = np.cos(2 * np.pi * month / 12.0)[:, None]
    season_code = cond_df["season_code"].astype(int).to_numpy(dtype=np.int64)
    season_onehot = _season_one_hot(season_code)

    low_wind = cond_df["low_wind_flag"].astype(float).to_numpy(dtype=np.float32)[:, None]
    low_irr = cond_df["low_irradiance_flag"].astype(float).to_numpy(dtype=np.float32)[:, None]
    duration = cond_df["duration_hours"].astype(float).to_numpy(dtype=np.float32)
    duration_z = ((duration - normalizers.duration_mean) / normalizers.duration_std)[:, None]

    if "window_start_time" in meta_df.columns:
        start_hour = pd.to_datetime(meta_df["window_start_time"]).dt.hour.to_numpy(dtype=np.float32)
    else:
        start_hour = np.zeros((len(cond_df),), dtype=np.float32)
    start_hour_sin = np.sin(2 * np.pi * start_hour / 24.0)[:, None]
    start_hour_cos = np.cos(2 * np.pi * start_hour / 24.0)[:, None]

    extreme_prob = cond_df["extreme_prob"].astype(float).clip(1e-8, 1.0).to_numpy(dtype=np.float32)[:, None]
    tail_score = cond_df["tail_score"].astype(float).to_numpy(dtype=np.float32)
    tail_score_z = ((tail_score - normalizers.tail_mean) / normalizers.tail_std)[:, None]
    severity = (cond_df["severity_level"].astype(float).to_numpy(dtype=np.float32) / 3.0)[:, None]

    bg = np.concatenate([event_onehot, month_sin, month_cos, season_onehot], axis=1).astype(np.float32)
    proc = np.concatenate([low_wind, low_irr, duration_z, start_hour_sin, start_hour_cos], axis=1).astype(np.float32)
    risk = np.concatenate([extreme_prob, tail_score_z, severity], axis=1).astype(np.float32)

    if ablation == "no_month":
        month_slice = slice(event_onehot.shape[1], event_onehot.shape[1] + 2)
        bg[:, month_slice] = 0.0
    if ablation in {"no_evt", "no_evt_continuous"}:
        # Keep severity as an ordinal risk hint, but remove continuous EVT features.
        risk[:, 0:2] = 0.0
    if ablation == "no_evt_strict":
        # Strict EVT ablation removes the whole risk layer signal.
        risk[:, :] = 0.0

    day_mask = infer_day_mask(meta_df, seq_len, daylight_start_hour, daylight_end_hour)
    risk_targets = cond_df[
        [
            "cum_deficit",
            "netload_ramp_max",
            "imbalance_duration",
            "imbalance_tau",
            "extreme_prob",
            "severity_level",
        ]
    ].to_numpy(dtype=np.float32)

    layout = {
        "background": {
            "event_onehot": list(range(0, event_onehot.shape[1])),
            "month_sin": [event_onehot.shape[1]],
            "month_cos": [event_onehot.shape[1] + 1],
            "season_onehot": list(range(event_onehot.shape[1] + 2, bg.shape[1])),
        },
        "process": {
            "low_wind_flag": [0],
            "low_irradiance_flag": [1],
            "duration_hours_zscore": [2],
            "start_hour_sin": [3],
            "start_hour_cos": [4],
        },
        "risk": {
            "extreme_prob": [0],
            "tail_score_zscore": [1],
            "severity_level_scaled": [2],
        },
        "ablation": ablation,
        "normalizers": {
            "duration_mean": normalizers.duration_mean,
            "duration_std": normalizers.duration_std,
            "tail_mean": normalizers.tail_mean,
            "tail_std": normalizers.tail_std,
        },
    }
    arrays = {
        "background": bg,
        "process": proc,
        "risk": risk,
        "day_mask": day_mask,
        "risk_targets": risk_targets,
    }
    return arrays, layout


class ConditionedWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        cond_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        seq_len: int,
        ablation: str = "full",
        normalizers: Optional[ConditionNormalizers] = None,
        daylight_start_hour: int = 6,
        daylight_end_hour: int = 18,
    ) -> None:
        if x.ndim != 3 or x.shape[1] != 3:
            raise ValueError("Expected X with shape [N, 3, T].")
        self.X = x.astype(np.float32)
        self.cond_df = cond_df.reset_index(drop=True).copy()
        self.meta_df = meta_df.reset_index(drop=True).copy()
        self.seq_len = seq_len
        arrays, layout = build_condition_bundle(
            self.cond_df,
            self.meta_df,
            seq_len=seq_len,
            ablation=ablation,
            normalizers=normalizers,
            expected_event_types=None,
            daylight_start_hour=daylight_start_hour,
            daylight_end_hour=daylight_end_hour,
        )
        self.background = arrays["background"]
        self.process = arrays["process"]
        self.risk = arrays["risk"]
        self.day_mask = arrays["day_mask"]
        self.risk_targets = arrays["risk_targets"]
        self.condition_meta = layout

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.background[idx]),
            torch.from_numpy(self.process[idx]),
            torch.from_numpy(self.risk[idx]),
            torch.from_numpy(self.risk_targets[idx]),
            torch.from_numpy(self.day_mask[idx]),
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(time_emb)[:, :, None] + self.cond_proj(cond_emb)[:, :, None]
        h = self.block2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock1D(in_ch, out_ch, time_dim, cond_dim)
        self.res2 = ResidualBlock1D(out_ch, out_ch, time_dim, cond_dim)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, time_emb, cond_emb)
        x = self.res2(x, time_emb, cond_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock1D(out_ch + skip_ch, out_ch, time_dim, cond_dim)
        self.res2 = ResidualBlock1D(out_ch, out_ch, time_dim, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb, cond_emb)
        x = self.res2(x, time_emb, cond_emb)
        return x


class HierarchicalConditionEncoder(nn.Module):
    def __init__(self, bg_dim: int, proc_dim: int, risk_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.bg_dim = bg_dim
        self.proc_dim = proc_dim
        self.risk_dim = risk_dim
        self.bg_mlp = nn.Sequential(nn.Linear(bg_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.proc_mlp = nn.Sequential(nn.Linear(proc_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.risk_mlp = nn.Sequential(nn.Linear(risk_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fuse_mlp = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, bg_cond: torch.Tensor, proc_cond: torch.Tensor, risk_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bg_emb = self.bg_mlp(bg_cond)
        proc_emb = self.proc_mlp(proc_cond)
        risk_emb = self.risk_mlp(risk_cond)
        fused = self.fuse_mlp(torch.cat([bg_emb, proc_emb, risk_emb], dim=1))
        return bg_emb, proc_emb, risk_emb, fused


class FlatConditionEncoder(nn.Module):
    def __init__(self, bg_dim: int, proc_dim: int, risk_dim: int, hidden_dim: int) -> None:
        super().__init__()
        total_dim = bg_dim + proc_dim + risk_dim
        self.flat_mlp = nn.Sequential(nn.Linear(total_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, bg_cond: torch.Tensor, proc_cond: torch.Tensor, risk_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.flat_mlp(torch.cat([bg_cond, proc_cond, risk_cond], dim=1))
        return fused, fused, fused, fused


class HierarchicalConditionalUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        time_dim: int,
        cond_dim: int,
        bg_dim: int,
        proc_dim: int,
        risk_dim: int,
        flat_condition: bool = False,
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
        if flat_condition:
            self.cond_encoder = FlatConditionEncoder(bg_dim, proc_dim, risk_dim, cond_dim)
        else:
            self.cond_encoder = HierarchicalConditionEncoder(bg_dim, proc_dim, risk_dim, cond_dim)
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels, time_dim, cond_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_dim, cond_dim)
        self.mid1 = ResidualBlock1D(base_channels * 2, base_channels * 4, time_dim, cond_dim)
        self.mid2 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_dim, cond_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, time_dim, cond_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels, time_dim, cond_dim)
        self.final_cond_proj = nn.Linear(cond_dim, base_channels)
        self.final = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=1),
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
        bg_cond: Optional[torch.Tensor],
        proc_cond: Optional[torch.Tensor],
        risk_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = x.size(0)
        time_emb = self.time_mlp(t)
        if bg_cond is None or proc_cond is None or risk_cond is None:
            bg_cond, proc_cond, risk_cond = self._zero_bundle(batch_size, x.device)
        bg_emb, proc_emb, risk_emb, fused_emb = self.cond_encoder(bg_cond, proc_cond, risk_cond)
        down_cond = bg_emb + fused_emb
        mid_cond = proc_emb + fused_emb
        up_cond = risk_emb + fused_emb

        x0 = self.init_conv(x)
        x1, skip1 = self.down1(x0, time_emb, down_cond)
        x2, skip2 = self.down2(x1, time_emb, down_cond)
        h = self.mid1(x2, time_emb, mid_cond)
        h = self.mid2(h, time_emb, mid_cond)
        h = self.up2(h, skip2, time_emb, up_cond)
        h = self.up1(h, skip1, time_emb, up_cond)
        h = h + self.final_cond_proj(up_cond)[:, :, None]
        return self.final(h)


class DiffusionScheduler(nn.Module):
    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_alpha_bars[t][:, None, None] * x0 + self.sqrt_one_minus_alpha_bars[t][:, None, None] * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        return (x_t - self.sqrt_one_minus_alpha_bars[t][:, None, None] * pred_noise) / self.sqrt_alpha_bars[t][:, None, None]

    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        bg_cond: Optional[torch.Tensor],
        proc_cond: Optional[torch.Tensor],
        risk_cond: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        pred_cond = model(x, t, bg_cond, proc_cond, risk_cond)
        if guidance_scale != 1.0:
            pred_uncond = model(x, t, None, None, None)
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_noise = pred_cond
        beta_t = self.betas[t][:, None, None]
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bars[t][:, None, None]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None, None]
        model_mean = sqrt_recip_alpha_t * (x - beta_t * pred_noise / sqrt_one_minus_ab_t)
        var_t = self.posterior_variance[t][:, None, None]
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float()[:, None, None]
        return model_mean + nonzero_mask * torch.sqrt(var_t) * noise


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def condition_dropout(
    bg_cond: torch.Tensor,
    proc_cond: torch.Tensor,
    risk_cond: torch.Tensor,
    dropout_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dropout_prob <= 0:
        return bg_cond, proc_cond, risk_cond
    keep_mask = (torch.rand(bg_cond.size(0), device=bg_cond.device) > dropout_prob).float().unsqueeze(1)
    return bg_cond * keep_mask, proc_cond * keep_mask, risk_cond * keep_mask


def denormalize_x_torch(x_norm: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
    return x_norm * x_std + x_mean


def denormalize_x_np(x_norm: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return x_norm * x_std + x_mean


def apply_physical_projection(x_denorm: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
    load = torch.clamp(x_denorm[:, 0, :], min=0.0)
    wind = torch.clamp(x_denorm[:, 1, :], min=0.0)
    solar = torch.clamp(x_denorm[:, 2, :], min=0.0) * day_mask
    return torch.stack([load, wind, solar], dim=1)


def physics_penalty(x_denorm: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
    neg_penalty = F.relu(-x_denorm).mean()
    night_mask = 1.0 - day_mask
    night_solar_penalty = (F.relu(x_denorm[:, 2, :]) * night_mask).mean()
    return neg_penalty + night_solar_penalty


def resource_consistency_loss(
    x_proj: torch.Tensor,
    x_true: torch.Tensor,
    day_mask: torch.Tensor,
    proc_cond: torch.Tensor,
) -> torch.Tensor:
    low_wind = proc_cond[:, 0]
    low_irr = proc_cond[:, 1]
    wind_pred = x_proj[:, 1, :].mean(dim=1)
    wind_true = x_true[:, 1, :].mean(dim=1)
    solar_pred = (x_proj[:, 2, :] * day_mask).sum(dim=1) / (day_mask.sum(dim=1) + 1e-6)
    solar_true = (x_true[:, 2, :] * day_mask).sum(dim=1) / (day_mask.sum(dim=1) + 1e-6)
    wind_loss = (low_wind * torch.abs(wind_pred - wind_true)).sum() / (low_wind.sum() + 1e-6)
    solar_loss = (low_irr * torch.abs(solar_pred - solar_true)).sum() / (low_irr.sum() + 1e-6)
    return wind_loss + solar_loss


def risk_consistency_loss(
    x_proj: torch.Tensor,
    risk_targets: torch.Tensor,
    delta_t_hours: float,
    duration_temp: float,
    risk_norm: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = risk_targets[:, 3]
    target_cum = risk_targets[:, 0]
    target_ramp = risk_targets[:, 1]
    target_dur = risk_targets[:, 2]

    cum_pred, ramp_pred, dur_pred = soft_risk_metrics_torch(
        x_proj,
        tau=tau,
        delta_t_hours=delta_t_hours,
        duration_temp=duration_temp,
    )
    cum_loss = F.l1_loss(
        (torch.log1p(cum_pred) - risk_norm["log_cum_mean"]) / risk_norm["log_cum_std"],
        (torch.log1p(target_cum) - risk_norm["log_cum_mean"]) / risk_norm["log_cum_std"],
    )
    ramp_loss = F.l1_loss(
        (ramp_pred - risk_norm["ramp_mean"]) / risk_norm["ramp_std"],
        (target_ramp - risk_norm["ramp_mean"]) / risk_norm["ramp_std"],
    )
    dur_loss = F.l1_loss(
        (dur_pred - risk_norm["dur_mean"]) / risk_norm["dur_std"],
        (target_dur - risk_norm["dur_mean"]) / risk_norm["dur_std"],
    )
    total = cum_loss + ramp_loss + dur_loss
    return total, cum_loss, ramp_loss, dur_loss


def sample_sequences(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    bg_cond: torch.Tensor,
    proc_cond: torch.Tensor,
    risk_cond: torch.Tensor,
    shape: tuple[int, int, int],
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(shape, device=device)
    for step in reversed(range(scheduler.steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x = scheduler.p_sample(model, x, t, bg_cond, proc_cond, risk_cond, guidance_scale)
    return x


def load_split_arrays(data_dir: str | Path, split: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    x = np.load(root / f"X_{split}.npy").astype(np.float32)
    cond_df = pd.read_csv(root / f"cond_{split}.csv")
    meta_df = pd.read_csv(root / f"meta_{split}.csv")
    return x, cond_df, meta_df
