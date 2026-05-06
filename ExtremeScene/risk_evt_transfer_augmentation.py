from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evt_fit import EVTConfig, fit_evt_and_label
from risk_metrics import batch_hard_risk_metrics


CHANNELS = ["load", "wind_power", "solar_power"]


@dataclass
class RiskEVTTransferConfig:
    """Configuration for risk-oriented EVT distribution-transfer augmentation."""

    data_dir: str
    out_dir: str
    seq_len: int = 36
    seed: int = 42
    device: str = "cpu"
    risk_mode: str = "composite"
    epsilon: float = 0.70
    num_transfer_iters: int = 3
    candidate_multiplier: int = 3
    resample_strategy: str = "mixed"
    keep_original_ratio: float = 0.30
    wgan_epochs: int = 120
    batch_size: int = 32
    noise_dim: int = 64
    gan_base_channels: int = 48
    n_critic: int = 3
    gp_lambda: float = 10.0
    lr_g: float = 1.0e-4
    lr_d: float = 1.0e-4
    channel_upper_factor: float = 1.05
    solar_night_zero: bool = False
    daylight_start_hour: int = 6
    daylight_end_hour: int = 18
    save_intermediate: bool = True
    strict_aux_quantile: float = 0.70
    evt_threshold_quantile: float = 0.90
    severity_q1: float = 0.60
    severity_q2: float = 0.80
    severity_q3: float = 0.92
    channel_max_train: Optional[list[float]] = field(default=None, repr=False)


@dataclass
class TransferBundle:
    """Current train split state.

    X: 风光荷样本序列，形状为 [样本数, 时间步, 通道数]，通道为
    load、wind_power、solar_power。
    meta: 样本条件标签和风险标签表。
    event_mask: 核心事件段掩码，增强阶段不作为 GAN 输入，只用于保存和诊断。
    """

    X: np.ndarray
    meta: pd.DataFrame
    event_mask: Optional[np.ndarray] = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_time_channel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got {x.shape}.")
    if x.shape[2] == 3:
        return x.astype(np.float32)
    if x.shape[1] == 3:
        return np.transpose(x, (0, 2, 1)).astype(np.float32)
    raise ValueError(f"Cannot infer channel axis from X shape {x.shape}.")


def _to_channel_time(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got {x.shape}.")
    if x.shape[1] == 3:
        return x.astype(np.float32)
    if x.shape[2] == 3:
        return np.transpose(x, (0, 2, 1)).astype(np.float32)
    raise ValueError(f"Cannot infer channel axis from X shape {x.shape}.")


def _infer_start_hour(meta: pd.DataFrame) -> pd.Series:
    if "start_hour" in meta.columns:
        return pd.to_numeric(meta["start_hour"], errors="coerce").fillna(0.0)
    if "window_start_time" in meta.columns:
        return pd.to_datetime(meta["window_start_time"], errors="coerce").dt.hour.fillna(0.0)
    return pd.Series(np.zeros(len(meta)), index=meta.index, dtype=float)


def _season_to_code(meta: pd.DataFrame) -> pd.Series:
    if "season_code" in meta.columns:
        return pd.to_numeric(meta["season_code"], errors="coerce").fillna(0).astype(int).clip(0, 3)
    mapping = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
    if "season" in meta.columns:
        return meta["season"].astype(str).str.lower().map(mapping).fillna(0).astype(int)
    month = pd.to_numeric(meta.get("month", 1), errors="coerce").fillna(1).astype(int)
    out = pd.Series(0, index=meta.index, dtype=int)
    out[month.isin([6, 7, 8])] = 1
    out[month.isin([9, 10, 11])] = 2
    out[month.isin([12, 1, 2])] = 3
    return out


def _quantile_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{name}": float("nan") for name in ["q50", "q75", "q90", "q95", "q99", "max", "mean"]}
    return {
        f"{prefix}_q50": float(np.quantile(arr, 0.50)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"{prefix}_q90": float(np.quantile(arr, 0.90)),
        f"{prefix}_q95": float(np.quantile(arr, 0.95)),
        f"{prefix}_q99": float(np.quantile(arr, 0.99)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
    }


def load_training_samples(data_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load train samples and merged metadata.

    Returns
    -------
    X:
        风光荷样本序列，shape=[N, L, C]。N 为样本数，L 当前为 36，
        C=3，对应 load、wind_power、solar_power。
    meta:
        样本条件标签和风险标签表，包含 event_type、month、severity_level、
        extreme_prob、tail_score、cum_deficit 等字段。
    """

    bundle = _load_training_bundle(data_dir)
    return bundle.X, bundle.meta


def _load_training_bundle(data_dir: str | Path) -> TransferBundle:
    root = Path(data_dir)
    x = _to_time_channel(np.load(root / "X_train.npy").astype(np.float32))
    cond = pd.read_csv(root / "cond_train.csv")
    meta_df = pd.read_csv(root / "meta_train.csv")
    meta = cond.reset_index(drop=True).copy()
    for col in meta_df.columns:
        if col not in meta.columns:
            meta[col] = meta_df[col].to_numpy()
    meta["_source_index"] = np.arange(len(meta), dtype=int)
    meta["source_sample_id"] = meta.get("sample_id", pd.Series(np.arange(len(meta)))).astype(str)
    mask_path = root / "event_mask_train.npy"
    event_mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else None
    return TransferBundle(X=x, meta=meta, event_mask=event_mask)


def compute_risk_metrics(X: np.ndarray, meta: pd.DataFrame, config: RiskEVTTransferConfig) -> pd.DataFrame:
    """Compute sample-level joint imbalance risk metrics.

    net_load: 净负荷，等于 load - wind_power - solar_power。
    cum_deficit: 累计缺额，衡量事件窗口内持续供需压力。
    netload_ramp_max: 最大净负荷爬坡，衡量短时调节压力。
    imbalance_duration: 持续失衡时长，衡量净负荷超过阈值的持续时间。
    """

    x_ct = _to_channel_time(X)
    if "imbalance_tau" in meta.columns:
        tau = pd.to_numeric(meta["imbalance_tau"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        net = x_ct[:, 0, :] - x_ct[:, 1, :] - x_ct[:, 2, :]
        tau = np.full((x_ct.shape[0],), float(np.quantile(net.reshape(-1), 0.75)), dtype=float)
    if "delta_t_hours" in meta.columns and len(meta):
        delta_t = float(pd.to_numeric(meta["delta_t_hours"], errors="coerce").fillna(1.0).iloc[0])
    else:
        delta_t = 1.0
    metrics = batch_hard_risk_metrics(x_ct, tau=tau, delta_t_hours=delta_t)
    return pd.DataFrame(metrics)


def _quantile_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    return ranks / max(len(arr) - 1, 1)


def compute_risk_score(metrics: pd.DataFrame, mode: str = "composite") -> np.ndarray:
    """Convert risk metrics into a scalar tail-selection score.

    risk_score: 综合风险极端程度得分，用于极值分布转移筛选。
    mode="cum" uses cum_deficit only. mode="composite" combines cumulative,
    ramp, and duration quantile ranks. mode="strict" prefers high cumulative
    deficit while requiring ramp or duration to be above an auxiliary quantile.
    """

    mode = str(mode).strip().lower()
    cum = pd.to_numeric(metrics["cum_deficit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ramp = pd.to_numeric(metrics["netload_ramp_max"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dur = pd.to_numeric(metrics["imbalance_duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if mode == "cum":
        return cum.astype(float)
    if mode == "composite":
        return (
            _quantile_rank(cum)
            + 0.25 * _quantile_rank(ramp)
            + 0.35 * _quantile_rank(dur)
        ).astype(float)
    if mode == "strict":
        ramp_q = float(np.quantile(ramp, 0.70)) if len(ramp) else 0.0
        dur_q = float(np.quantile(dur, 0.70)) if len(dur) else 0.0
        aux_ok = (ramp >= ramp_q) | (dur >= dur_q)
        base = cum.astype(float).copy()
        if np.any(~aux_ok):
            floor = float(np.nanmin(base)) - 1.0
            base[~aux_ok] = floor + 1e-3 * _quantile_rank(cum[~aux_ok])
        return base
    raise ValueError("risk_mode must be one of {'cum', 'composite', 'strict'}.")


class ChannelMinMaxScaler:
    def __init__(self, lo: Optional[np.ndarray] = None, hi: Optional[np.ndarray] = None) -> None:
        self.lo = lo
        self.hi = hi

    def fit(self, X: np.ndarray) -> "ChannelMinMaxScaler":
        x = _to_time_channel(X)
        self.lo = np.quantile(x, 0.002, axis=(0, 1)).astype(np.float32)
        self.hi = np.quantile(x, 0.998, axis=(0, 1)).astype(np.float32)
        self.hi = np.maximum(self.hi, self.lo + 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = _to_time_channel(X)
        y = 2.0 * (x - self.lo[None, None, :]) / (self.hi[None, None, :] - self.lo[None, None, :]) - 1.0
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        y = _to_time_channel(Y)
        x = 0.5 * (y + 1.0) * (self.hi[None, None, :] - self.lo[None, None, :]) + self.lo[None, None, :]
        return np.maximum(x, 0.0).astype(np.float32)


class ConditionEncoder:
    def __init__(self) -> None:
        self.event_cardinality = 1
        self.duration_mean = 0.0
        self.duration_std = 1.0
        self.tail_mean = 0.0
        self.tail_std = 1.0
        self.dim = 0

    def fit(self, meta: pd.DataFrame) -> "ConditionEncoder":
        event = pd.to_numeric(meta.get("event_type_code", 0), errors="coerce").fillna(0).astype(int)
        self.event_cardinality = int(max(event.max() + 1, 1)) if len(event) else 1
        duration = pd.to_numeric(meta.get("duration_hours", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        tail = pd.to_numeric(meta.get("tail_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        self.duration_mean = float(np.mean(duration)) if len(duration) else 0.0
        self.duration_std = float(np.std(duration) + 1e-6)
        self.tail_mean = float(np.mean(tail)) if len(tail) else 0.0
        self.tail_std = float(np.std(tail) + 1e-6)
        self.dim = self.event_cardinality + 2 + 4 + 2 + 1 + 2 + 3
        return self

    def transform(self, meta: pd.DataFrame) -> np.ndarray:
        event = pd.to_numeric(meta.get("event_type_code", 0), errors="coerce").fillna(0).astype(int).clip(0, self.event_cardinality - 1)
        event_oh = np.eye(self.event_cardinality, dtype=np.float32)[event.to_numpy(dtype=int)]
        month = pd.to_numeric(meta.get("month", 1), errors="coerce").fillna(1).astype(float).to_numpy()
        month_feat = np.stack([np.sin(2 * np.pi * month / 12.0), np.cos(2 * np.pi * month / 12.0)], axis=1)
        season = _season_to_code(meta).clip(0, 3).to_numpy(dtype=int)
        season_oh = np.eye(4, dtype=np.float32)[season]
        low_wind = pd.to_numeric(meta.get("low_wind_flag", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None]
        low_irr = pd.to_numeric(meta.get("low_irradiance_flag", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None]
        duration = pd.to_numeric(meta.get("duration_hours", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        duration_z = ((duration - self.duration_mean) / self.duration_std)[:, None]
        start_hour = _infer_start_hour(meta).to_numpy(dtype=float)
        start_feat = np.stack([np.sin(2 * np.pi * start_hour / 24.0), np.cos(2 * np.pi * start_hour / 24.0)], axis=1)
        severity = pd.to_numeric(meta.get("severity_level", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None] / 3.0
        extreme_prob = pd.to_numeric(meta.get("extreme_prob", 1.0), errors="coerce").fillna(1.0).clip(1e-8, 1.0).to_numpy(dtype=float)[:, None]
        tail = pd.to_numeric(meta.get("tail_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        tail_z = ((tail - self.tail_mean) / self.tail_std)[:, None]
        out = np.concatenate(
            [event_oh, month_feat, season_oh, low_wind, low_irr, duration_z, start_feat, severity, extreme_prob, tail_z],
            axis=1,
        )
        return out.astype(np.float32)


class TransferGenerator(nn.Module):
    def __init__(self, noise_dim: int, cond_dim: int, seq_len: int, base_channels: int) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, base_channels * seq_len),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z, cond], dim=1)).view(z.size(0), -1, self.seq_len)
        y = self.net(h)
        return y.transpose(1, 2)


class TransferDiscriminator(nn.Module):
    def __init__(self, cond_dim: int, seq_len: int, base_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, base_channels, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels, base_channels, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cond = nn.Sequential(nn.Linear(cond_dim, base_channels), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(nn.Linear(base_channels * 2, base_channels), nn.LeakyReLU(0.2), nn.Linear(base_channels, 1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x.transpose(1, 2)).squeeze(-1)
        c = self.cond(cond)
        return self.head(torch.cat([feat, c], dim=1)).view(-1)


@dataclass
class WGANState:
    generator: TransferGenerator
    discriminator: TransferDiscriminator
    scaler: ChannelMinMaxScaler
    condition_encoder: ConditionEncoder
    summary: dict


def _gradient_penalty(discriminator: TransferDiscriminator, real: torch.Tensor, fake: torch.Tensor, cond: torch.Tensor, gp_lambda: float) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = discriminator(interpolated, cond)
    grad = autograd.grad(
        outputs=score.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = grad.reshape(real.size(0), -1).norm(2, dim=1)
    return float(gp_lambda) * ((norm - 1.0) ** 2).mean()


def train_conditional_wgan_gp(
    X_i: np.ndarray,
    meta_i: pd.DataFrame,
    prev_model: Optional[WGANState],
    config: RiskEVTTransferConfig,
) -> WGANState:
    """Train a lightweight conditional 1D-CNN WGAN-GP.

    X_i: 当前第 i 轮训练集，shape=[N, 36, 3]。
    meta_i: 当前样本条件标签，作为生成器和判别器条件输入。
    prev_model: 上一轮 WGANState，用于 hot start。
    """

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    scaler = ChannelMinMaxScaler().fit(X_i)
    x_norm = scaler.transform(X_i)
    cond_encoder = ConditionEncoder().fit(meta_i)
    cond = cond_encoder.transform(meta_i)
    dataset = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(cond))
    loader = DataLoader(dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True, drop_last=False)
    generator = TransferGenerator(config.noise_dim, cond_encoder.dim, config.seq_len, config.gan_base_channels).to(device)
    discriminator = TransferDiscriminator(cond_encoder.dim, config.seq_len, config.gan_base_channels).to(device)

    if prev_model is not None:
        try:
            generator.load_state_dict(prev_model.generator.state_dict())
            discriminator.load_state_dict(prev_model.discriminator.state_dict())
        except RuntimeError:
            pass

    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(0.5, 0.9))
    history: list[dict] = []
    for epoch in range(1, int(config.wgan_epochs) + 1):
        d_losses: list[float] = []
        g_losses: list[float] = []
        for real, cond_batch in loader:
            real = real.to(device)
            cond_batch = cond_batch.to(device)
            for _ in range(max(1, int(config.n_critic))):
                z = torch.randn(real.size(0), config.noise_dim, device=device)
                fake = generator(z, cond_batch).detach()
                d_real = discriminator(real, cond_batch).mean()
                d_fake = discriminator(fake, cond_batch).mean()
                gp = _gradient_penalty(discriminator, real, fake, cond_batch, config.gp_lambda)
                d_loss = d_fake - d_real + gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
            z = torch.randn(real.size(0), config.noise_dim, device=device)
            fake = generator(z, cond_batch)
            g_loss = -discriminator(fake, cond_batch).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            d_losses.append(float(d_loss.item()))
            g_losses.append(float(g_loss.item()))
        if epoch == 1 or epoch == config.wgan_epochs or epoch % 50 == 0:
            history.append({"epoch": epoch, "d_loss": float(np.mean(d_losses)), "g_loss": float(np.mean(g_losses))})
    return WGANState(
        generator=generator,
        discriminator=discriminator,
        scaler=scaler,
        condition_encoder=cond_encoder,
        summary={"wgan_epochs": int(config.wgan_epochs), "history": history, "condition_dim": int(cond_encoder.dim)},
    )


def generate_candidates(
    generator: WGANState | TransferGenerator,
    meta_condition_pool: pd.DataFrame,
    num_candidates: int,
    config: RiskEVTTransferConfig,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Generate candidate samples from condition rows sampled with replacement.

    candidate_multiplier: 候选生成倍数，通常候选数量 = candidate_multiplier * N。
    """

    if not isinstance(generator, WGANState):
        raise TypeError("generate_candidates expects the WGANState returned by train_conditional_wgan_gp.")
    state = generator
    device = next(state.generator.parameters()).device
    rng = np.random.default_rng(int(config.seed) + int(num_candidates))
    sampled_idx = rng.integers(0, len(meta_condition_pool), size=int(num_candidates))
    meta_g = meta_condition_pool.iloc[sampled_idx].reset_index(drop=True).copy()
    meta_g["synthetic"] = 1
    meta_g["condition_source_index"] = sampled_idx.astype(int)
    cond = torch.from_numpy(state.condition_encoder.transform(meta_g)).to(device)
    outs: list[np.ndarray] = []
    state.generator.eval()
    with torch.no_grad():
        for start in range(0, len(meta_g), config.batch_size):
            cond_b = cond[start:start + config.batch_size]
            z = torch.randn(cond_b.size(0), config.noise_dim, device=device)
            y = state.generator(z, cond_b).cpu().numpy()
            outs.append(y)
    x_norm = np.concatenate(outs, axis=0).astype(np.float32)
    x = state.scaler.inverse_transform(x_norm)
    return x, meta_g, sampled_idx.astype(int)


def _physical_keep_mask(X: np.ndarray, config: RiskEVTTransferConfig) -> np.ndarray:
    x = _to_time_channel(X)
    finite = np.isfinite(x).all(axis=(1, 2))
    nonflat = x.std(axis=(1, 2)) > 1e-6
    nonnegative_soft = (x >= -1e-3).all(axis=(1, 2))
    if config.channel_max_train is None:
        upper = np.maximum(np.nanmax(x, axis=(0, 1)), 1e-6) * config.channel_upper_factor
    else:
        upper = np.asarray(config.channel_max_train, dtype=np.float32) * config.channel_upper_factor
    not_extreme_bound = (x <= upper[None, None, :] * 1.50).all(axis=(1, 2))
    return finite & nonflat & nonnegative_soft & not_extreme_bound


def physical_filter_and_clip(
    X_g: np.ndarray,
    meta_g: pd.DataFrame,
    config: RiskEVTTransferConfig,
    event_mask_g: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, pd.DataFrame, Optional[np.ndarray], dict]:
    """Filter physically invalid candidates and clip valid ones to train bounds.

    load、wind_power、solar_power 不得为负；三通道上界使用训练集最大值
    的合理放大系数，避免用 clip 掩盖明显异常样本。
    """

    keep = _physical_keep_mask(X_g, config)
    x = _to_time_channel(X_g)[keep].copy()
    meta = meta_g.loc[keep].reset_index(drop=True).copy()
    mask = event_mask_g[keep] if event_mask_g is not None else None
    upper = np.asarray(config.channel_max_train if config.channel_max_train is not None else np.nanmax(x, axis=(0, 1)), dtype=np.float32)
    upper = upper * float(config.channel_upper_factor)
    x = np.clip(x, 0.0, upper[None, None, :]).astype(np.float32)
    if config.solar_night_zero and "window_start_time" in meta.columns:
        hours0 = pd.to_datetime(meta["window_start_time"], errors="coerce").dt.hour.fillna(0).to_numpy(dtype=int)
        for i, hour0 in enumerate(hours0):
            hours = (hour0 + np.arange(x.shape[1])) % 24
            night = (hours < config.daylight_start_hour) | (hours > config.daylight_end_hour)
            x[i, night, 2] = 0.0
    summary = {
        "generated_count": int(len(X_g)),
        "kept_count": int(len(x)),
        "filtered_count": int(len(X_g) - len(x)),
        "filtered_ratio": float(1.0 - len(x) / max(len(X_g), 1)),
    }
    return x, meta, mask, summary


def _resample_from_pool(
    original: TransferBundle,
    pool: TransferBundle,
    target_size: int,
    config: RiskEVTTransferConfig,
    iteration: int,
) -> TransferBundle:
    rng = np.random.default_rng(int(config.seed) + 7919 * int(iteration))
    if len(pool.X) == 0:
        return original
    if config.resample_strategy == "pure_tail":
        pool_n = target_size
        orig_n = 0
    else:
        orig_n = int(round(target_size * float(config.keep_original_ratio)))
        orig_n = min(max(orig_n, 0), target_size)
        pool_n = target_size - orig_n
    orig_idx = rng.integers(0, len(original.X), size=orig_n) if orig_n else np.array([], dtype=int)
    pool_idx = rng.integers(0, len(pool.X), size=pool_n) if pool_n else np.array([], dtype=int)
    parts_x = []
    parts_meta = []
    parts_mask = []
    if orig_n:
        parts_x.append(original.X[orig_idx])
        parts_meta.append(original.meta.iloc[orig_idx].copy())
        if original.event_mask is not None:
            parts_mask.append(original.event_mask[orig_idx])
    if pool_n:
        parts_x.append(pool.X[pool_idx])
        parts_meta.append(pool.meta.iloc[pool_idx].copy())
        if pool.event_mask is not None:
            parts_mask.append(pool.event_mask[pool_idx])
    x_new = np.concatenate(parts_x, axis=0).astype(np.float32)
    meta_new = pd.concat(parts_meta, ignore_index=True)
    meta_new["transfer_iteration"] = int(iteration)
    meta_new["transfer_resampled"] = 1
    if parts_mask and sum(len(m) for m in parts_mask) == len(x_new):
        mask_new = np.concatenate(parts_mask, axis=0).astype(np.float32)
    else:
        mask_new = None
    return TransferBundle(X=x_new, meta=meta_new.reset_index(drop=True), event_mask=mask_new)


def evt_distribution_transfer(X_i: np.ndarray, meta_i: pd.DataFrame, config: RiskEVTTransferConfig) -> tuple[np.ndarray, pd.DataFrame, Optional[np.ndarray], pd.DataFrame]:
    """Run iterative risk EVT distribution transfer.

    epsilon: 分位筛选因子，用于确定保留尾部样本的相对极端阈值。
    keep_original_ratio: 保留原始样本比例，防止分布过度偏移。
    num_transfer_iters: 极值分布转移迭代轮数。
    """

    _set_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_bundle = _load_training_bundle(config.data_dir)
    current = TransferBundle(X=_to_time_channel(X_i), meta=meta_i.reset_index(drop=True).copy(), event_mask=original_bundle.event_mask)
    config.channel_max_train = np.max(original_bundle.X, axis=(0, 1)).astype(float).tolist()
    target_size = len(current.X)
    state: Optional[WGANState] = None
    iter_rows: list[dict] = []
    inter_dir = out_dir / "transfer_iterations"
    inter_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, int(config.num_transfer_iters) + 1):
        metrics_real = compute_risk_metrics(current.X, current.meta, config)
        score_real = compute_risk_score(metrics_real, mode=config.risk_mode)
        state = train_conditional_wgan_gp(current.X, current.meta, prev_model=state, config=config)
        num_candidates = int(config.candidate_multiplier) * len(current.X)
        X_g, meta_g, source_idx = generate_candidates(state, current.meta, num_candidates, config)
        mask_g = current.event_mask[source_idx] if current.event_mask is not None else None
        X_g, meta_g, mask_g, filter_summary = physical_filter_and_clip(X_g, meta_g, config, event_mask_g=mask_g)
        if len(X_g) == 0:
            raise RuntimeError("All generated candidates were filtered out; inspect WGAN output scale and physical filters.")
        metrics_gen = compute_risk_metrics(X_g, meta_g, config)
        score_gen = compute_risk_score(metrics_gen, mode=config.risk_mode)

        tau_real = float(np.quantile(score_real, float(config.epsilon)))
        tau_gen = float(np.quantile(score_gen, float(config.epsilon)))
        real_tail_mask = score_real >= tau_real
        gen_tail_mask = score_gen >= tau_gen
        tail_real = TransferBundle(
            X=current.X[real_tail_mask],
            meta=current.meta.loc[real_tail_mask].reset_index(drop=True).copy(),
            event_mask=current.event_mask[real_tail_mask] if current.event_mask is not None else None,
        )
        tail_gen = TransferBundle(
            X=X_g[gen_tail_mask],
            meta=meta_g.loc[gen_tail_mask].reset_index(drop=True).copy(),
            event_mask=mask_g[gen_tail_mask] if mask_g is not None else None,
        )
        tail_gen.meta["synthetic"] = 1
        pool_X = np.concatenate([tail_real.X, tail_gen.X], axis=0).astype(np.float32)
        pool_meta = pd.concat([tail_real.meta, tail_gen.meta], ignore_index=True)
        if tail_real.event_mask is not None and tail_gen.event_mask is not None:
            pool_mask = np.concatenate([tail_real.event_mask, tail_gen.event_mask], axis=0).astype(np.float32)
        else:
            pool_mask = None
        pool = TransferBundle(pool_X, pool_meta, pool_mask)
        current = _resample_from_pool(original_bundle, pool, target_size=target_size, config=config, iteration=iteration)

        row = {
            "iteration": int(iteration),
            "tau_real": tau_real,
            "tau_gen": tau_gen,
            "real_tail_count": int(real_tail_mask.sum()),
            "gen_tail_count": int(gen_tail_mask.sum()),
            "pool_count": int(len(pool_X)),
            **filter_summary,
            **_quantile_summary(metrics_real["cum_deficit"].to_numpy(), "real_cum_deficit"),
            **_quantile_summary(metrics_gen["cum_deficit"].to_numpy(), "gen_cum_deficit"),
            **_quantile_summary(score_real, "real_risk_score"),
            **_quantile_summary(score_gen, "gen_risk_score"),
            "event_type_distribution": json.dumps(current.meta.get("event_type", pd.Series(dtype=str)).value_counts().to_dict(), ensure_ascii=False),
            "severity_distribution": json.dumps(current.meta.get("severity_level", pd.Series(dtype=int)).value_counts().sort_index().to_dict(), ensure_ascii=False),
            "wgan_summary": json.dumps(state.summary, ensure_ascii=False),
        }
        iter_rows.append(row)
        if config.save_intermediate:
            np.save(inter_dir / f"iteration_{iteration:02d}_X_time_channel.npy", current.X.astype(np.float32))
            current.meta.to_csv(inter_dir / f"iteration_{iteration:02d}_meta.csv", index=False, encoding="utf-8-sig")
            if current.event_mask is not None:
                np.save(inter_dir / f"iteration_{iteration:02d}_event_mask.npy", current.event_mask.astype(np.float32))

    stats_df = pd.DataFrame(iter_rows)
    stats_df.to_csv(out_dir / "transfer_iteration_stats.csv", index=False, encoding="utf-8-sig")
    return current.X, current.meta, current.event_mask, stats_df


def recompute_evt_labels(X_final: np.ndarray, meta_final: pd.DataFrame, config: RiskEVTTransferConfig) -> tuple[pd.DataFrame, dict]:
    """Recompute risk metrics and EVT labels only from augmented train data.

    extreme_prob: EVT 或经验尾部概率得到的连续极端概率。
    severity_level: 离散极端等级。
    """

    meta = meta_final.reset_index(drop=True).copy()
    metrics = compute_risk_metrics(X_final, meta, config)
    for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
        meta[col] = metrics[col].to_numpy(dtype=float)
    evt_cfg = EVTConfig(
        metric_col="cum_deficit",
        threshold_quantile=config.evt_threshold_quantile,
        severity_mode="hybrid",
        severity_q1=config.severity_q1,
        severity_q2=config.severity_q2,
        severity_q3=config.severity_q3,
    )
    labeled, evt_info = fit_evt_and_label(meta, evt_cfg)
    labeled["sample_id"] = [f"EVTTR_{i + 1:05d}" for i in range(len(labeled))]
    return labeled, evt_info


def _split_cond_meta_columns(full_meta: pd.DataFrame, base_data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(base_data_dir)
    cond_cols = pd.read_csv(root / "cond_train.csv", nrows=0).columns.tolist()
    meta_cols = pd.read_csv(root / "meta_train.csv", nrows=0).columns.tolist()
    cond_out = pd.DataFrame()
    meta_out = pd.DataFrame()
    for col in cond_cols:
        cond_out[col] = full_meta[col] if col in full_meta.columns else ""
    for col in meta_cols:
        meta_out[col] = full_meta[col] if col in full_meta.columns else ""
    return cond_out, meta_out


def _plot_before_after(before: np.ndarray, after: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(before, bins=24, alpha=0.55, label="before")
    plt.hist(after, bins=24, alpha=0.55, label="after")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_augmented_dataset(
    X_original: np.ndarray,
    meta_original: pd.DataFrame,
    X_final: np.ndarray,
    meta_final: pd.DataFrame,
    event_mask_final: Optional[np.ndarray],
    config: RiskEVTTransferConfig,
    iteration_stats: pd.DataFrame,
    evt_info: dict,
) -> dict:
    """Save project-compatible augmented train files.

    Saves both [N,L,C] diagnostic arrays and project-compatible [N,3,L]
    `X_train_aug.npy` for the existing proposed diffusion trainer.
    """

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_aug, meta_aug = _split_cond_meta_columns(meta_final, config.data_dir)
    np.save(out_dir / "augmented_train_X.npy", _to_time_channel(X_final).astype(np.float32))
    np.save(out_dir / "X_train_aug.npy", _to_channel_time(X_final).astype(np.float32))
    cond_aug.to_csv(out_dir / "cond_train_aug.csv", index=False, encoding="utf-8-sig")
    meta_aug.to_csv(out_dir / "meta_train_aug.csv", index=False, encoding="utf-8-sig")
    meta_final.to_csv(out_dir / "augmented_train_meta.csv", index=False, encoding="utf-8-sig")
    if event_mask_final is not None:
        np.save(out_dir / "event_mask_train_aug.npy", event_mask_final.astype(np.float32))
    iteration_stats.to_csv(out_dir / "transfer_iteration_stats.csv", index=False, encoding="utf-8-sig")

    before_metrics = compute_risk_metrics(X_original, meta_original, config)
    after_metrics = compute_risk_metrics(X_final, meta_final, config)
    before_score = compute_risk_score(before_metrics, mode=config.risk_mode)
    after_score = compute_risk_score(after_metrics, mode=config.risk_mode)
    _plot_before_after(before_score, after_score, "risk_score before/after EVT transfer", out_dir / "risk_distribution_before_after.png")
    _plot_before_after(before_metrics["cum_deficit"].to_numpy(), after_metrics["cum_deficit"].to_numpy(), "cum_deficit before/after EVT transfer", out_dir / "cum_deficit_hist_before_after.png")

    event_dist = pd.DataFrame(
        {
            "before": meta_original.get("event_type", pd.Series(dtype=str)).value_counts(),
            "after": meta_final.get("event_type", pd.Series(dtype=str)).value_counts(),
        }
    ).fillna(0).astype(int)
    event_dist.to_csv(out_dir / "event_type_distribution_before_after.csv", encoding="utf-8-sig")
    sev_dist = pd.DataFrame(
        {
            "before": meta_original.get("severity_level", pd.Series(dtype=int)).value_counts().sort_index(),
            "after": meta_final.get("severity_level", pd.Series(dtype=int)).value_counts().sort_index(),
        }
    ).fillna(0).astype(int)
    sev_dist.to_csv(out_dir / "severity_distribution_before_after.csv", encoding="utf-8-sig")

    summary = {
        "config": asdict(config),
        "original_train_size": int(len(X_original)),
        "augmented_train_size": int(len(X_final)),
        "target_size_kept": bool(len(X_original) == len(X_final)),
        "risk_mode": config.risk_mode,
        "epsilon": float(config.epsilon),
        "candidate_multiplier": int(config.candidate_multiplier),
        "keep_original_ratio": float(config.keep_original_ratio),
        "num_transfer_iters": int(config.num_transfer_iters),
        "cum_deficit_before": _quantile_summary(before_metrics["cum_deficit"].to_numpy(), "cum_deficit_before"),
        "cum_deficit_after": _quantile_summary(after_metrics["cum_deficit"].to_numpy(), "cum_deficit_after"),
        "risk_score_before": _quantile_summary(before_score, "risk_score_before"),
        "risk_score_after": _quantile_summary(after_score, "risk_score_after"),
        "evt_info": evt_info,
        "note": "Only train split is augmented. Validation/test files are untouched and read from the original dataset directory.",
    }
    (out_dir / "augmentation_log.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_risk_evt_transfer_augmentation(config: RiskEVTTransferConfig) -> dict:
    """High-level API used by experiment runners."""

    original = _load_training_bundle(config.data_dir)
    X_final, meta_final, event_mask_final, iteration_stats = evt_distribution_transfer(original.X, original.meta, config)
    labeled_meta, evt_info = recompute_evt_labels(X_final, meta_final, config)
    return save_augmented_dataset(
        X_original=original.X,
        meta_original=original.meta,
        X_final=X_final,
        meta_final=labeled_meta,
        event_mask_final=event_mask_final,
        config=config,
        iteration_stats=iteration_stats,
        evt_info=evt_info,
    )
