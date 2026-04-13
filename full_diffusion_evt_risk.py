from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class TrainConfig:
    data_dir: str = "mock_dataset_outputs"
    out_dir: str = "mock_full_diffusion_evt_risk_outputs"
    seed: int = 42

    seq_len: int = 24
    in_channels: int = 3
    cond_dropout: float = 0.10

    batch_size: int = 64
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.85

    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    guidance_scale: float = 1.6

    base_channels: int = 64
    time_emb_dim: int = 128
    cond_emb_dim: int = 128

    ema_decay: float = 0.995
    num_generated_samples: int = 12

    # === 新增三类机制的权重 ===
    tail_alpha: float = 2.0                 # 尾部敏感训练强度
    tail_gamma: float = 1.0                 # 尾部权重曲率
    lambda_risk: float = 0.08               # 联合失衡约束总权重
    lambda_cum: float = 1.0                 # 累计缺额权重
    lambda_ramp: float = 0.5                # 爬坡权重
    lambda_dur: float = 0.4                 # 持续失衡时长权重
    duration_temp: float = 12.0             # soft duration 温度系数
    use_severity_aux: bool = True           # 是否保留 severity 作为辅助条件


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceConditionDataset(Dataset):
    def __init__(self, x_path: Path, cond_path: Path, seq_len: int = 24, use_severity_aux: bool = True) -> None:
        self.X = np.load(x_path).astype(np.float32)
        self.cond_df = pd.read_csv(cond_path)
        self.use_severity_aux = use_severity_aux

        if self.X.ndim != 3:
            raise ValueError(f"X.npy 应该是 (N, C, T)，当前得到 {self.X.shape}")
        if self.X.shape[0] != len(self.cond_df):
            raise ValueError("X.npy 与 cond.csv 样本数不一致")
        if self.X.shape[2] != seq_len:
            raise ValueError(f"序列长度应为 {seq_len}，当前为 {self.X.shape[2]}")

        self.cond_features, self.cond_meta = self._build_condition_matrix(self.cond_df)
        self.target_metrics = self._build_target_metrics(self.cond_df)

        self.X_mean = self.X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.X_std = (self.X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        self.X_norm = ((self.X - self.X_mean) / self.X_std).astype(np.float32)

    @staticmethod
    def _safe_zscore(col: pd.Series) -> Tuple[np.ndarray, dict]:
        arr = col.to_numpy(dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-6:
            std = 1.0
        return ((arr - mean) / std).astype(np.float32), {"mean": mean, "std": std}

    def _build_condition_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        required_cols = [
            "event_type_code",
            "low_wind_flag",
            "low_irradiance_flag",
            "severity_level",
            "duration_hours",
            "month",
            "extreme_prob",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"cond.csv 缺少必要字段: {missing}")

        n_events = int(df["event_type_code"].max()) + 1
        event_onehot = np.eye(n_events, dtype=np.float32)[df["event_type_code"].astype(int).to_numpy()]

        low_wind = df["low_wind_flag"].astype(np.float32).to_numpy()[:, None]
        low_irr = df["low_irradiance_flag"].astype(np.float32).to_numpy()[:, None]

        # 连续尾部概率引导：extreme_prob 越小越极端，这里同时构造原概率和强度量
        p_ext = df["extreme_prob"].astype(np.float32).clip(1e-6, 1.0).to_numpy()
        p_ext_col = p_ext[:, None]
        extremeness = (1.0 - p_ext)[:, None]  # 越大越极端

        severity = (df["severity_level"].astype(np.float32).to_numpy() / 3.0)[:, None]
        duration_z, duration_meta = self._safe_zscore(df["duration_hours"])
        month = df["month"].astype(np.float32).to_numpy()
        month_sin = np.sin(2 * np.pi * month / 12.0)[:, None]
        month_cos = np.cos(2 * np.pi * month / 12.0)[:, None]

        cond_parts = [event_onehot, low_wind, low_irr, p_ext_col, extremeness]
        feature_layout = {
            "event_onehot": list(range(0, n_events)),
            "low_wind_flag": [n_events],
            "low_irradiance_flag": [n_events + 1],
            "extreme_prob": [n_events + 2],
            "extremeness_strength": [n_events + 3],
        }

        next_idx = n_events + 4
        if self.use_severity_aux:
            cond_parts.append(severity)
            feature_layout["severity_level_scaled"] = [next_idx]
            next_idx += 1

        cond_parts.extend([duration_z[:, None], month_sin, month_cos])
        feature_layout["duration_hours_zscore"] = [next_idx]
        feature_layout["month_sin"] = [next_idx + 1]
        feature_layout["month_cos"] = [next_idx + 2]

        cond = np.concatenate(cond_parts, axis=1).astype(np.float32)
        meta = {
            "n_event_types": n_events,
            "feature_layout": feature_layout,
            "duration_hours": duration_meta,
        }
        return cond, meta

    @staticmethod
    def _build_target_metrics(df: pd.DataFrame) -> np.ndarray:
        required = ["cum_deficit", "netload_ramp_max", "imbalance_duration", "extreme_prob"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"cond.csv 缺少风险约束字段: {missing}")
        metrics = df[["cum_deficit", "netload_ramp_max", "imbalance_duration", "extreme_prob"]].to_numpy(dtype=np.float32)
        return metrics

    def __len__(self) -> int:
        return len(self.X_norm)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X_norm[idx])
        cond = torch.from_numpy(self.cond_features[idx])
        metrics = torch.from_numpy(self.target_metrics[idx])
        return x, cond, metrics


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
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

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, time_dim: int, cond_input_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.cond_input_dim = cond_input_dim
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels, time_dim, cond_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_dim, cond_dim)
        self.mid1 = ResidualBlock1D(base_channels * 2, base_channels * 4, time_dim, cond_dim)
        self.mid2 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_dim, cond_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, time_dim, cond_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels, time_dim, cond_dim)
        self.final = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        if cond is None:
            cond = torch.zeros(x.size(0), self.cond_input_dim, device=x.device)
        cond_emb = self.cond_mlp(cond)
        x0 = self.init_conv(x)
        x1, skip1 = self.down1(x0, time_emb, cond_emb)
        x2, skip2 = self.down2(x1, time_emb, cond_emb)
        h = self.mid1(x2, time_emb, cond_emb)
        h = self.mid2(h, time_emb, cond_emb)
        h = self.up2(h, skip2, time_emb, cond_emb)
        h = self.up1(h, skip1, time_emb, cond_emb)
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

    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        pred_noise_cond = model(x, t, cond)
        if guidance_scale != 1.0:
            pred_noise_uncond = model(x, t, None)
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
        else:
            pred_noise = pred_noise_cond

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


def denormalize_x_np(x_norm: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return x_norm * x_std + x_mean


def denormalize_x_torch(x_norm: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
    return x_norm * x_std + x_mean


def compute_risk_metrics_from_seq(x_denorm: torch.Tensor, duration_temp: float = 12.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    load = x_denorm[:, 0, :]
    wind = x_denorm[:, 1, :]
    solar = x_denorm[:, 2, :]
    net_load = load - wind - solar
    cum_deficit = F.relu(net_load).sum(dim=1)
    ramp = net_load[:, 1:] - net_load[:, :-1]
    zero = torch.zeros(net_load.size(0), 1, device=net_load.device, dtype=net_load.dtype)
    ramp = torch.cat([zero, ramp], dim=1)
    ramp_max = ramp.max(dim=1).values
    imbalance_duration = torch.sigmoid(net_load / duration_temp).sum(dim=1)
    return cum_deficit, ramp_max, imbalance_duration


def weighted_noise_loss(pred_noise: torch.Tensor, noise: torch.Tensor, extreme_prob: torch.Tensor, alpha: float, gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    per_sample_mse = ((pred_noise - noise) ** 2).mean(dim=(1, 2))
    tail_weight = 1.0 + alpha * ((1.0 - extreme_prob.clamp(1e-6, 1.0)) ** gamma)
    loss = (tail_weight * per_sample_mse).mean()
    return loss, tail_weight.detach()


@torch.no_grad()
def sample_sequences(model: nn.Module, scheduler: DiffusionScheduler, cond: torch.Tensor, shape: Tuple[int, int, int], guidance_scale: float, device: torch.device) -> torch.Tensor:
    model.eval()
    x = torch.randn(shape, device=device)
    for step in reversed(range(scheduler.steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x = scheduler.p_sample(model, x, t, cond, guidance_scale)
    return x


def plot_loss_curve(history: dict[str, list[float]], out_path: Path) -> None:
    plt.figure(figsize=(9.2, 5.2))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Conditional DDPM Training Loss (EVT + Tail + Risk)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def format_demo_text(meta: pd.Series, event_name: str, best_val: float, best_mae: Optional[float] = None) -> tuple[str, str]:
    line1 = f"EVT-guided DDPM | sample_id={meta.get('sample_id', 'N/A')} | event={event_name}"
    line2 = (
        f"p_ext={float(meta.get('extreme_prob', 0.0)):.4f}, sev={int(meta.get('severity_level', 0))}, "
        f"low_wind={int(meta.get('low_wind_flag', 0))}, low_irr={int(meta.get('low_irradiance_flag', 0))}, "
        f"duration={int(meta.get('duration_hours', 0))}h, month={int(meta.get('month', 0))}, val_loss={best_val:.4f}"
    )
    if best_mae is not None:
        line2 += f", best_MAE={best_mae:.2f}"
    return line1, line2


def plot_generated_vs_real(real_seq: np.ndarray, gen_seq: np.ndarray, title1: str, title2: str, out_path: Path) -> None:
    names = ["Load", "Wind", "Solar"]
    x = np.arange(real_seq.shape[-1])
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(x, real_seq[i], label="Real", linewidth=2.0)
        ax.plot(x, gen_seq[i], label="Generated", linestyle="--", linewidth=2.0)
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Hour index")
    fig.suptitle(title1, fontsize=16, y=0.985)
    fig.text(0.5, 0.955, title2, ha="center", va="top", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_netload(real_seq: np.ndarray, gen_seq: np.ndarray, title1: str, title2: str, out_path: Path) -> None:
    x = np.arange(real_seq.shape[-1])
    net_real = real_seq[0] - real_seq[1] - real_seq[2]
    net_gen = gen_seq[0] - gen_seq[1] - gen_seq[2]
    plt.figure(figsize=(10.8, 5.2))
    plt.plot(x, net_real, label="Real net load", linewidth=2.0)
    plt.plot(x, net_gen, label="Generated net load", linestyle="--", linewidth=2.0)
    plt.xlabel("Hour index")
    plt.ylabel("Net load")
    plt.suptitle(title1, fontsize=15, y=0.98)
    plt.figtext(0.5, 0.93, title2, ha="center", va="top", fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def event_name_from_code(code: int, mapping: dict[str, int]) -> str:
    inv = {v: k for k, v in mapping.items()}
    return inv.get(code, str(code))


def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SequenceConditionDataset(
        data_dir / "X.npy",
        data_dir / "cond.csv",
        seq_len=cfg.seq_len,
        use_severity_aux=cfg.use_severity_aux,
    )

    n_total = len(dataset)
    n_train = max(1, int(n_total * cfg.train_ratio))
    n_val = max(1, n_total - n_train)
    if n_train + n_val > n_total:
        n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    cond_input_dim = dataset.cond_features.shape[1]
    model = ConditionalUNet1D(cfg.in_channels, cfg.base_channels, cfg.time_emb_dim, cond_input_dim, cfg.cond_emb_dim).to(device)
    ema_model = ConditionalUNet1D(cfg.in_channels, cfg.base_channels, cfg.time_emb_dim, cond_input_dim, cfg.cond_emb_dim).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema = EMA(model, cfg.ema_decay)

    scheduler = DiffusionScheduler(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_mean_t = torch.from_numpy(dataset.X_mean).to(device)
    x_std_t = torch.from_numpy(dataset.X_std).to(device)

    history = {"train_total": [], "train_eps": [], "train_risk": [], "val_total": [], "val_eps": [], "val_risk": []}
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        stats = {"total": 0.0, "eps": 0.0, "risk": 0.0, "n": 0}

        for x, cond, metrics in train_loader:
            x = x.to(device)
            cond = cond.to(device)
            metrics = metrics.to(device)
            extreme_prob = metrics[:, 3].clamp(1e-6, 1.0)

            if cfg.cond_dropout > 0:
                keep_mask = (torch.rand(cond.size(0), device=device) > cfg.cond_dropout).float().unsqueeze(1)
                cond_in = cond * keep_mask
            else:
                cond_in = cond

            t = torch.randint(0, cfg.diffusion_steps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = scheduler.q_sample(x, t, noise)
            pred_noise = model(x_t, t, cond_in)

            eps_loss, _ = weighted_noise_loss(pred_noise, noise, extreme_prob, cfg.tail_alpha, cfg.tail_gamma)

            x0_pred_norm = scheduler.predict_x0(x_t, t, pred_noise).clamp(-4.0, 4.0)
            x0_pred = denormalize_x_torch(x0_pred_norm, x_mean_t, x_std_t)
            cum_pred, ramp_pred, dur_pred = compute_risk_metrics_from_seq(x0_pred, cfg.duration_temp)

            target_cum = metrics[:, 0]
            target_ramp = metrics[:, 1]
            target_dur = metrics[:, 2]

            risk_cum = F.l1_loss(torch.log1p(cum_pred), torch.log1p(target_cum))
            risk_ramp = F.l1_loss(ramp_pred, target_ramp)
            risk_dur = F.l1_loss(dur_pred, target_dur)
            risk_loss = cfg.lambda_cum * risk_cum + cfg.lambda_ramp * risk_ramp + cfg.lambda_dur * risk_dur
            total_loss = eps_loss + cfg.lambda_risk * risk_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            bs = x.size(0)
            stats["total"] += total_loss.item() * bs
            stats["eps"] += eps_loss.item() * bs
            stats["risk"] += risk_loss.item() * bs
            stats["n"] += bs

        history["train_total"].append(stats["total"] / max(1, stats["n"]))
        history["train_eps"].append(stats["eps"] / max(1, stats["n"]))
        history["train_risk"].append(stats["risk"] / max(1, stats["n"]))

        ema.copy_to(ema_model)
        ema_model.eval()
        vstats = {"total": 0.0, "eps": 0.0, "risk": 0.0, "n": 0}
        with torch.no_grad():
            for x, cond, metrics in val_loader:
                x = x.to(device)
                cond = cond.to(device)
                metrics = metrics.to(device)
                extreme_prob = metrics[:, 3].clamp(1e-6, 1.0)

                t = torch.randint(0, cfg.diffusion_steps, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                x_t = scheduler.q_sample(x, t, noise)
                pred_noise = ema_model(x_t, t, cond)

                eps_loss, _ = weighted_noise_loss(pred_noise, noise, extreme_prob, cfg.tail_alpha, cfg.tail_gamma)
                x0_pred_norm = scheduler.predict_x0(x_t, t, pred_noise).clamp(-4.0, 4.0)
                x0_pred = denormalize_x_torch(x0_pred_norm, x_mean_t, x_std_t)
                cum_pred, ramp_pred, dur_pred = compute_risk_metrics_from_seq(x0_pred, cfg.duration_temp)

                target_cum = metrics[:, 0]
                target_ramp = metrics[:, 1]
                target_dur = metrics[:, 2]
                risk_cum = F.l1_loss(torch.log1p(cum_pred), torch.log1p(target_cum))
                risk_ramp = F.l1_loss(ramp_pred, target_ramp)
                risk_dur = F.l1_loss(dur_pred, target_dur)
                risk_loss = cfg.lambda_cum * risk_cum + cfg.lambda_ramp * risk_ramp + cfg.lambda_dur * risk_dur
                total_loss = eps_loss + cfg.lambda_risk * risk_loss

                bs = x.size(0)
                vstats["total"] += total_loss.item() * bs
                vstats["eps"] += eps_loss.item() * bs
                vstats["risk"] += risk_loss.item() * bs
                vstats["n"] += bs

        val_total = vstats["total"] / max(1, vstats["n"])
        history["val_total"].append(val_total)
        history["val_eps"].append(vstats["eps"] / max(1, vstats["n"]))
        history["val_risk"].append(vstats["risk"] / max(1, vstats["n"]))

        if val_total < best_val:
            best_val = val_total
            torch.save(
                {
                    "model_state": ema_model.state_dict(),
                    "config": asdict(cfg),
                    "cond_meta": dataset.cond_meta,
                    "x_mean": dataset.X_mean,
                    "x_std": dataset.X_std,
                },
                out_dir / "best_evt_risk_ddpm.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | train_total={history['train_total'][-1]:.6f} | "
                f"train_eps={history['train_eps'][-1]:.6f} | train_risk={history['train_risk'][-1]:.6f} | "
                f"val_total={history['val_total'][-1]:.6f}"
            )

    plot_loss_curve(history, out_dir / "loss_curve.png")

    # 生成展示样本：优先选更极端且持续更长的验证样本
    cond_df = pd.read_csv(data_dir / "cond.csv")
    mapping_path = data_dir / "event_type_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}

    val_indices = val_ds.indices if hasattr(val_ds, "indices") else list(range(len(dataset)))
    val_meta = cond_df.loc[val_indices].copy()
    for c in ["extreme_prob", "severity_level", "low_wind_flag", "low_irradiance_flag", "duration_hours"]:
        if c not in val_meta.columns:
            val_meta[c] = 0
    val_meta["extremeness"] = 1.0 - val_meta["extreme_prob"].astype(float)
    val_meta = val_meta.sort_values(["extremeness", "severity_level", "duration_hours", "low_wind_flag", "low_irradiance_flag"], ascending=[False, False, False, False, False])
    demo_idx = int(val_meta.index[0])

    demo_cond = torch.from_numpy(dataset.cond_features[demo_idx: demo_idx + 1]).to(device)
    demo_real_norm = dataset.X_norm[demo_idx]
    real_seq = denormalize_x_np(demo_real_norm, dataset.X_mean[0], dataset.X_std[0])

    ema.copy_to(ema_model)
    ema_model.eval()
    candidate_num = max(8, cfg.num_generated_samples)
    multi_cond = demo_cond.repeat(candidate_num, 1)
    multi_gen_norm = sample_sequences(
        ema_model, scheduler, cond=multi_cond,
        shape=(candidate_num, cfg.in_channels, cfg.seq_len),
        guidance_scale=cfg.guidance_scale, device=device
    ).cpu().numpy()
    multi_gen = denormalize_x_np(multi_gen_norm, dataset.X_mean, dataset.X_std)
    maes = np.mean(np.abs(multi_gen - real_seq[None, :, :]), axis=(1, 2))
    best_idx = int(np.argmin(maes))
    gen_seq = multi_gen[best_idx]
    gen_mean_seq = multi_gen.mean(axis=0)

    meta_row = cond_df.loc[demo_idx]
    event_name = event_name_from_code(int(meta_row["event_type_code"]), mapping)
    title1, title2 = format_demo_text(meta_row, event_name, best_val=best_val, best_mae=float(maes[best_idx]))
    plot_generated_vs_real(real_seq, gen_seq, title1, title2, out_dir / "generated_vs_real_curve.png")
    plot_generated_vs_real(real_seq, gen_mean_seq, title1, title2 + " | curve=mean_of_generated", out_dir / "generated_mean_vs_real_curve.png")
    plot_netload(real_seq, gen_seq, title1, title2, out_dir / "netload_curve.png")

    np.save(out_dir / "generated_samples.npy", multi_gen.astype(np.float32))
    summary = {
        "device": str(device),
        "num_samples": len(dataset),
        "train_size": n_train,
        "val_size": n_val,
        "seq_shape": list(dataset.X.shape),
        "condition_dim": int(dataset.cond_features.shape[1]),
        "best_val_loss": float(best_val),
        "demo_index": int(demo_idx),
        "demo_event": event_name,
        "demo_meta": meta_row.to_dict(),
        "guidance_scale": cfg.guidance_scale,
        "candidate_num": candidate_num,
        "best_demo_mae": float(maes[best_idx]),
        "innovation_flags": {
            "continuous_extreme_prob_guidance": True,
            "tail_sensitive_weighted_training": True,
            "joint_imbalance_consistency_constraint": True,
        },
        "loss_weights": {
            "tail_alpha": cfg.tail_alpha,
            "tail_gamma": cfg.tail_gamma,
            "lambda_risk": cfg.lambda_risk,
            "lambda_cum": cfg.lambda_cum,
            "lambda_ramp": cfg.lambda_ramp,
            "lambda_dur": cfg.lambda_dur,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "condition_meta.json").write_text(json.dumps(dataset.cond_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(out_dir / "normalization_stats.npz", x_mean=dataset.X_mean.astype(np.float32), x_std=dataset.X_std.astype(np.float32))

    print("\n训练完成，输出文件位于:")
    print(out_dir.resolve())
    print("- best_evt_risk_ddpm.pt")
    print("- loss_curve.png")
    print("- generated_vs_real_curve.png")
    print("- generated_mean_vs_real_curve.png")
    print("- netload_curve.png")
    print("- generated_samples.npy")
    print("- summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加入 EVT 连续概率引导 + 尾部敏感训练 + 联合失衡约束的条件扩散模型")
    parser.add_argument("--data-dir", type=str, default="mock_dataset_outputs")
    parser.add_argument("--out-dir", type=str, default="mock_full_diffusion_evt_risk_outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-risk", type=float, default=0.08)
    parser.add_argument("--tail-alpha", type=float, default=2.0)
    parser.add_argument("--tail-gamma", type=float, default=1.0)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        diffusion_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        lambda_risk=args.lambda_risk,
        tail_alpha=args.tail_alpha,
        tail_gamma=args.tail_gamma,
    )
    main(cfg)
