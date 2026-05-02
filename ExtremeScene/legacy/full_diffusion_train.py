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

BASE_DIR = Path(__file__).resolve().parent


# =========================
# 配置
# =========================
@dataclass
class TrainConfig:
    data_dir: str = str(BASE_DIR / "mock_dataset_outputs")
    out_dir: str = str(BASE_DIR / "mock_full_diffusion_outputs")
    seed: int = 42

    seq_len: int = 24
    in_channels: int = 3
    cond_dropout: float = 0.10

    batch_size: int = 64
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.85

    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    guidance_scale: float = 1.8

    base_channels: int = 64
    time_emb_dim: int = 128
    cond_emb_dim: int = 128

    ema_decay: float = 0.995
    sample_every: int = 25
    num_generated_samples: int = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 数据集与条件处理
# =========================
class SequenceConditionDataset(Dataset):
    def __init__(self, x_path: Path, cond_path: Path, seq_len: int = 24) -> None:
        self.X = np.load(x_path).astype(np.float32)
        self.cond_df = pd.read_csv(cond_path)

        if self.X.ndim != 3:
            raise ValueError(f"X.npy 应该是 (N, C, T)，当前得到 {self.X.shape}")
        if self.X.shape[0] != len(self.cond_df):
            raise ValueError("X.npy 与 cond.csv 样本数不一致")
        if self.X.shape[2] != seq_len:
            raise ValueError(f"序列长度应为 {seq_len}，当前为 {self.X.shape[2]}")

        self.cond_features, self.cond_meta = self._build_condition_matrix(self.cond_df)
        self.X_mean = self.X.mean(axis=(0, 2), keepdims=True)
        self.X_std = self.X.std(axis=(0, 2), keepdims=True) + 1e-6
        self.X_norm = (self.X - self.X_mean) / self.X_std

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
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"cond.csv 缺少必要字段: {missing}")

        n_events = int(df["event_type_code"].max()) + 1
        event_onehot = np.eye(n_events, dtype=np.float32)[df["event_type_code"].astype(int).to_numpy()]

        low_wind = df["low_wind_flag"].astype(np.float32).to_numpy()[:, None]
        low_irr = df["low_irradiance_flag"].astype(np.float32).to_numpy()[:, None]
        severity = (df["severity_level"].astype(np.float32).to_numpy() / 3.0)[:, None]
        duration_z, duration_meta = self._safe_zscore(df["duration_hours"])
        month_sin = np.sin(2 * np.pi * df["month"].astype(np.float32).to_numpy() / 12.0)[:, None]
        month_cos = np.cos(2 * np.pi * df["month"].astype(np.float32).to_numpy() / 12.0)[:, None]

        cond = np.concatenate(
            [
                event_onehot,
                low_wind,
                low_irr,
                severity,
                duration_z[:, None],
                month_sin,
                month_cos,
            ],
            axis=1,
        ).astype(np.float32)

        meta = {
            "n_event_types": n_events,
            "feature_layout": {
                "event_onehot": list(range(0, n_events)),
                "low_wind_flag": [n_events],
                "low_irradiance_flag": [n_events + 1],
                "severity_level_scaled": [n_events + 2],
                "duration_hours_zscore": [n_events + 3],
                "month_sin": [n_events + 4],
                "month_cos": [n_events + 5],
            },
            "duration_hours": duration_meta,
        }
        return cond, meta

    def __len__(self) -> int:
        return len(self.X_norm)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X_norm[idx])
        cond = torch.from_numpy(self.cond_features[idx])
        return x, cond


# =========================
# 模型组件
# =========================
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
            cond = torch.zeros(x.size(0), self.cond_mlp[0].in_features, device=x.device)
        cond_emb = self.cond_mlp(cond)

        x0 = self.init_conv(x)
        x1, skip1 = self.down1(x0, time_emb, cond_emb)
        x2, skip2 = self.down2(x1, time_emb, cond_emb)
        h = self.mid1(x2, time_emb, cond_emb)
        h = self.mid2(h, time_emb, cond_emb)
        h = self.up2(h, skip2, time_emb, cond_emb)
        h = self.up1(h, skip1, time_emb, cond_emb)
        return self.final(h)


# =========================
# DDPM 调度与 EMA
# =========================
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
        return (
            self.sqrt_alpha_bars[t][:, None, None] * x0
            + self.sqrt_one_minus_alpha_bars[t][:, None, None] * noise
        )

    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
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


# =========================
# 训练与采样
# =========================
def denormalize_x(x_norm: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return x_norm * x_std + x_mean


@torch.no_grad()
def sample_sequences(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    cond: torch.Tensor,
    shape: Tuple[int, int, int],
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(shape, device=device)
    for step in reversed(range(scheduler.steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x = scheduler.p_sample(model, x, t, cond, guidance_scale)
    return x


def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Conditional DDPM Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_generated_vs_real(real_seq: np.ndarray, gen_seq: np.ndarray, title: str, out_path: Path) -> None:
    names = ["Load", "Wind", "Solar"]
    x = np.arange(real_seq.shape[-1])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(x, real_seq[i], label="Real")
        ax.plot(x, gen_seq[i], label="Generated", linestyle="--")
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Hour index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)



def plot_netload(real_seq: np.ndarray, gen_seq: np.ndarray, title: str, out_path: Path) -> None:
    x = np.arange(real_seq.shape[-1])
    net_real = real_seq[0] - real_seq[1] - real_seq[2]
    net_gen = gen_seq[0] - gen_seq[1] - gen_seq[2]

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, net_real, label="Real net load")
    plt.plot(x, net_gen, label="Generated net load", linestyle="--")
    plt.xlabel("Hour index")
    plt.ylabel("Net load")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
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

    x_path = data_dir / "X.npy"
    cond_path = data_dir / "cond.csv"
    mapping_path = data_dir / "event_type_mapping.json"

    dataset = SequenceConditionDataset(x_path, cond_path, seq_len=cfg.seq_len)

    n_total = len(dataset)
    n_train = max(1, int(n_total * cfg.train_ratio))
    n_val = n_total - n_train
    if n_val == 0:
        n_train = n_total - 1
        n_val = 1

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    cond_input_dim = dataset.cond_features.shape[1]
    model = ConditionalUNet1D(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        time_dim=cfg.time_emb_dim,
        cond_input_dim=cond_input_dim,
        cond_dim=cfg.cond_emb_dim,
    ).to(device)
    ema_model = ConditionalUNet1D(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        time_dim=cfg.time_emb_dim,
        cond_input_dim=cond_input_dim,
        cond_dim=cfg.cond_emb_dim,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema = EMA(model, cfg.ema_decay)

    scheduler = DiffusionScheduler(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_train = 0.0
        n_train_items = 0

        for x, cond in train_loader:
            x = x.to(device)
            cond = cond.to(device)

            if cfg.cond_dropout > 0:
                keep_mask = (torch.rand(cond.size(0), device=device) > cfg.cond_dropout).float().unsqueeze(1)
                cond_in = cond * keep_mask
            else:
                cond_in = cond

            t = torch.randint(0, cfg.diffusion_steps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = scheduler.q_sample(x, t, noise)
            pred_noise = model(x_t, t, cond_in)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            epoch_train += loss.item() * x.size(0)
            n_train_items += x.size(0)

        train_loss = epoch_train / max(1, n_train_items)
        train_losses.append(train_loss)

        ema.copy_to(ema_model)
        ema_model.eval()
        epoch_val = 0.0
        n_val_items = 0
        with torch.no_grad():
            for x, cond in val_loader:
                x = x.to(device)
                cond = cond.to(device)
                t = torch.randint(0, cfg.diffusion_steps, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                x_t = scheduler.q_sample(x, t, noise)
                pred_noise = ema_model(x_t, t, cond)
                loss = F.mse_loss(pred_noise, noise)
                epoch_val += loss.item() * x.size(0)
                n_val_items += x.size(0)

        val_loss = epoch_val / max(1, n_val_items)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": ema_model.state_dict(),
                    "config": asdict(cfg),
                    "cond_meta": dataset.cond_meta,
                    "x_mean": dataset.X_mean,
                    "x_std": dataset.X_std,
                },
                out_dir / "best_conditional_ddpm.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    plot_loss_curve(train_losses, val_losses, out_dir / "loss_curve.png")

    # 用验证集中的第一个样本做条件生成演示
    ema.copy_to(ema_model)
    ema_model.eval()

    cond_df = pd.read_csv(cond_path)
    val_indices = val_ds.indices if hasattr(val_ds, "indices") else list(range(len(dataset)))
    demo_idx = val_indices[0]

    demo_cond = torch.from_numpy(dataset.cond_features[demo_idx : demo_idx + 1]).to(device)
    demo_real_norm = dataset.X_norm[demo_idx]

    gen_norm = sample_sequences(
        ema_model,
        scheduler,
        cond=demo_cond,
        shape=(1, cfg.in_channels, cfg.seq_len),
        guidance_scale=cfg.guidance_scale,
        device=device,
    ).cpu().numpy()[0]

    real_seq = denormalize_x(demo_real_norm, dataset.X_mean[0], dataset.X_std[0])
    gen_seq = denormalize_x(gen_norm, dataset.X_mean[0], dataset.X_std[0])

    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    event_code = int(cond_df.loc[demo_idx, "event_type_code"])
    event_name = event_name_from_code(event_code, mapping)
    sev = int(cond_df.loc[demo_idx, "severity_level"])
    lw = int(cond_df.loc[demo_idx, "low_wind_flag"])
    li = int(cond_df.loc[demo_idx, "low_irradiance_flag"])

    title = f"Conditional DDPM Sample | event={event_name}, severity={sev}, low_wind={lw}, low_irr={li}"
    plot_generated_vs_real(real_seq, gen_seq, title, out_dir / "generated_vs_real_curve.png")
    plot_netload(real_seq, gen_seq, title, out_dir / "netload_curve.png")

    # 额外生成若干条样本，便于汇报时展示“同条件多样性”
    multi_cond = demo_cond.repeat(cfg.num_generated_samples, 1)
    multi_gen_norm = sample_sequences(
        ema_model,
        scheduler,
        cond=multi_cond,
        shape=(cfg.num_generated_samples, cfg.in_channels, cfg.seq_len),
        guidance_scale=cfg.guidance_scale,
        device=device,
    ).cpu().numpy()
    multi_gen = denormalize_x(multi_gen_norm, dataset.X_mean, dataset.X_std)
    np.save(out_dir / "generated_samples.npy", multi_gen.astype(np.float32))

    # 保存一个简单摘要
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
        "guidance_scale": cfg.guidance_scale,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "normalization_stats.npz",
        x_mean=dataset.X_mean.astype(np.float32),
        x_std=dataset.X_std.astype(np.float32),
    )
    (out_dir / "condition_meta.json").write_text(json.dumps(dataset.cond_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n训练完成，输出文件位于:")
    print(out_dir.resolve())
    print("- best_conditional_ddpm.pt")
    print("- loss_curve.png")
    print("- generated_vs_real_curve.png")
    print("- netload_curve.png")
    print("- generated_samples.npy")
    print("- summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整条件扩散模型（mock 数据版）")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "mock_dataset_outputs"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "mock_full_diffusion_outputs"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=100, help="扩散步数")
    parser.add_argument("--guidance", type=float, default=1.8, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42)
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
    )
    main(cfg)
