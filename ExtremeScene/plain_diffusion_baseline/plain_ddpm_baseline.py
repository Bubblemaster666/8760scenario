"""
Plain DDPM baseline for wind-solar-load extreme scenario generation.

This is a deliberately clean baseline:
- no weather/event labels
- no conditional input
- no EVT extreme_prob / tail_score
- no hierarchical condition encoder
- no tail-sensitive loss
- no joint imbalance risk loss
- no spatio-temporal enhanced branch

Unified data format:
    X.npy: [N, 3, T] or [N, T, 3]
    channel order: load, wind_power, solar_power

Commands:
    train, generate, evaluate, pipeline
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


CHANNELS = ["load", "wind_power", "solar_power"]
EPS = 1e-8


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_x(path: str | Path) -> np.ndarray:
    x = np.load(path)
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got shape {x.shape}")
    # Accept [N, T, 3] and convert to [N, 3, T].
    if x.shape[1] != 3 and x.shape[2] == 3:
        x = np.transpose(x, (0, 2, 1))
    if x.shape[1] != 3:
        raise ValueError(f"X must have 3 channels, got shape {x.shape}")
    return x.astype(np.float32)


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def train_val_test_split(
    x: np.ndarray,
    cond: Optional[pd.DataFrame],
    meta: Optional[pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, object]:
    n = len(x)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }
    out: Dict[str, object] = {"indices": splits}
    for name, ids in splits.items():
        out[f"X_{name}"] = x[ids]
        if cond is not None:
            out[f"cond_{name}"] = cond.iloc[ids].reset_index(drop=True)
        if meta is not None:
            out[f"meta_{name}"] = meta.iloc[ids].reset_index(drop=True)
    return out


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class PlainNoisePredictor(nn.Module):
    """Small unconditional DDPM noise predictor for [B, 3, T]."""

    def __init__(self, channels: int = 3, base_channels: int = 64, n_blocks: int = 6, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_proj = nn.Conv1d(channels, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResBlock1D(base_channels, time_dim) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(1, base_channels)
        self.out_proj = nn.Conv1d(base_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(F.silu(self.out_norm(h)))


@dataclass
class TrainConfig:
    diffusion_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    epochs: int = 100
    batch_size: int = 64
    lr: float = 2e-4
    base_channels: int = 64
    n_blocks: int = 6
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    grad_clip: float = 1.0


class GaussianDiffusion:
    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device):
        self.steps = steps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return a * x0 + b * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (xt - b * eps) / (a + EPS)

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, int, int]) -> torch.Tensor:
        model.eval()
        x = torch.randn(shape, device=self.device)
        for step in reversed(range(self.steps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            eps = model(x, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - alpha_bar) * eps)
            if step > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean
        return x


def normalize_train(x_train: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], dict]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
    x_train_n = (x_train - mean) / std
    arrs_n = [(a - mean) / std for a in arrays]
    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return x_train_n.astype(np.float32), [a.astype(np.float32) for a in arrs_n], stats


def denormalize(x: np.ndarray, stats: dict) -> np.ndarray:
    return x * stats["std"] + stats["mean"]


def infer_dt_hours(meta: Optional[pd.DataFrame], t_len: int) -> float:
    if meta is not None and "window_start_time" in meta.columns and "window_end_time" in meta.columns and len(meta) > 0:
        try:
            s = pd.to_datetime(meta["window_start_time"].iloc[0])
            e = pd.to_datetime(meta["window_end_time"].iloc[0])
            hours = (e - s).total_seconds() / 3600.0
            if hours > 0:
                return hours / max(t_len - 1, 1)
        except Exception:
            pass
    # Common fallback: 24 hourly points or 96 quarter-hourly points over one day.
    if t_len == 96:
        return 0.25
    if t_len == 48:
        return 0.5
    return 1.0


def physical_projection(x: np.ndarray, meta: Optional[pd.DataFrame] = None, night_solar_zero: bool = True) -> np.ndarray:
    x = x.copy()
    x[:, 0, :] = np.maximum(x[:, 0, :], 0.0)
    x[:, 1, :] = np.maximum(x[:, 1, :], 0.0)
    x[:, 2, :] = np.maximum(x[:, 2, :], 0.0)
    if night_solar_zero:
        n, _, t_len = x.shape
        if meta is not None and "window_start_time" in meta.columns and len(meta) >= n:
            dt = infer_dt_hours(meta, t_len)
            for i in range(n):
                try:
                    start = pd.to_datetime(meta["window_start_time"].iloc[i])
                    hours = np.array([(start + pd.Timedelta(hours=j * dt)).hour for j in range(t_len)])
                    night = (hours < 6) | (hours >= 20)
                    x[i, 2, night] = 0.0
                except Exception:
                    pass
    return x


def train_plain_ddpm(data_dir: str | Path, output_dir: str | Path, cfg: TrainConfig, device_str: str) -> None:
    set_seed(cfg.seed)
    data_dir = Path(data_dir)
    out = ensure_dir(output_dir)
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    # If split files already exist, use them. Otherwise split X.npy.
    if (data_dir / "X_train.npy").exists():
        x_train = load_x(data_dir / "X_train.npy")
        x_val = load_x(data_dir / "X_val.npy") if (data_dir / "X_val.npy").exists() else x_train[: max(1, len(x_train)//10)]
        x_test = load_x(data_dir / "X_test.npy") if (data_dir / "X_test.npy").exists() else x_val
        cond_test = pd.read_csv(data_dir / "cond_test.csv") if (data_dir / "cond_test.csv").exists() else None
        meta_test = pd.read_csv(data_dir / "meta_test.csv") if (data_dir / "meta_test.csv").exists() else None
    else:
        x = load_x(data_dir / "X.npy")
        cond = pd.read_csv(data_dir / "cond.csv") if (data_dir / "cond.csv").exists() else None
        meta = pd.read_csv(data_dir / "meta.csv") if (data_dir / "meta.csv").exists() else None
        split = train_val_test_split(x, cond, meta, cfg.train_ratio, cfg.val_ratio, cfg.seed)
        ds = ensure_dir(out / "dataset_split")
        for name in ["train", "val", "test"]:
            np.save(ds / f"X_{name}.npy", split[f"X_{name}"])
            if f"cond_{name}" in split:
                split[f"cond_{name}"].to_csv(ds / f"cond_{name}.csv", index=False)
            if f"meta_{name}" in split:
                split[f"meta_{name}"].to_csv(ds / f"meta_{name}.csv", index=False)
        x_train = split["X_train"]
        x_val = split["X_val"]
        x_test = split["X_test"]
        cond_test = split.get("cond_test")
        meta_test = split.get("meta_test")

    x_train_n, [x_val_n, x_test_n], stats = normalize_train(x_train, x_val, x_test)
    np.savez(out / "normalization_stats.npz", mean=stats["mean"], std=stats["std"])
    np.save(out / "X_test.npy", x_test)
    if cond_test is not None:
        cond_test.to_csv(out / "cond_test.csv", index=False)
    if meta_test is not None:
        meta_test.to_csv(out / "meta_test.csv", index=False)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_n)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_tensor = torch.from_numpy(x_val_n).to(device)

    model = PlainNoisePredictor(base_channels=cfg.base_channels, n_blocks=cfg.n_blocks).to(device)
    diffusion = GaussianDiffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    best_val = float("inf")
    history: List[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            b = xb.size(0)
            t = torch.randint(0, cfg.diffusion_steps, (b,), device=device)
            noise = torch.randn_like(xb)
            xt = diffusion.q_sample(xb, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            if len(val_tensor) > 0:
                vt = torch.randint(0, cfg.diffusion_steps, (val_tensor.size(0),), device=device)
                vnoise = torch.randn_like(val_tensor)
                vxt = diffusion.q_sample(val_tensor, vt, vnoise)
                vpred = model(vxt, vt)
                val_loss = float(F.mse_loss(vpred, vnoise).detach().cpu())
            else:
                val_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_eps_loss": float(np.mean(losses)), "val_eps_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "shape": x_train.shape}, out / "best_model.pt")
        if epoch == 1 or epoch % max(1, cfg.epochs // 10) == 0:
            print(f"epoch={epoch:04d} train={row['train_eps_loss']:.5f} val={val_loss:.5f}")

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    save_json({
        "model_name": "plain_ddpm",
        "baseline_role": "ordinary diffusion baseline",
        "used_condition": False,
        "used_evt": False,
        "used_tail_loss": False,
        "used_risk_loss": False,
        "used_hierarchical_condition": False,
        "used_spatiotemporal_enhancement": False,
        "train_config": asdict(cfg),
        "best_val_eps_loss": best_val,
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
    }, out / "summary.json")


def generate_plain_ddpm(model_dir: str | Path, output_dir: str | Path, n_samples: int, device_str: str, use_test_count: bool) -> None:
    model_dir = Path(model_dir)
    out = ensure_dir(output_dir)
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device)
    cfg = TrainConfig(**ckpt["config"])
    shape_train = ckpt.get("shape", [1, 3, 24])
    c, t_len = int(shape_train[1]), int(shape_train[2])
    if use_test_count and (model_dir / "X_test.npy").exists():
        n_samples = len(load_x(model_dir / "X_test.npy"))

    model = PlainNoisePredictor(base_channels=cfg.base_channels, n_blocks=cfg.n_blocks).to(device)
    model.load_state_dict(ckpt["model"])
    diffusion = GaussianDiffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device)

    stats_npz = np.load(model_dir / "normalization_stats.npz")
    stats = {"mean": stats_npz["mean"], "std": stats_npz["std"]}
    xs = []
    batch = min(256, max(1, n_samples))
    remaining = n_samples
    while remaining > 0:
        b = min(batch, remaining)
        sample_n = diffusion.sample(model, (b, c, t_len)).detach().cpu().numpy().astype(np.float32)
        xs.append(sample_n)
        remaining -= b
    x_gen_n = np.concatenate(xs, axis=0)
    x_gen = denormalize(x_gen_n, stats).astype(np.float32)

    meta = pd.read_csv(model_dir / "meta_test.csv") if (model_dir / "meta_test.csv").exists() else None
    if meta is not None and len(meta) >= len(x_gen):
        meta_used = meta.iloc[:len(x_gen)].reset_index(drop=True)
    else:
        meta_used = None
    x_gen = physical_projection(x_gen, meta_used)
    np.save(out / "generated_samples.npy", x_gen)

    cond = pd.read_csv(model_dir / "cond_test.csv") if (model_dir / "cond_test.csv").exists() else None
    if cond is not None:
        cond_out = cond.iloc[:len(x_gen)].copy().reset_index(drop=True)
    else:
        cond_out = pd.DataFrame({"sample_id": np.arange(len(x_gen))})
    cond_out["generated_id"] = np.arange(len(x_gen))
    cond_out.to_csv(out / "generated_conditions.csv", index=False)

    # Long table output.
    rows = []
    for i in range(len(x_gen)):
        for tt in range(t_len):
            rows.append({
                "sample_id": cond_out["sample_id"].iloc[i] if "sample_id" in cond_out.columns else i,
                "generated_id": i,
                "t": tt,
                "load": float(x_gen[i, 0, tt]),
                "wind_power": float(x_gen[i, 1, tt]),
                "solar_power": float(x_gen[i, 2, tt]),
            })
    pd.DataFrame(rows).to_csv(out / "generated_samples_long.csv", index=False)
    save_json({"n_samples": int(n_samples), "shape": list(x_gen.shape), "model_name": "plain_ddpm"}, out / "generation_summary.json")


def _flatten_var(x: np.ndarray, ch: int) -> np.ndarray:
    return x[:, ch, :].reshape(-1)


def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    lo = np.nanmin([np.nanmin(a), np.nanmin(b)])
    hi = np.nanmax([np.nanmax(a), np.nanmax(b)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    pa, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa.astype(np.float64) + EPS
    pb = pb.astype(np.float64) + EPS
    pa /= pa.sum()
    pb /= pb.sum()
    return float(jensenshannon(pa, pb, base=2.0) ** 2)


def acf_1d(v: np.ndarray, max_lag: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = v - np.nanmean(v)
    denom = np.nansum(v * v) + EPS
    vals = []
    for lag in range(1, max_lag + 1):
        if lag >= len(v):
            vals.append(0.0)
        else:
            vals.append(float(np.nansum(v[:-lag] * v[lag:]) / denom))
    return np.array(vals)


def mean_acf(x: np.ndarray, ch: int, max_lag: int) -> np.ndarray:
    return np.nanmean(np.stack([acf_1d(x[i, ch, :], max_lag) for i in range(len(x))], axis=0), axis=0)


def corr_matrix(x: np.ndarray) -> np.ndarray:
    # concatenate all sample time points as observations of variables [load, wind, solar]
    data = np.transpose(x, (0, 2, 1)).reshape(-1, 3)
    if len(data) < 2:
        return np.eye(3)
    c = np.corrcoef(data.T)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


def compute_risk_metrics(x: np.ndarray, tau: float = 0.0, dt_hours: float = 1.0) -> pd.DataFrame:
    net = x[:, 0, :] - x[:, 1, :] - x[:, 2, :]
    cum = np.maximum(net - tau, 0.0).sum(axis=1) * dt_hours
    ramp = np.diff(net, axis=1).max(axis=1) if net.shape[1] > 1 else np.zeros(len(x))
    dur = (net > tau).sum(axis=1) * dt_hours
    return pd.DataFrame({
        "cum_deficit": cum,
        "netload_ramp_max": ramp,
        "imbalance_duration": dur,
    })


def rel_err(a: float, b: float) -> float:
    return float(abs(b - a) / (abs(a) + EPS))


def evaluate(real_path: str | Path, gen_path: str | Path, output_dir: str | Path, model_name: str, cond_path: Optional[str], acf_max_lag: int, tau: float) -> None:
    out = ensure_dir(output_dir)
    real = load_x(real_path)
    gen = load_x(gen_path)
    n = min(len(real), len(gen))
    real = real[:n]
    gen = gen[:n]
    rows: Dict[str, float | str] = {"model_name": model_name, "n_eval": n}
    for ci, name in enumerate(CHANNELS):
        ra = _flatten_var(real, ci)
        ga = _flatten_var(gen, ci)
        rows[f"wasserstein_{name}"] = float(wasserstein_distance(ra, ga))
        rows[f"js_{name}"] = js_divergence(ra, ga)
        rows[f"acf_mae_{name}"] = float(np.mean(np.abs(mean_acf(real, ci, acf_max_lag) - mean_acf(gen, ci, acf_max_lag))))
    rows["wasserstein_mean"] = float(np.mean([rows[f"wasserstein_{n}"] for n in CHANNELS]))
    rows["js_mean"] = float(np.mean([rows[f"js_{n}"] for n in CHANNELS]))
    rows["acf_mae_mean"] = float(np.mean([rows[f"acf_mae_{n}"] for n in CHANNELS]))
    cm_r = corr_matrix(real)
    cm_g = corr_matrix(gen)
    rows["corr_matrix_error_fro"] = float(np.linalg.norm(cm_r - cm_g, ord="fro"))
    rows["corr_matrix_error_mae"] = float(np.mean(np.abs(cm_r - cm_g)))

    dt = 1.0
    risk_r = compute_risk_metrics(real, tau=tau, dt_hours=dt)
    risk_g = compute_risk_metrics(gen, tau=tau, dt_hours=dt)
    risk_comp_rows = []
    for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
        rr = risk_r[col].to_numpy()
        gg = risk_g[col].to_numpy()
        rows[f"{col}_mae"] = float(np.mean(np.abs(rr - gg)))
        rows[f"{col}_relative_error_mean"] = rel_err(float(np.mean(rr)), float(np.mean(gg)))
        rows[f"{col}_q95_error"] = float(abs(np.quantile(rr, 0.95) - np.quantile(gg, 0.95)))
        rows[f"{col}_q99_error"] = float(abs(np.quantile(rr, 0.99) - np.quantile(gg, 0.99)))
        risk_comp_rows.append({
            "metric": col,
            "real_mean": float(np.mean(rr)),
            "generated_mean": float(np.mean(gg)),
            "mae": rows[f"{col}_mae"],
            "relative_error_mean": rows[f"{col}_relative_error_mean"],
            "real_q95": float(np.quantile(rr, 0.95)),
            "generated_q95": float(np.quantile(gg, 0.95)),
            "real_q99": float(np.quantile(rr, 0.99)),
            "generated_q99": float(np.quantile(gg, 0.99)),
        })
    pd.DataFrame([rows]).to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(risk_comp_rows).to_csv(out / "risk_metrics_real_vs_generated.csv", index=False)
    np.savetxt(out / "corr_matrix_real.csv", cm_r, delimiter=",")
    np.savetxt(out / "corr_matrix_generated.csv", cm_g, delimiter=",")

    # Optional group metrics by event_type/severity_level if cond exists.
    if cond_path and Path(cond_path).exists():
        cond = pd.read_csv(cond_path).iloc[:n].reset_index(drop=True)
        eval_df = pd.concat([cond, risk_r.add_prefix("real_"), risk_g.add_prefix("gen_")], axis=1)
        for group_col, fname in [("event_type", "metrics_by_event_type.csv"), ("severity_level", "metrics_by_severity_level.csv")]:
            if group_col in eval_df.columns:
                group_rows = []
                for gval, sub in eval_df.groupby(group_col):
                    gr = {group_col: gval, "n": len(sub)}
                    for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
                        gr[f"{col}_mae"] = float(np.mean(np.abs(sub[f"real_{col}"] - sub[f"gen_{col}"])))
                    group_rows.append(gr)
                pd.DataFrame(group_rows).to_csv(out / fname, index=False)
    save_json(dict(rows), out / "evaluation_summary.json")


def pipeline(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    cfg = TrainConfig(
        diffusion_steps=args.diffusion_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        n_blocks=args.n_blocks,
        seed=args.seed,
    )
    train_plain_ddpm(args.data_dir, out, cfg, args.device)
    gen_out = ensure_dir(out / "generation")
    generate_plain_ddpm(out, gen_out, args.n_samples, args.device, use_test_count=True)
    real = out / "X_test.npy"
    cond = out / "cond_test.csv" if (out / "cond_test.csv").exists() else None
    evaluate(real, gen_out / "generated_samples.npy", out / "evaluation", "plain_ddpm", str(cond) if cond else None, args.acf_max_lag, args.imbalance_tau)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plain DDPM baseline for wind-solar-load scenario generation")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", required=True)
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--diffusion-steps", type=int, default=200)
    tr.add_argument("--base-channels", type=int, default=64)
    tr.add_argument("--n-blocks", type=int, default=6)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--device", default="auto")

    ge = sub.add_parser("generate")
    ge.add_argument("--model-dir", required=True)
    ge.add_argument("--output-dir", required=True)
    ge.add_argument("--n-samples", type=int, default=100)
    ge.add_argument("--use-test-count", action="store_true")
    ge.add_argument("--device", default="auto")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--real", required=True)
    ev.add_argument("--generated", required=True)
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--model-name", default="plain_ddpm")
    ev.add_argument("--cond", default=None)
    ev.add_argument("--acf-max-lag", type=int, default=12)
    ev.add_argument("--imbalance-tau", type=float, default=0.0)

    pi = sub.add_parser("pipeline")
    pi.add_argument("--data-dir", required=True)
    pi.add_argument("--output-dir", required=True)
    pi.add_argument("--epochs", type=int, default=100)
    pi.add_argument("--batch-size", type=int, default=64)
    pi.add_argument("--lr", type=float, default=2e-4)
    pi.add_argument("--diffusion-steps", type=int, default=200)
    pi.add_argument("--base-channels", type=int, default=64)
    pi.add_argument("--n-blocks", type=int, default=6)
    pi.add_argument("--seed", type=int, default=42)
    pi.add_argument("--n-samples", type=int, default=100)
    pi.add_argument("--acf-max-lag", type=int, default=12)
    pi.add_argument("--imbalance-tau", type=float, default=0.0)
    pi.add_argument("--device", default="auto")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        cfg = TrainConfig(
            diffusion_steps=args.diffusion_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            n_blocks=args.n_blocks,
            seed=args.seed,
        )
        train_plain_ddpm(args.data_dir, args.output_dir, cfg, args.device)
    elif args.cmd == "generate":
        generate_plain_ddpm(args.model_dir, args.output_dir, args.n_samples, args.device, args.use_test_count)
    elif args.cmd == "evaluate":
        evaluate(args.real, args.generated, args.output_dir, args.model_name, args.cond, args.acf_max_lag, args.imbalance_tau)
    elif args.cmd == "pipeline":
        pipeline(args)


if __name__ == "__main__":
    main()
