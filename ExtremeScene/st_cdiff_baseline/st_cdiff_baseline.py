from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# Avoid very slow CPU oversubscription on small time-series workloads.
torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Configuration
# ============================================================

@dataclass
class STCDiffConfig:
    # model / diffusion
    seq_len: int = 24
    channels: int = 3
    base_channels: int = 64
    cond_dim: int = 0
    cond_emb_dim: int = 128
    time_emb_dim: int = 128
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # training
    epochs: int = 80
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-5
    device: str = "cpu"
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # generation / physical projection
    n_per_condition: int = 1
    solar_zero_before_hour: int = 6
    solar_zero_after_hour: int = 20
    enforce_nonnegative: bool = True
    enforce_night_solar_zero: bool = True

    # evaluation
    acf_max_lag: int = 12
    imbalance_tau: float = 0.0
    delta_t: float = 1.0

    # condition fields; deliberately exclude EVT continuous probability and risk metrics
    use_event_type: bool = True
    use_month: bool = True
    use_resource_flags: bool = True
    use_duration: bool = True
    use_severity_level: bool = True


# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_nct(x: np.ndarray) -> np.ndarray:
    """Ensure array is [N, 3, T]. Accepts [N, T, 3] as well."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got shape {x.shape}")
    if x.shape[1] == 3:
        return x
    if x.shape[2] == 3:
        return np.transpose(x, (0, 2, 1)).astype(np.float32)
    raise ValueError(f"Cannot infer channel dimension from shape {x.shape}. Expected [N,3,T] or [N,T,3].")


def to_ntc(x: np.ndarray) -> np.ndarray:
    x = ensure_nct(x)
    return np.transpose(x, (0, 2, 1))


class ChannelStandardScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x_nct: np.ndarray) -> "ChannelStandardScaler":
        x = ensure_nct(x_nct)
        self.mean_ = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = x.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = np.maximum(self.std_, 1e-6).astype(np.float32)
        return self

    def transform(self, x_nct: np.ndarray) -> np.ndarray:
        x = ensure_nct(x_nct)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x_nct: np.ndarray) -> np.ndarray:
        x = ensure_nct(x_nct)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        return (x * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean_.reshape(-1).tolist(), "std": self.std_.reshape(-1).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStandardScaler":
        obj = cls()
        obj.mean_ = np.asarray(d["mean"], dtype=np.float32).reshape(1, 3, 1)
        obj.std_ = np.asarray(d["std"], dtype=np.float32).reshape(1, 3, 1)
        return obj


# ============================================================
# Condition builder
# ============================================================

class ConditionBuilder:
    """Build a flat condition vector for ST-CDiff.

    This baseline uses condition gating, but deliberately does not use
    extreme_prob, tail_score, cum_deficit, ramp, or duration risk losses.
    """

    def __init__(self, cfg: STCDiffConfig):
        self.cfg = cfg
        self.event_values: list[int] = []
        self.severity_values: list[int] = []
        self.duration_mean: float = 0.0
        self.duration_std: float = 1.0
        self.fields: list[str] = []

    def fit(self, cond: pd.DataFrame) -> "ConditionBuilder":
        if self.cfg.use_event_type:
            if "event_type_code" in cond.columns:
                vals = sorted(pd.Series(cond["event_type_code"]).fillna(0).astype(int).unique().tolist())
            elif "event_type" in cond.columns:
                vals = sorted(pd.Series(cond["event_type"]).astype("category").cat.codes.unique().tolist())
            else:
                vals = [0]
            self.event_values = vals
        if self.cfg.use_severity_level:
            if "severity_level" in cond.columns:
                self.severity_values = sorted(pd.Series(cond["severity_level"]).fillna(0).astype(int).unique().tolist())
            else:
                self.severity_values = [0]
        if self.cfg.use_duration:
            if "duration_hours" in cond.columns:
                v = pd.Series(cond["duration_hours"]).astype(float).fillna(0).to_numpy()
            else:
                v = np.zeros(len(cond), dtype=np.float32)
            self.duration_mean = float(np.mean(v))
            self.duration_std = float(max(np.std(v), 1e-6))
        self._build_fields()
        return self

    def _build_fields(self) -> None:
        fields: list[str] = []
        if self.cfg.use_event_type:
            fields += [f"event_type_onehot_{v}" for v in self.event_values]
        if self.cfg.use_month:
            fields += ["month_sin", "month_cos"]
        if self.cfg.use_resource_flags:
            fields += ["low_wind_flag", "low_irradiance_flag"]
        if self.cfg.use_duration:
            fields += ["duration_hours_zscore"]
        if self.cfg.use_severity_level:
            fields += [f"severity_onehot_{v}" for v in self.severity_values]
        self.fields = fields

    def transform(self, cond: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        n = len(cond)
        if self.cfg.use_event_type:
            if "event_type_code" in cond.columns:
                raw = pd.Series(cond["event_type_code"]).fillna(0).astype(int).to_numpy()
            elif "event_type" in cond.columns:
                raw = pd.Series(cond["event_type"]).astype("category").cat.codes.to_numpy()
            else:
                raw = np.zeros(n, dtype=int)
            one = np.zeros((n, len(self.event_values)), dtype=np.float32)
            idx_map = {v: i for i, v in enumerate(self.event_values)}
            for i, v in enumerate(raw):
                if int(v) in idx_map:
                    one[i, idx_map[int(v)]] = 1.0
            parts.append(one)
        if self.cfg.use_month:
            if "month" in cond.columns:
                m = pd.Series(cond["month"]).fillna(1).astype(float).to_numpy()
            else:
                m = np.ones(n, dtype=np.float32)
            parts.append(np.sin(2 * np.pi * (m - 1) / 12.0).reshape(-1, 1).astype(np.float32))
            parts.append(np.cos(2 * np.pi * (m - 1) / 12.0).reshape(-1, 1).astype(np.float32))
        if self.cfg.use_resource_flags:
            lw = pd.Series(cond["low_wind_flag"]).fillna(0).astype(float).to_numpy() if "low_wind_flag" in cond.columns else np.zeros(n)
            li = pd.Series(cond["low_irradiance_flag"]).fillna(0).astype(float).to_numpy() if "low_irradiance_flag" in cond.columns else np.zeros(n)
            parts.append(lw.reshape(-1, 1).astype(np.float32))
            parts.append(li.reshape(-1, 1).astype(np.float32))
        if self.cfg.use_duration:
            dur = pd.Series(cond["duration_hours"]).fillna(self.duration_mean).astype(float).to_numpy() if "duration_hours" in cond.columns else np.full(n, self.duration_mean)
            z = (dur - self.duration_mean) / self.duration_std
            parts.append(z.reshape(-1, 1).astype(np.float32))
        if self.cfg.use_severity_level:
            raw = pd.Series(cond["severity_level"]).fillna(0).astype(int).to_numpy() if "severity_level" in cond.columns else np.zeros(n, dtype=int)
            one = np.zeros((n, len(self.severity_values)), dtype=np.float32)
            idx_map = {v: i for i, v in enumerate(self.severity_values)}
            for i, v in enumerate(raw):
                if int(v) in idx_map:
                    one[i, idx_map[int(v)]] = 1.0
            parts.append(one)
        if not parts:
            return np.zeros((n, 0), dtype=np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "event_values": self.event_values,
            "severity_values": self.severity_values,
            "duration_mean": self.duration_mean,
            "duration_std": self.duration_std,
            "fields": self.fields,
            "note": "ST-CDiff condition vector excludes extreme_prob, tail_score and risk metrics by design.",
        }

    @classmethod
    def from_dict(cls, d: dict, cfg: STCDiffConfig) -> "ConditionBuilder":
        obj = cls(cfg)
        obj.event_values = list(map(int, d.get("event_values", [])))
        obj.severity_values = list(map(int, d.get("severity_values", [])))
        obj.duration_mean = float(d.get("duration_mean", 0.0))
        obj.duration_std = float(d.get("duration_std", 1.0))
        obj.fields = list(d.get("fields", []))
        return obj


# ============================================================
# Dataset / split
# ============================================================

class ScenarioDataset(Dataset):
    def __init__(self, x: np.ndarray, cond_vec: np.ndarray):
        self.x = torch.tensor(ensure_nct(x), dtype=torch.float32)
        self.cond = torch.tensor(cond_vec, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.cond[idx]


def load_raw_dataset(data_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    x_path = data_dir / "X.npy"
    cond_path = data_dir / "cond.csv"
    meta_path = data_dir / "meta.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"Cannot find {x_path}")
    if not cond_path.exists():
        raise FileNotFoundError(f"Cannot find {cond_path}")
    x = ensure_nct(np.load(x_path))
    cond = pd.read_csv(cond_path)
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame({"sample_id": cond.get("sample_id", np.arange(len(cond)))})
    if len(cond) != x.shape[0]:
        raise ValueError(f"cond.csv rows {len(cond)} != X samples {x.shape[0]}")
    return x, cond, meta


def stratified_split(cond: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(cond)
    keys = []
    for _, row in cond.iterrows():
        et = int(row["event_type_code"]) if "event_type_code" in cond.columns and pd.notna(row.get("event_type_code")) else 0
        sv = int(row["severity_level"]) if "severity_level" in cond.columns and pd.notna(row.get("severity_level")) else 0
        keys.append(f"{et}_{sv}")
    keys = np.asarray(keys)
    train_idx, val_idx, test_idx = [], [], []
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        rng.shuffle(idx)
        if len(idx) < 5:
            # avoid tiny classes being emptied; assign mostly to train
            n_test = 1 if len(idx) >= 3 else 0
            n_val = 1 if len(idx) >= 4 else 0
        else:
            n_test = max(1, int(round(len(idx) * test_ratio)))
            n_val = max(1, int(round(len(idx) * val_ratio)))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    if train_idx.size == 0:
        raise ValueError("Train split is empty. Please provide more samples.")
    return train_idx, val_idx, test_idx


def save_split_dataset(out_dir: Path, x: np.ndarray, cond: pd.DataFrame, meta: pd.DataFrame, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> None:
    ds_dir = out_dir / "dataset_split"
    ds_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in [("train", tr), ("val", va), ("test", te)]:
        np.save(ds_dir / f"X_{name}.npy", x[idx])
        cond.iloc[idx].reset_index(drop=True).to_csv(ds_dir / f"cond_{name}.csv", index=False, encoding="utf-8-sig")
        if len(meta) == len(cond):
            meta.iloc[idx].reset_index(drop=True).to_csv(ds_dir / f"meta_{name}.csv", index=False, encoding="utf-8-sig")


# ============================================================
# Diffusion schedule
# ============================================================

class DiffusionSchedule:
    def __init__(self, cfg: STCDiffConfig, device: torch.device):
        self.cfg = cfg
        beta = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.sqrt_ab = torch.sqrt(alpha_bar)
        self.sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.sqrt_ab[t].view(-1, 1, 1)
        om = self.sqrt_1m_ab[t].view(-1, 1, 1)
        return ab * x0 + om * noise


# ============================================================
# Model: ST-CDiff
# ============================================================

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ChannelRelationModule(nn.Module):
    """Light channel-mixing module over [load, wind, solar]."""
    def __init__(self, channels: int = 3):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(channels, channels))
        nn.init.normal_(self.raw, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        a = torch.tanh(self.raw) * 0.25
        y = torch.einsum("ij,bjt->bit", a, x)
        return x + y


class TemporalAttention(nn.Module):
    """Very small temporal attention/gating. Not a full transformer."""
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h), dim=-1)  # [B,1,T]
        context = torch.sum(h * w, dim=-1, keepdim=True)
        return h + self.proj(context).expand_as(h) * 0.15


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
        )
        # start close to identity
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, h: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        gb = self.net(cond_emb)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = 1.0 + gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * h + beta


class TemporalResBlock(nn.Module):
    def __init__(self, channels: int, cond_emb_dim: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=8 if channels >= 8 else 1, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=8 if channels >= 8 else 1, num_channels=channels)
        self.film = FiLM(cond_emb_dim, channels)
        self.attn = TemporalAttention(channels)

    def forward(self, h: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        r = h
        h = self.conv1(F.silu(self.norm1(h)))
        h = self.film(h, cond_emb)
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.attn(h)
        return r + h


class STCDiffDenoiser(nn.Module):
    """Joint spatio-temporal enhanced conditional denoiser.

    Enhancements over vanilla conditional diffusion:
      1) ChannelRelationModule for load-wind-solar coupling.
      2) Dilated residual Conv1D blocks for temporal continuity.
      3) FiLM condition gating for adaptive condition fusion.

    It intentionally excludes EVT probability, tail-score, risk loss,
    and hierarchical background-process-risk condition structure.
    """
    def __init__(self, cfg: STCDiffConfig):
        super().__init__()
        self.cfg = cfg
        self.channel_relation = ChannelRelationModule(cfg.channels)
        self.in_conv = nn.Conv1d(cfg.channels, cfg.base_channels, kernel_size=3, padding=1)
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.time_emb_dim, cfg.cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_emb_dim, cfg.cond_emb_dim),
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(max(cfg.cond_dim, 1), cfg.cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_emb_dim, cfg.cond_emb_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(cfg.cond_emb_dim * 2, cfg.cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_emb_dim, cfg.cond_emb_dim),
        )
        dilations = [1, 2, 4, 8, 4, 2, 1]
        self.blocks = nn.ModuleList([TemporalResBlock(cfg.base_channels, cfg.cond_emb_dim, d) for d in dilations])
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=8 if cfg.base_channels >= 8 else 1, num_channels=cfg.base_channels),
            nn.SiLU(),
            nn.Conv1d(cfg.base_channels, cfg.channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.channel_relation(x)
        h = self.in_conv(x)
        te = sinusoidal_timestep_embedding(t, self.cfg.time_emb_dim)
        te = self.t_mlp(te)
        if cond.shape[1] == 0:
            cond = torch.zeros((cond.shape[0], 1), dtype=cond.dtype, device=cond.device)
        ce = self.c_mlp(cond)
        emb = self.fuse(torch.cat([te, ce], dim=1))
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out(h)


# ============================================================
# Training / generation
# ============================================================

def save_model_bundle(out_dir: Path, model: nn.Module, cfg: STCDiffConfig, scaler: ChannelStandardScaler, cond_builder: ConditionBuilder) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "st_cdiff_model.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "condition_meta.json").write_text(json.dumps(cond_builder.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_bundle(model_dir: str | Path, device: str = "cpu") -> tuple[STCDiffDenoiser, STCDiffConfig, ChannelStandardScaler, ConditionBuilder]:
    model_dir = Path(model_dir)
    cfg = STCDiffConfig(**json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
    cfg.device = device or cfg.device
    scaler = ChannelStandardScaler.from_dict(json.loads((model_dir / "scaler.json").read_text(encoding="utf-8")))
    cond_builder = ConditionBuilder.from_dict(json.loads((model_dir / "condition_meta.json").read_text(encoding="utf-8")), cfg)
    model = STCDiffDenoiser(cfg).to(torch.device(cfg.device))
    state = torch.load(model_dir / "st_cdiff_model.pt", map_location=torch.device(cfg.device))
    model.load_state_dict(state)
    model.eval()
    return model, cfg, scaler, cond_builder


def train_model(data_dir: str | Path, output_dir: str | Path, cfg: STCDiffConfig) -> dict:
    seed_all(cfg.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, cond, meta = load_raw_dataset(data_dir)
    cfg.seq_len = int(x.shape[2])

    tr, va, te = stratified_split(cond, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    save_split_dataset(output_dir, x, cond, meta, tr, va, te)

    scaler = ChannelStandardScaler().fit(x[tr])
    x_tr = scaler.transform(x[tr])
    x_va = scaler.transform(x[va]) if va.size else scaler.transform(x[tr])[:0]

    cond_builder = ConditionBuilder(cfg).fit(cond.iloc[tr].reset_index(drop=True))
    cond_tr = cond_builder.transform(cond.iloc[tr].reset_index(drop=True))
    cond_va = cond_builder.transform(cond.iloc[va].reset_index(drop=True)) if va.size else np.zeros((0, cond_tr.shape[1]), dtype=np.float32)
    cfg.cond_dim = int(cond_tr.shape[1])

    device = torch.device(cfg.device)
    model = STCDiffDenoiser(cfg).to(device)
    schedule = DiffusionSchedule(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(ScenarioDataset(x_tr, cond_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ScenarioDataset(x_va, cond_va), batch_size=cfg.batch_size, shuffle=False, drop_last=False) if va.size else None

    logs = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for xb, cb in train_loader:
            xb = xb.to(device)
            cb = cb.to(device)
            b = xb.size(0)
            t = torch.randint(0, cfg.diffusion_steps, (b,), device=device)
            eps = torch.randn_like(xb)
            xt = schedule.q_sample(xb, t, eps)
            pred = model(xt, t, cb)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n_batches += 1
        tr_loss = loss_sum / max(n_batches, 1)

        val_loss = np.nan
        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            vs = 0.0
            vn = 0
            with torch.no_grad():
                for xb, cb in val_loader:
                    xb = xb.to(device); cb = cb.to(device)
                    b = xb.size(0)
                    t = torch.randint(0, cfg.diffusion_steps, (b,), device=device)
                    eps = torch.randn_like(xb)
                    xt = schedule.q_sample(xb, t, eps)
                    pred = model(xt, t, cb)
                    vs += float(F.mse_loss(pred, eps).item())
                    vn += 1
            val_loss = vs / max(vn, 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logs.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})
        if epoch % max(1, cfg.epochs // 10) == 0 or epoch == 1:
            print(f"[train] epoch={epoch:04d} train_loss={tr_loss:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    save_model_bundle(output_dir, model, cfg, scaler, cond_builder)
    pd.DataFrame(logs).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 4))
    plt.plot([r["epoch"] for r in logs], [r["train_loss"] for r in logs], label="train")
    if not all(np.isnan(r["val_loss"]) for r in logs):
        plt.plot([r["epoch"] for r in logs], [r["val_loss"] for r in logs], label="val")
    plt.xlabel("Epoch"); plt.ylabel("DDPM noise MSE"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    summary = {
        "model_name": "ST-CDiff",
        "role": "improved diffusion baseline with temporal/channel/condition-gating enhancement",
        "important_exclusions": ["extreme_prob", "tail_score", "tail-sensitive loss", "risk consistency loss", "hierarchical background-process-risk condition"],
        "n_total": int(x.shape[0]),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "seq_len": int(cfg.seq_len),
        "condition_fields": cond_builder.fields,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def sample_diffusion(model: STCDiffDenoiser, cfg: STCDiffConfig, cond_vec: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    schedule = DiffusionSchedule(cfg, device)
    n = cond_vec.shape[0]
    x = torch.randn(n, cfg.channels, cfg.seq_len, device=device)
    cond = torch.tensor(cond_vec, dtype=torch.float32, device=device)
    for step in reversed(range(cfg.diffusion_steps)):
        t = torch.full((n,), step, dtype=torch.long, device=device)
        eps = model(x, t, cond)
        beta_t = schedule.beta[step]
        alpha_t = schedule.alpha[step]
        ab_t = schedule.alpha_bar[step]
        coef = beta_t / torch.sqrt(1.0 - ab_t)
        mean = (x - coef * eps) / torch.sqrt(alpha_t)
        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    return x.detach().cpu().numpy().astype(np.float32)


def physical_projection(samples_nct: np.ndarray, cfg: STCDiffConfig) -> np.ndarray:
    x = ensure_nct(samples_nct).copy()
    if cfg.enforce_nonnegative:
        x = np.clip(x, 0.0, None)
    if cfg.enforce_night_solar_zero and x.shape[1] >= 3:
        T = x.shape[2]
        # Works for hourly T=24 and quarter-hour T=96 by mapping index to hour of day.
        hour = np.floor(np.arange(T) * 24.0 / max(T, 1)).astype(int) % 24
        night = (hour < cfg.solar_zero_before_hour) | (hour >= cfg.solar_zero_after_hour)
        x[:, 2, night] = 0.0
    return x.astype(np.float32)


def generated_long_dataframe(samples_nct: np.ndarray, cond: pd.DataFrame) -> pd.DataFrame:
    x = ensure_nct(samples_nct)
    rows = []
    for i in range(x.shape[0]):
        sid = cond.iloc[i].get("sample_id", i) if i < len(cond) else i
        for t in range(x.shape[2]):
            rows.append({
                "generated_id": i,
                "sample_id": sid,
                "t": t,
                "load": float(x[i, 0, t]),
                "wind_power": float(x[i, 1, t]),
                "solar_power": float(x[i, 2, t]),
                "event_type": cond.iloc[i].get("event_type", "") if i < len(cond) else "",
                "event_type_code": cond.iloc[i].get("event_type_code", np.nan) if i < len(cond) else np.nan,
                "month": cond.iloc[i].get("month", np.nan) if i < len(cond) else np.nan,
                "severity_level": cond.iloc[i].get("severity_level", np.nan) if i < len(cond) else np.nan,
            })
    return pd.DataFrame(rows)


def generate_samples(model_dir: str | Path, cond_csv: str | Path, output_dir: str | Path, n_per_condition: int = 1, device: str = "cpu") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, scaler, cond_builder = load_model_bundle(model_dir, device=device)
    device_t = torch.device(cfg.device)
    cond = pd.read_csv(cond_csv)
    if n_per_condition > 1:
        cond_gen = pd.concat([cond] * n_per_condition, ignore_index=True)
        if "sample_id" in cond_gen.columns:
            cond_gen["source_sample_id"] = cond_gen["sample_id"]
            cond_gen["sample_id"] = [f"{s}_g{i}" for i, s in enumerate(cond_gen["source_sample_id"].astype(str))]
    else:
        cond_gen = cond.copy()
    cond_vec = cond_builder.transform(cond_gen)
    x_norm = sample_diffusion(model, cfg, cond_vec, device_t)
    x = scaler.inverse_transform(x_norm)
    x = physical_projection(x, cfg)
    np.save(output_dir / "generated_samples.npy", x)
    cond_gen.to_csv(output_dir / "generated_conditions.csv", index=False, encoding="utf-8-sig")
    generated_long_dataframe(x, cond_gen).to_csv(output_dir / "generated_samples_long.csv", index=False, encoding="utf-8-sig")
    summary = {"n_generated": int(x.shape[0]), "shape": list(x.shape), "n_per_condition": int(n_per_condition)}
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ============================================================
# Evaluation metrics
# ============================================================

def _js_divergence(x: np.ndarray, y: np.ndarray, bins: int = 80) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo = min(float(np.nanmin(x)), float(np.nanmin(y)))
    hi = max(float(np.nanmax(x)), float(np.nanmax(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    px, edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    py, _ = np.histogram(y, bins=edges, density=False)
    px = px.astype(np.float64) + 1e-12
    py = py.astype(np.float64) + 1e-12
    px /= px.sum(); py /= py.sum()
    return float(jensenshannon(px, py, base=np.e) ** 2)


def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)
    denom = float(np.dot(x, x)) + 1e-12
    out = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            out.append(0.0)
        else:
            out.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(out, dtype=np.float64)


def _mean_acf(samples_nct: np.ndarray, channel: int, max_lag: int) -> np.ndarray:
    x = ensure_nct(samples_nct)
    acfs = [_acf_1d(x[i, channel], max_lag) for i in range(x.shape[0])]
    return np.mean(np.stack(acfs, axis=0), axis=0)


def corr_matrix(samples_nct: np.ndarray) -> np.ndarray:
    x = ensure_nct(samples_nct)
    flat = np.transpose(x, (0, 2, 1)).reshape(-1, 3)
    if flat.shape[0] < 3:
        return np.eye(3)
    c = np.corrcoef(flat.T)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def compute_risk_metrics(samples_nct: np.ndarray, tau: float = 0.0, delta_t: float = 1.0) -> pd.DataFrame:
    x = ensure_nct(samples_nct)
    load = x[:, 0, :]
    wind = x[:, 1, :]
    solar = x[:, 2, :]
    net = load - wind - solar
    cum_deficit = np.maximum(net - tau, 0.0).sum(axis=1) * delta_t
    ramp = np.diff(net, axis=1)
    ramp_max = ramp.max(axis=1) if ramp.shape[1] > 0 else np.zeros(x.shape[0])
    duration = (net > tau).sum(axis=1) * delta_t
    return pd.DataFrame({
        "cum_deficit": cum_deficit,
        "netload_ramp_max": ramp_max,
        "imbalance_duration": duration,
    })


def metric_distribution_error(real: np.ndarray, gen: np.ndarray) -> dict:
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    q_grid = np.linspace(0.01, 0.99, 99)
    rq = np.quantile(real, q_grid)
    gq = np.quantile(gen, q_grid)
    return {
        "mean_real": float(np.mean(real)),
        "mean_generated": float(np.mean(gen)),
        "relative_error": float(abs(np.mean(gen) - np.mean(real)) / (abs(np.mean(real)) + 1e-12)),
        "quantile_mae": float(np.mean(np.abs(rq - gq))),
        "q95_error": float(abs(np.quantile(gen, 0.95) - np.quantile(real, 0.95))),
        "q99_error": float(abs(np.quantile(gen, 0.99) - np.quantile(real, 0.99))),
    }


def severity_match_rate(gen_risk: pd.DataFrame, cond: pd.DataFrame) -> dict:
    if "severity_level" not in cond.columns or len(cond) != len(gen_risk):
        return {"severity_match_rate_strict": np.nan, "severity_match_rate_adjacent": np.nan}
    sev_true = pd.Series(cond["severity_level"]).fillna(0).astype(int).to_numpy()
    # Recompute generated severity by generated cum_deficit quartiles: simple comparable diagnostic.
    cd = gen_risk["cum_deficit"].to_numpy()
    if np.unique(cd).size < 4:
        pred = np.zeros_like(sev_true)
    else:
        q = np.quantile(cd, [0.5, 0.75, 0.9])
        pred = np.digitize(cd, q, right=False).astype(int)
    return {
        "severity_match_rate_strict": float(np.mean(pred == sev_true[: len(pred)])),
        "severity_match_rate_adjacent": float(np.mean(np.abs(pred - sev_true[: len(pred)]) <= 1)),
    }


def evaluate_generation(real_path: str | Path, generated_path: str | Path, cond_path: str | Path, output_dir: str | Path, model_name: str = "ST-CDiff", acf_max_lag: int = 12, tau: float = 0.0, delta_t: float = 1.0) -> dict:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    real = ensure_nct(np.load(real_path))
    gen = ensure_nct(np.load(generated_path))
    cond = pd.read_csv(cond_path)
    if len(cond) != gen.shape[0]:
        # Use repeated/trimmed conditions for grouping when n_per_condition differs.
        reps = int(math.ceil(gen.shape[0] / max(len(cond), 1)))
        cond = pd.concat([cond] * reps, ignore_index=True).iloc[: gen.shape[0]].reset_index(drop=True)

    names = ["load", "wind", "solar"]
    summary: dict[str, Any] = {"model": model_name, "n_real": int(real.shape[0]), "n_generated": int(gen.shape[0]), "seq_len": int(real.shape[2])}

    w_vals = []
    js_vals = []
    acf_vals = []
    for c, name in enumerate(names):
        rv = real[:, c, :].reshape(-1)
        gv = gen[:, c, :].reshape(-1)
        w = float(wasserstein_distance(rv, gv))
        js = _js_divergence(rv, gv)
        max_lag = int(min(acf_max_lag, real.shape[2] - 1, gen.shape[2] - 1))
        acf_err = float(np.mean(np.abs(_mean_acf(real, c, max_lag) - _mean_acf(gen, c, max_lag)))) if max_lag >= 1 else 0.0
        summary[f"wasserstein_{name}"] = w
        summary[f"js_{name}"] = js
        summary[f"acf_mae_{name}"] = acf_err
        w_vals.append(w); js_vals.append(js); acf_vals.append(acf_err)
    summary["wasserstein_mean"] = float(np.mean(w_vals))
    summary["js_mean"] = float(np.mean(js_vals))
    summary["acf_mae_mean"] = float(np.mean(acf_vals))

    corr_r = corr_matrix(real)
    corr_g = corr_matrix(gen)
    summary["corr_matrix_error_mae"] = float(np.mean(np.abs(corr_r - corr_g)))
    summary["corr_matrix_error_fro"] = float(np.linalg.norm(corr_r - corr_g, ord="fro"))

    real_risk = compute_risk_metrics(real, tau=tau, delta_t=delta_t)
    gen_risk = compute_risk_metrics(gen, tau=tau, delta_t=delta_t)
    rows = []
    for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
        e = metric_distribution_error(real_risk[col].to_numpy(), gen_risk[col].to_numpy())
        summary[f"{col}_relative_error"] = e["relative_error"]
        summary[f"{col}_quantile_mae"] = e["quantile_mae"]
        summary[f"{col}_q95_error"] = e["q95_error"]
        summary[f"{col}_q99_error"] = e["q99_error"]
        rows.append({"metric": col, **e})
    summary.update(severity_match_rate(gen_risk, cond))

    pd.DataFrame([summary]).to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output_dir / "risk_metrics_real_vs_generated.csv", index=False, encoding="utf-8-sig")

    # Grouped metrics: use generated cond grouping and compare to all real or matching real cond if possible.
    for group_col, out_name in [("event_type_code", "metrics_by_event_type.csv"), ("severity_level", "metrics_by_severity_level.csv")]:
        if group_col in cond.columns:
            grow = []
            for val, sub in cond.groupby(group_col):
                idx = sub.index.to_numpy()
                if len(idx) < 2:
                    continue
                gm = gen[idx]
                rr = real
                grow.append({
                    group_col: val,
                    "n_generated": int(len(idx)),
                    "wasserstein_mean": float(np.mean([wasserstein_distance(rr[:, c, :].reshape(-1), gm[:, c, :].reshape(-1)) for c in range(3)])),
                    "js_mean": float(np.mean([_js_divergence(rr[:, c, :].reshape(-1), gm[:, c, :].reshape(-1)) for c in range(3)])),
                    "corr_matrix_error_mae": float(np.mean(np.abs(corr_matrix(rr) - corr_matrix(gm)))),
                })
            pd.DataFrame(grow).to_csv(output_dir / out_name, index=False, encoding="utf-8-sig")

    # Figures
    pd.DataFrame(corr_r, index=names, columns=names).to_csv(output_dir / "corr_matrix_real.csv", encoding="utf-8-sig")
    pd.DataFrame(corr_g, index=names, columns=names).to_csv(output_dir / "corr_matrix_generated.csv", encoding="utf-8-sig")

    plt.figure(figsize=(8, 4))
    for c, name in enumerate(names):
        max_lag = int(min(acf_max_lag, real.shape[2] - 1, gen.shape[2] - 1))
        if max_lag >= 1:
            plt.plot(np.arange(1, max_lag + 1), _mean_acf(real, c, max_lag), label=f"real {name}")
            plt.plot(np.arange(1, max_lag + 1), _mean_acf(gen, c, max_lag), linestyle="--", label=f"gen {name}")
    plt.xlabel("Lag"); plt.ylabel("ACF"); plt.grid(alpha=0.25); plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(fig_dir / "acf_comparison.png", dpi=160); plt.close()

    i = 0
    plt.figure(figsize=(10, 6))
    t = np.arange(real.shape[2])
    for c, name in enumerate(names):
        ax = plt.subplot(3, 1, c + 1)
        ax.plot(t, real[i % real.shape[0], c], label="real", linewidth=1.5)
        ax.plot(t, gen[i % gen.shape[0], c], label="generated", linewidth=1.5, linestyle="--")
        ax.set_ylabel(name); ax.grid(alpha=0.25)
        if c == 0: ax.legend()
    plt.xlabel("time step"); plt.tight_layout(); plt.savefig(fig_dir / "typical_generated_curve.png", dpi=160); plt.close()

    net_r = real[i % real.shape[0], 0] - real[i % real.shape[0], 1] - real[i % real.shape[0], 2]
    net_g = gen[i % gen.shape[0], 0] - gen[i % gen.shape[0], 1] - gen[i % gen.shape[0], 2]
    plt.figure(figsize=(8, 4)); plt.plot(t, net_r, label="real net load"); plt.plot(t, net_g, label="generated net load", linestyle="--"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / "netload_curve.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 4))
    data = [real_risk["cum_deficit"], gen_risk["cum_deficit"], real_risk["netload_ramp_max"], gen_risk["netload_ramp_max"], real_risk["imbalance_duration"], gen_risk["imbalance_duration"]]
    plt.boxplot(data, labels=["real\ncum", "gen\ncum", "real\nramp", "gen\nramp", "real\ndur", "gen\ndur"], showfliers=False)
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(fig_dir / "risk_metric_boxplot.png", dpi=160); plt.close()

    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ============================================================
# Demo/mock data
# ============================================================

def create_mock_dataset(out_dir: str | Path, n: int = 120, seq_len: int = 24, seed: int = 42) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, seq_len, endpoint=False)
    x = np.zeros((n, 3, seq_len), dtype=np.float32)
    rows = []
    for i in range(n):
        event = int(rng.integers(0, 4))
        severity = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
        month = int(rng.integers(1, 13))
        low_wind = int(event in [0, 3] or rng.random() < 0.25)
        low_irr = int(event in [1, 2] or rng.random() < 0.25)
        dur = float(rng.integers(12, 72))
        sev_gain = 1.0 + 0.14 * severity
        load = (1.0 + 0.18 * np.sin(t - 0.8) + 0.06 * rng.normal(size=seq_len)) * sev_gain
        wind = 0.45 + 0.13 * np.sin(t + rng.uniform(-1, 1)) + 0.08 * rng.normal(size=seq_len)
        solar = 0.58 * np.maximum(0.0, np.sin(t - np.pi / 2)) + 0.04 * rng.normal(size=seq_len)
        if low_wind:
            wind *= 0.55 - 0.06 * severity
        if low_irr:
            solar *= 0.50 - 0.05 * severity
        if event == 3:  # high temperature: load high, wind low
            load += 0.15 + 0.05 * severity
            wind *= 0.82
        if event == 1:  # snow/blizzard: solar sharply reduced
            solar *= 0.45
            load += 0.08 * severity
        solar[np.floor(np.arange(seq_len) * 24.0 / seq_len).astype(int) < 6] = 0.0
        solar[np.floor(np.arange(seq_len) * 24.0 / seq_len).astype(int) >= 20] = 0.0
        x[i, 0] = np.clip(load, 0, None)
        x[i, 1] = np.clip(wind, 0, None)
        x[i, 2] = np.clip(solar, 0, None)
        net = x[i, 0] - x[i, 1] - x[i, 2]
        rows.append({
            "sample_id": f"mock_{i:04d}",
            "event_type_code": event,
            "event_type": ["cold_wave", "blizzard", "sandstorm", "heatwave"][event],
            "month": month,
            "low_wind_flag": low_wind,
            "low_irradiance_flag": low_irr,
            "duration_hours": dur,
            "severity_level": severity,
            "cum_deficit": float(np.maximum(net, 0).sum()),
            "netload_ramp_max": float(np.diff(net).max()),
            "imbalance_duration": float((net > 0).sum()),
        })
    np.save(out_dir / "X.npy", x)
    pd.DataFrame(rows).to_csv(out_dir / "cond.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "sample_id": [r["sample_id"] for r in rows],
        "window_start_time": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "window_end_time": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
    }).to_csv(out_dir / "meta.csv", index=False, encoding="utf-8-sig")
    return {"data_dir": str(out_dir), "shape": list(x.shape)}


def run_pipeline(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    if args.use_mock:
        data_dir = out_dir / "mock_dataset"
        create_mock_dataset(data_dir, n=args.mock_n, seq_len=args.seq_len, seed=args.seed)
    else:
        data_dir = Path(args.data_dir)
    cfg = STCDiffConfig(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        device=args.device,
        seed=args.seed,
        acf_max_lag=args.acf_max_lag,
        imbalance_tau=args.imbalance_tau,
        n_per_condition=args.n_per_condition,
    )
    model_dir = out_dir / "model"
    train_summary = train_model(data_dir, model_dir, cfg)
    cond_test = model_dir / "dataset_split" / "cond_test.csv"
    x_test = model_dir / "dataset_split" / "X_test.npy"
    gen_dir = model_dir / "generation"
    generate_samples(model_dir, cond_test, gen_dir, n_per_condition=args.n_per_condition, device=args.device)
    eval_dir = model_dir / "evaluation"
    evaluate_generation(x_test, gen_dir / "generated_samples.npy", gen_dir / "generated_conditions.csv", eval_dir, model_name="ST-CDiff", acf_max_lag=args.acf_max_lag, tau=args.imbalance_tau)
    summary = {"output_dir": str(out_dir), "train_summary": train_summary}
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ST-CDiff baseline: spatio-temporal enhanced conditional diffusion for wind-solar-load scenarios.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_train = argparse.ArgumentParser(add_help=False)
    common_train.add_argument("--epochs", type=int, default=80)
    common_train.add_argument("--batch-size", type=int, default=64)
    common_train.add_argument("--lr", type=float, default=2e-4)
    common_train.add_argument("--diffusion-steps", type=int, default=100)
    common_train.add_argument("--base-channels", type=int, default=64)
    common_train.add_argument("--device", type=str, default="cpu")
    common_train.add_argument("--seed", type=int, default=42)
    common_train.add_argument("--seq-len", type=int, default=24)
    common_train.add_argument("--acf-max-lag", type=int, default=12)
    common_train.add_argument("--imbalance-tau", type=float, default=0.0)

    sp = sub.add_parser("make-mock", parents=[common_train])
    sp.add_argument("--output-dir", type=str, required=True)
    sp.add_argument("--mock-n", type=int, default=120)

    sp = sub.add_parser("train", parents=[common_train])
    sp.add_argument("--data-dir", type=str, required=True)
    sp.add_argument("--output-dir", type=str, required=True)

    sp = sub.add_parser("generate")
    sp.add_argument("--model-dir", type=str, required=True)
    sp.add_argument("--cond-csv", type=str, required=True)
    sp.add_argument("--output-dir", type=str, required=True)
    sp.add_argument("--n-per-condition", type=int, default=1)
    sp.add_argument("--device", type=str, default="cpu")

    sp = sub.add_parser("evaluate")
    sp.add_argument("--real", type=str, required=True)
    sp.add_argument("--generated", type=str, required=True)
    sp.add_argument("--cond", type=str, required=True)
    sp.add_argument("--output-dir", type=str, required=True)
    sp.add_argument("--model-name", type=str, default="ST-CDiff")
    sp.add_argument("--acf-max-lag", type=int, default=12)
    sp.add_argument("--imbalance-tau", type=float, default=0.0)
    sp.add_argument("--delta-t", type=float, default=1.0)

    sp = sub.add_parser("pipeline", parents=[common_train])
    sp.add_argument("--data-dir", type=str, default="")
    sp.add_argument("--output-dir", type=str, required=True)
    sp.add_argument("--use-mock", action="store_true")
    sp.add_argument("--mock-n", type=int, default=120)
    sp.add_argument("--n-per-condition", type=int, default=1)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "make-mock":
        print(json.dumps(create_mock_dataset(args.output_dir, n=args.mock_n, seq_len=args.seq_len, seed=args.seed), ensure_ascii=False, indent=2))
    elif args.cmd == "train":
        cfg = STCDiffConfig(seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, diffusion_steps=args.diffusion_steps, base_channels=args.base_channels, device=args.device, seed=args.seed, acf_max_lag=args.acf_max_lag, imbalance_tau=args.imbalance_tau)
        print(json.dumps(train_model(args.data_dir, args.output_dir, cfg), ensure_ascii=False, indent=2))
    elif args.cmd == "generate":
        print(json.dumps(generate_samples(args.model_dir, args.cond_csv, args.output_dir, n_per_condition=args.n_per_condition, device=args.device), ensure_ascii=False, indent=2))
    elif args.cmd == "evaluate":
        evaluate_generation(args.real, args.generated, args.cond, args.output_dir, model_name=args.model_name, acf_max_lag=args.acf_max_lag, tau=args.imbalance_tau, delta_t=args.delta_t)
    elif args.cmd == "pipeline":
        print(json.dumps(run_pipeline(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
