from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Config
# ============================================================

@dataclass
class EnhancedGANConfig:
    data_dir: str = "outputs/dataset"
    output_dir: str = "outputs/enhanced_gan_extreme"
    random_seed: int = 42
    device: str = "cpu"

    # data
    seq_len: int = 24
    test_ratio: float = 0.2
    val_ratio: float = 0.15
    use_extreme_prob_condition: bool = False
    use_month_condition: bool = True
    use_severity_condition: bool = True
    use_resource_flags: bool = True
    use_duration_condition: bool = True

    # model
    z_dim: int = 64
    cond_emb_dim: int = 32
    hidden_dim: int = 128
    n_res_blocks: int = 3
    batch_size: int = 64
    epochs: int = 100
    n_critic: int = 4
    gp_lambda: float = 10.0
    drift_lambda: float = 1e-3

    # variable learning rate; D generally faster than G
    lr_g: float = 1.0e-4
    lr_d: float = 2.0e-4
    lr_decay_r: float = 0.15

    # physical correction / penalties
    lambda_range: float = 0.05
    lambda_ramp: float = 0.05
    ramp_quantile: float = 0.995
    range_quantile: float = 0.998
    solar_zero_before_hour: int = 6
    solar_zero_after_hour: int = 20

    # evaluation
    acf_max_lag: int = 12
    js_bins: int = 80
    imbalance_tau_mode: str = "quantile"  # quantile / fixed
    imbalance_tau_quantile: float = 0.75
    imbalance_tau_fixed: float = 0.0

    # outputs
    generate_per_test_condition: int = 1
    make_figures: bool = True


# ============================================================
# Utils
# ============================================================

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    """Return [N, 3, T]. Accepts [N, 3, T] or [N, T, 3]."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got shape={x.shape}")
    if x.shape[1] == 3:
        return x
    if x.shape[2] == 3:
        return np.transpose(x, (0, 2, 1)).astype(np.float32)
    raise ValueError(f"Cannot infer channel axis from shape={x.shape}; expected [N,3,T] or [N,T,3]")


class ChannelMinMaxScaler:
    def __init__(self, q_low: float = 0.002, q_high: float = 0.998):
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.lo: Optional[np.ndarray] = None
        self.hi: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "ChannelMinMaxScaler":
        x = ensure_channel_first(x)
        self.lo = np.quantile(x, self.q_low, axis=(0, 2)).astype(np.float32)
        self.hi = np.quantile(x, self.q_high, axis=(0, 2)).astype(np.float32)
        self.hi = np.maximum(self.hi, self.lo + 1e-6).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = ensure_channel_first(x)
        lo = self.lo[None, :, None]
        hi = self.hi[None, :, None]
        y = 2.0 * (x - lo) / (hi - lo) - 1.0
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = ensure_channel_first(y)
        lo = self.lo[None, :, None]
        hi = self.hi[None, :, None]
        x = 0.5 * (y + 1.0) * (hi - lo) + lo
        return np.clip(x, 0.0, None).astype(np.float32)

    def to_json_dict(self) -> dict:
        return {"q_low": self.q_low, "q_high": self.q_high, "lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_json_dict(cls, d: dict) -> "ChannelMinMaxScaler":
        obj = cls(d.get("q_low", 0.002), d.get("q_high", 0.998))
        obj.lo = np.asarray(d["lo"], dtype=np.float32)
        obj.hi = np.asarray(d["hi"], dtype=np.float32)
        return obj


# ============================================================
# Condition builder
# ============================================================

class ConditionBuilder:
    """Build paper-style classification labels plus optional numeric context.

    Enhanced GAN uses classification label c to guide extreme scenario generation.
    This builder maps cond.csv into a compact condition vector while keeping the
    proposed-method-specific EVT/risk losses out of the GAN baseline.
    """

    def __init__(self, cfg: EnhancedGANConfig):
        self.cfg = cfg
        self.meta: Dict[str, object] = {}
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.cat_cardinalities: Dict[str, int] = {}
        self.num_mean: Optional[np.ndarray] = None
        self.num_std: Optional[np.ndarray] = None
        self.total_dim: int = 0

    @staticmethod
    def _coerce_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.full(len(df), default), index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    def fit(self, cond: pd.DataFrame) -> "ConditionBuilder":
        cond = cond.copy()
        cat_cols = []
        if "event_type_code" in cond.columns:
            cat_cols.append("event_type_code")
        elif "event_type" in cond.columns:
            # create deterministic codes from string labels
            values = sorted(cond["event_type"].astype(str).unique().tolist())
            mapping = {v: i for i, v in enumerate(values)}
            cond["event_type_code"] = cond["event_type"].astype(str).map(mapping).astype(int)
            self.meta["event_type_mapping"] = mapping
            cat_cols.append("event_type_code")
        else:
            cond["event_type_code"] = 0
            cat_cols.append("event_type_code")

        if self.cfg.use_month_condition:
            cond["month_code"] = self._coerce_col(cond, "month", 1).astype(int).clip(1, 12) - 1
            cat_cols.append("month_code")

        if self.cfg.use_severity_condition:
            cond["severity_code"] = self._coerce_col(cond, "severity_level", 0).astype(int).clip(0, 10)
            cat_cols.append("severity_code")

        numeric_cols = []
        if self.cfg.use_resource_flags:
            for c in ["low_wind_flag", "low_irradiance_flag"]:
                cond[c] = self._coerce_col(cond, c, 0.0).astype(float)
                numeric_cols.append(c)
        if self.cfg.use_duration_condition:
            cond["duration_hours"] = self._coerce_col(cond, "duration_hours", float(self.cfg.seq_len)).astype(float)
            numeric_cols.append("duration_hours")
        if self.cfg.use_extreme_prob_condition:
            cond["extreme_prob"] = self._coerce_col(cond, "extreme_prob", 0.1).clip(1e-6, 1.0).astype(float)
            numeric_cols.append("extreme_prob")

        self.cat_cols = cat_cols
        self.numeric_cols = numeric_cols
        self.cat_cardinalities = {}
        for c in cat_cols:
            vals = self._coerce_col(cond, c, 0).astype(int).to_numpy()
            self.cat_cardinalities[c] = int(max(vals.max() + 1, 1))

        if numeric_cols:
            mat = cond[numeric_cols].astype(float).to_numpy(dtype=np.float32)
            self.num_mean = mat.mean(axis=0, keepdims=True).astype(np.float32)
            self.num_std = np.maximum(mat.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
        else:
            self.num_mean = np.zeros((1, 0), dtype=np.float32)
            self.num_std = np.ones((1, 0), dtype=np.float32)

        self.total_dim = sum(self.cat_cardinalities[c] for c in cat_cols) + len(numeric_cols)
        self.meta.update(
            {
                "cat_cols": self.cat_cols,
                "numeric_cols": self.numeric_cols,
                "cat_cardinalities": self.cat_cardinalities,
                "num_mean": self.num_mean.reshape(-1).tolist(),
                "num_std": self.num_std.reshape(-1).tolist(),
                "total_dim": self.total_dim,
                "use_extreme_prob_condition": self.cfg.use_extreme_prob_condition,
                "note": "Enhanced GAN baseline uses classification-style labels; no EVT tail loss or risk consistency loss is used.",
            }
        )
        return self

    def transform(self, cond: pd.DataFrame) -> np.ndarray:
        cond = cond.copy()
        if "event_type_code" not in cond.columns:
            if "event_type_mapping" in self.meta and "event_type" in cond.columns:
                mapping = self.meta["event_type_mapping"]
                cond["event_type_code"] = cond["event_type"].astype(str).map(mapping).fillna(0).astype(int)
            else:
                cond["event_type_code"] = 0
        if "month_code" in self.cat_cols:
            cond["month_code"] = self._coerce_col(cond, "month", 1).astype(int).clip(1, 12) - 1
        if "severity_code" in self.cat_cols:
            cond["severity_code"] = self._coerce_col(cond, "severity_level", 0).astype(int).clip(0, 10)

        parts = []
        for c in self.cat_cols:
            vals = self._coerce_col(cond, c, 0).astype(int).clip(0, self.cat_cardinalities[c] - 1).to_numpy()
            oh = np.zeros((len(cond), self.cat_cardinalities[c]), dtype=np.float32)
            oh[np.arange(len(cond)), vals] = 1.0
            parts.append(oh)
        if self.numeric_cols:
            for c in self.numeric_cols:
                if c not in cond.columns:
                    cond[c] = 0.0
            mat = cond[self.numeric_cols].astype(float).to_numpy(dtype=np.float32)
            mat = (mat - self.num_mean) / self.num_std
            mat = np.clip(mat, -5.0, 5.0)
            parts.append(mat.astype(np.float32))
        if not parts:
            return np.zeros((len(cond), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, cfg: EnhancedGANConfig) -> "ConditionBuilder":
        obj = cls(cfg)
        meta = json.loads(path.read_text(encoding="utf-8"))
        obj.meta = meta
        obj.cat_cols = list(meta["cat_cols"])
        obj.numeric_cols = list(meta["numeric_cols"])
        obj.cat_cardinalities = {k: int(v) for k, v in meta["cat_cardinalities"].items()}
        obj.num_mean = np.asarray(meta.get("num_mean", []), dtype=np.float32).reshape(1, -1)
        obj.num_std = np.asarray(meta.get("num_std", []), dtype=np.float32).reshape(1, -1)
        obj.total_dim = int(meta["total_dim"])
        return obj


# ============================================================
# Dataset loading
# ============================================================

def read_dataset(data_dir: str, cfg: EnhancedGANConfig) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.DataFrame], dict]:
    d = Path(data_dir)
    info = {"data_dir": str(d.resolve())}
    if (d / "X_train.npy").exists() and (d / "X_test.npy").exists():
        x_train = ensure_channel_first(np.load(d / "X_train.npy"))
        x_test = ensure_channel_first(np.load(d / "X_test.npy"))
        cond_train = pd.read_csv(d / "cond_train.csv")
        cond_test = pd.read_csv(d / "cond_test.csv")
        meta_train = pd.read_csv(d / "meta_train.csv") if (d / "meta_train.csv").exists() else None
        meta_test = pd.read_csv(d / "meta_test.csv") if (d / "meta_test.csv").exists() else None
        x = np.concatenate([x_train, x_test], axis=0)
        cond = pd.concat([cond_train, cond_test], ignore_index=True)
        meta = pd.concat([meta_train, meta_test], ignore_index=True) if meta_train is not None and meta_test is not None else None
        info.update({"mode": "pre_split", "n_train": int(len(x_train)), "n_test": int(len(x_test))})
        return x, cond, meta, info

    if not (d / "X.npy").exists():
        raise FileNotFoundError(f"Cannot find X.npy or X_train/X_test in {d}")
    x = ensure_channel_first(np.load(d / "X.npy"))
    if not (d / "cond.csv").exists():
        raise FileNotFoundError(f"Cannot find cond.csv in {d}")
    cond = pd.read_csv(d / "cond.csv")
    meta = pd.read_csv(d / "meta.csv") if (d / "meta.csv").exists() else None
    if len(cond) != x.shape[0]:
        raise ValueError(f"cond.csv rows {len(cond)} != X samples {x.shape[0]}")
    info.update({"mode": "single_split", "n_total": int(x.shape[0])})
    return x, cond, meta, info


def split_indices(cond: pd.DataFrame, cfg: EnhancedGANConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    n = len(cond)
    # stratify by event_type_code if possible
    if "event_type_code" in cond.columns:
        labels = pd.to_numeric(cond["event_type_code"], errors="coerce").fillna(0).astype(int).to_numpy()
    elif "event_type" in cond.columns:
        labels = pd.Categorical(cond["event_type"].astype(str)).codes
    else:
        labels = np.zeros(n, dtype=int)

    train_idx, val_idx, test_idx = [], [], []
    for lab in np.unique(labels):
        ids = np.where(labels == lab)[0]
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * cfg.test_ratio))) if len(ids) >= 5 else max(1, len(ids) // 5)
        n_val = max(1, int(round(len(ids) * cfg.val_ratio))) if len(ids) - n_test >= 5 else max(0, (len(ids) - n_test) // 5)
        test_idx.append(ids[:n_test])
        val_idx.append(ids[n_test:n_test + n_val])
        train_idx.append(ids[n_test + n_val:])
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
    if len(train_idx) == 0:
        # fall back for tiny demo data
        ids = np.arange(n)
        rng.shuffle(ids)
        n_test = max(1, int(round(n * cfg.test_ratio)))
        n_val = max(1, int(round(n * cfg.val_ratio))) if n - n_test > 2 else 0
        test_idx = ids[:n_test]
        val_idx = ids[n_test:n_test+n_val]
        train_idx = ids[n_test+n_val:]
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)


class ExtremeGANDataset(Dataset):
    def __init__(self, x_norm: np.ndarray, cond_vec: np.ndarray):
        self.x = torch.tensor(x_norm, dtype=torch.float32)
        self.c = torch.tensor(cond_vec, dtype=torch.float32)
    def __len__(self) -> int:
        return self.x.shape[0]
    def __getitem__(self, idx: int):
        return self.x[idx], self.c[idx]


# ============================================================
# Model
# ============================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, channels)
        self.norm1 = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.norm2 = nn.GroupNorm(8 if channels >= 8 else 1, channels)
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.norm1(x), 0.2))
        h = h + self.cond(c).unsqueeze(-1)
        h = self.conv2(F.leaky_relu(self.norm2(h), 0.2))
        return x + h


class EnhancedGenerator(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, seq_len: int, hidden_dim: int = 128, n_res_blocks: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, hidden_dim * seq_len),
            nn.LeakyReLU(0.2),
        )
        self.blocks = nn.ModuleList([ResidualBlock1D(hidden_dim, hidden_dim) for _ in range(n_res_blocks)])
        self.out = nn.Sequential(
            nn.GroupNorm(8 if hidden_dim >= 8 else 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond_proj(cond)
        h = self.fc(torch.cat([z, c], dim=1)).view(z.size(0), -1, self.seq_len)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


class EnhancedCritic(nn.Module):
    def __init__(self, cond_dim: int, seq_len: int, hidden_dim: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.cond_line = nn.Sequential(nn.Linear(cond_dim, seq_len), nn.Tanh())
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cl = self.cond_line(cond).unsqueeze(1)
        h = torch.cat([x, cl], dim=1)
        return self.head(self.conv(h)).squeeze(1)


# ============================================================
# Physical constraints
# ============================================================

def make_day_mask(seq_len: int, solar_zero_before_hour: int = 6, solar_zero_after_hour: int = 20, start_hour: int = 0) -> np.ndarray:
    hours = (np.arange(seq_len) + int(start_hour)) % 24
    return ((hours >= solar_zero_before_hour) & (hours < solar_zero_after_hour)).astype(np.float32)


def physical_project_np(x: np.ndarray, cfg: EnhancedGANConfig, channel_limits: Optional[np.ndarray] = None, ramp_limits: Optional[np.ndarray] = None) -> np.ndarray:
    x = ensure_channel_first(x).copy()
    x = np.clip(x, 0.0, None)
    if channel_limits is not None:
        x = np.minimum(x, channel_limits[None, :, None])
    # night PV zero
    day_mask = make_day_mask(x.shape[2], cfg.solar_zero_before_hour, cfg.solar_zero_after_hour)
    x[:, 2, :] *= day_mask[None, :]
    # ramp correction: clip first differences channel-wise
    if ramp_limits is not None:
        for n in range(x.shape[0]):
            for c in range(3):
                lim = float(max(ramp_limits[c], 1e-6))
                for t in range(1, x.shape[2]):
                    delta = x[n, c, t] - x[n, c, t - 1]
                    if delta > lim:
                        x[n, c, t] = x[n, c, t - 1] + lim
                    elif delta < -lim:
                        x[n, c, t] = x[n, c, t - 1] - lim
        x[:, 2, :] *= day_mask[None, :]
    return np.clip(x, 0.0, None).astype(np.float32)


def physical_penalty_torch(x_denorm: torch.Tensor, channel_limits: torch.Tensor, ramp_limits: torch.Tensor, day_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x_denorm: [B,3,T]
    below = F.relu(-x_denorm).mean()
    above = F.relu(x_denorm - channel_limits[None, :, None]).mean()
    solar_night = (x_denorm[:, 2, :] * (1.0 - day_mask[None, :])).pow(2).mean()
    range_pen = below + above + 0.5 * solar_night
    ramp = torch.abs(x_denorm[:, :, 1:] - x_denorm[:, :, :-1])
    ramp_pen = F.relu(ramp - ramp_limits[None, :, None]).mean()
    return range_pen, ramp_pen


def denorm_torch(x_norm: torch.Tensor, scaler: ChannelMinMaxScaler, device: torch.device) -> torch.Tensor:
    lo = torch.tensor(scaler.lo, dtype=x_norm.dtype, device=device)[None, :, None]
    hi = torch.tensor(scaler.hi, dtype=x_norm.dtype, device=device)[None, :, None]
    return 0.5 * (x_norm + 1.0) * (hi - lo) + lo


# ============================================================
# Training
# ============================================================

def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, device=device)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)
    out = critic(interp, cond)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(b, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def lr_lambda(epoch: int, r: float) -> float:
    return float((epoch + 1) ** (-r))


def train_enhanced_gan(cfg: EnhancedGANConfig) -> dict:
    seed_all(cfg.random_seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all, cond_all, meta_all, data_info = read_dataset(cfg.data_dir, cfg)
    cfg.seq_len = int(x_all.shape[2])

    # split
    if data_info.get("mode") == "pre_split":
        n_train = int(data_info["n_train"])
        n_test = int(data_info["n_test"])
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_train + n_test)
        # split train into train/val
        rng = np.random.default_rng(cfg.random_seed)
        rng.shuffle(train_idx)
        n_val = max(1, int(round(len(train_idx) * cfg.val_ratio))) if len(train_idx) > 5 else 0
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]
    else:
        train_idx, val_idx, test_idx = split_indices(cond_all, cfg)

    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    cond_train, cond_val, cond_test = cond_all.iloc[train_idx].reset_index(drop=True), cond_all.iloc[val_idx].reset_index(drop=True), cond_all.iloc[test_idx].reset_index(drop=True)

    scaler = ChannelMinMaxScaler(q_high=cfg.range_quantile).fit(x_train)
    x_train_n = scaler.transform(x_train)
    x_val_n = scaler.transform(x_val) if len(x_val) else np.empty((0, 3, cfg.seq_len), dtype=np.float32)

    cb = ConditionBuilder(cfg).fit(cond_train)
    c_train = cb.transform(cond_train)
    c_val = cb.transform(cond_val) if len(cond_val) else np.empty((0, cb.total_dim), dtype=np.float32)
    c_test = cb.transform(cond_test)

    channel_limits = np.quantile(x_train, cfg.range_quantile, axis=(0, 2)).astype(np.float32)
    ramp_limits = np.quantile(np.abs(np.diff(x_train, axis=2)), cfg.ramp_quantile, axis=(0, 2)).astype(np.float32)
    ramp_limits = np.maximum(ramp_limits, 1e-6)

    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    gen = EnhancedGenerator(cfg.z_dim, cb.total_dim, cfg.seq_len, cfg.hidden_dim, cfg.n_res_blocks).to(device)
    dis = EnhancedCritic(cb.total_dim, cfg.seq_len, cfg.hidden_dim).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(dis.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lambda e: lr_lambda(e, cfg.lr_decay_r))
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lambda e: lr_lambda(e, cfg.lr_decay_r))

    ds = ExtremeGANDataset(x_train_n, c_train)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    ch_lim_t = torch.tensor(channel_limits, dtype=torch.float32, device=device)
    rp_lim_t = torch.tensor(ramp_limits, dtype=torch.float32, device=device)
    day_mask_t = torch.tensor(make_day_mask(cfg.seq_len, cfg.solar_zero_before_hour, cfg.solar_zero_after_hour), dtype=torch.float32, device=device)

    logs = []
    best_score = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        gen.train(); dis.train()
        d_loss_sum = 0.0; g_loss_sum = 0.0; gp_sum = 0.0; range_sum = 0.0; ramp_sum = 0.0; nb = 0
        for real_x, cond in loader:
            real_x = real_x.to(device)
            cond = cond.to(device)
            b = real_x.size(0)

            for _ in range(cfg.n_critic):
                z = torch.randn(b, cfg.z_dim, device=device)
                fake = gen(z, cond).detach()
                d_real = dis(real_x, cond).mean()
                d_fake = dis(fake, cond).mean()
                gp = gradient_penalty(dis, real_x, fake, cond, device)
                drift = (d_real.pow(2) + d_fake.pow(2)) * cfg.drift_lambda
                loss_d = d_fake - d_real + cfg.gp_lambda * gp + drift
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            z = torch.randn(b, cfg.z_dim, device=device)
            fake = gen(z, cond)
            adv = -dis(fake, cond).mean()
            fake_den = denorm_torch(fake, scaler, device)
            range_pen, ramp_pen = physical_penalty_torch(fake_den, ch_lim_t, rp_lim_t, day_mask_t)
            loss_g = adv + cfg.lambda_range * range_pen + cfg.lambda_ramp * ramp_pen
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_loss_sum += float(loss_d.detach().cpu())
            g_loss_sum += float(loss_g.detach().cpu())
            gp_sum += float(gp.detach().cpu())
            range_sum += float(range_pen.detach().cpu())
            ramp_sum += float(ramp_pen.detach().cpu())
            nb += 1

        sch_g.step(); sch_d.step()

        # validation proxy: Wasserstein in normalized space
        val_score = np.nan
        if len(x_val_n) > 0:
            gen.eval()
            with torch.no_grad():
                cv = torch.tensor(c_val, dtype=torch.float32, device=device)
                zv = torch.randn(len(c_val), cfg.z_dim, device=device)
                fake_v = gen(zv, cv).cpu().numpy()
            val_score = float(np.mean([wasserstein_distance(x_val_n[:, c, :].ravel(), fake_v[:, c, :].ravel()) for c in range(3)]))
            if val_score < best_score:
                best_score = val_score
                best_state = {"gen": {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}, "dis": {k: v.detach().cpu().clone() for k, v in dis.state_dict().items()}}
        else:
            best_state = {"gen": {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}, "dis": {k: v.detach().cpu().clone() for k, v in dis.state_dict().items()}}

        logs.append(
            {
                "epoch": epoch + 1,
                "loss_d": d_loss_sum / max(nb, 1),
                "loss_g": g_loss_sum / max(nb, 1),
                "gp": gp_sum / max(nb, 1),
                "range_pen": range_sum / max(nb, 1),
                "ramp_pen": ramp_sum / max(nb, 1),
                "val_wasserstein_norm": val_score,
                "lr_g": sch_g.get_last_lr()[0],
                "lr_d": sch_d.get_last_lr()[0],
            }
        )

    if best_state is not None:
        gen.load_state_dict(best_state["gen"])
        dis.load_state_dict(best_state["dis"])

    # save artifacts
    torch.save({"generator": gen.state_dict(), "critic": dis.state_dict(), "config": asdict(cfg)}, out_dir / "enhanced_gan_model.pt")
    (out_dir / "scaler.json").write_text(json.dumps(scaler.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    cb.save(out_dir / "condition_meta.json")
    pd.DataFrame(logs).to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    np.savez(out_dir / "physical_limits.npz", channel_limits=channel_limits, ramp_limits=ramp_limits)

    # save split data for reproducibility
    np.save(out_dir / "X_train.npy", x_train)
    np.save(out_dir / "X_val.npy", x_val)
    np.save(out_dir / "X_test.npy", x_test)
    cond_train.to_csv(out_dir / "cond_train.csv", index=False, encoding="utf-8-sig")
    cond_val.to_csv(out_dir / "cond_val.csv", index=False, encoding="utf-8-sig")
    cond_test.to_csv(out_dir / "cond_test.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Enhanced GAN extreme baseline",
        "paper_style_features": ["classification label conditioning", "WGAN-GP", "variable learning rate", "range/ramp/solar-night constraints"],
        "not_included": ["EVT tail-sensitive loss", "joint imbalance risk consistency loss", "hierarchical diffusion condition injection"],
        "data_info": data_info,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "seq_len": int(cfg.seq_len),
        "condition_dim": int(cb.total_dim),
        "channel_limits": channel_limits.tolist(),
        "ramp_limits": ramp_limits.tolist(),
        "best_val_wasserstein_norm": float(best_score) if np.isfinite(best_score) else None,
        "config": asdict(cfg),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ============================================================
# Generation
# ============================================================

def load_model_for_generation(model_dir: str, cfg_override: Optional[EnhancedGANConfig] = None):
    d = Path(model_dir)
    ckpt = torch.load(d / "enhanced_gan_model.pt", map_location="cpu")
    cfg = EnhancedGANConfig(**ckpt.get("config", {}))
    if cfg_override is not None:
        # override only runtime fields
        cfg.device = cfg_override.device
        cfg.output_dir = cfg_override.output_dir
    scaler = ChannelMinMaxScaler.from_json_dict(json.loads((d / "scaler.json").read_text(encoding="utf-8")))
    cb = ConditionBuilder.load(d / "condition_meta.json", cfg)
    gen = EnhancedGenerator(cfg.z_dim, cb.total_dim, cfg.seq_len, cfg.hidden_dim, cfg.n_res_blocks)
    gen.load_state_dict(ckpt["generator"])
    limits = np.load(d / "physical_limits.npz")
    return gen, scaler, cb, cfg, limits["channel_limits"], limits["ramp_limits"]


def generate_from_conditions(model_dir: str, cond_csv: str, output_dir: str, n_per_condition: int = 1, device: str = "cpu") -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen, scaler, cb, cfg, channel_limits, ramp_limits = load_model_for_generation(model_dir, EnhancedGANConfig(device=device, output_dir=output_dir))
    device_t = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    gen = gen.to(device_t).eval()

    cond = pd.read_csv(cond_csv)
    cond_rep = pd.concat([cond] * int(n_per_condition), ignore_index=True)
    cond_vec = cb.transform(cond_rep)
    with torch.no_grad():
        c = torch.tensor(cond_vec, dtype=torch.float32, device=device_t)
        z = torch.randn(len(cond_rep), cfg.z_dim, device=device_t)
        y = gen(z, c).cpu().numpy()
    x = scaler.inverse_transform(y)
    x = physical_project_np(x, cfg, channel_limits=channel_limits, ramp_limits=ramp_limits)

    np.save(out_dir / "generated_samples.npy", x)
    # long table
    rows = []
    for i in range(x.shape[0]):
        src_idx = i % len(cond)
        base = cond.iloc[src_idx].to_dict()
        for t in range(x.shape[2]):
            rows.append(
                {
                    "generated_id": i,
                    "source_condition_index": int(src_idx),
                    "t": t,
                    "load": float(x[i, 0, t]),
                    "wind_power": float(x[i, 1, t]),
                    "solar_power": float(x[i, 2, t]),
                    "event_type": base.get("event_type", ""),
                    "event_type_code": base.get("event_type_code", ""),
                    "month": base.get("month", ""),
                    "severity_level": base.get("severity_level", ""),
                    "extreme_prob": base.get("extreme_prob", ""),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "generated_samples_long.csv", index=False, encoding="utf-8-sig")
    cond_rep.to_csv(out_dir / "generated_conditions.csv", index=False, encoding="utf-8-sig")
    summary = {"n_generated": int(x.shape[0]), "shape": list(x.shape), "model_dir": str(Path(model_dir).resolve())}
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ============================================================
# Evaluation
# ============================================================

def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    denom = np.dot(x, x) + 1e-12
    out = np.ones(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(x[:-lag], x[lag:]) / denom if len(x) > lag else 0.0
    return out


def js_divergence_hist(x: np.ndarray, y: np.ndarray, bins: int = 80) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    px, edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    py, _ = np.histogram(y, bins=edges, density=False)
    px = px.astype(np.float64) + 1e-12
    py = py.astype(np.float64) + 1e-12
    px /= px.sum(); py /= py.sum()
    return float(jensenshannon(px, py, base=np.e) ** 2)


def resolve_tau(x_real: np.ndarray, cfg: EnhancedGANConfig) -> float:
    net = x_real[:, 0, :] - x_real[:, 1, :] - x_real[:, 2, :]
    if cfg.imbalance_tau_mode.lower() == "fixed":
        return float(cfg.imbalance_tau_fixed)
    return float(np.quantile(net, cfg.imbalance_tau_quantile))


def compute_risk_metrics_np(x: np.ndarray, tau: float, delta_t: float = 1.0) -> pd.DataFrame:
    x = ensure_channel_first(x)
    net = x[:, 0, :] - x[:, 1, :] - x[:, 2, :]
    excess = np.maximum(0.0, net - tau)
    cum = excess.sum(axis=1) * delta_t
    ramp = np.diff(net, axis=1)
    ramp_max = ramp.max(axis=1) if ramp.shape[1] > 0 else np.zeros(net.shape[0])
    dur = (net > tau).sum(axis=1) * delta_t
    return pd.DataFrame({"cum_deficit": cum, "netload_ramp_max": ramp_max, "imbalance_duration": dur})


def _risk_error(real_v: np.ndarray, gen_v: np.ndarray) -> dict:
    real_v = np.asarray(real_v, dtype=np.float64)
    gen_v = np.asarray(gen_v, dtype=np.float64)
    return {
        "real_mean": float(np.mean(real_v)),
        "gen_mean": float(np.mean(gen_v)),
        "mae_to_mean": float(abs(np.mean(gen_v) - np.mean(real_v))),
        "relative_error": float(abs(np.mean(gen_v) - np.mean(real_v)) / (abs(np.mean(real_v)) + 1e-12)),
        "q95_error": float(abs(np.quantile(gen_v, 0.95) - np.quantile(real_v, 0.95))),
        "q99_error": float(abs(np.quantile(gen_v, 0.99) - np.quantile(real_v, 0.99))),
    }


def evaluate_generation(real_path: str, generated_path: str, cond_csv: Optional[str], output_dir: str, model_name: str = "enhanced_gan", cfg: Optional[EnhancedGANConfig] = None) -> dict:
    cfg = cfg or EnhancedGANConfig(output_dir=output_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    x_real = ensure_channel_first(np.load(real_path))
    x_gen = ensure_channel_first(np.load(generated_path))
    # If generated has more samples than real because n_per_condition>1, compare all distributions directly.
    T = min(x_real.shape[2], x_gen.shape[2])
    x_real = x_real[:, :, :T]
    x_gen = x_gen[:, :, :T]
    names = ["load", "wind", "solar"]

    summary = {"model": model_name, "n_real": int(x_real.shape[0]), "n_generated": int(x_gen.shape[0]), "seq_len": int(T)}
    for ci, name in enumerate(names):
        r = x_real[:, ci, :].ravel()
        g = x_gen[:, ci, :].ravel()
        summary[f"wasserstein_{name}"] = float(wasserstein_distance(r, g))
        summary[f"js_{name}"] = js_divergence_hist(r, g, cfg.js_bins)
    summary["wasserstein_mean"] = float(np.mean([summary[f"wasserstein_{n}"] for n in names]))
    summary["js_mean"] = float(np.mean([summary[f"js_{n}"] for n in names]))

    # ACF error: mean ACF per sample then compare
    for ci, name in enumerate(names):
        max_lag = min(cfg.acf_max_lag, T - 1)
        if max_lag < 1:
            err = 0.0
        else:
            acf_r = np.mean([_acf_1d(x_real[i, ci, :], max_lag) for i in range(x_real.shape[0])], axis=0)
            acf_g = np.mean([_acf_1d(x_gen[i, ci, :], max_lag) for i in range(x_gen.shape[0])], axis=0)
            err = float(np.mean(np.abs(acf_r[1:] - acf_g[1:])))
        summary[f"acf_mae_{name}"] = err
    summary["acf_mae_mean"] = float(np.mean([summary[f"acf_mae_{n}"] for n in names]))

    corr_r = np.corrcoef(x_real.transpose(0, 2, 1).reshape(-1, 3).T)
    corr_g = np.corrcoef(x_gen.transpose(0, 2, 1).reshape(-1, 3).T)
    summary["corr_matrix_error_fro"] = float(np.linalg.norm(corr_r - corr_g, ord="fro"))
    summary["corr_matrix_error_mae"] = float(np.mean(np.abs(corr_r - corr_g)))

    tau = resolve_tau(x_real, cfg)
    summary["imbalance_tau"] = float(tau)
    risk_r = compute_risk_metrics_np(x_real, tau)
    risk_g = compute_risk_metrics_np(x_gen, tau)
    risk_rows = []
    for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
        err = _risk_error(risk_r[col].to_numpy(), risk_g[col].to_numpy())
        for k, v in err.items():
            summary[f"{col}_{k}"] = v
        risk_rows.append({"metric": col, **err})
    pd.DataFrame(risk_rows).to_csv(out_dir / "risk_metrics_real_vs_generated.csv", index=False, encoding="utf-8-sig")

    # extreme degree match proxy if cond exists and severity_level exists
    if cond_csv and Path(cond_csv).exists():
        cond = pd.read_csv(cond_csv)
        if "severity_level" in cond.columns:
            real_cum = risk_r["cum_deficit"].to_numpy()
            gen_cum = risk_g["cum_deficit"].to_numpy()
            q1, q2, q3 = np.quantile(real_cum, [0.70, 0.90, 0.97])
            def map_level(v):
                return np.where(v >= q3, 3, np.where(v >= q2, 2, np.where(v >= q1, 1, 0)))
            gen_level = map_level(gen_cum)
            cond_level = pd.to_numeric(cond["severity_level"], errors="coerce").fillna(0).astype(int).to_numpy()
            if len(cond_level) != len(gen_level):
                reps = int(math.ceil(len(gen_level) / max(len(cond_level), 1)))
                cond_level = np.tile(cond_level, reps)[: len(gen_level)]
            summary["extreme_degree_match_rate_strict"] = float(np.mean(gen_level == cond_level))
            summary["extreme_degree_match_rate_adjacent"] = float(np.mean(np.abs(gen_level - cond_level) <= 1))

    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    # by event/severity metrics if conditions available
    if cond_csv and Path(cond_csv).exists():
        cond = pd.read_csv(cond_csv)
        if len(cond) != x_gen.shape[0]:
            reps = int(math.ceil(x_gen.shape[0] / max(len(cond), 1)))
            cond = pd.concat([cond] * reps, ignore_index=True).iloc[:x_gen.shape[0]].reset_index(drop=True)
        group_cols = [c for c in ["event_type", "event_type_code", "severity_level"] if c in cond.columns]
        for gc in group_cols:
            rows = []
            for val, idx in cond.groupby(gc).groups.items():
                idx = np.asarray(list(idx), dtype=int)
                rg = compute_risk_metrics_np(x_gen[idx], tau)
                rows.append({
                    gc: val,
                    "n_generated": int(len(idx)),
                    "cum_deficit_mean": float(rg["cum_deficit"].mean()),
                    "netload_ramp_max_mean": float(rg["netload_ramp_max"].mean()),
                    "imbalance_duration_mean": float(rg["imbalance_duration"].mean()),
                })
            pd.DataFrame(rows).to_csv(out_dir / f"metrics_by_{gc}.csv", index=False, encoding="utf-8-sig")

    if cfg.make_figures:
        try:
            import matplotlib.pyplot as plt
            idx = 0
            t = np.arange(T)
            plt.figure(figsize=(12, 7))
            for ci, name in enumerate(names):
                ax = plt.subplot(3, 1, ci + 1)
                ax.plot(t, x_real[idx, ci, :], label="real")
                ax.plot(t, x_gen[idx, ci, :], label="generated", alpha=0.85)
                ax.set_ylabel(name)
                ax.grid(alpha=0.25)
                if ci == 0:
                    ax.legend()
            plt.tight_layout(); plt.savefig(fig_dir / "typical_generated_curve.png", dpi=180); plt.close()

            plt.figure(figsize=(12, 4))
            net_r = x_real[idx, 0, :] - x_real[idx, 1, :] - x_real[idx, 2, :]
            net_g = x_gen[idx, 0, :] - x_gen[idx, 1, :] - x_gen[idx, 2, :]
            plt.plot(t, net_r, label="real net load")
            plt.plot(t, net_g, label="generated net load")
            plt.legend(); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(fig_dir / "netload_curve.png", dpi=180); plt.close()

            max_lag = min(cfg.acf_max_lag, T - 1)
            plt.figure(figsize=(12, 6))
            for ci, name in enumerate(names):
                acf_r = np.mean([_acf_1d(x_real[i, ci, :], max_lag) for i in range(x_real.shape[0])], axis=0)
                acf_g = np.mean([_acf_1d(x_gen[i, ci, :], max_lag) for i in range(x_gen.shape[0])], axis=0)
                plt.plot(np.arange(max_lag + 1), acf_r, label=f"real {name}")
                plt.plot(np.arange(max_lag + 1), acf_g, linestyle="--", label=f"gen {name}")
            plt.legend(ncol=2); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(fig_dir / "acf_comparison.png", dpi=180); plt.close()

            for mat, title, fp in [(corr_r, "Real corr", "corr_matrix_real.png"), (corr_g, "Generated corr", "corr_matrix_generated.png")]:
                plt.figure(figsize=(4.5, 4))
                plt.imshow(mat, vmin=-1, vmax=1)
                plt.colorbar()
                plt.xticks([0,1,2], names); plt.yticks([0,1,2], names)
                plt.title(title); plt.tight_layout(); plt.savefig(fig_dir / fp, dpi=180); plt.close()

            risk_plot = pd.DataFrame({
                "real_cum_deficit": risk_r["cum_deficit"],
                "gen_cum_deficit": pd.Series(risk_g["cum_deficit"]),
            })
            plt.figure(figsize=(8, 4))
            plt.boxplot([risk_r["cum_deficit"].to_numpy(), risk_g["cum_deficit"].to_numpy()], labels=["real", "generated"])
            plt.title("Cum deficit distribution"); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(fig_dir / "risk_metric_boxplot.png", dpi=180); plt.close()

            plt.figure(figsize=(8, 4))
            plt.hist(risk_r["cum_deficit"], bins=30, density=True, alpha=0.5, label="real")
            plt.hist(risk_g["cum_deficit"], bins=30, density=True, alpha=0.5, label="generated")
            plt.legend(); plt.title("Tail distribution comparison: cum_deficit"); plt.tight_layout(); plt.savefig(fig_dir / "tail_distribution_comparison.png", dpi=180); plt.close()
        except Exception as e:
            warnings.warn(f"Figure generation failed: {e}")

    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced GAN baseline for extreme wind-solar-load scenario generation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--data-dir", default="outputs/dataset")
        sp.add_argument("--output-dir", default="outputs/enhanced_gan_extreme")
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--seed", type=int, default=42)

    tr = sub.add_parser("train")
    add_common(tr)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--z-dim", type=int, default=64)
    tr.add_argument("--n-critic", type=int, default=4)
    tr.add_argument("--gp-lambda", type=float, default=10.0)
    tr.add_argument("--lr-g", type=float, default=1e-4)
    tr.add_argument("--lr-d", type=float, default=2e-4)
    tr.add_argument("--use-extreme-prob-condition", action="store_true")

    ge = sub.add_parser("generate")
    ge.add_argument("--model-dir", required=True)
    ge.add_argument("--cond-csv", required=True)
    ge.add_argument("--output-dir", required=True)
    ge.add_argument("--n-per-condition", type=int, default=1)
    ge.add_argument("--device", default="cpu")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--real", required=True)
    ev.add_argument("--generated", required=True)
    ev.add_argument("--cond", default="")
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--model-name", default="enhanced_gan")
    ev.add_argument("--acf-max-lag", type=int, default=12)
    ev.add_argument("--js-bins", type=int, default=80)

    pipe = sub.add_parser("pipeline")
    add_common(pipe)
    pipe.add_argument("--epochs", type=int, default=100)
    pipe.add_argument("--batch-size", type=int, default=64)
    pipe.add_argument("--n-per-condition", type=int, default=1)
    pipe.add_argument("--use-extreme-prob-condition", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        cfg = EnhancedGANConfig(
            data_dir=args.data_dir, output_dir=args.output_dir, device=args.device, random_seed=args.seed,
            epochs=args.epochs, batch_size=args.batch_size, z_dim=args.z_dim, n_critic=args.n_critic,
            gp_lambda=args.gp_lambda, lr_g=args.lr_g, lr_d=args.lr_d,
            use_extreme_prob_condition=bool(args.use_extreme_prob_condition),
        )
        summary = train_enhanced_gan(cfg)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.cmd == "generate":
        summary = generate_from_conditions(args.model_dir, args.cond_csv, args.output_dir, args.n_per_condition, args.device)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.cmd == "evaluate":
        cfg = EnhancedGANConfig(output_dir=args.output_dir, acf_max_lag=args.acf_max_lag, js_bins=args.js_bins)
        summary = evaluate_generation(args.real, args.generated, args.cond or None, args.output_dir, args.model_name, cfg)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.cmd == "pipeline":
        cfg = EnhancedGANConfig(
            data_dir=args.data_dir, output_dir=args.output_dir, device=args.device, random_seed=args.seed,
            epochs=args.epochs, batch_size=args.batch_size,
            use_extreme_prob_condition=bool(args.use_extreme_prob_condition),
            generate_per_test_condition=args.n_per_condition,
        )
        summary_train = train_enhanced_gan(cfg)
        model_dir = args.output_dir
        cond_test = str(Path(model_dir) / "cond_test.csv")
        gen_dir = str(Path(model_dir) / "generation")
        summary_gen = generate_from_conditions(model_dir, cond_test, gen_dir, args.n_per_condition, args.device)
        eval_dir = str(Path(model_dir) / "evaluation")
        summary_eval = evaluate_generation(str(Path(model_dir) / "X_test.npy"), str(Path(gen_dir) / "generated_samples.npy"), str(Path(gen_dir) / "generated_conditions.csv"), eval_dir, "enhanced_gan", cfg)
        all_summary = {"train": summary_train, "generate": summary_gen, "evaluate": summary_eval}
        (Path(model_dir) / "pipeline_summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
