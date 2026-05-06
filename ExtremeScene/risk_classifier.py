from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class RiskClassifier1D(nn.Module):
    """Lightweight risk morphology classifier.

    Input:
        x: [B, 3, T], where channels are load, wind_power and solar_power.

    Output:
        logits for cum_level, ramp_level, duration_level and severity_level.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 48) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cum_head = nn.Linear(hidden, 4)
        self.ramp_head = nn.Linear(hidden, 4)
        self.duration_head = nn.Linear(hidden, 4)
        self.severity_head = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x).squeeze(-1)
        return {
            "cum_level": self.cum_head(h),
            "ramp_level": self.ramp_head(h),
            "duration_level": self.duration_head(h),
            "severity_level": self.severity_head(h),
        }


class MaskPriorMLP(nn.Module):
    """Condition-to-exceedance-mask prior.

    Input:
        cond_features: [B, F] numeric condition vector.
    Output:
        mask logits [B, T], where each value represents net_load > tau probability.
    """

    def __init__(self, input_dim: int, seq_len: int = 36, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, cond_features: torch.Tensor) -> torch.Tensor:
        return self.net(cond_features)


def _numeric_col(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(np.full((len(frame),), default), index=frame.index)


def build_mask_prior_features(cond_df: pd.DataFrame, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Build compact numeric condition features for the mask prior.

    The feature vector uses only condition labels, never real test curves.
    """

    frame = cond_df.reset_index(drop=True).copy()
    month = _numeric_col(frame, "month", 1).to_numpy(dtype=np.float32)
    season = _numeric_col(frame, "season_code", 0).clip(0, 3).astype(int).to_numpy()
    event_code = _numeric_col(frame, "event_type_code", 0).clip(0, 12).astype(int).to_numpy()
    max_event = int(stats.get("max_event_code", event_code.max() if len(event_code) else 0)) if stats else int(event_code.max() if len(event_code) else 0)
    event_onehot = np.eye(max_event + 1, dtype=np.float32)[np.clip(event_code, 0, max_event)]
    season_onehot = np.eye(4, dtype=np.float32)[season]

    duration = _numeric_col(frame, "duration_hours", 0).to_numpy(dtype=np.float32)
    tail = _numeric_col(frame, "tail_score", 0).to_numpy(dtype=np.float32)
    if stats is None:
        stats = {
            "max_event_code": max_event,
            "duration_mean": float(duration.mean()),
            "duration_std": float(duration.std() + 1e-6),
            "tail_mean": float(tail.mean()),
            "tail_std": float(tail.std() + 1e-6),
        }
    duration_z = ((duration - float(stats["duration_mean"])) / float(stats["duration_std"]))[:, None]
    tail_z = ((tail - float(stats["tail_mean"])) / float(stats["tail_std"]))[:, None]
    start_hour = _numeric_col(frame, "start_hour", 0).to_numpy(dtype=np.float32)
    if "window_start_time" in frame.columns:
        start_hour = pd.to_datetime(frame["window_start_time"]).dt.hour.to_numpy(dtype=np.float32)
    features = [
        event_onehot,
        np.sin(2 * np.pi * month / 12.0)[:, None],
        np.cos(2 * np.pi * month / 12.0)[:, None],
        season_onehot,
        _numeric_col(frame, "low_wind_flag", 0).to_numpy(dtype=np.float32)[:, None],
        _numeric_col(frame, "low_irradiance_flag", 0).to_numpy(dtype=np.float32)[:, None],
        duration_z,
        np.sin(2 * np.pi * start_hour / 24.0)[:, None],
        np.cos(2 * np.pi * start_hour / 24.0)[:, None],
        _numeric_col(frame, "extreme_prob", 0).to_numpy(dtype=np.float32)[:, None],
        tail_z,
        (_numeric_col(frame, "severity_level", 0).clip(0, 3).to_numpy(dtype=np.float32) / 3.0)[:, None],
        (_numeric_col(frame, "cum_level", 0).clip(0, 3).to_numpy(dtype=np.float32) / 3.0)[:, None],
        (_numeric_col(frame, "ramp_level", 0).clip(0, 3).to_numpy(dtype=np.float32) / 3.0)[:, None],
        (_numeric_col(frame, "duration_level", 0).clip(0, 3).to_numpy(dtype=np.float32) / 3.0)[:, None],
    ]
    return np.concatenate(features, axis=1).astype(np.float32), stats


@dataclass
class ClassifierTrainConfig:
    data_dir: str
    out_dir: str
    seq_len: int = 36
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    device: str = "cpu"


def train_risk_classifier(cfg: ClassifierTrainConfig) -> dict:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(data_dir / "X_train.npy").astype(np.float32)
    cond = pd.read_csv(data_dir / "cond_train.csv")
    x_mean = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    x_std = (x.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    targets = {
        "cum_level": _numeric_col(cond, "cum_level", 0).clip(0, 3).to_numpy(dtype=np.int64),
        "ramp_level": _numeric_col(cond, "ramp_level", 0).clip(0, 3).to_numpy(dtype=np.int64),
        "duration_level": _numeric_col(cond, "duration_level", 0).clip(0, 3).to_numpy(dtype=np.int64),
        "severity_level": _numeric_col(cond, "severity_level", 0).clip(0, 3).to_numpy(dtype=np.int64),
    }
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = TensorDataset(
        torch.from_numpy(x_norm),
        torch.from_numpy(targets["cum_level"]),
        torch.from_numpy(targets["ramp_level"]),
        torch.from_numpy(targets["duration_level"]),
        torch.from_numpy(targets["severity_level"]),
    )
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True)
    model = RiskClassifier1D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    for _ in range(int(cfg.epochs)):
        model.train()
        for xb, cum, ramp, dur, sev in loader:
            xb = xb.to(device)
            logits = model(xb)
            loss = (
                F.cross_entropy(logits["cum_level"], cum.to(device))
                + F.cross_entropy(logits["ramp_level"], ramp.to(device))
                + F.cross_entropy(logits["duration_level"], dur.to(device))
                + F.cross_entropy(logits["severity_level"], sev.to(device))
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x_norm).to(device)
        logits = model(xb)
        metrics = {}
        for key, target in targets.items():
            pred = logits[key].argmax(dim=1).cpu().numpy()
            metrics[f"{key}_acc"] = float((pred == target).mean())
    ckpt = {"model_state": model.state_dict(), "x_mean": x_mean, "x_std": x_std, "metrics": metrics}
    torch.save(ckpt, out_dir / "risk_classifier.pt")
    (out_dir / "risk_classifier_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def train_mask_prior(data_dir: str | Path, out_dir: str | Path, seq_len: int = 36, epochs: int = 80, seed: int = 42, device: str = "cpu") -> dict:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_train = pd.read_csv(data_dir / "cond_train.csv")
    mask_train = np.load(data_dir / "exceed_mask_train.npy").astype(np.float32)
    features, stats = build_mask_prior_features(cond_train)
    model = MaskPriorMLP(features.shape[1], seq_len=seq_len)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    ds = TensorDataset(torch.from_numpy(features), torch.from_numpy(mask_train))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(int(epochs)):
        model.train()
        for xb, mb in loader:
            logits = model(xb.to(dev))
            loss = F.binary_cross_entropy_with_logits(logits, mb.to(dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(dev))
        train_prob = torch.sigmoid(logits).cpu().numpy()
        train_bce = float(F.binary_cross_entropy(torch.from_numpy(train_prob).clamp(1e-6, 1 - 1e-6), torch.from_numpy(mask_train)).item())
    ckpt = {"model_state": model.state_dict(), "input_dim": int(features.shape[1]), "seq_len": int(seq_len), "feature_stats": stats, "train_bce": train_bce}
    torch.save(ckpt, out_dir / "mask_prior.pt")
    (out_dir / "mask_prior_metrics.json").write_text(json.dumps({"train_bce": train_bce}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train_bce": train_bce, "checkpoint": str(out_dir / "mask_prior.pt")}


def predict_mask_prior(checkpoint_path: str | Path, cond_df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    features, _ = build_mask_prior_features(cond_df, stats=ckpt["feature_stats"])
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MaskPriorMLP(int(ckpt["input_dim"]), seq_len=int(ckpt["seq_len"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(features).to(dev))).cpu().numpy()
    return prob.astype(np.float32)
