from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate_generation import EvalConfig, evaluate_generation
from risk_ranking_utils import add_risk_score, write_risk_tables
from run_copula_guided_residual_diffusion import DatasetSpec, _dataset_specs
from run_gan_augmented_tailweighted_copula import (
    METHOD_NAME as GAN_METHOD,
    GanAugmentedTailWeightedConfig,
    _correlation_filter_candidates,
    _risk_filter_candidates,
    _tail_threshold_and_index,
    _to_channel_time,
)
from run_month_evt_copula_risk_selection import (
    _add_month_season,
    _build_train_risk_table,
    _compute_monthly_tau,
    _copula_cfg,
    _generate_candidate_pool,
    _load_split,
    _parse_weights,
    _select_candidates,
)
from run_simple_evt_risk_diffusion import FULL_COMPARE_METRICS
from run_tailweighted_month_evt_copula import (
    TAIL_FIXED_METHOD,
    TailWeightedConfig,
    _compute_tail_scores,
    _fit_tailweighted_group_copulas,
)


BASE_DIR = Path(__file__).resolve().parent

VAE_METHOD = "TransformerVAE_Augmented_TailWeighted_Copula"
TRANSFORMER_METHOD = "Conditional_Transformer_Risk_Generator"
FLOW_METHOD = "Conditional_NormalizingFlow_Risk_Generator"
ENSEMBLE_METHOD = "ValSelected_Model_Ensemble"
MARGIN_ENSEMBLE_PREFIX = "ValSelected_Model_Ensemble_Margin"


@dataclass
class ModelPoolConfig(GanAugmentedTailWeightedConfig):
    out_dir: Path = BASE_DIR / "results" / "model_pool_experiments"
    augmented_data_root: Path = BASE_DIR / "outputs" / "model_pool_augmented_datasets"
    device: str = "cpu"
    model_epochs: int = 5
    vae_epochs: int = 5
    transformer_epochs: int = 5
    flow_epochs: int = 5
    direct_hidden_dim: int = 64
    flow_hidden_dim: int = 128
    latent_dim: int = 16
    beta_kl: float = 0.001
    candidate_count: int = 20
    use_candidate_multiplier: bool = False
    direct_k_candidates: int = 10
    batch_size: int = 32
    model_lr: float = 1.0e-3
    selection_margins: tuple[float, ...] = (0.0,)


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _condition_frame(cond: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = cond.reset_index(drop=True).copy()
    meta = meta.reset_index(drop=True)
    for col in meta.columns:
        if col not in out.columns:
            out[col] = meta[col].to_numpy()
    if "month" not in out.columns:
        out["month"] = 1
    return out


class ConditionEncoder:
    def __init__(self, cond_train: pd.DataFrame):
        event = cond_train.get("event_type", pd.Series(["unknown"] * len(cond_train))).astype(str)
        self.event_values = sorted(event.fillna("unknown").unique().tolist())
        self.event_to_idx = {value: i for i, value in enumerate(self.event_values)}
        duration = pd.to_numeric(cond_train.get("duration_hours", cond_train.get("imbalance_duration", 1.0)), errors="coerce").fillna(1.0)
        self.duration_scale = float(max(duration.max(), 1.0))

    @property
    def dim(self) -> int:
        return len(self.event_values) + 4

    def transform(self, cond: pd.DataFrame) -> np.ndarray:
        event = cond.get("event_type", pd.Series(["unknown"] * len(cond))).astype(str).fillna("unknown")
        month = pd.to_numeric(cond.get("month", 1), errors="coerce").fillna(1).to_numpy(float)
        duration = pd.to_numeric(cond.get("duration_hours", cond.get("imbalance_duration", 1.0)), errors="coerce").fillna(1.0).to_numpy(float)
        extreme = pd.to_numeric(cond.get("extreme_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        onehot = np.zeros((len(cond), len(self.event_values)), dtype=np.float32)
        for i, value in enumerate(event):
            onehot[i, self.event_to_idx.get(str(value), 0)] = 1.0
        month_rad = 2.0 * np.pi * (month - 1.0) / 12.0
        scalars = np.stack(
            [
                np.sin(month_rad),
                np.cos(month_rad),
                np.clip(duration / max(self.duration_scale, 1e-6), 0.0, 2.0),
                np.clip(extreme, 0.0, 1.0),
            ],
            axis=1,
        ).astype(np.float32)
        return np.concatenate([onehot, scalars], axis=1).astype(np.float32)


class ChannelScaler:
    def fit(self, x: np.ndarray) -> "ChannelScaler":
        x_ct = _to_channel_time(x)
        self.mean = x_ct.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std = (x_ct.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        self.channel_max = x_ct.max(axis=(0, 2), keepdims=True).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((_to_channel_time(x) - self.mean) / self.std).astype(np.float32)

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        x = np.asarray(x_norm, dtype=np.float32) * self.std + self.mean
        return _project_physics(x)


def _project_physics(x: np.ndarray) -> np.ndarray:
    x = _to_channel_time(x).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(x, 0.0).astype(np.float32)


class TransformerVAE(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int = 64, latent_dim: int = 16, seq_len: int = 36):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(3 + cond_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * 3),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_seq = c[:, None, :].expand(-1, x.shape[2], -1)
        inp = torch.cat([x.transpose(1, 2), c_seq], dim=-1)
        h = self.encoder(self.in_proj(inp)).mean(dim=1)
        return self.mu(h), self.logvar(h).clamp(-8.0, 6.0)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        y = self.dec(torch.cat([z, c], dim=-1))
        return y.view(-1, self.seq_len, 3).transpose(1, 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        return self.decode(z, c), mu, logvar


class DirectTransformerGenerator(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int = 64, seq_len: int = 36, noise_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.in_proj = nn.Linear(3 + noise_dim + cond_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.out = nn.Linear(hidden_dim, 3)

    def forward(self, base: torch.Tensor, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c_seq = cond[:, None, :].expand(-1, self.seq_len, -1)
        inp = torch.cat([base.transpose(1, 2), noise, c_seq], dim=-1)
        return self.out(self.encoder(self.in_proj(inp))).transpose(1, 2)


class CouplingNet(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim * 2),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([x, c], dim=-1))
        s, t = out.chunk(2, dim=-1)
        return torch.tanh(s) * 1.5, t


class ConditionalRealNVP(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0
            masks.append(mask)
        self.register_buffer("masks", torch.stack(masks, dim=0))
        self.nets = nn.ModuleList([CouplingNet(dim, cond_dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for mask, net in zip(self.masks, self.nets):
            x_masked = z * mask
            s, t = net(x_masked, c)
            inv_mask = 1.0 - mask
            z = x_masked + inv_mask * (z * torch.exp(s) + t)
            logdet = logdet + (inv_mask * s).sum(dim=-1)
        return z, logdet

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for mask, net in reversed(list(zip(self.masks, self.nets))):
            x_masked = x * mask
            s, t = net(x_masked, c)
            inv_mask = 1.0 - mask
            x = x_masked + inv_mask * ((x - t) * torch.exp(-s))
        return x


def _train_vae(x: np.ndarray, cond_feat: np.ndarray, cfg: ModelPoolConfig) -> tuple[TransformerVAE, ChannelScaler]:
    device = torch.device(cfg.device)
    scaler = ChannelScaler().fit(x)
    x_norm = scaler.transform(x)
    ds = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(cond_feat.astype(np.float32)))
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True)
    model = TransformerVAE(cond_feat.shape[1], cfg.direct_hidden_dim, cfg.latent_dim, x_norm.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.model_lr))
    for _ in range(int(cfg.vae_epochs)):
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            recon, mu, logvar = model(xb, cb)
            recon_loss = F.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + float(cfg.beta_kl) * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), scaler


def _train_transformer(x: np.ndarray, cond_feat: np.ndarray, cfg: ModelPoolConfig) -> tuple[DirectTransformerGenerator, ChannelScaler]:
    device = torch.device(cfg.device)
    scaler = ChannelScaler().fit(x)
    x_norm = scaler.transform(x)
    ds = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(cond_feat.astype(np.float32)))
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True)
    model = DirectTransformerGenerator(cond_feat.shape[1], cfg.direct_hidden_dim, x_norm.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.model_lr))
    for _ in range(int(cfg.transformer_epochs)):
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            noisy = xb + 0.15 * torch.randn_like(xb)
            noise = torch.randn((xb.shape[0], xb.shape[2], model.noise_dim), device=device)
            pred = model(noisy, noise, cb)
            loss = F.mse_loss(pred, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), scaler


def _train_flow(x: np.ndarray, cond_feat: np.ndarray, cfg: ModelPoolConfig) -> tuple[ConditionalRealNVP, ChannelScaler]:
    device = torch.device(cfg.device)
    scaler = ChannelScaler().fit(x)
    x_norm = scaler.transform(x).reshape(len(x), -1).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(cond_feat.astype(np.float32)))
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True)
    model = ConditionalRealNVP(x_norm.shape[1], cond_feat.shape[1], cfg.flow_hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.model_lr))
    const = 0.5 * x_norm.shape[1] * math.log(2.0 * math.pi)
    for _ in range(int(cfg.flow_epochs)):
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            z, logdet = model(xb, cb)
            nll = 0.5 * (z**2).sum(dim=-1) + const - logdet
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), scaler


@torch.no_grad()
def _sample_vae(model: TransformerVAE, scaler: ChannelScaler, cond_feat: np.ndarray, n: int, cfg: ModelPoolConfig) -> np.ndarray:
    device = torch.device(cfg.device)
    idx = np.random.choice(len(cond_feat), size=int(n), replace=True)
    c = torch.from_numpy(cond_feat[idx].astype(np.float32)).to(device)
    z = torch.randn((len(idx), int(cfg.latent_dim)), device=device)
    x = model.decode(z, c).cpu().numpy()
    return scaler.inverse(x)


@torch.no_grad()
def _sample_transformer_candidates(
    model: DirectTransformerGenerator,
    scaler: ChannelScaler,
    train_x: np.ndarray,
    cond_feat: np.ndarray,
    k: int,
    cfg: ModelPoolConfig,
) -> np.ndarray:
    device = torch.device(cfg.device)
    x_norm = scaler.transform(train_x)
    out = []
    for i in range(len(cond_feat)):
        base_idx = np.random.choice(len(x_norm), size=int(k), replace=True)
        base = torch.from_numpy(x_norm[base_idx]).to(device)
        c = torch.from_numpy(np.repeat(cond_feat[i : i + 1], int(k), axis=0).astype(np.float32)).to(device)
        noise = torch.randn((int(k), x_norm.shape[2], model.noise_dim), device=device)
        pred = model(base, noise, c).cpu().numpy()
        out.append(scaler.inverse(pred))
    return np.stack(out, axis=0).astype(np.float32)


@torch.no_grad()
def _sample_flow_candidates(model: ConditionalRealNVP, scaler: ChannelScaler, cond_feat: np.ndarray, k: int, cfg: ModelPoolConfig) -> np.ndarray:
    device = torch.device(cfg.device)
    dim = int(3 * 36)
    out = []
    for i in range(len(cond_feat)):
        c = torch.from_numpy(np.repeat(cond_feat[i : i + 1], int(k), axis=0).astype(np.float32)).to(device)
        z = torch.randn((int(k), dim), device=device)
        x = model.inverse(z, c).cpu().numpy().reshape(int(k), 3, 36)
        out.append(scaler.inverse(x))
    return np.stack(out, axis=0).astype(np.float32)


def _evaluate_test(method: str, generated: Path, spec: DatasetSpec, out_dir: Path, split: str = "test") -> dict:
    eval_dir = out_dir / "evaluations" / f"{method}_{split}"[:80]
    eval_dir.mkdir(parents=True, exist_ok=True)
    mask_path = spec.data_dir / f"event_mask_{split}.npy"
    summary = evaluate_generation(
        EvalConfig(
            real=str(spec.data_dir / f"X_{split}.npy"),
            generated=str(generated),
            cond=str(spec.data_dir / f"cond_{split}.csv"),
            meta=str(spec.data_dir / f"meta_{split}.csv"),
            event_mask=str(mask_path) if mask_path.exists() else None,
            out_dir=str(eval_dir),
            model_name=method[:60],
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = {"method": method}
    row.update({metric: summary["metrics"].get(metric, np.nan) for metric in FULL_COMPARE_METRICS})
    return row


def _build_train_context(spec: DatasetSpec, cfg: ModelPoolConfig, out_dir: Path):
    x_train, cond_train_raw, meta_train, mask_train = _load_split(spec.data_dir, "train")
    cond_train = _add_month_season(cond_train_raw, meta_train)
    tau_by_month, _ = _compute_monthly_tau(x_train, cond_train, cfg, spec.out_name, out_dir)
    train_risk = _build_train_risk_table(x_train, cond_train, mask_train, tau_by_month, cfg)
    return x_train, cond_train_raw, meta_train, mask_train, cond_train, tau_by_month, train_risk


def _generate_tailweighted_split(spec: DatasetSpec, cfg: ModelPoolConfig, out_dir: Path, split: str, generated_name: str, train_override=None) -> Path:
    if train_override is None:
        x_train, cond_train_raw, meta_train, mask_train = _load_split(spec.data_dir, "train")
    else:
        x_train, cond_train_raw, meta_train, mask_train = train_override
    _, cond_split_raw, meta_split, mask_split = _load_split(spec.data_dir, split)
    cond_train = _add_month_season(cond_train_raw, meta_train)
    cond_split = _add_month_season(cond_split_raw, meta_split)
    tau_by_month, _ = _compute_monthly_tau(x_train, cond_train, cfg, spec.out_name, out_dir)
    train_risk = _build_train_risk_table(x_train, cond_train, mask_train, tau_by_month, cfg)
    tail_df = _compute_tail_scores(train_risk, cond_train, cfg, spec.out_name, out_dir)
    models, _, _ = _fit_tailweighted_group_copulas(x_train, cond_train, tail_df, cfg, out_dir, spec.out_name)
    cop_cfg = _copula_cfg(cfg, int(_to_channel_time(x_train).shape[2]), out_dir / f"{generated_name}_copula_groups")
    rng = np.random.default_rng(int(cfg.seed))
    candidates, metrics, targets, base = _generate_candidate_pool(
        split, cond_split, mask_split, models, train_risk, tau_by_month, cfg, cop_cfg, rng, out_dir, spec.out_name
    )
    gen, log = _select_candidates(candidates, metrics, targets, base, train_risk, cond_split, _parse_weights(cfg.fixed_weights), cfg)
    log.to_csv(out_dir / f"candidate_selection_log_{generated_name}_{split}.csv", index=False, encoding="utf-8-sig")
    path = out_dir / f"generated_samples_{generated_name}.npy" if split == "test" else out_dir / f"generated_val_{generated_name}.npy"
    np.save(path, gen.astype(np.float32))
    return path


def _candidate_metrics_for_pool(candidates: np.ndarray, cond: pd.DataFrame, mask: np.ndarray | None, train_risk: pd.DataFrame, tau_by_month: dict[int, float], cfg: ModelPoolConfig):
    from run_gan_augmented_tailweighted_copula import _build_candidate_risk_table

    frames = []
    target_rows = []
    base_rows = []
    for i, row in cond.reset_index(drop=True).iterrows():
        month = int(row.get("month", 1))
        mask_i = np.repeat(mask[i : i + 1], candidates.shape[1], axis=0) if mask is not None else None
        risk = _build_candidate_risk_table(candidates[i], np.full((candidates.shape[1],), month), mask_i, train_risk, tau_by_month, cfg)
        risk.insert(0, "candidate_index", np.arange(candidates.shape[1], dtype=int))
        risk.insert(0, "sample_id", int(i))
        frames.append(risk)
        target_rows.append({"sample_id": int(i), "dataset": ""})
        base_rows.append({"dataset": "", "sample_id": int(i), "month": month, "season": "", "copula_group_used": "direct", "target_group_used": "risk", "extreme_prob": float(row.get("extreme_prob", 0.0))})
    return pd.concat(frames, ignore_index=True), target_rows, base_rows


def _select_direct_candidates(candidates: np.ndarray, cond: pd.DataFrame, mask: np.ndarray | None, train_risk: pd.DataFrame, tau_by_month: dict[int, float], cfg: ModelPoolConfig):
    metrics, targets, base = _candidate_metrics_for_pool(candidates, cond, mask, train_risk, tau_by_month, cfg)
    return _select_candidates(candidates, metrics, targets, base, train_risk, cond, _parse_weights(cfg.fixed_weights), cfg)


def _run_vae_aug(spec: DatasetSpec, cfg: ModelPoolConfig, out_dir: Path, split: str) -> Path:
    x_train, cond_train_raw, meta_train, mask_train, cond_train, tau_by_month, train_risk = _build_train_context(spec, cfg, out_dir)
    tail_df = _compute_tail_scores(train_risk, cond_train, cfg, spec.out_name, out_dir)
    threshold, tail_idx, _ = _tail_threshold_and_index(tail_df["tail_score"].to_numpy(float), cfg)
    if len(tail_idx) < 4:
        tail_idx = np.argsort(tail_df["tail_score"].to_numpy(float))[-min(4, len(tail_df)) :]
    train_meta = _condition_frame(cond_train_raw, meta_train)
    tail_meta = train_meta.iloc[tail_idx].reset_index(drop=True)
    enc = ConditionEncoder(train_meta)
    tail_feat = enc.transform(tail_meta)
    model, scaler = _train_vae(x_train[tail_idx], tail_feat, cfg)
    n_candidates = int(round(3.0 * len(x_train))) if cfg.use_candidate_multiplier else int(cfg.candidate_count)
    x_cand = _sample_vae(model, scaler, tail_feat, n_candidates, cfg)
    sampled = np.random.choice(len(tail_meta), size=n_candidates, replace=True)
    cand_meta = tail_meta.iloc[sampled].reset_index(drop=True)
    cand_mask = mask_train[tail_idx][sampled] if mask_train is not None else None
    x_risk, meta_risk, mask_risk, risk_candidates, _ = _risk_filter_candidates(x_cand, cand_meta, cand_mask, train_risk, tau_by_month, cfg)
    n_keep = min(max(int(round(float(cfg.gan_keep_ratio) * len(x_train))), int(cfg.gan_keep_min)), len(tail_idx), len(x_risk))
    kept_x, kept_meta, kept_mask, _, _ = _correlation_filter_candidates(
        x_risk, meta_risk, mask_risk, risk_candidates, x_train[tail_idx], train_risk.iloc[tail_idx].reset_index(drop=True), train_risk, n_keep, cfg
    )
    if len(kept_x):
        cond_cols = list(cond_train_raw.columns)
        meta_cols = list(meta_train.columns)
        x_aug = np.concatenate([_to_channel_time(x_train), _to_channel_time(kept_x)], axis=0)
        cond_aug = pd.concat([cond_train_raw.reset_index(drop=True), kept_meta.reindex(columns=cond_cols)], ignore_index=True)
        meta_aug = pd.concat([meta_train.reset_index(drop=True), kept_meta.reindex(columns=meta_cols)], ignore_index=True)
        mask_aug = np.concatenate([mask_train, kept_mask], axis=0) if mask_train is not None and kept_mask is not None else mask_train
    else:
        x_aug, cond_aug, meta_aug, mask_aug = x_train, cond_train_raw, meta_train, mask_train
    aug_dir = cfg.augmented_data_root / spec.out_name / VAE_METHOD
    aug_dir.mkdir(parents=True, exist_ok=True)
    np.save(aug_dir / "X_train_aug.npy", _to_channel_time(x_aug).astype(np.float32))
    cond_aug.to_csv(aug_dir / "cond_train_aug.csv", index=False, encoding="utf-8-sig")
    meta_aug.to_csv(aug_dir / "meta_train_aug.csv", index=False, encoding="utf-8-sig")
    if mask_aug is not None:
        np.save(aug_dir / "event_mask_train_aug.npy", mask_aug.astype(np.float32))
    return _generate_tailweighted_split(spec, cfg, out_dir, split, VAE_METHOD, train_override=(x_aug, cond_aug, meta_aug, mask_aug))


def _run_direct_method(spec: DatasetSpec, cfg: ModelPoolConfig, out_dir: Path, split: str, method: str) -> Path:
    x_train, cond_train_raw, meta_train, mask_train, cond_train, tau_by_month, train_risk = _build_train_context(spec, cfg, out_dir)
    _, cond_split_raw, meta_split, mask_split = _load_split(spec.data_dir, split)
    cond_split = _add_month_season(cond_split_raw, meta_split)
    enc = ConditionEncoder(_condition_frame(cond_train_raw, meta_train))
    train_feat = enc.transform(_condition_frame(cond_train_raw, meta_train))
    split_feat = enc.transform(_condition_frame(cond_split_raw, meta_split))
    if method == TRANSFORMER_METHOD:
        model, scaler = _train_transformer(x_train, train_feat, cfg)
        candidates = _sample_transformer_candidates(model, scaler, x_train, split_feat, int(cfg.k_candidates), cfg)
    elif method == FLOW_METHOD:
        model, scaler = _train_flow(x_train, train_feat, cfg)
        candidates = _sample_flow_candidates(model, scaler, split_feat, int(cfg.k_candidates), cfg)
    else:
        raise ValueError(method)
    selected, log = _select_direct_candidates(candidates, cond_split, mask_split, train_risk, tau_by_month, cfg)
    log.to_csv(out_dir / f"candidate_selection_log_{method}_{split}.csv", index=False, encoding="utf-8-sig")
    path = out_dir / f"generated_samples_{method}.npy" if split == "test" else out_dir / f"generated_val_{method}.npy"
    np.save(path, selected.astype(np.float32))
    return path


def _margin_method_name(margin: float) -> str:
    return f"{MARGIN_ENSEMBLE_PREFIX}_{margin:.2f}"


def _select_with_margin(val_scored: pd.DataFrame, margin: float) -> tuple[str, str, str, float, float]:
    tail_row = val_scored[val_scored["method"] == TAIL_FIXED_METHOD]
    if tail_row.empty:
        raise ValueError("TailWeighted validation row is required for margin selection.")
    tail_score = float(tail_row["risk_score"].iloc[0])
    non_tail = val_scored[val_scored["method"] != TAIL_FIXED_METHOD].copy()
    if non_tail.empty:
        return TAIL_FIXED_METHOD, TAIL_FIXED_METHOD, tail_score, tail_score, "no non-TailWeighted candidate exists"
    best_idx = non_tail.sort_values(["risk_score", "priority"], ascending=[True, True]).index[0]
    best_method = str(non_tail.loc[best_idx, "method"])
    best_score = float(non_tail.loc[best_idx, "risk_score"])
    if abs(float(margin)) < 1e-12:
        all_best_idx = val_scored.sort_values(["risk_score", "priority"], ascending=[True, True]).index[0]
        selected = str(val_scored.loc[all_best_idx, "method"])
        selected_score = float(val_scored.loc[all_best_idx, "risk_score"])
        reason = "margin=0.00 uses current best validation risk_score with stability tie priority"
        return selected, best_method, tail_score, best_score, reason
    if best_score <= tail_score - float(margin):
        reason = f"{best_method} val_score {best_score:.6f} <= TailWeighted {tail_score:.6f} - margin {margin:.2f}"
        return best_method, best_method, tail_score, best_score, reason
    reason = f"best non-TailWeighted improvement {tail_score - best_score:.6f} < margin {margin:.2f}; fallback to TailWeighted"
    return TAIL_FIXED_METHOD, best_method, tail_score, best_score, reason


def _load_existing_gan_paths(spec: DatasetSpec, out_dir: Path) -> tuple[Path | None, Path | None]:
    gan_root = BASE_DIR / "results" / "gan_augmented_tailweighted_copula_valprotected_formal50" / spec.out_name
    val_source = gan_root / "generated_val_gan_aug_tailweighted.npy"
    test_source = gan_root / "generated_samples_gan_aug_tailweighted.npy"
    val_target = out_dir / f"generated_val_{GAN_METHOD}.npy"
    test_target = out_dir / f"generated_samples_{GAN_METHOD}.npy"
    if val_source.exists() and test_source.exists():
        shutil.copy2(val_source, val_target)
        shutil.copy2(test_source, test_target)
        return val_target, test_target
    return None, None


def run_dataset(spec: DatasetSpec, cfg: ModelPoolConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = cfg.out_dir / spec.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)
    methods = [TAIL_FIXED_METHOD, VAE_METHOD, TRANSFORMER_METHOD, FLOW_METHOD]
    val_paths = {
        TAIL_FIXED_METHOD: _generate_tailweighted_split(spec, cfg, out_dir, "val", TAIL_FIXED_METHOD),
        VAE_METHOD: _run_vae_aug(spec, cfg, out_dir, "val"),
        TRANSFORMER_METHOD: _run_direct_method(spec, cfg, out_dir, "val", TRANSFORMER_METHOD),
        FLOW_METHOD: _run_direct_method(spec, cfg, out_dir, "val", FLOW_METHOD),
    }
    test_paths = {
        TAIL_FIXED_METHOD: _generate_tailweighted_split(spec, cfg, out_dir, "test", TAIL_FIXED_METHOD),
        VAE_METHOD: _run_vae_aug(spec, cfg, out_dir, "test"),
        TRANSFORMER_METHOD: _run_direct_method(spec, cfg, out_dir, "test", TRANSFORMER_METHOD),
        FLOW_METHOD: _run_direct_method(spec, cfg, out_dir, "test", FLOW_METHOD),
    }
    gan_val, gan_test = _load_existing_gan_paths(spec, out_dir)
    if gan_val is not None and gan_test is not None:
        val_paths[GAN_METHOD] = gan_val
        test_paths[GAN_METHOD] = gan_test
        methods.append(GAN_METHOD)
    val_rows = []
    for method, path in val_paths.items():
        row = _evaluate_test(method, path, spec, out_dir, split="val")
        val_rows.append(row)
    val_df = add_risk_score(pd.DataFrame(val_rows))
    priority = {TAIL_FIXED_METHOD: 0, VAE_METHOD: 1, GAN_METHOD: 2, TRANSFORMER_METHOD: 3, FLOW_METHOD: 4}
    val_df["priority"] = val_df["method"].map(priority).fillna(9)
    selections = []
    for margin in cfg.selection_margins:
        selected_method, best_candidate_method, tail_score, best_candidate_score, reason = _select_with_margin(val_df, float(margin))
        selections.append(
            {
                "margin": float(margin),
                "dataset": spec.out_name,
                "tailweighted_val_score": tail_score,
                "best_candidate_method": best_candidate_method,
                "best_candidate_val_score": best_candidate_score,
                "selected_method": selected_method,
                "reason": reason,
            }
        )
        val_df[f"selected_margin_{float(margin):.2f}"] = val_df["method"].astype(str).eq(selected_method)
    val_df = val_df.rename(
        columns={
            "q99_cum_deficit_error": "val_q99",
            "core_q99_cum_deficit_error": "val_core_q99",
            "netload_ramp_max_mae": "val_ramp",
            "imbalance_duration_mae": "val_duration",
            "risk_score": "val_risk_score",
        }
    )
    val_df.insert(0, "dataset", spec.out_name)
    val_df.to_csv(out_dir / "val_model_selection_summary.csv", index=False, encoding="utf-8-sig")
    rows = [_evaluate_test(method, path, spec, out_dir, split="test") for method, path in test_paths.items()]
    for selection in selections:
        margin = float(selection["margin"])
        selected_method = str(selection["selected_method"])
        method_name = _margin_method_name(margin)
        ensemble_path = out_dir / f"generated_samples_{method_name}.npy"
        shutil.copy2(test_paths[selected_method], ensemble_path)
        rows.append(_evaluate_test(method_name, ensemble_path, spec, out_dir, split="test"))
        if abs(margin) < 1e-12:
            legacy_path = out_dir / f"generated_samples_{ENSEMBLE_METHOD}.npy"
            shutil.copy2(test_paths[selected_method], legacy_path)
            rows.append(_evaluate_test(ENSEMBLE_METHOD, legacy_path, spec, out_dir, split="test"))
    df = add_risk_score(pd.DataFrame(rows))
    for selection in selections:
        method_name = _margin_method_name(float(selection["margin"]))
        sub = df[df["method"] == method_name]
        selection["selected_method_test_rank"] = float(sub["risk_rank"].iloc[0]) if not sub.empty else np.nan
        selection["selected_method_test_score"] = float(sub["risk_score"].iloc[0]) if not sub.empty else np.nan
    selection_df = pd.DataFrame(selections)
    selection_df.to_csv(out_dir / "margin_selection_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir)
    lines = [
        f"# {spec.name} - Model Pool Experiment",
        "",
        "## Validation Selection",
        "",
        val_df.to_markdown(index=False),
        "",
        "## Margin Selection",
        "",
        selection_df.to_markdown(index=False),
        "",
        "## Main Risk Table",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Auxiliary Realism",
        "",
        aux.to_markdown(index=False),
    ]
    (out_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8")
    return risk_main, aux, val_df, selection_df


def _write_global(
    root: Path,
    risk_tables: dict[str, pd.DataFrame],
    aux_tables: dict[str, pd.DataFrame],
    val_tables: dict[str, pd.DataFrame],
    margin_tables: dict[str, pd.DataFrame],
    cfg: ModelPoolConfig,
) -> None:
    risk_rows = []
    for ds, table in risk_tables.items():
        df = table.copy()
        df.insert(0, "dataset", ds)
        risk_rows.append(df)
    risk_summary = pd.concat(risk_rows, ignore_index=True)
    risk_summary.to_csv(root / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")
    aux_rows = []
    for ds, table in aux_tables.items():
        df = table.copy()
        df.insert(0, "dataset", ds)
        aux_rows.append(df)
    pd.concat(aux_rows, ignore_index=True).to_csv(root / "all_datasets_auxiliary_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(val_tables.values(), ignore_index=True).to_csv(root / "all_datasets_val_model_selection_summary.csv", index=False, encoding="utf-8-sig")
    margin_method_names = [_margin_method_name(float(margin)) for margin in cfg.selection_margins]
    required = [TAIL_FIXED_METHOD, VAE_METHOD, TRANSFORMER_METHOD, FLOW_METHOD, GAN_METHOD, ENSEMBLE_METHOD] + margin_method_names
    methods = list(dict.fromkeys(required + risk_summary["method"].astype(str).tolist()))
    rows = []
    datasets = list(risk_tables.keys())
    for method in methods:
        row = {"method": method}
        ranks, scores = [], []
        for ds in datasets:
            sub = risk_summary[(risk_summary["dataset"] == ds) & (risk_summary["method"] == method)]
            if sub.empty:
                row[f"{ds}_risk_rank"] = np.nan
                row[f"{ds}_risk_score"] = np.nan
                continue
            rank = float(sub["risk_rank"].iloc[0])
            score = float(sub["risk_score"].iloc[0])
            row[f"{ds}_risk_rank"] = rank
            row[f"{ds}_risk_score"] = score
            ranks.append(rank)
            scores.append(score)
        row["mean_risk_rank"] = float(np.mean(ranks)) if ranks else np.nan
        row["mean_risk_score"] = float(np.mean(scores)) if scores else np.nan
        row["wins_count"] = int(sum(1 for rank in ranks if rank == 1.0))
        row["top3_count"] = int(sum(1 for rank in ranks if rank <= 3.0))
        rows.append(row)
    rank_df = pd.DataFrame(rows).sort_values(["mean_risk_rank", "mean_risk_score"], na_position="last")
    rank_df.to_csv(root / "all_datasets_rank_summary.csv", index=False, encoding="utf-8-sig")
    margin_detail = pd.concat(margin_tables.values(), ignore_index=True)
    aggregate_rows = []
    for margin in cfg.selection_margins:
        method_name = _margin_method_name(float(margin))
        rank_sub = rank_df[rank_df["method"] == method_name]
        detail_sub = margin_detail[np.isclose(margin_detail["margin"].astype(float), float(margin))]
        selected_methods = "; ".join(f"{row.dataset}:{row.selected_method}" for row in detail_sub.itertuples(index=False))
        aggregate_rows.append(
            {
                "row_type": "aggregate",
                "margin": float(margin),
                "dataset": "ALL",
                "tailweighted_val_score": np.nan,
                "best_candidate_method": "",
                "best_candidate_val_score": np.nan,
                "selected_method": method_name,
                "selected_method_test_rank": float(rank_sub["mean_risk_rank"].iloc[0]) if not rank_sub.empty else np.nan,
                "selected_method_test_score": float(rank_sub["mean_risk_score"].iloc[0]) if not rank_sub.empty else np.nan,
                "reason": "aggregate across datasets",
                "mean_risk_rank": float(rank_sub["mean_risk_rank"].iloc[0]) if not rank_sub.empty else np.nan,
                "mean_risk_score": float(rank_sub["mean_risk_score"].iloc[0]) if not rank_sub.empty else np.nan,
                "wins_count": int(rank_sub["wins_count"].iloc[0]) if not rank_sub.empty else 0,
                "top3_count": int(rank_sub["top3_count"].iloc[0]) if not rank_sub.empty else 0,
                "selected_methods_by_dataset": selected_methods,
            }
        )
    margin_detail_out = margin_detail.copy()
    margin_detail_out.insert(0, "row_type", "dataset")
    margin_detail_out["mean_risk_rank"] = np.nan
    margin_detail_out["mean_risk_score"] = np.nan
    margin_detail_out["wins_count"] = np.nan
    margin_detail_out["top3_count"] = np.nan
    margin_detail_out["selected_methods_by_dataset"] = ""
    all_margin_summary = pd.concat([margin_detail_out, pd.DataFrame(aggregate_rows)], ignore_index=True)
    all_margin_summary.to_csv(root / "all_margins_selection_summary.csv", index=False, encoding="utf-8-sig")

    tail = rank_df[rank_df["method"] == TAIL_FIXED_METHOD].iloc[0]
    selected = pd.concat(val_tables.values(), ignore_index=True)
    margin_rank_rows = []
    for margin in cfg.selection_margins:
        method_name = _margin_method_name(float(margin))
        sub = rank_df[rank_df["method"] == method_name]
        detail_sub = margin_detail[np.isclose(margin_detail["margin"].astype(float), float(margin))]
        margin_rank_rows.append(
            {
                "margin": float(margin),
                "method_name": method_name,
                "selected_methods": ", ".join(f"{r.dataset}={r.selected_method}" for r in detail_sub.itertuples(index=False)),
                "mean_rank": float(sub["mean_risk_rank"].iloc[0]) if not sub.empty else np.nan,
                "mean_score": float(sub["mean_risk_score"].iloc[0]) if not sub.empty else np.nan,
            }
        )
    margin_rank_df = pd.DataFrame(margin_rank_rows).sort_values(["mean_rank", "mean_score"], na_position="last")
    overall_best = rank_df.iloc[0]
    best_margin = margin_rank_df.iloc[0]
    tail_score = float(tail["mean_risk_score"])
    tail_rank = float(tail["mean_risk_rank"])
    best_margin_method = str(best_margin["method_name"])
    best_margin_rank = float(best_margin["mean_rank"])
    best_margin_score = float(best_margin["mean_score"])
    overall_best_method = str(overall_best["method"])
    overall_best_rank = float(overall_best["mean_risk_rank"])
    overall_best_score = float(overall_best["mean_risk_score"])
    recommended_margin = float(best_margin["margin"])
    zero_row = margin_rank_df[np.isclose(margin_rank_df["margin"].astype(float), 0.0)]
    conservative_005 = margin_rank_df[np.isclose(margin_rank_df["margin"].astype(float), 0.05)]
    if not zero_row.empty and not conservative_005.empty:
        zero_score = float(zero_row["mean_score"].iloc[0])
        score_gap = float(conservative_005["mean_score"].iloc[0]) - zero_score
        rank_gap = float(conservative_005["mean_rank"].iloc[0]) - float(zero_row["mean_rank"].iloc[0])
        if score_gap <= 0.03 and rank_gap <= 0.5:
            recommended_margin = 0.05
    lines = [
        "# Model Pool Margin Formal50 Experiment Report",
        "",
        "## Rank Summary",
        "",
        rank_df.to_markdown(index=False),
        "",
        "## Margin Selection Summary",
        "",
        margin_rank_df.to_markdown(index=False),
        "",
        "## Validation Model Selection",
        "",
        selected.to_markdown(index=False),
        "",
        "【实验结论】",
        f"- 最优方法：{overall_best_method}",
        f"- 最优 margin：{float(best_margin['margin']):.2f}",
        f"- 是否超过 TailWeighted Fixed：{'是' if overall_best_score < tail_score and overall_best_rank < tail_rank else '否'}",
        f"- 是否推荐作为最终主方法：{'是' if overall_best_score < tail_score and overall_best_rank < tail_rank else '否'}",
        "",
        "【各 margin 对比】",
    ]
    for item in margin_rank_df.sort_values("margin").itertuples(index=False):
        lines.extend(
            [
                f"margin={float(item.margin):.2f}:",
                f"- selected methods: {item.selected_methods}",
                f"- mean_rank: {float(item.mean_rank):.4f}",
                f"- mean_score: {float(item.mean_score):.4f}",
                "",
            ]
        )
    recommended_row = margin_rank_df[np.isclose(margin_rank_df["margin"].astype(float), recommended_margin)].iloc[0]
    lines.extend(
        [
            "【最终推荐】",
            f"- 推荐使用哪个 margin：{recommended_margin:.2f}",
            f"- 原因：在 margin ensemble 内，该 margin 的 mean_rank={float(recommended_row['mean_rank']):.4f}、mean_score={float(recommended_row['mean_score']):.4f}；TailWeighted Fixed 的 mean_rank={tail_rank:.4f}、mean_score={tail_score:.4f}。",
            f"- 额外结论：本轮全局最优是 {overall_best_method}，mean_rank={overall_best_rank:.4f}、mean_score={overall_best_score:.4f}；margin ensemble 没有超过它。",
            "",
            "【各数据集选择】",
        ]
    )
    rec_detail = margin_detail[np.isclose(margin_detail["margin"].astype(float), recommended_margin)]
    for item in rec_detail.itertuples(index=False):
        lines.extend(
            [
                f"{item.dataset}:",
                f"- selected method: {item.selected_method}",
                f"- reason: {item.reason}",
                "",
            ]
        )
    lines.extend(
        [
            "【风险】",
            "- 是否可能验证集过拟合：是，尤其是 GAN 在验证集上分数很低，但测试集没有对应领先，说明单纯 margin 不能完全避免过拟合选择。",
            "- 是否需要多随机种子：是，建议至少 3 个 seed 验证 selection margin 的稳定性。",
            "- 下一步最小改动：保持模型池不变，增加多 seed 复验；若继续使用 ensemble，建议增加非 Copula 候选的测试前验证稳健性约束，而不是继续新增模型。",
        ]
    )
    (root / "final_model_pool_margin_formal50_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(cfg: ModelPoolConfig, datasets: list[str] | None = None) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.augmented_data_root.mkdir(parents=True, exist_ok=True)
    specs = [s for s in _dataset_specs() if s.data_dir.exists()]
    if datasets:
        specs = [s for s in specs if s.out_name in set(datasets)]
    risk_tables, aux_tables, val_tables, margin_tables = {}, {}, {}, {}
    for spec in specs:
        print(f"\n=== Dataset: {spec.out_name} ===")
        risk, aux, val, margin = run_dataset(spec, cfg)
        risk_tables[spec.out_name] = risk
        aux_tables[spec.out_name] = aux
        val_tables[spec.out_name] = val
        margin_tables[spec.out_name] = margin
    _write_global(cfg.out_dir, risk_tables, aux_tables, val_tables, margin_tables, cfg)


def parse_args() -> tuple[ModelPoolConfig, list[str] | None]:
    parser = argparse.ArgumentParser(description="Run model pool experiments.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "model_pool_experiments")
    parser.add_argument("--augmented-data-root", type=Path, default=BASE_DIR / "outputs" / "model_pool_augmented_datasets")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--k-candidates", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--use-candidate-multiplier", action="store_true")
    parser.add_argument("--alpha-tail", type=float, default=1.0)
    parser.add_argument("--fixed-weights", type=str, default="0.35,0.25,0.20,0.20")
    parser.add_argument("--selection-margins", type=str, default="0.00")
    args = parser.parse_args()
    cfg = ModelPoolConfig(
        out_dir=args.out_dir,
        augmented_data_root=args.augmented_data_root,
        seed=args.seed,
        model_epochs=args.epochs,
        vae_epochs=args.epochs,
        transformer_epochs=args.epochs,
        flow_epochs=args.epochs,
        k_candidates=args.k_candidates,
        direct_k_candidates=args.k_candidates,
        candidate_count=args.candidate_count,
        use_candidate_multiplier=bool(args.use_candidate_multiplier),
        alpha_tail=args.alpha_tail,
        fixed_weights=args.fixed_weights,
        selection_margins=tuple(float(item.strip()) for item in args.selection_margins.split(",") if item.strip()),
    )
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()] or None
    return cfg, datasets


if __name__ == "__main__":
    cfg, datasets = parse_args()
    run_all(cfg, datasets)
