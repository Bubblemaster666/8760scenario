from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Dataset

from kmeans_markov_monthly_rebalance import (
    Build8760Config as KMeansBuildConfig,
    _load_profiles,
    _month_vector_365,
    build_background_8760,
)


@dataclass
class GanPipelineConfig:
    output_dir: str = "gan_background_outputs"
    kmeans_output_dir: str = "background_8760_outputs"
    reuse_gan_csv: str = ""
    random_seed: int = 42
    start_time: str = "2025-01-01 00:00:00"

    # Step 1: remove extreme windows
    extreme_q_net: float = 0.97
    extreme_q_peak: float = 0.98
    extreme_q_ramp: float = 0.98
    extreme_q_load: float = 0.97
    extreme_q_resource_low: float = 0.05
    extreme_buffer_days: int = 1

    # Step 2/3/6
    n_day_types: int = 6
    state_transition_weight: float = 0.78
    continuity_top_k: int = 14
    continuity_temp: float = 0.24
    smooth_hours: int = 4
    pool_per_month: int = 420

    # paper-inspired representation for day-state modeling
    use_paper_embed_for_state: bool = True
    embed_dim: int = 16
    embed_stage1_epochs: int = 16
    embed_stage3_epochs: int = 16
    embed_batch_size: int = 256
    embed_lr: float = 8e-4
    embed_recon_weight: float = 1.0
    embed_cluster_weight: float = 0.9
    state_embed_weight: float = 0.85
    state_feature_weight: float = 1.0
    state_auto_weight_search: bool = False
    state_embed_weight_grid: tuple[float, ...] = (0.45, 0.65, 0.85, 1.05, 1.25)

    # Step 5: joint conditional WGAN-GP
    z_dim: int = 64
    month_emb_dim: int = 8
    batch_size: int = 256
    epochs: int = 55
    auto_tune_trials: int = 3
    n_critic: int = 4
    gp_lambda: float = 10.0
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    device: str = "cpu"

    # compare metrics
    acf_max_lag: int = 168
    cost_thermal: float = 420.0
    cost_ramp: float = 95.0
    cost_curtail: float = 130.0

    # postprocess refinement
    use_postprocess_refine: bool = True
    refine_random_trials: int = 260
    refine_objective: str = "max_ratio"  # max_ratio / weighted_ratio / multi_metric_norm
    use_cost_match_candidates: bool = True
    cost_match_top_n: int = 14
    cost_match_max_abs_shift: float = 0.06


class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x: np.ndarray) -> "SimpleStandardScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.scale_ = x.std(axis=0, keepdims=True)
        self.scale_ = np.maximum(self.scale_, 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_


class SimpleKMeans:
    def __init__(self, n_clusters: int, random_seed: int = 42, n_init: int = 20, max_iter: int = 120):
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.n_init = n_init
        self.max_iter = max_iter
        self.centers_ = None

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        n = x.shape[0]
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        for _ in range(self.n_init):
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            centers = x[idx].copy()
            for _ in range(self.max_iter):
                dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                labels = dist2.argmin(axis=1)
                new_centers = np.zeros_like(centers)
                for k in range(self.n_clusters):
                    mk = labels == k
                    if mk.any():
                        new_centers[k] = x[mk].mean(axis=0)
                    else:
                        far_idx = np.argmax(dist2.min(axis=1))
                        new_centers[k] = x[far_idx]
                if np.linalg.norm(new_centers - centers) < 1e-6:
                    centers = new_centers
                    break
                centers = new_centers
            dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist2.argmin(axis=1)
            inertia = ((x - centers[labels]) ** 2).sum()
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()
        self.centers_ = best_centers
        return best_labels.astype(np.int32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        dist2 = ((x[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return dist2.argmin(axis=1).astype(np.int32)


def _cluster_validity_scores(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> dict:
    """Lightweight DBI/CHI/SSE quality scores for k-means clustering."""
    n = int(x.shape[0])
    k = int(centers.shape[0])
    if n <= k or k <= 1:
        return {"dbi": np.inf, "chi": 0.0, "sse": np.inf}

    sse = 0.0
    scatters = np.zeros(k, dtype=np.float64)
    counts = np.zeros(k, dtype=np.int32)
    for i in range(k):
        mk = labels == i
        ni = int(np.sum(mk))
        counts[i] = ni
        if ni == 0:
            scatters[i] = 0.0
            continue
        diff = x[mk] - centers[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        scatters[i] = float(np.mean(dist))
        sse += float(np.sum(diff * diff))

    center_dist = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    dbi_terms = []
    for i in range(k):
        if counts[i] == 0:
            continue
        rij = []
        for j in range(k):
            if i == j or counts[j] == 0:
                continue
            denom = max(float(center_dist[i, j]), 1e-12)
            rij.append((scatters[i] + scatters[j]) / denom)
        if rij:
            dbi_terms.append(max(rij))
    dbi = float(np.mean(dbi_terms)) if dbi_terms else np.inf

    mu = np.mean(x, axis=0)
    between = 0.0
    for i in range(k):
        if counts[i] == 0:
            continue
        between += float(counts[i]) * float(np.sum((centers[i] - mu) ** 2))
    within = float(max(sse, 1e-12))
    chi = float((between / max(k - 1, 1)) / (within / max(n - k, 1)))
    return {"dbi": dbi, "chi": chi, "sse": float(sse)}


def _stratified_split_idx(labels: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    tr = []
    va = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * test_ratio)))
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    tr_idx = np.concatenate(tr) if tr else np.array([], dtype=np.int32)
    va_idx = np.concatenate(va) if va else np.array([], dtype=np.int32)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    return tr_idx, va_idx


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _day_features(day_profiles: np.ndarray) -> np.ndarray:
    load = day_profiles[:, :, 0]
    wind = day_profiles[:, :, 1]
    solar = day_profiles[:, :, 2]
    net = load - wind - solar
    ramp = np.abs(np.diff(net, axis=1))
    return np.column_stack(
        [
            load.sum(axis=1),
            wind.sum(axis=1),
            solar.sum(axis=1),
            net.sum(axis=1),
            net.max(axis=1),
            ramp.max(axis=1),
            ramp.mean(axis=1),
        ]
    )


def _detect_extreme_days(day_profiles: np.ndarray, cfg: GanPipelineConfig) -> np.ndarray:
    load = day_profiles[:, :, 0]
    wind = day_profiles[:, :, 1]
    solar = day_profiles[:, :, 2]
    net = load - wind - solar

    load_e = load.sum(axis=1)
    res_e = wind.sum(axis=1) + solar.sum(axis=1)
    net_e = net.sum(axis=1)
    net_peak = net.max(axis=1)
    net_ramp = np.abs(np.diff(net, axis=1)).max(axis=1)

    mask = (
        (net_e >= np.quantile(net_e, cfg.extreme_q_net))
        | (net_peak >= np.quantile(net_peak, cfg.extreme_q_peak))
        | (net_ramp >= np.quantile(net_ramp, cfg.extreme_q_ramp))
        | ((load_e >= np.quantile(load_e, cfg.extreme_q_load)) & (res_e <= np.quantile(res_e, cfg.extreme_q_resource_low)))
    )

    if cfg.extreme_buffer_days > 0:
        spread = mask.copy()
        idx = np.where(mask)[0]
        for t in idx:
            l = max(0, t - cfg.extreme_buffer_days)
            r = min(mask.size, t + cfg.extreme_buffer_days + 1)
            spread[l:r] = True
        mask = spread
    return mask


class ChannelMinMaxScaler:
    def __init__(self, q_low: float = 0.002, q_high: float = 0.998):
        self.q_low = q_low
        self.q_high = q_high
        self.lo = None
        self.hi = None

    def fit(self, x: np.ndarray) -> "ChannelMinMaxScaler":
        self.lo = np.quantile(x, self.q_low, axis=(0, 1))
        self.hi = np.quantile(x, self.q_high, axis=(0, 1))
        self.hi = np.maximum(self.hi, self.lo + 1e-6)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = 2.0 * (x - self.lo[None, None, :]) / (self.hi[None, None, :] - self.lo[None, None, :]) - 1.0
        return np.clip(y, -1.0, 1.0)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        x = 0.5 * (y + 1.0) * (self.hi[None, None, :] - self.lo[None, None, :]) + self.lo[None, None, :]
        return np.clip(x, 0.0, None)


class DayDataset(Dataset):
    def __init__(self, x: np.ndarray, m: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.m = torch.tensor(m.astype(np.int64) - 1, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.m[idx]


def _spline_fill_missing(day_profiles: np.ndarray) -> np.ndarray:
    out = day_profiles.copy()
    t_grid = np.arange(out.shape[1], dtype=np.float64)
    for i in range(out.shape[0]):
        for c in range(out.shape[2]):
            x = out[i, :, c]
            if not np.isnan(x).any():
                continue
            ok = np.where(np.isfinite(x))[0]
            if ok.size == 0:
                out[i, :, c] = 0.0
                continue
            if ok.size == 1:
                out[i, :, c] = x[ok[0]]
                continue
            try:
                cs = CubicSpline(t_grid[ok], x[ok], bc_type="natural")
                y = cs(t_grid)
            except Exception:
                y = np.interp(t_grid, t_grid[ok], x[ok])
            out[i, :, c] = y
    return out


def _zscore_norm_days(day_profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = day_profiles.mean(axis=(0, 1), keepdims=True)
    sd = day_profiles.std(axis=(0, 1), keepdims=True)
    sd = np.maximum(sd, 1e-8)
    x = (day_profiles - mu) / sd
    x = np.clip(x, -4.0, 4.0)
    return x.astype(np.float32), mu.reshape(3), sd.reshape(3)


def _zscore_apply_days(day_profiles: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = (day_profiles - mu[None, None, :]) / sd[None, None, :]
    return np.clip(x, -4.0, 4.0).astype(np.float32)


class STConvAttnAE(nn.Module):
    def __init__(self, emb_dim: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 1), stride=(1, 2, 1), padding=(1, 1, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), stride=(1, 2, 1), padding=(1, 1, 0)),
            nn.LeakyReLU(0.2),
        )
        self.time_attn = nn.Conv1d(32, 1, kernel_size=1)
        self.space_attn = nn.Conv1d(32, 1, kernel_size=1)
        self.emb_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 6, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, emb_dim),
        )
        self.dec_in = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32 * 3 * 6),
            nn.LeakyReLU(0.2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 4, 1), stride=(1, 2, 1), padding=(1, 1, 0)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, 16, kernel_size=(3, 4, 1), stride=(1, 2, 1), padding=(1, 1, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 1, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.Tanh(),
        )

    def _to_3d(self, x: torch.Tensor) -> torch.Tensor:
        # [B,24,3] -> [B,1,3,24,1]
        return x.permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)

    def _to_day(self, y: torch.Tensor) -> torch.Tensor:
        # [B,1,3,24,1] -> [B,24,3]
        return y.squeeze(1).squeeze(-1).permute(0, 2, 1)

    def _st_attend(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,C,3,6,1]
        x = h.squeeze(-1)  # [B,C,3,6]
        t_feat = x.mean(dim=2)  # [B,C,6]
        s_feat = x.mean(dim=3)  # [B,C,3]
        t_w = torch.softmax(self.time_attn(t_feat), dim=-1)  # [B,1,6]
        s_w = torch.softmax(self.space_attn(s_feat), dim=-1)  # [B,1,3]
        return h * t_w[:, :, None, :, None] * s_w[:, :, :, None, None]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self._to_3d(x))
        h = self._st_attend(h)
        return self.emb_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z).view(z.size(0), 32, 3, 6, 1)
        y = self.dec(h)
        return self._to_day(y)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        y = self.decode(z)
        return y, z


def _student_t_distribution(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    dist2 = torch.cdist(z, centers, p=2) ** 2
    q = 1.0 / (1.0 + dist2 / alpha)
    q = q ** ((alpha + 1.0) / 2.0)
    q = q / torch.clamp(q.sum(dim=1, keepdim=True), min=1e-12)
    return q


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    f = torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
    p = (q ** 2) / f
    p = p / torch.clamp(p.sum(dim=1, keepdim=True), min=1e-12)
    return p


def _train_paper_embedder_three_stage(
    regular_day: np.ndarray,
    cfg: GanPipelineConfig,
    out_dir: Path,
) -> tuple[np.ndarray, dict]:
    x_fill = _spline_fill_missing(regular_day)
    x_norm, mu, sd = _zscore_norm_days(x_fill)
    ds = DayDataset(x_norm, np.ones(x_norm.shape[0], dtype=np.int32))
    loader = DataLoader(ds, batch_size=cfg.embed_batch_size, shuffle=True, drop_last=False)
    device = torch.device(cfg.device)

    model = STConvAttnAE(emb_dim=cfg.embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.embed_lr, betas=(0.9, 0.99))
    logs = []

    # Stage 1: reconstruction pre-train
    model.train()
    for ep in range(cfg.embed_stage1_epochs):
        loss_sum = 0.0
        n = 0
        for xb, _ in loader:
            xb = xb.to(device)
            rec, _ = model(xb)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n += 1
        logs.append({"stage": 1, "epoch": ep + 1, "loss_rec": loss_sum / max(n, 1), "loss_kl": 0.0, "loss_total": loss_sum / max(n, 1)})

    # Stage 2: initialize centers by KMeans on latent embeddings
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_norm, dtype=torch.float32, device=device)
        z_all = model.encode(xb).cpu().numpy()
    km = SimpleKMeans(n_clusters=cfg.n_day_types, random_seed=cfg.random_seed, n_init=25, max_iter=120)
    pseudo = km.fit_predict(z_all).astype(np.int64)
    centers = torch.tensor(km.centers_, dtype=torch.float32, device=device, requires_grad=True)

    # Stage 3: joint optimize reconstruction + clustering KL
    opt_joint = torch.optim.Adam(list(model.parameters()) + [centers], lr=cfg.embed_lr * 0.7, betas=(0.9, 0.99))
    idx_all = np.arange(x_norm.shape[0], dtype=np.int64)
    bs = cfg.embed_batch_size
    model.train()
    for ep in range(cfg.embed_stage3_epochs):
        # update target distribution on full-set each epoch
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(x_norm, dtype=torch.float32, device=device)
            z_full = model.encode(xb)
            q_full = _student_t_distribution(z_full, centers)
            p_full = _target_distribution(q_full).detach()
        model.train()

        np.random.shuffle(idx_all)
        loss_rec_sum = 0.0
        loss_kl_sum = 0.0
        loss_tot_sum = 0.0
        n = 0
        for st in range(0, idx_all.size, bs):
            ix = idx_all[st : st + bs]
            xb = torch.tensor(x_norm[ix], dtype=torch.float32, device=device)
            pb = p_full[ix]
            rec, z = model(xb)
            q = _student_t_distribution(z, centers)
            loss_rec = F.mse_loss(rec, xb)
            loss_kl = F.kl_div(torch.log(torch.clamp(q, min=1e-12)), pb, reduction="batchmean")
            loss = cfg.embed_recon_weight * loss_rec + cfg.embed_cluster_weight * loss_kl
            opt_joint.zero_grad(set_to_none=True)
            loss.backward()
            opt_joint.step()
            loss_rec_sum += float(loss_rec.item())
            loss_kl_sum += float(loss_kl.item())
            loss_tot_sum += float(loss.item())
            n += 1
        logs.append(
            {
                "stage": 3,
                "epoch": ep + 1,
                "loss_rec": loss_rec_sum / max(n, 1),
                "loss_kl": loss_kl_sum / max(n, 1),
                "loss_total": loss_tot_sum / max(n, 1),
            }
        )

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_norm, dtype=torch.float32, device=device)
        z_final = model.encode(xb).cpu().numpy()
    pd.DataFrame(logs).to_csv(out_dir / "paper_embed_train_log.csv", index=False, encoding="utf-8-sig")

    bundle = {
        "method": "paper_3dconv_stattn_three_stage",
        "model": model,
        "norm_mu": mu.astype(np.float32),
        "norm_sd": sd.astype(np.float32),
    }
    return z_final, bundle


def _embed_day_profiles(day_profiles: np.ndarray, bundle: dict, cfg: GanPipelineConfig) -> np.ndarray:
    x = _spline_fill_missing(day_profiles)
    xz = _zscore_apply_days(x, bundle["norm_mu"], bundle["norm_sd"])
    device = torch.device(cfg.device)
    model = bundle["model"].to(device)
    model.eval()
    out = []
    bs = max(128, min(1024, cfg.embed_batch_size))
    with torch.no_grad():
        for st in range(0, xz.shape[0], bs):
            xb = torch.tensor(xz[st : st + bs], dtype=torch.float32, device=device)
            zb = model.encode(xb).cpu().numpy()
            out.append(zb)
    return np.vstack(out)


class CondGenerator(nn.Module):
    def __init__(self, z_dim: int, month_emb_dim: int):
        super().__init__()
        self.month_emb = nn.Embedding(12, month_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(z_dim + month_emb_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 24 * 64),
            nn.LeakyReLU(0.2),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, month_idx: torch.Tensor) -> torch.Tensor:
        emb = self.month_emb(month_idx)
        h = self.fc(torch.cat([z, emb], dim=1))
        h = h.view(h.size(0), 64, 24)
        y = self.conv(h).permute(0, 2, 1)
        return y


class CondCritic(nn.Module):
    def __init__(self, month_emb_dim: int):
        super().__init__()
        self.month_emb = nn.Embedding(12, month_emb_dim)
        self.month_proj = nn.Linear(month_emb_dim, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(4, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 24, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, month_idx: torch.Tensor) -> torch.Tensor:
        xm = x.permute(0, 2, 1)
        emb = self.month_emb(month_idx)
        m_line = self.month_proj(emb).unsqueeze(1)
        inp = torch.cat([xm, m_line], dim=1)
        h = self.conv(inp)
        return self.head(h).squeeze(1)


def _gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    month_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, 1, device=device)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)
    out = critic(interp, month_idx)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(bsz, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def _train_one_trial(
    x_train: np.ndarray,
    m_train: np.ndarray,
    x_val: np.ndarray,
    m_val: np.ndarray,
    cfg: GanPipelineConfig,
    trial_seed: int,
    trial_idx: int,
    out_dir: Path,
) -> tuple[dict, Dict[str, torch.Tensor]]:
    _seed_all(trial_seed)
    device = torch.device(cfg.device)
    ds = DayDataset(x_train, m_train)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    gen = CondGenerator(cfg.z_dim, cfg.month_emb_dim).to(device)
    dis = CondCritic(cfg.month_emb_dim).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(dis.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))

    logs = []
    for ep in range(cfg.epochs):
        d_sum = 0.0
        g_sum = 0.0
        n_batch = 0
        for real_x, month_idx in loader:
            real_x = real_x.to(device)
            month_idx = month_idx.to(device)
            bsz = real_x.size(0)

            for _ in range(cfg.n_critic):
                z = torch.randn(bsz, cfg.z_dim, device=device)
                fake = gen(z, month_idx).detach()
                d_real = dis(real_x, month_idx).mean()
                d_fake = dis(fake, month_idx).mean()
                gp = _gradient_penalty(dis, real_x, fake, month_idx, device)
                loss_d = d_fake - d_real + cfg.gp_lambda * gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            z = torch.randn(bsz, cfg.z_dim, device=device)
            fake = gen(z, month_idx)
            loss_g = -dis(fake, month_idx).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_sum += float(loss_d.item())
            g_sum += float(loss_g.item())
            n_batch += 1

        logs.append({"epoch": ep + 1, "loss_d": d_sum / max(n_batch, 1), "loss_g": g_sum / max(n_batch, 1)})

    pd.DataFrame(logs).to_csv(out_dir / f"train_log_trial_{trial_idx}.csv", index=False, encoding="utf-8-sig")

    gen.eval()
    with torch.no_grad():
        mv = torch.tensor(m_val.astype(np.int64) - 1, dtype=torch.long, device=device)
        zv = torch.randn(len(m_val), cfg.z_dim, device=device)
        xv = gen(zv, mv).cpu().numpy()
    score_w = np.mean([wasserstein_distance(x_val[:, :, c].reshape(-1), xv[:, :, c].reshape(-1)) for c in range(3)])
    corr_real = np.corrcoef(x_val.reshape(-1, 3).T)
    corr_fake = np.corrcoef(xv.reshape(-1, 3).T)
    score_corr = float(np.mean(np.abs(corr_real - corr_fake)))
    score = float(score_w + 0.45 * score_corr)

    info = {
        "trial_idx": trial_idx,
        "trial_seed": trial_seed,
        "score_wasserstein_norm": float(score_w),
        "score_corr_l1": score_corr,
        "score_total": score,
        "last_loss_d": float(logs[-1]["loss_d"]) if logs else np.nan,
        "last_loss_g": float(logs[-1]["loss_g"]) if logs else np.nan,
    }
    state = {"gen": {k: v.cpu().clone() for k, v in gen.state_dict().items()}}
    return info, state


def _train_gan_auto_tune(
    regular_day: np.ndarray,
    regular_month: np.ndarray,
    cfg: GanPipelineConfig,
    out_dir: Path,
) -> tuple[nn.Module, ChannelMinMaxScaler, dict]:
    scaler = ChannelMinMaxScaler().fit(regular_day)
    x_norm = scaler.transform(regular_day)
    tr_idx, va_idx = _stratified_split_idx(regular_month, test_ratio=0.15, seed=cfg.random_seed)
    x_train, x_val = x_norm[tr_idx], x_norm[va_idx]
    m_train, m_val = regular_month[tr_idx], regular_month[va_idx]

    trials = []
    states = []
    for t in range(cfg.auto_tune_trials):
        info, state = _train_one_trial(
            x_train=x_train,
            m_train=m_train,
            x_val=x_val,
            m_val=m_val,
            cfg=cfg,
            trial_seed=cfg.random_seed + 101 * t + 7,
            trial_idx=t,
            out_dir=out_dir,
        )
        trials.append(info)
        states.append(state)

    trial_df = pd.DataFrame(trials).sort_values("score_total", ascending=True).reset_index(drop=True)
    trial_df.to_csv(out_dir / "gan_trial_scores.csv", index=False, encoding="utf-8-sig")
    best_idx = int(trial_df.loc[0, "trial_idx"])

    device = torch.device(cfg.device)
    gen = CondGenerator(cfg.z_dim, cfg.month_emb_dim).to(device)
    gen.load_state_dict(states[best_idx]["gen"])
    gen.eval()

    meta = {
        "best_trial_idx": best_idx,
        "trials": trials,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
    }
    return gen, scaler, meta


def _build_state_models(
    regular_day: np.ndarray,
    regular_month: np.ndarray,
    regular_year_day_idx: np.ndarray,
    n_years: int,
    cfg: GanPipelineConfig,
    out_dir: Path,
) -> tuple[SimpleKMeans, SimpleStandardScaler, np.ndarray, np.ndarray, Optional[dict]]:
    embed_bundle = None
    fit_meta = {}
    if cfg.use_paper_embed_for_state:
        # Paper-inspired path: 3D-conv + spatiotemporal attention + three-stage training.
        emb, embed_bundle = _train_paper_embedder_three_stage(regular_day, cfg, out_dir)
        hand = _day_features(regular_day)
        n_emb = int(emb.shape[1])
        feat_raw = np.hstack([emb, hand])
        feat_scaler_base = SimpleStandardScaler().fit(feat_raw)
        feat_z_base = feat_scaler_base.transform(feat_raw)
        candidate_weights = [float(cfg.state_embed_weight)]
        if cfg.state_auto_weight_search:
            candidate_weights = sorted({float(cfg.state_embed_weight), *[float(v) for v in cfg.state_embed_weight_grid]})

        best = None
        rows = []
        for w_emb in candidate_weights:
            feat_z_i = feat_z_base.copy()
            feat_z_i[:, :n_emb] *= float(w_emb)
            feat_z_i[:, n_emb:] *= float(cfg.state_feature_weight)
            day_kmeans_i = SimpleKMeans(n_clusters=cfg.n_day_types, random_seed=cfg.random_seed, n_init=28)
            reg_state_i = day_kmeans_i.fit_predict(feat_z_i).astype(np.int32)
            q = _cluster_validity_scores(feat_z_i, reg_state_i, day_kmeans_i.centers_)
            balance = np.bincount(reg_state_i, minlength=cfg.n_day_types).astype(np.float64) / max(reg_state_i.size, 1)
            balance_penalty = float(np.std(balance))
            score = float(q["dbi"] / max(np.log1p(q["chi"]), 1e-12) + 0.12 * balance_penalty)
            rec = {
                "embed_weight": float(w_emb),
                "feature_weight": float(cfg.state_feature_weight),
                "dbi": float(q["dbi"]),
                "chi": float(q["chi"]),
                "sse": float(q["sse"]),
                "balance_std": balance_penalty,
                "score": score,
            }
            rows.append(rec)
            if (best is None) or (score < best["score"]):
                best = {
                    "score": score,
                    "embed_weight": float(w_emb),
                    "day_kmeans": day_kmeans_i,
                    "reg_state": reg_state_i,
                }

        day_kmeans = best["day_kmeans"]
        feat_scaler = feat_scaler_base
        reg_state = best["reg_state"]
        fit_meta = {
            "state_embed_weight": float(best["embed_weight"]),
            "state_feature_weight": float(cfg.state_feature_weight),
            "state_weight_candidates": rows,
        }
        embed_bundle["state_embed_weight"] = float(best["embed_weight"])
        embed_bundle["state_feature_weight"] = float(cfg.state_feature_weight)
        embed_bundle["state_n_embed"] = n_emb
        embed_bundle["state_weight_candidates"] = rows
    else:
        feat = _day_features(regular_day)
        feat_scaler = SimpleStandardScaler().fit(feat)
        feat_z = feat_scaler.transform(feat)
        day_kmeans = SimpleKMeans(n_clusters=cfg.n_day_types, random_seed=cfg.random_seed, n_init=30)
        reg_state = day_kmeans.fit_predict(feat_z).astype(np.int32)

    state_yd = np.full((n_years, 365), -1, dtype=np.int32)
    for i in range(regular_day.shape[0]):
        y, d = regular_year_day_idx[i]
        state_yd[y, d] = reg_state[i]

    month_doy = _month_vector_365()
    month_freq = np.zeros((12, cfg.n_day_types), dtype=np.float64)
    for m in range(1, 13):
        mask = regular_month == m
        cnt = np.bincount(reg_state[mask], minlength=cfg.n_day_types).astype(np.float64)
        month_freq[m - 1] = cnt / max(cnt.sum(), 1.0)

    trans = np.zeros((12, cfg.n_day_types, cfg.n_day_types), dtype=np.float64)
    for y in range(n_years):
        for d in range(1, 365):
            s0 = state_yd[y, d - 1]
            s1 = state_yd[y, d]
            if s0 < 0 or s1 < 0:
                continue
            m = month_doy[d] - 1
            trans[m, s0, s1] += 1.0

    for m in range(12):
        for s in range(cfg.n_day_types):
            rs = trans[m, s].sum()
            if rs > 0:
                trans[m, s] /= rs
            else:
                trans[m, s] = month_freq[m]
    if fit_meta:
        pd.DataFrame(fit_meta["state_weight_candidates"]).to_csv(out_dir / "paper_state_weight_search.csv", index=False, encoding="utf-8-sig")
    return day_kmeans, feat_scaler, month_freq, trans, embed_bundle


def _sample_state_path(month_doy: np.ndarray, month_freq: np.ndarray, month_trans: np.ndarray, weight: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(month_doy.size, dtype=np.int32)
    n_states = month_freq.shape[1]
    for d in range(month_doy.size):
        m = month_doy[d] - 1
        if d == 0:
            p = month_freq[m]
        else:
            p = weight * month_trans[m, out[d - 1]] + (1.0 - weight) * month_freq[m]
        p = np.clip(p, 0.0, None)
        p = p / max(p.sum(), 1e-12)
        out[d] = int(rng.choice(n_states, p=p))
    return out


def _generate_month_pools(gen: nn.Module, scaler: ChannelMinMaxScaler, cfg: GanPipelineConfig) -> Dict[int, np.ndarray]:
    device = torch.device(cfg.device)
    pools: Dict[int, np.ndarray] = {}
    gen.eval()
    with torch.no_grad():
        for m in range(1, 13):
            month_idx = torch.full((cfg.pool_per_month,), m - 1, dtype=torch.long, device=device)
            z = torch.randn(cfg.pool_per_month, cfg.z_dim, device=device)
            y = gen(z, month_idx).cpu().numpy()
            x = scaler.inverse_transform(y)
            night_hours = np.array([0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23], dtype=np.int32)
            x[:, night_hours, 2] = 0.0
            pools[m] = np.clip(x, 0.0, None)
    return pools


def _choose_days_from_pool(
    pools: Dict[int, np.ndarray],
    pool_states: Dict[int, np.ndarray],
    month_doy: np.ndarray,
    state_path: np.ndarray,
    cfg: GanPipelineConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_days = month_doy.size
    out_daily = np.zeros((n_days, 24, 3), dtype=np.float64)
    out_choice = np.zeros(n_days, dtype=np.int32)
    scale = np.array([0.08, 0.08, 0.05], dtype=np.float64)

    for d in range(n_days):
        m = int(month_doy[d])
        s = int(state_path[d])
        pool = pools[m]
        states = pool_states[m]
        cands = np.where(states == s)[0]
        if cands.size == 0:
            cands = np.arange(pool.shape[0], dtype=np.int32)

        if d == 0:
            idx = int(rng.choice(cands))
        else:
            prev_last = out_daily[d - 1, -1, :]
            start = pool[cands, 0, :]
            score = np.sum(np.abs(start - prev_last[None, :]) / scale[None, :], axis=1)
            k = min(cfg.continuity_top_k, cands.size)
            if cands.size > k:
                keep = np.argpartition(score, k - 1)[:k]
                cands = cands[keep]
                score = score[keep]
            score = score - score.min()
            w = np.exp(-score / max(cfg.continuity_temp, 1e-6))
            if np.isfinite(w).all() and w.sum() > 0:
                w /= w.sum()
                idx = int(rng.choice(cands, p=w))
            else:
                idx = int(rng.choice(cands))

        out_daily[d] = pool[idx]
        out_choice[d] = idx
    return out_daily, out_choice


def _smooth_boundaries(hourly: np.ndarray, smooth_hours: int) -> np.ndarray:
    if smooth_hours <= 0:
        return hourly
    out = hourly.copy()
    for b in range(24, out.shape[0], 24):
        k = min(smooth_hours, out.shape[0] - b)
        if k <= 0:
            continue
        taper = np.linspace(1.0, 0.0, k, endpoint=False)
        delta = out[b] - out[b - 1]
        out[b : b + k] -= taper[:, None] * delta[None, :]
    return np.clip(out, 0.0, None)


def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    out = np.ones(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return out


def _js_divergence(x: np.ndarray, y: np.ndarray, bins: int = 100) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if hi <= lo:
        return 0.0
    px, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    py, _ = np.histogram(y, bins=edges, density=True)
    px = np.clip(px, 1e-12, None)
    py = np.clip(py, 1e-12, None)
    px = px / px.sum()
    py = py / py.sum()
    m = 0.5 * (px + py)
    return float(0.5 * np.sum(px * np.log(px / m)) + 0.5 * np.sum(py * np.log(py / m)))


def _annual_cost(seq: np.ndarray, cfg: GanPipelineConfig) -> float:
    net = seq[:, 0] - seq[:, 1] - seq[:, 2]
    thermal = np.maximum(net, 0.0).sum()
    curtail = np.maximum(-net, 0.0).sum()
    ramp = np.abs(np.diff(net)).sum()
    return float(cfg.cost_thermal * thermal + cfg.cost_curtail * curtail + cfg.cost_ramp * ramp)


def _monthly_rebalance(hourly: np.ndarray, hist_y: np.ndarray, month_doy: np.ndarray, strength: float = 0.9, clip_min: float = 0.75, clip_max: float = 1.25) -> np.ndarray:
    out = hourly.copy()
    month_hour = np.repeat(month_doy, 24)
    for c in range(3):
        h_m = np.array([np.median(hist_y[:, month_hour == m, c].sum(axis=1)) for m in range(1, 13)], dtype=np.float64)
        g_m = np.array([out[month_hour == m, c].sum() for m in range(1, 13)], dtype=np.float64)
        ratio = np.clip(h_m / (g_m + 1e-12), clip_min, clip_max)
        scale = 1.0 + strength * (ratio - 1.0)
        for m in range(1, 13):
            out[month_hour == m, c] *= scale[m - 1]
    return np.clip(out, 0.0, None)


def _monthly_quantile_map(hourly: np.ndarray, hist_y: np.ndarray, month_doy: np.ndarray, strength: float = 0.5, preserve_month_sum: bool = True) -> np.ndarray:
    out = hourly.copy()
    month_hour = np.repeat(month_doy, 24)
    for m in range(1, 13):
        mk = month_hour == m
        for c in range(3):
            g = out[mk, c]
            h = hist_y[:, mk, c].reshape(-1)
            if g.size < 8 or h.size < 32:
                continue
            before = float(g.sum())
            gs = np.sort(g)
            cdf = np.searchsorted(gs, g, side="right").astype(np.float64) / max(len(gs), 1)
            cdf = np.clip(cdf, 0.005, 0.995)
            target = np.quantile(h, cdf)
            mapped = (1.0 - strength) * g + strength * target
            if preserve_month_sum and before > 0:
                mapped *= before / max(float(mapped.sum()), 1e-12)
            out[mk, c] = mapped
    return np.clip(out, 0.0, None)


def _monthly_net_correction(hourly: np.ndarray, hist_y: np.ndarray, month_doy: np.ndarray, strength: float = 0.75) -> np.ndarray:
    out = hourly.copy()
    month_hour = np.repeat(month_doy, 24)
    for m in range(1, 13):
        mk = month_hour == m
        l = float(out[mk, 0].sum())
        w = float(out[mk, 1].sum())
        s = float(out[mk, 2].sum())
        l_t = float(np.median(hist_y[:, mk, 0].sum(axis=1)))
        w_t = float(np.median(hist_y[:, mk, 1].sum(axis=1)))
        n_t = float(np.median((hist_y[:, mk, 0] - hist_y[:, mk, 1] - hist_y[:, mk, 2]).sum(axis=1)))

        A = np.array([[l, 0], [0, w], [l, -w], [1, 0], [0, 1]], dtype=np.float64)
        y = np.array([l_t, w_t, n_t + s, 1.0, 1.0], dtype=np.float64)
        W = np.diag([1.0, 1.0, 2.4, 0.35, 0.35])
        sol, _, _, _ = np.linalg.lstsq(W @ A, W @ y, rcond=None)
        a = 1.0 + strength * (float(sol[0]) - 1.0)
        b = 1.0 + strength * (float(sol[1]) - 1.0)
        a = float(np.clip(a, 0.93, 1.07))
        b = float(np.clip(b, 0.85, 1.15))
        out[mk, 0] *= a
        out[mk, 1] *= b
    return np.clip(out, 0.0, None)


def _monthly_joint_correction(hourly: np.ndarray, hist_y: np.ndarray, month_doy: np.ndarray, strength: float = 0.85) -> np.ndarray:
    out = hourly.copy()
    month_hour = np.repeat(month_doy, 24)
    for m in range(1, 13):
        mk = month_hour == m
        l = float(out[mk, 0].sum())
        w = float(out[mk, 1].sum())
        s = float(out[mk, 2].sum())
        l_t = float(np.median(hist_y[:, mk, 0].sum(axis=1)))
        w_t = float(np.median(hist_y[:, mk, 1].sum(axis=1)))
        s_t = float(np.median(hist_y[:, mk, 2].sum(axis=1)))
        n_t = float(np.median((hist_y[:, mk, 0] - hist_y[:, mk, 1] - hist_y[:, mk, 2]).sum(axis=1)))

        A = np.array(
            [
                [l, 0, 0],
                [0, w, 0],
                [0, 0, s],
                [l, -w, -s],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        y = np.array([l_t, w_t, s_t, n_t, 1.0, 1.0, 1.0], dtype=np.float64)
        W = np.diag([1.0, 1.0, 1.0, 2.8, 0.32, 0.32, 0.32])
        sol, _, _, _ = np.linalg.lstsq(W @ A, W @ y, rcond=None)
        a = 1.0 + strength * (float(sol[0]) - 1.0)
        b = 1.0 + strength * (float(sol[1]) - 1.0)
        c = 1.0 + strength * (float(sol[2]) - 1.0)
        a = float(np.clip(a, 0.90, 1.10))
        b = float(np.clip(b, 0.78, 1.22))
        c = float(np.clip(c, 0.78, 1.22))
        out[mk, 0] *= a
        out[mk, 1] *= b
        out[mk, 2] *= c
    return np.clip(out, 0.0, None)


def _std_shrink_adjust(
    hourly: np.ndarray,
    hist_y: np.ndarray,
    month_doy: np.ndarray,
    strength: float = 0.1,
    by_month: bool = True,
    only_over: bool = True,
) -> np.ndarray:
    out = hourly.reshape(365, 24, 3).copy()
    hist_net = hist_y[:, :, 0] - hist_y[:, :, 1] - hist_y[:, :, 2]
    hist_stack = np.concatenate([hist_y, hist_net[:, :, None]], axis=2)
    h_day = hist_stack.reshape(hist_stack.shape[0], 365, 24, 4)
    target_std = np.stack(
        [np.median(np.stack([h_day[y, :, :, i].std(axis=0) for y in range(h_day.shape[0])], axis=0), axis=0) for i in range(4)],
        axis=0,
    )

    out_net = out[:, :, 0] - out[:, :, 1] - out[:, :, 2]
    out4 = np.concatenate([out, out_net[:, :, None]], axis=2)
    if by_month:
        for m in range(1, 13):
            d_idx = np.where(month_doy == m)[0]
            for c in range(3):
                for h in range(24):
                    vals = out4[d_idx, h, c]
                    mu = float(vals.mean())
                    sd = float(vals.std())
                    if sd < 1e-9:
                        continue
                    ratio = float(target_std[c, h] / sd)
                    if only_over:
                        ratio = min(1.0, ratio)
                    gamma = 1.0 + strength * (ratio - 1.0)
                    out4[d_idx, h, c] = mu + gamma * (vals - mu)
    else:
        for c in range(3):
            for h in range(24):
                vals = out4[:, h, c]
                mu = float(vals.mean())
                sd = float(vals.std())
                if sd < 1e-9:
                    continue
                ratio = float(target_std[c, h] / sd)
                if only_over:
                    ratio = min(1.0, ratio)
                gamma = 1.0 + strength * (ratio - 1.0)
                out4[:, h, c] = mu + gamma * (vals - mu)
    return np.clip(out4[:, :, :3].reshape(365 * 24, 3), 0.0, None)


def _prepare_postprocess_refs(hist_y: np.ndarray, cfg: GanPipelineConfig) -> dict:
    hist_net = hist_y[:, :, 0] - hist_y[:, :, 1] - hist_y[:, :, 2]
    hist_stack = np.concatenate([hist_y, hist_net[:, :, None]], axis=2)
    h_day = hist_stack.reshape(hist_stack.shape[0], 365, 24, 4)
    hist_acf_med = np.stack(
        [np.median(np.stack([_acf_1d(hist_stack[y, :, i], cfg.acf_max_lag) for y in range(hist_stack.shape[0])], axis=0), axis=0) for i in range(4)],
        axis=0,
    )
    hist_std_med = np.stack(
        [np.median(np.stack([h_day[y, :, :, i].std(axis=0) for y in range(h_day.shape[0])], axis=0), axis=0) for i in range(4)],
        axis=0,
    )
    hist_corr = np.corrcoef(hist_stack.reshape(-1, 4).T)
    hist_cost = np.array([_annual_cost(hist_y[y], cfg) for y in range(hist_y.shape[0])], dtype=np.float64)
    cost_target = float(np.median(hist_cost))
    return {
        "hist_stack": hist_stack,
        "hist_acf_med": hist_acf_med,
        "hist_std_med": hist_std_med,
        "hist_corr": hist_corr,
        "cost_target": cost_target,
    }


def _postprocess_metrics(seq: np.ndarray, hist_y: np.ndarray, cfg: GanPipelineConfig, refs: dict | None = None) -> dict:
    if refs is None:
        refs = _prepare_postprocess_refs(hist_y, cfg)
    hist_stack = refs["hist_stack"]
    hist_acf_med = refs["hist_acf_med"]
    hist_std_med = refs["hist_std_med"]
    hist_corr = refs["hist_corr"]
    cost_target = refs["cost_target"]

    seq_net = seq[:, 0] - seq[:, 1] - seq[:, 2]
    ss = np.concatenate([seq, seq_net[:, None]], axis=1)
    ss_day = ss.reshape(365, 24, 4)

    w_mean = float(np.mean([wasserstein_distance(ss[:, i], hist_stack[:, :, i].reshape(-1)) for i in range(4)]))
    js_mean = float(np.mean([_js_divergence(ss[:, i], hist_stack[:, :, i].reshape(-1)) for i in range(4)]))
    acf_mean = float(np.mean([np.mean(np.abs(_acf_1d(ss[:, i], cfg.acf_max_lag) - hist_acf_med[i])) for i in range(4)]))
    std_mae_mean = float(np.mean([np.mean(np.abs(ss_day[:, :, i].std(axis=0) - hist_std_med[i])) for i in range(4)]))
    corr_l1 = float(np.mean(np.abs(np.corrcoef(ss.T) - hist_corr)))
    cost_rel = float(abs(_annual_cost(seq, cfg) - cost_target) / (cost_target + 1e-12))
    return {
        "w_mean": w_mean,
        "js_mean": js_mean,
        "acf_mean": acf_mean,
        "std_mae_mean": std_mae_mean,
        "corr_l1": corr_l1,
        "cost_rel": cost_rel,
    }


def _daily_library_projection(
    hourly: np.ndarray,
    hist_y: np.ndarray,
    month_doy: np.ndarray,
    cfg: GanPipelineConfig,
    alpha: float = 0.4,
    top_k: int = 8,
    temp: float = 0.12,
) -> np.ndarray:
    seq_day = hourly.reshape(365, 24, 3).copy()
    hist_day = hist_y.reshape(-1, 24, 3)
    hist_month = np.tile(month_doy, hist_y.shape[0])

    # Restrict library to regular days, so we do not re-inject extreme behavior.
    regular_mask = ~_detect_extreme_days(hist_day, cfg)
    hist_day = hist_day[regular_mask]
    hist_month = hist_month[regular_mask]
    if hist_day.shape[0] < 365:
        return hourly.copy()

    feat_hist = _day_features(hist_day)
    feat_seq = _day_features(seq_day)
    mu = feat_hist.mean(axis=0, keepdims=True)
    sd = np.maximum(feat_hist.std(axis=0, keepdims=True), 1e-8)
    feat_hist_z = (feat_hist - mu) / sd
    feat_seq_z = (feat_seq - mu) / sd

    out = seq_day.copy()
    rng = np.random.default_rng(cfg.random_seed + 9173)
    prev_net_end = None
    for d in range(365):
        m = month_doy[d]
        ids = np.where(hist_month == m)[0]
        if ids.size == 0:
            continue

        dist = ((feat_hist_z[ids] - feat_seq_z[d]) ** 2).sum(axis=1)
        if prev_net_end is not None:
            net_open = hist_day[ids, 0, 0] - hist_day[ids, 0, 1] - hist_day[ids, 0, 2]
            scale = np.maximum(np.std(net_open), 1e-6)
            dist = dist + 0.35 * np.abs(net_open - prev_net_end) / scale

        k = int(min(max(top_k, 1), ids.size))
        if k == 1:
            chosen = int(ids[int(np.argmin(dist))])
        else:
            top_local = np.argpartition(dist, k - 1)[:k]
            top_dist = dist[top_local]
            weights = np.exp(-(top_dist - top_dist.min()) / max(float(temp), 1e-6))
            weights = weights / np.maximum(weights.sum(), 1e-12)
            picked = int(rng.choice(top_local, p=weights))
            chosen = int(ids[picked])

        out[d] = (1.0 - alpha) * seq_day[d] + alpha * hist_day[chosen]
        prev_net_end = float(out[d, -1, 0] - out[d, -1, 1] - out[d, -1, 2])
    return np.clip(out.reshape(365 * 24, 3), 0.0, None)


def _collect_candidate_metrics(candidates: Dict[str, np.ndarray], hist_y: np.ndarray, cfg: GanPipelineConfig, refs: dict) -> pd.DataFrame:
    rows = []
    for k, seq in candidates.items():
        m = _postprocess_metrics(seq, hist_y, cfg, refs=refs)
        rows.append({"candidate": k, **m})
    return pd.DataFrame(rows)


def _apply_cost_shift(seq: np.ndarray, delta: float, mode: str) -> np.ndarray:
    out = seq.copy()
    if mode == "load":
        out[:, 0] += delta
    elif mode == "split":
        eta = 0.55
        out[:, 0] += eta * delta
        ren = out[:, 1] + out[:, 2]
        share_w = np.where(ren > 1e-9, out[:, 1] / ren, 0.5)
        out[:, 1] -= (1.0 - eta) * delta * share_w
        out[:, 2] -= (1.0 - eta) * delta * (1.0 - share_w)
    else:
        raise ValueError(f"unsupported cost shift mode: {mode}")
    return np.clip(out, 0.0, None)


def _annual_cost_match_shift(
    seq: np.ndarray,
    cfg: GanPipelineConfig,
    cost_target: float,
    mode: str = "load",
    max_abs_shift: float = 0.06,
) -> tuple[np.ndarray, float, float]:
    """Search a small global net-shift surrogate to better align annual cost."""
    best_seq = seq
    best_delta = 0.0
    best_gap = abs(_annual_cost(seq, cfg) - cost_target)

    for delta in np.linspace(-max_abs_shift, max_abs_shift, 41):
        cand = _apply_cost_shift(seq, float(delta), mode=mode)
        gap = abs(_annual_cost(cand, cfg) - cost_target)
        if gap < best_gap:
            best_gap = float(gap)
            best_delta = float(delta)
            best_seq = cand

    # Local coordinate refinement around the best coarse point.
    step = max_abs_shift / 8.0
    for _ in range(10):
        improved = False
        for sgn in (-1.0, 1.0):
            delta_try = float(np.clip(best_delta + sgn * step, -max_abs_shift, max_abs_shift))
            cand = _apply_cost_shift(seq, delta_try, mode=mode)
            gap = abs(_annual_cost(cand, cfg) - cost_target)
            if gap + 1e-12 < best_gap:
                best_gap = float(gap)
                best_delta = delta_try
                best_seq = cand
                improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return best_seq, best_delta, best_gap


def _postprocess_score(seq: np.ndarray, hist_y: np.ndarray, cfg: GanPipelineConfig) -> dict:
    m = _postprocess_metrics(seq, hist_y, cfg)
    total = float(m["w_mean"] + 0.35 * m["acf_mean"] + 0.25 * m["cost_rel"])
    return {
        "score_total": total,
        "score_w_mean": m["w_mean"],
        "score_acf_mean": m["acf_mean"],
        "score_cost_rel": m["cost_rel"],
        "score_js_mean": m["js_mean"],
        "score_std_mae_mean": m["std_mae_mean"],
        "score_corr_l1": m["corr_l1"],
    }


def _refine_sequence(
    hourly: np.ndarray,
    hist_y: np.ndarray,
    month_doy: np.ndarray,
    cfg: GanPipelineConfig,
    out_dir: Path,
    km_hourly: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if not cfg.use_postprocess_refine:
        return hourly, {"chosen": "raw"}

    rng = np.random.default_rng(cfg.random_seed + 25001)
    candidates = {"raw": hourly}
    candidates["reb_0.75"] = _monthly_rebalance(hourly, hist_y, month_doy, strength=0.75)
    candidates["reb_0.90_q_0.35"] = _monthly_quantile_map(_monthly_rebalance(hourly, hist_y, month_doy, 0.90), hist_y, month_doy, strength=0.35)
    candidates["reb_0.90_q_0.50"] = _monthly_quantile_map(_monthly_rebalance(hourly, hist_y, month_doy, 0.90), hist_y, month_doy, strength=0.50)
    candidates["reb_0.90_q_0.50_net"] = _monthly_net_correction(candidates["reb_0.90_q_0.50"], hist_y, month_doy, strength=0.75)
    candidates["reb_0.90_q_0.50_net_proj_a0.30"] = _monthly_quantile_map(
        _daily_library_projection(candidates["reb_0.90_q_0.50_net"], hist_y, month_doy, cfg, alpha=0.30),
        hist_y,
        month_doy,
        strength=0.30,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.45"] = _monthly_quantile_map(
        _daily_library_projection(candidates["reb_0.90_q_0.50_net"], hist_y, month_doy, cfg, alpha=0.45),
        hist_y,
        month_doy,
        strength=0.30,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.30_net2"] = _monthly_quantile_map(
        _monthly_net_correction(candidates["reb_0.90_q_0.50_net_proj_a0.30"], hist_y, month_doy, strength=0.90),
        hist_y,
        month_doy,
        strength=0.20,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.45_net2"] = _monthly_quantile_map(
        _monthly_net_correction(candidates["reb_0.90_q_0.50_net_proj_a0.45"], hist_y, month_doy, strength=0.90),
        hist_y,
        month_doy,
        strength=0.20,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.30_joint2"] = _monthly_quantile_map(
        _monthly_joint_correction(candidates["reb_0.90_q_0.50_net_proj_a0.30"], hist_y, month_doy, strength=0.90),
        hist_y,
        month_doy,
        strength=0.15,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.45_joint2"] = _monthly_quantile_map(
        _monthly_joint_correction(candidates["reb_0.90_q_0.50_net_proj_a0.45"], hist_y, month_doy, strength=0.90),
        hist_y,
        month_doy,
        strength=0.15,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2"] = _monthly_quantile_map(
        _monthly_joint_correction(
            _daily_library_projection(candidates["reb_0.90_q_0.50_net"], hist_y, month_doy, cfg, alpha=0.65),
            hist_y,
            month_doy,
            strength=0.90,
        ),
        hist_y,
        month_doy,
        strength=0.15,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2_joint3_s1.50_q0.35"] = _monthly_quantile_map(
        _monthly_joint_correction(candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2"], hist_y, month_doy, strength=1.50),
        hist_y,
        month_doy,
        strength=0.35,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2_joint3_s1.45_q0.35"] = _monthly_quantile_map(
        _monthly_joint_correction(candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2"], hist_y, month_doy, strength=1.45),
        hist_y,
        month_doy,
        strength=0.35,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2_joint3_s1.50_q0.325"] = _monthly_quantile_map(
        _monthly_joint_correction(candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2"], hist_y, month_doy, strength=1.50),
        hist_y,
        month_doy,
        strength=0.325,
    )
    candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2_net3_s1.50_q0.35"] = _monthly_quantile_map(
        _monthly_net_correction(candidates["reb_0.90_q_0.50_net_proj_a0.65_joint2"], hist_y, month_doy, strength=1.50),
        hist_y,
        month_doy,
        strength=0.35,
    )

    # Automatic local search around strong seeds:
    # target 24h-std improvement while pulling back cost and preserving distribution/ACF.
    if cfg.refine_random_trials > 0:
        seed_keys = [
            "reb_0.90_q_0.50_net_proj_a0.65_joint2_joint3_s1.50_q0.35",
            "reb_0.90_q_0.50_net_proj_a0.65_joint2_net3_s1.50_q0.35",
            "reb_0.90_q_0.50_net_proj_a0.45_net2",
            "reb_0.90_q_0.50_net_proj_a0.45_joint2",
            "reb_0.90_q_0.50_net_proj_a0.65_joint2",
        ]
        for i in range(cfg.refine_random_trials):
            seed = candidates[str(seed_keys[int(rng.integers(0, len(seed_keys)))])]
            s = float(rng.uniform(0.02, 0.35))
            by_month = bool(rng.integers(0, 2))
            only_over = bool(rng.integers(0, 2))
            seq = _std_shrink_adjust(seed, hist_y, month_doy, strength=s, by_month=by_month, only_over=only_over)

            corr_mode = str(rng.choice(["none", "net", "joint"]))
            corr_s = float(rng.uniform(0.15, 1.80))
            if corr_mode == "net":
                seq = _monthly_net_correction(seq, hist_y, month_doy, strength=corr_s)
            elif corr_mode == "joint":
                seq = _monthly_joint_correction(seq, hist_y, month_doy, strength=corr_s)

            q = float(rng.uniform(0.0, 0.25))
            if q > 1e-8:
                seq = _monthly_quantile_map(seq, hist_y, month_doy, strength=q)
            candidates[f"search_{i:04d}_s{s:.3f}_{corr_mode}"] = seq

    refs = _prepare_postprocess_refs(hist_y, cfg)
    df = _collect_candidate_metrics(candidates, hist_y, cfg, refs)
    metric_cols = ["w_mean", "js_mean", "acf_mean", "std_mae_mean", "corr_l1", "cost_rel"]

    chosen_mode = "multi_metric_norm"
    if km_hourly is not None:
        km_m = _postprocess_metrics(km_hourly, hist_y, cfg, refs=refs)
        ratio_weights = {
            "w_mean": 1.00,
            "js_mean": 1.00,
            "acf_mean": 1.50,
            "std_mae_mean": 1.20,
            "corr_l1": 1.00,
            "cost_rel": 2.00,
        }
        for c in metric_cols:
            df[f"ratio_{c}"] = df[c] / max(float(km_m[c]), 1e-12)
        df["max_ratio"] = df[[f"ratio_{c}" for c in metric_cols]].max(axis=1)
        wsum = float(sum(ratio_weights.values()))
        df["weighted_ratio"] = sum(ratio_weights[c] * df[f"ratio_{c}"] for c in metric_cols) / wsum

        # Cost-aware augmentation: add shifted candidates around promising seeds.
        if cfg.use_cost_match_candidates and not df.empty:
            pick_n = int(max(cfg.cost_match_top_n, 1))
            seed_keys = set(df.nsmallest(pick_n, "corr_l1")["candidate"].astype(str).tolist())
            seed_keys.update(df.nsmallest(pick_n, "max_ratio")["candidate"].astype(str).tolist())
            seed_keys.update(df.nsmallest(pick_n, "cost_rel")["candidate"].astype(str).tolist())
            for key in seed_keys:
                seq0 = candidates[key]
                for mode in ("load", "split"):
                    adj, delta, _ = _annual_cost_match_shift(
                        seq0,
                        cfg=cfg,
                        cost_target=refs["cost_target"],
                        mode=mode,
                        max_abs_shift=float(cfg.cost_match_max_abs_shift),
                    )
                    candidates[f"{key}_cost_{mode}_{delta:+.4f}"] = adj
            df = _collect_candidate_metrics(candidates, hist_y, cfg, refs)
            for c in metric_cols:
                df[f"ratio_{c}"] = df[c] / max(float(km_m[c]), 1e-12)
            df["max_ratio"] = df[[f"ratio_{c}" for c in metric_cols]].max(axis=1)
            df["weighted_ratio"] = sum(ratio_weights[c] * df[f"ratio_{c}"] for c in metric_cols) / wsum

        obj = str(cfg.refine_objective).lower()
        # Guardrail: prefer candidates that avoid clear regressions on cost/correlation.
        guarded = df[(df["ratio_cost_rel"] <= 1.08) & (df["ratio_corr_l1"] <= 1.10)].copy()
        if guarded.empty:
            guarded = df[(df["ratio_cost_rel"] <= 1.35) & (df["ratio_corr_l1"] <= 1.35)].copy()
        if guarded.empty:
            guarded = df.copy()

        if obj == "weighted_ratio":
            chosen_mode = "weighted_ratio"
            sorted_rows = guarded.sort_values(["weighted_ratio", "max_ratio", "w_mean"], ascending=True)
        elif obj == "multi_metric_norm":
            chosen_mode = "multi_metric_norm"
            for c in metric_cols:
                cmin = float(guarded[c].min())
                cmax = float(guarded[c].max())
                guarded[f"norm_{c}"] = (guarded[c] - cmin) / max(cmax - cmin, 1e-12)
            norm_weights = {"w_mean": 1.0, "js_mean": 1.0, "acf_mean": 1.8, "std_mae_mean": 1.0, "corr_l1": 0.8, "cost_rel": 4.0}
            nsum = float(sum(norm_weights.values()))
            guarded["score_total"] = sum(norm_weights[c] * guarded[f"norm_{c}"] for c in metric_cols) / nsum
            sorted_rows = guarded.sort_values(["score_total", "max_ratio", "weighted_ratio"], ascending=True)
        else:
            chosen_mode = "max_ratio"
            sorted_rows = guarded.sort_values(["max_ratio", "weighted_ratio", "w_mean"], ascending=True)
    else:
        for c in metric_cols:
            cmin = float(df[c].min())
            cmax = float(df[c].max())
            df[f"norm_{c}"] = (df[c] - cmin) / max(cmax - cmin, 1e-12)
        norm_weights = {"w_mean": 1.0, "js_mean": 1.0, "acf_mean": 1.8, "std_mae_mean": 1.0, "corr_l1": 0.8, "cost_rel": 4.0}
        nsum = float(sum(norm_weights.values()))
        df["score_total"] = sum(norm_weights[c] * df[f"norm_{c}"] for c in metric_cols) / nsum
        sorted_rows = df.sort_values("score_total", ascending=True)

    best_key = str(sorted_rows.iloc[0]["candidate"])
    sorted_rows.to_csv(out_dir / "gan_refine_candidates.csv", index=False, encoding="utf-8-sig")
    return candidates[best_key], {"chosen": best_key, "chosen_mode": chosen_mode, "candidates": sorted_rows.to_dict(orient="records")}


def _plot_corr_matrix(ax, mat: np.ndarray, labels: list[str], title: str) -> None:
    ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)


def _run_comparison(hist_y: np.ndarray, gan_hourly: np.ndarray, km_hourly: np.ndarray, cfg: GanPipelineConfig, out_dir: Path) -> dict:
    names = ["load", "wind", "solar", "net_load"]
    hist_net = hist_y[:, :, 0] - hist_y[:, :, 1] - hist_y[:, :, 2]
    gan_net = gan_hourly[:, 0] - gan_hourly[:, 1] - gan_hourly[:, 2]
    km_net = km_hourly[:, 0] - km_hourly[:, 1] - km_hourly[:, 2]
    hist_stack = np.concatenate([hist_y, hist_net[:, :, None]], axis=2)
    gan_stack = np.concatenate([gan_hourly, gan_net[:, None]], axis=1)
    km_stack = np.concatenate([km_hourly, km_net[:, None]], axis=1)

    # Wasserstein / JS
    rows = []
    for i, n in enumerate(names):
        h = hist_stack[:, :, i].reshape(-1)
        g = gan_stack[:, i]
        k = km_stack[:, i]
        rows.extend(
            [
                {"metric": "Wasserstein", "series": n, "method": "GAN", "value": wasserstein_distance(g, h)},
                {"metric": "Wasserstein", "series": n, "method": "KMeans", "value": wasserstein_distance(k, h)},
                {"metric": "JS", "series": n, "method": "GAN", "value": _js_divergence(g, h)},
                {"metric": "JS", "series": n, "method": "KMeans", "value": _js_divergence(k, h)},
            ]
        )
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(out_dir / "compare_distance_metrics.csv", index=False, encoding="utf-8-sig")

    # ACF
    lags = np.arange(cfg.acf_max_lag + 1)
    acf_rows = []
    fig = plt.figure(figsize=(12, 9))
    for i, n in enumerate(names):
        ax = fig.add_subplot(2, 2, i + 1)
        h_acf = np.stack([_acf_1d(hist_stack[y, :, i], cfg.acf_max_lag) for y in range(hist_stack.shape[0])], axis=0)
        h_med = np.median(h_acf, axis=0)
        g_acf = _acf_1d(gan_stack[:, i], cfg.acf_max_lag)
        k_acf = _acf_1d(km_stack[:, i], cfg.acf_max_lag)
        ax.plot(lags, h_med, label="Historical")
        ax.plot(lags, g_acf, label="GAN")
        ax.plot(lags, k_acf, label="KMeans")
        ax.set_title(f"ACF - {n}")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend()
        acf_rows.extend(
            [
                {"series": n, "method": "GAN", "acf_mae": float(np.mean(np.abs(g_acf - h_med)))},
                {"series": n, "method": "KMeans", "acf_mae": float(np.mean(np.abs(k_acf - h_med)))},
            ]
        )
    fig.tight_layout()
    fig.savefig(out_dir / "compare_acf.png", dpi=180)
    plt.close(fig)
    acf_df = pd.DataFrame(acf_rows)
    acf_df.to_csv(out_dir / "compare_acf_mae.csv", index=False, encoding="utf-8-sig")

    # 24-point std
    std_rows = []
    h_day = hist_stack.reshape(hist_stack.shape[0], 365, 24, 4)
    g_day = gan_stack.reshape(365, 24, 4)
    k_day = km_stack.reshape(365, 24, 4)
    fig = plt.figure(figsize=(12, 9))
    for i, n in enumerate(names):
        ax = fig.add_subplot(2, 2, i + 1)
        h_std = np.stack([h_day[y, :, :, i].std(axis=0) for y in range(h_day.shape[0])], axis=0)
        h_med = np.median(h_std, axis=0)
        g_std = g_day[:, :, i].std(axis=0)
        k_std = k_day[:, :, i].std(axis=0)
        ax.plot(np.arange(24), h_med, label="Historical")
        ax.plot(np.arange(24), g_std, label="GAN")
        ax.plot(np.arange(24), k_std, label="KMeans")
        ax.set_title(f"24h Std - {n}")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend()
        std_rows.extend(
            [
                {"series": n, "method": "GAN", "std_curve_mae": float(np.mean(np.abs(g_std - h_med)))},
                {"series": n, "method": "KMeans", "std_curve_mae": float(np.mean(np.abs(k_std - h_med)))},
            ]
        )
    fig.tight_layout()
    fig.savefig(out_dir / "compare_24h_std_curve.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(std_rows).to_csv(out_dir / "compare_std_curve_mae.csv", index=False, encoding="utf-8-sig")

    # correlation matrix
    labels = ["load", "wind", "solar", "net"]
    hist_corr = np.corrcoef(hist_stack.reshape(-1, 4).T)
    gan_corr = np.corrcoef(gan_stack.T)
    km_corr = np.corrcoef(km_stack.T)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    _plot_corr_matrix(axs[0], hist_corr, labels, "Historical")
    _plot_corr_matrix(axs[1], gan_corr, labels, "GAN")
    _plot_corr_matrix(axs[2], km_corr, labels, "KMeans")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_corr_matrix.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(
        [
            {"method": "GAN", "corr_l1_error": float(np.mean(np.abs(gan_corr - hist_corr)))},
            {"method": "KMeans", "corr_l1_error": float(np.mean(np.abs(km_corr - hist_corr)))},
        ]
    ).to_csv(out_dir / "compare_corr_error.csv", index=False, encoding="utf-8-sig")

    # cost
    hist_cost = np.array([_annual_cost(hist_y[y], cfg) for y in range(hist_y.shape[0])], dtype=np.float64)
    target = float(np.median(hist_cost))
    cost_rows = [
        {"method": "Historical median", "annual_cost": target, "relative_error": 0.0},
        {"method": "GAN", "annual_cost": _annual_cost(gan_hourly, cfg), "relative_error": abs(_annual_cost(gan_hourly, cfg) - target) / (target + 1e-12)},
        {"method": "KMeans", "annual_cost": _annual_cost(km_hourly, cfg), "relative_error": abs(_annual_cost(km_hourly, cfg) - target) / (target + 1e-12)},
    ]
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(out_dir / "compare_annual_cost.csv", index=False, encoding="utf-8-sig")
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    bar = cost_df[cost_df["method"] != "Historical median"]
    ax.bar(bar["method"], bar["relative_error"] * 100.0, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Annual Cost Relative Error")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_annual_cost_error.png", dpi=180)
    plt.close(fig)

    return {
        "wasserstein_mean_gan": float(dist_df[(dist_df.metric == "Wasserstein") & (dist_df.method == "GAN")]["value"].mean()),
        "wasserstein_mean_kmeans": float(dist_df[(dist_df.metric == "Wasserstein") & (dist_df.method == "KMeans")]["value"].mean()),
        "js_mean_gan": float(dist_df[(dist_df.metric == "JS") & (dist_df.method == "GAN")]["value"].mean()),
        "js_mean_kmeans": float(dist_df[(dist_df.metric == "JS") & (dist_df.method == "KMeans")]["value"].mean()),
        "acf_mae_mean_gan": float(acf_df[acf_df.method == "GAN"]["acf_mae"].mean()),
        "acf_mae_mean_kmeans": float(acf_df[acf_df.method == "KMeans"]["acf_mae"].mean()),
        "cost_rel_err_gan": float(cost_df[cost_df.method == "GAN"]["relative_error"].iloc[0]),
        "cost_rel_err_kmeans": float(cost_df[cost_df.method == "KMeans"]["relative_error"].iloc[0]),
    }


def run_pipeline(cfg: GanPipelineConfig) -> dict:
    _seed_all(cfg.random_seed)
    rng = np.random.default_rng(cfg.random_seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_y_d_h, wind_y_d_h, solar_y_d_h, source_meta = _load_profiles(KMeansBuildConfig())
    n_years = load_y_d_h.shape[0]
    hist_y = np.stack([load_y_d_h, wind_y_d_h, solar_y_d_h], axis=-1).reshape(n_years, 365 * 24, 3)
    day_profiles = hist_y.reshape(n_years * 365, 24, 3)
    month_doy = _month_vector_365()
    month_flat = np.tile(month_doy, n_years)
    year_day_idx = np.stack([np.repeat(np.arange(n_years), 365), np.tile(np.arange(365), n_years)], axis=1)

    # Step 1: remove extreme windows
    extreme_mask = _detect_extreme_days(day_profiles, cfg)
    regular_mask = ~extreme_mask
    regular_day = day_profiles[regular_mask]
    regular_month = month_flat[regular_mask]
    regular_year_day = year_day_idx[regular_mask]

    pd.DataFrame(
        {
            "flat_day_index": np.where(extreme_mask)[0],
            "year_index": year_day_idx[extreme_mask, 0],
            "day_of_year": year_day_idx[extreme_mask, 1] + 1,
            "month": month_flat[extreme_mask],
        }
    ).to_csv(out_dir / "extreme_days_removed.csv", index=False, encoding="utf-8-sig")

    train_meta = {}
    refine_meta = {}
    if cfg.reuse_gan_csv:
        reuse_path = Path(cfg.reuse_gan_csv)
        reuse_df = pd.read_csv(reuse_path)
        hourly_raw = reuse_df[["load", "wind_power", "solar_power"]].to_numpy(dtype=np.float64)
        state_path = np.full(365, -1, dtype=np.int32)
        day_choice = np.full(365, -1, dtype=np.int32)
        train_meta = {"reused_gan_csv": str(reuse_path)}
    else:
        # Step 2/3/4/5
        gen, scaler, train_meta = _train_gan_auto_tune(regular_day, regular_month, cfg, out_dir)
        day_kmeans, day_feat_scaler, month_freq, month_trans, embed_bundle = _build_state_models(
            regular_day=regular_day,
            regular_month=regular_month,
            regular_year_day_idx=regular_year_day,
            n_years=n_years,
            cfg=cfg,
            out_dir=out_dir,
        )

        pools = _generate_month_pools(gen, scaler, cfg)
        pool_states = {}
        for m in range(1, 13):
            if embed_bundle is not None:
                emb = _embed_day_profiles(pools[m], embed_bundle, cfg)
                w_emb = float(embed_bundle.get("state_embed_weight", 1.0))
                w_feat = float(embed_bundle.get("state_feature_weight", 1.0))
                feat = np.hstack([emb, _day_features(pools[m])])
                feat_z = day_feat_scaler.transform(feat)
                n_emb = int(embed_bundle.get("state_n_embed", emb.shape[1]))
                feat_z[:, :n_emb] *= w_emb
                feat_z[:, n_emb:] *= w_feat
            else:
                feat = _day_features(pools[m])
                feat_z = day_feat_scaler.transform(feat)
            pool_states[m] = day_kmeans.predict(feat_z).astype(np.int32)

        # Step 6: build annual sequence
        state_path = _sample_state_path(month_doy, month_freq, month_trans, cfg.state_transition_weight, rng)
        daily_seq, day_choice = _choose_days_from_pool(pools, pool_states, month_doy, state_path, cfg, rng)
        hourly_raw = daily_seq.reshape(365 * 24, 3)

    kmeans_csv = Path(cfg.kmeans_output_dir) / "background_8760.csv"
    if not kmeans_csv.exists():
        build_background_8760(KMeansBuildConfig(output_dir=cfg.kmeans_output_dir))
    km_df = pd.read_csv(kmeans_csv)
    km_hourly = km_df[["load", "wind_power", "solar_power"]].to_numpy(dtype=np.float64)

    hourly = _smooth_boundaries(hourly_raw, cfg.smooth_hours)
    hourly, refine_meta = _refine_sequence(hourly, hist_y, month_doy, cfg, out_dir, km_hourly=km_hourly)

    time = pd.date_range(pd.Timestamp(cfg.start_time), periods=365 * 24, freq="1h")
    gan_df = pd.DataFrame({"time": time, "load": hourly[:, 0], "wind_power": hourly[:, 1], "solar_power": hourly[:, 2]})
    gan_df["net_load"] = gan_df["load"] - gan_df["wind_power"] - gan_df["solar_power"]
    gan_df["month"] = np.repeat(month_doy, 24)
    gan_df.to_csv(out_dir / "gan_background_8760.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "day_index": np.arange(1, 366),
            "month": month_doy,
            "state_id": state_path,
            "pool_choice_index": day_choice,
        }
    ).to_csv(out_dir / "gan_background_day_meta.csv", index=False, encoding="utf-8-sig")

    compare_summary = _run_comparison(hist_y, hourly, km_hourly, cfg, out_dir)

    summary = {
        "output_dir": str(out_dir.resolve()),
        "n_hist_years": int(n_years),
        "n_regular_days": int(regular_day.shape[0]),
        "n_extreme_removed_days": int(extreme_mask.sum()),
        "train_meta": train_meta,
        "refine_meta": refine_meta,
        "compare_summary": compare_summary,
    }
    (out_dir / "gan_pipeline_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "gan_pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "gan_source_meta.json").write_text(json.dumps(source_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== GAN pipeline done ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved key files:")
    print("- gan_background_8760.csv")
    print("- extreme_days_removed.csv")
    if not cfg.reuse_gan_csv:
        print("- gan_trial_scores.csv")
        if cfg.use_paper_embed_for_state:
            print("- paper_embed_train_log.csv")
            if cfg.state_auto_weight_search:
                print("- paper_state_weight_search.csv")
    print("- gan_refine_candidates.csv")
    print("- compare_wasserstein_js.png")
    print("- compare_acf.png")
    print("- compare_24h_std_curve.png")
    print("- compare_corr_matrix.png")
    print("- compare_annual_cost_error.png")
    return summary


def parse_args() -> GanPipelineConfig:
    p = argparse.ArgumentParser(description="GAN-based regular background 8760 generation + KMeans comparison.")
    p.add_argument("--output-dir", type=str, default="gan_background_outputs")
    p.add_argument("--kmeans-output-dir", type=str, default="background_8760_outputs")
    p.add_argument("--reuse-gan-csv", type=str, default="")
    p.add_argument("--start-time", type=str, default="2025-01-01 00:00:00")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=55)
    p.add_argument("--auto-tune-trials", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-critic", type=int, default=4)
    p.add_argument("--pool-per-month", type=int, default=420)
    p.add_argument("--disable-paper-embed-for-state", action="store_true")
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--embed-stage1-epochs", type=int, default=16)
    p.add_argument("--embed-stage3-epochs", type=int, default=16)
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embed-lr", type=float, default=8e-4)
    p.add_argument("--embed-cluster-weight", type=float, default=0.9)
    p.add_argument("--embed-recon-weight", type=float, default=1.0)
    p.add_argument("--state-embed-weight", type=float, default=0.85)
    p.add_argument("--state-feature-weight", type=float, default=1.0)
    p.add_argument("--enable-state-auto-weight-search", action="store_true")
    p.add_argument("--disable-postprocess-refine", action="store_true")
    p.add_argument("--refine-random-trials", type=int, default=260)
    p.add_argument("--refine-objective", type=str, default="max_ratio", choices=["max_ratio", "weighted_ratio", "multi_metric_norm"])
    p.add_argument("--disable-cost-match-candidates", action="store_true")
    p.add_argument("--cost-match-top-n", type=int, default=14)
    p.add_argument("--cost-match-max-abs-shift", type=float, default=0.06)
    p.add_argument("--device", type=str, default="cpu")
    a = p.parse_args()
    return GanPipelineConfig(
        output_dir=a.output_dir,
        kmeans_output_dir=a.kmeans_output_dir,
        reuse_gan_csv=a.reuse_gan_csv,
        start_time=a.start_time,
        random_seed=a.random_seed,
        epochs=a.epochs,
        auto_tune_trials=a.auto_tune_trials,
        batch_size=a.batch_size,
        n_critic=a.n_critic,
        pool_per_month=a.pool_per_month,
        use_paper_embed_for_state=not a.disable_paper_embed_for_state,
        embed_dim=a.embed_dim,
        embed_stage1_epochs=a.embed_stage1_epochs,
        embed_stage3_epochs=a.embed_stage3_epochs,
        embed_batch_size=a.embed_batch_size,
        embed_lr=a.embed_lr,
        embed_cluster_weight=a.embed_cluster_weight,
        embed_recon_weight=a.embed_recon_weight,
        state_embed_weight=a.state_embed_weight,
        state_feature_weight=a.state_feature_weight,
        state_auto_weight_search=a.enable_state_auto_weight_search,
        use_postprocess_refine=not a.disable_postprocess_refine,
        refine_random_trials=a.refine_random_trials,
        refine_objective=a.refine_objective,
        use_cost_match_candidates=not a.disable_cost_match_candidates,
        cost_match_top_n=a.cost_match_top_n,
        cost_match_max_abs_shift=a.cost_match_max_abs_shift,
        device=a.device,
    )


def main() -> None:
    cfg = parse_args()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
