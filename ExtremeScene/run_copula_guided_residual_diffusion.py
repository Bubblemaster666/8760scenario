from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate_generation import EvalConfig, evaluate_generation
from hierarchical_diffusion import (
    DiffusionScheduler,
    EMA,
    HierarchicalConditionalUNet1D,
    apply_physical_projection,
    condition_dropout,
    denormalize_x_torch,
    physics_penalty,
)
from risk_metrics import soft_risk_metrics_torch
from risk_ranking_utils import (
    AUXILIARY_REALISM_METRICS,
    RISK_EVALUATION_EXPLANATION,
    RISK_MAIN_METRICS,
    RISK_RANKING_EXPLANATION,
    add_risk_score,
    write_risk_tables,
)
from run_simple_evt_risk_diffusion import (
    FULL_COMPARE_METRICS,
    SimpleEVTRiskDataset,
    _normalizers,
    evaluate_method,
    generate_simple,
    train_simple_model,
)
from traditional_statistical_extreme_baseline.traditional_copula_baseline import (
    CopulaConfig,
    TraditionalCopulaBaseline,
)


BASE_DIR = Path(__file__).resolve().parent
METHOD_NAME = "Copula_Guided_EVT_Risk_Residual_Diffusion"
PROTECTED_METHOD_NAME = "Copula_Guided_EVT_Risk_Residual_Diffusion_ValProtected"
CONSERVATIVE_METHOD_NAME = "Copula_Guided_EVT_Risk_Residual_Diffusion_ValProtectedConservative"


@dataclass
class DatasetSpec:
    name: str
    data_dir: Path
    out_name: str
    existing_results_dir: Path | None = None
    existing_sources: dict[str, Path] | None = None


@dataclass
class ResidualRunConfig:
    out_dir: Path = BASE_DIR / "results" / "copula_guided_residual_diffusion"
    device: str = "cpu"
    seed: int = 42
    stage1_epochs: int = 12
    stage2_epochs: int = 8
    batch_size: int = 32
    diffusion_steps: int = 100
    base_channels: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    ema_decay: float = 0.995
    cond_dropout: float = 0.10
    duration_temp: float = 12.0
    lambda_cum: float = 0.8
    lambda_ramp: float = 0.20
    lambda_dur: float = 0.25
    lambda_anchor: float = 0.08
    lambda_res: float = 0.10
    lambda_phy: float = 0.02
    copula_covariance_shrinkage: float = 0.08
    copula_temporal_smooth_strength: float = 0.15
    residual_scale_candidates: str = "0.0,0.25,0.5,0.75,1.0"
    residual_clip_quantile_candidates: str = "none,0.95,0.90"
    background_residual_scale_candidates: str = "1.0,0.5,0.25"
    run_g2_g3: bool = False


class CopulaResidualDataset(Dataset):
    """Copula-guided residual diffusion dataset.

    x_real: real load/wind/solar sequence, shape [N, 3, 36].
    x_base: Copula-generated base sequence with statistical dependence.
    residual_real: x_real - x_base, the EVT risk residual learned by diffusion.
    condition: simplified EVT condition from Simple_EVT_Risk_Diffusion.
    event_mask: core weather-event mask; used for background Copula anchoring.
    """

    def __init__(
        self,
        x_real: np.ndarray,
        x_base: np.ndarray,
        cond: pd.DataFrame,
        meta: pd.DataFrame,
        normalizers,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        residual_mean: np.ndarray,
        residual_std: np.ndarray,
        event_mask: np.ndarray | None,
    ) -> None:
        self.x_real = x_real.astype(np.float32)
        self.x_base = x_base.astype(np.float32)
        self.residual = (self.x_real - self.x_base).astype(np.float32)
        self.base_norm = ((self.x_base - x_mean) / x_std).astype(np.float32)
        self.residual_norm = ((self.residual - residual_mean) / residual_std).astype(np.float32)
        helper = SimpleEVTRiskDataset(
            np.zeros_like(self.x_real, dtype=np.float32),
            self.x_real,
            cond,
            meta,
            normalizers,
            event_mask,
            use_ramp_level_condition=False,
        )
        self.bg = helper.bg
        self.proc = helper.proc
        self.risk = helper.risk
        self.day_mask = helper.day_mask
        self.risk_targets = helper.risk_targets
        self.event_mask = helper.event_mask

    def __len__(self) -> int:
        return int(self.x_real.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "residual": torch.from_numpy(self.residual_norm[idx]),
            "base_norm": torch.from_numpy(self.base_norm[idx]),
            "base_raw": torch.from_numpy(self.x_base[idx]),
            "real_raw": torch.from_numpy(self.x_real[idx]),
            "bg": torch.from_numpy(self.bg[idx]),
            "proc": torch.from_numpy(self.proc[idx]),
            "risk": torch.from_numpy(self.risk[idx]),
            "risk_targets": torch.from_numpy(self.risk_targets[idx]),
            "day_mask": torch.from_numpy(self.day_mask[idx]),
            "event_mask": torch.from_numpy(self.event_mask[idx]),
        }


class CopulaResidualDenoiser(nn.Module):
    """Denoiser for residual diffusion.

    The network receives noisy_residual_t and x_base as six input channels.
    It predicts the diffusion noise of residual_real only. The final scenario
    is x_gen = x_base + residual_gen.
    """

    def __init__(self, base_channels: int, bg_dim: int, proc_dim: int, risk_dim: int) -> None:
        super().__init__()
        self.core = HierarchicalConditionalUNet1D(
            in_channels=6,
            base_channels=base_channels,
            time_dim=128,
            cond_dim=128,
            bg_dim=bg_dim,
            proc_dim=proc_dim,
            risk_dim=risk_dim,
            flat_condition=False,
            profile_channels=0,
        )

    def forward(
        self,
        noisy_residual: torch.Tensor,
        base_norm: torch.Tensor,
        t: torch.Tensor,
        bg_cond: torch.Tensor | None,
        proc_cond: torch.Tensor | None,
        risk_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        model_in = torch.cat([noisy_residual, base_norm], dim=1)
        # The wrapped U-Net emits six channels because its input has six; only
        # the first three channels are trained as residual-noise prediction.
        return self.core(model_in, t, bg_cond, proc_cond, risk_cond, None)[:, :3, :]


def _load_split(data_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
    cond = pd.read_csv(data_dir / f"cond_{split}.csv")
    meta = pd.read_csv(data_dir / f"meta_{split}.csv")
    mask_path = data_dir / f"event_mask_{split}.npy"
    mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else None
    return x, cond, meta, mask


def _risk_norm(cond_train: pd.DataFrame, device: torch.device) -> dict[str, torch.Tensor]:
    cum = pd.to_numeric(cond_train.get("cum_deficit", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ramp = pd.to_numeric(cond_train.get("netload_ramp_max", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    dur = pd.to_numeric(cond_train.get("imbalance_duration", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    log_cum = np.log1p(np.maximum(cum, 0.0))
    return {
        "log_cum_std": torch.tensor(float(log_cum.std() + 1e-6), device=device),
        "ramp_mean": torch.tensor(float(ramp.mean()), device=device),
        "ramp_std": torch.tensor(float(ramp.std() + 1e-6), device=device),
        "dur_std": torch.tensor(float(dur.std() + 1e-6), device=device),
    }


def _fit_and_sample_copula(
    data_dir: Path,
    out_dir: Path,
    cfg: ResidualRunConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x_train, cond_train, _, _ = _load_split(data_dir, "train")
    x_val, cond_val, _, _ = _load_split(data_dir, "val")
    x_test, cond_test, _, _ = _load_split(data_dir, "test")
    copula_cfg = CopulaConfig(
        data_dir=str(data_dir),
        output_dir=str(out_dir / "copula_model"),
        seed=int(cfg.seed),
        n_per_condition=1,
        seq_len=int(x_train.shape[2]),
        covariance_shrinkage=float(cfg.copula_covariance_shrinkage),
        temporal_smooth_strength=float(cfg.copula_temporal_smooth_strength),
        make_plots=False,
    )
    rng = np.random.default_rng(int(cfg.seed))
    model = TraditionalCopulaBaseline(copula_cfg)
    fit_summary = model.fit(x_train, cond_train)
    (out_dir / "copula_model").mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "copula_model" / "copula_model.pkl")
    x_base_train, _ = model.sample_conditions(cond_train, 1, rng)
    x_base_val, _ = model.sample_conditions(cond_val, 1, rng)
    x_base_test, _ = model.sample_conditions(cond_test, 1, rng)
    x_base_train = x_base_train[: len(x_train)].astype(np.float32)
    x_base_val = x_base_val[: len(x_val)].astype(np.float32)
    x_base_test = x_base_test[: len(x_test)].astype(np.float32)
    np.save(out_dir / "x_base_copula_train.npy", x_base_train)
    np.save(out_dir / "x_base_copula_val.npy", x_base_val)
    np.save(out_dir / "x_base_copula.npy", x_base_test)
    summary = {
        "fit_uses": "X_train.npy and cond_train.csv only",
        "validation_selection_uses": "cond_val.csv and X_val.npy only for residual-protection selection; X_test.npy is not used",
        "test_generation_uses": "cond_test.csv only; X_test.npy is not used by Copula sampling",
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "group_summary": fit_summary.get("group_summary", []),
        "copula_config": asdict(copula_cfg),
    }
    (out_dir / "copula_fit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return x_base_train, x_base_val, x_base_test, summary


def _make_loaders(
    data_dir: Path,
    x_base_train: np.ndarray,
    x_base_val: np.ndarray,
    x_base_test: np.ndarray,
    cfg: ResidualRunConfig,
    device: torch.device,
):
    x_train, cond_train, meta_train, mask_train = _load_split(data_dir, "train")
    x_val, cond_val, meta_val, mask_val = _load_split(data_dir, "val")
    x_test, cond_test, meta_test, mask_test = _load_split(data_dir, "test")

    x_mean = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    residual_train = (x_train - x_base_train).astype(np.float32)
    residual_mean = residual_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    residual_std = (residual_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    norms = _normalizers(cond_train)

    train_ds = CopulaResidualDataset(x_train, x_base_train, cond_train, meta_train, norms, x_mean, x_std, residual_mean, residual_std, mask_train)
    val_ds = CopulaResidualDataset(x_val, x_base_val.astype(np.float32), cond_val, meta_val, norms, x_mean, x_std, residual_mean, residual_std, mask_val)
    test_ds = CopulaResidualDataset(x_test, x_base_test, cond_test, meta_test, norms, x_mean, x_std, residual_mean, residual_std, mask_test)
    return {
        "train": DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False),
        "val": DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False),
        "val_ds": val_ds,
        "test_ds": test_ds,
        "x_mean": x_mean,
        "x_std": x_std,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_train_abs": np.abs(residual_train).astype(np.float32),
        "risk_norm": _risk_norm(cond_train, device),
        "bg_dim": train_ds.bg.shape[1],
        "proc_dim": train_ds.proc.shape[1],
        "risk_dim": train_ds.risk.shape[1],
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "test_shape": list(x_test.shape),
    }


def _anchor_loss(x_gen: torch.Tensor, x_base: torch.Tensor, x_std_t: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs((x_gen - x_base) / (x_std_t + 1e-6))
    if event_mask is not None:
        bg_mask = (1.0 - event_mask).clamp(0.0, 1.0)[:, None, :]
        denom = (bg_mask.sum() * diff.shape[1]).clamp(min=1.0)
        return (diff * bg_mask).sum() / denom
    return diff.mean()


def _p_sample_residual(
    model: CopulaResidualDenoiser,
    scheduler: DiffusionScheduler,
    residual_x: torch.Tensor,
    base_norm: torch.Tensor,
    t: torch.Tensor,
    bg: torch.Tensor,
    proc: torch.Tensor,
    risk: torch.Tensor,
) -> torch.Tensor:
    pred_noise = model(residual_x, base_norm, t, bg, proc, risk)
    beta_t = scheduler.betas[t][:, None, None]
    sqrt_one_minus_ab_t = scheduler.sqrt_one_minus_alpha_bars[t][:, None, None]
    sqrt_recip_alpha_t = scheduler.sqrt_recip_alphas[t][:, None, None]
    model_mean = sqrt_recip_alpha_t * (residual_x - beta_t * pred_noise / sqrt_one_minus_ab_t)
    var_t = scheduler.posterior_variance[t][:, None, None]
    noise = torch.randn_like(residual_x)
    nonzero_mask = (t != 0).float()[:, None, None]
    return model_mean + nonzero_mask * torch.sqrt(var_t) * noise


def _run_epoch(
    model: CopulaResidualDenoiser,
    scheduler: DiffusionScheduler,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    stage: str,
    device: torch.device,
    residual_mean_t: torch.Tensor,
    residual_std_t: torch.Tensor,
    x_std_t: torch.Tensor,
    risk_norm: dict[str, torch.Tensor],
    cfg: ResidualRunConfig,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    buckets = {k: [] for k in ["total_loss", "eps_loss", "cum_shortfall_loss", "ramp_loss", "dur_shortfall_loss", "anchor_loss", "residual_mag_loss", "physics_loss"]}
    for batch in loader:
        residual = batch["residual"].to(device)
        base_norm = batch["base_norm"].to(device)
        base_raw = batch["base_raw"].to(device)
        bg = batch["bg"].to(device)
        proc = batch["proc"].to(device)
        risk = batch["risk"].to(device)
        targets = batch["risk_targets"].to(device)
        day_mask = batch["day_mask"].to(device)
        event_mask = batch["event_mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(residual)
        t = torch.randint(0, scheduler.steps, (residual.shape[0],), device=device)
        residual_t = scheduler.q_sample(residual, t, noise)
        bg_in, proc_in, risk_in = condition_dropout(bg, proc, risk, float(cfg.cond_dropout)) if train else (bg, proc, risk)
        pred = model(residual_t, base_norm, t, bg_in, proc_in, risk_in)
        eps_loss = F.mse_loss(pred, noise)
        total = eps_loss
        cum_short = ramp_loss = dur_short = anchor = res_mag = phy = residual.new_tensor(0.0)
        if stage in {"stage2_risk", "stage2_risk_residual"}:
            residual0_norm = scheduler.predict_x0(residual_t, t, pred).clamp(-5.0, 5.0)
            residual0 = denormalize_x_torch(residual0_norm, residual_mean_t, residual_std_t)
            x_gen_raw = base_raw + residual0
            x_proj = apply_physical_projection(x_gen_raw, day_mask)
            tau = targets[:, 3]
            cum_pred, ramp_pred, dur_pred = soft_risk_metrics_torch(
                x_proj,
                tau=tau,
                delta_t_hours=1.0,
                duration_temp=float(cfg.duration_temp),
                ramp_metric_mode="window_3h",
                ramp_window_hours=3.0,
            )
            target_cum = targets[:, 0]
            target_ramp = targets[:, 1]
            target_dur = targets[:, 2]
            cum_short = torch.relu(torch.log1p(target_cum) - torch.log1p(cum_pred)).mean() / risk_norm["log_cum_std"].clamp(min=1e-6)
            ramp_loss = F.smooth_l1_loss(
                (ramp_pred - risk_norm["ramp_mean"]) / risk_norm["ramp_std"],
                (target_ramp - risk_norm["ramp_mean"]) / risk_norm["ramp_std"],
            )
            dur_short = (torch.relu(target_dur - dur_pred) / risk_norm["dur_std"].clamp(min=1e-6)).mean()
            anchor = _anchor_loss(x_proj, base_raw, x_std_t, event_mask)
            res_mag = torch.abs(residual0 / (x_std_t + 1e-6)).mean()
            phy = physics_penalty(x_gen_raw, day_mask)
            total = (
                eps_loss
                + float(cfg.lambda_cum) * cum_short
                + float(cfg.lambda_ramp) * ramp_loss
                + float(cfg.lambda_dur) * dur_short
                + float(cfg.lambda_anchor) * anchor
                + float(cfg.lambda_res) * res_mag
                + float(cfg.lambda_phy) * phy
            )
        if train:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        values = {
            "total_loss": total,
            "eps_loss": eps_loss,
            "cum_shortfall_loss": cum_short,
            "ramp_loss": ramp_loss,
            "dur_shortfall_loss": dur_short,
            "anchor_loss": anchor,
            "residual_mag_loss": res_mag,
            "physics_loss": phy,
        }
        for key, value in values.items():
            buckets[key].append(float(value.detach().cpu()))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in buckets.items()}


def train_residual_model(data_dir: Path, out_dir: Path, cfg: ResidualRunConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.set_num_threads(1)
    x_base_train, x_base_val, x_base_test, copula_summary = _fit_and_sample_copula(data_dir, out_dir, cfg)
    bundle = _make_loaders(data_dir, x_base_train, x_base_val, x_base_test, cfg, device)
    residual_mean_t = torch.from_numpy(bundle["residual_mean"]).to(device)
    residual_std_t = torch.from_numpy(bundle["residual_std"]).to(device)
    x_std_t = torch.from_numpy(bundle["x_std"]).to(device)
    model = CopulaResidualDenoiser(int(cfg.base_channels), bundle["bg_dim"], bundle["proc_dim"], bundle["risk_dim"]).to(device)
    scheduler = DiffusionScheduler(int(cfg.diffusion_steps), 1e-4, 0.02, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    ema = EMA(model, decay=float(cfg.ema_decay))
    history = []
    best_stage2 = float("inf")
    best_state = None
    total_epochs = int(cfg.stage1_epochs) + int(cfg.stage2_epochs)
    for epoch in range(1, total_epochs + 1):
        stage = "stage1_residual_diffusion" if epoch <= int(cfg.stage1_epochs) else "stage2_risk_residual"
        train_metrics = _run_epoch(model, scheduler, bundle["train"], optimizer, stage, device, residual_mean_t, residual_std_t, x_std_t, bundle["risk_norm"], cfg)
        ema.update(model)
        ema_model = CopulaResidualDenoiser(int(cfg.base_channels), bundle["bg_dim"], bundle["proc_dim"], bundle["risk_dim"]).to(device)
        ema.copy_to(ema_model)
        with torch.no_grad():
            val_metrics = _run_epoch(ema_model, scheduler, bundle["val"], None, stage, device, residual_mean_t, residual_std_t, x_std_t, bundle["risk_norm"], cfg)
        row = {"epoch": epoch, "stage": stage}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if stage == "stage2_risk_residual" and val_metrics["total_loss"] < best_stage2:
            best_stage2 = val_metrics["total_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        print(f"{METHOD_NAME} epoch {epoch:03d}/{total_epochs} | {stage} | train={train_metrics['total_loss']:.4f} | val={val_metrics['total_loss']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ema.copy_to(model)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "residual_mean": bundle["residual_mean"],
            "residual_std": bundle["residual_std"],
            "x_mean": bundle["x_mean"],
            "x_std": bundle["x_std"],
            "dims": {"bg_dim": bundle["bg_dim"], "proc_dim": bundle["proc_dim"], "risk_dim": bundle["risk_dim"]},
            "copula_summary": copula_summary,
            "selected_checkpoint_stage": "stage2_risk_residual" if best_state is not None else "stage1_residual_diffusion",
        },
        out_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": METHOD_NAME,
                "config": asdict(cfg),
                "data_shapes": {k: bundle[k] for k in ["train_shape", "val_shape", "test_shape"]},
                "selected_checkpoint_stage": "stage2_risk_residual" if best_state is not None else "stage1_residual_diffusion",
                "copula_summary": copula_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {"model": model, "scheduler": scheduler, "bundle": bundle, "device": device}


def _parse_float_candidates(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_clip_candidates(text: str) -> list[float | None]:
    out: list[float | None] = []
    for item in str(text).split(","):
        value = item.strip().lower()
        if not value:
            continue
        out.append(None if value in {"none", "null", "off"} else float(value))
    return out


@torch.no_grad()
def _sample_residual_raw(trained: dict, split: str = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the raw residual before protection.

    residual_raw: 扩散模型生成的风光荷残差序列，形状 [样本数, 3, 36]。
    x_base: Copula 统计先验基础场景。
    day_mask: 光伏日间掩码，用于物理投影。
    event_mask: 核心极端事件段标志，用于背景段残差保护。
    residual_real: 真实序列相对 Copula 基础场景的残差。
    """
    model: CopulaResidualDenoiser = trained["model"].eval().to(trained["device"])
    scheduler: DiffusionScheduler = trained["scheduler"]
    ds_key = "val_ds" if split == "val" else "test_ds"
    dataset: CopulaResidualDataset = trained["bundle"][ds_key]
    device = trained["device"]
    bg = torch.from_numpy(dataset.bg).to(device)
    proc = torch.from_numpy(dataset.proc).to(device)
    risk = torch.from_numpy(dataset.risk).to(device)
    base_norm = torch.from_numpy(dataset.base_norm).to(device)
    residual_mean_t = torch.from_numpy(trained["bundle"]["residual_mean"]).to(device)
    residual_std_t = torch.from_numpy(trained["bundle"]["residual_std"]).to(device)
    residual = torch.randn((len(dataset), 3, dataset.x_real.shape[2]), device=device)
    for step in reversed(range(scheduler.steps)):
        t = torch.full((residual.shape[0],), step, device=device, dtype=torch.long)
        residual = _p_sample_residual(model, scheduler, residual, base_norm, t, bg, proc, risk)
    residual_raw = denormalize_x_torch(residual, residual_mean_t, residual_std_t)
    return (
        residual_raw.detach().cpu().numpy().astype(np.float32),
        dataset.x_base.astype(np.float32),
        dataset.day_mask.astype(np.float32),
        dataset.event_mask.astype(np.float32),
        (dataset.x_real - dataset.x_base).astype(np.float32),
    )


def _protected_generation_from_residual(
    residual_raw: np.ndarray,
    x_base: np.ndarray,
    day_mask: np.ndarray,
    event_mask: np.ndarray | None,
    residual_abs_reference: np.ndarray,
    residual_scale: float = 1.0,
    clip_quantile: float | None = None,
    background_residual_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply validation-selected residual protection and physical projection.

    residual_scale: 残差整体缩放系数，防止扩散残差过度改写 Copula 统计结构。
    clip_quantile: 基于训练残差幅度的分位裁剪阈值，抑制异常尖峰残差。
    background_residual_scale: event_mask=0 背景段残差缩放系数，核心事件段保留修正能力。
    """
    residual = residual_raw.astype(np.float32).copy()
    if clip_quantile is not None:
        q = float(clip_quantile)
        thresholds = np.quantile(residual_abs_reference, q, axis=(0, 2)).astype(np.float32).reshape(1, 3, 1)
        residual = np.clip(residual, -thresholds, thresholds)
    if event_mask is not None and float(background_residual_scale) != 1.0:
        mask = np.asarray(event_mask, dtype=np.float32)[:, None, :]
        residual = residual * (mask + float(background_residual_scale) * (1.0 - mask))
    residual = residual * float(residual_scale)
    generated = x_base.astype(np.float32) + residual
    generated = np.maximum(generated, 0.0)
    generated[:, 2, :] = generated[:, 2, :] * np.asarray(day_mask, dtype=np.float32)
    return generated.astype(np.float32), residual.astype(np.float32)


def _save_protected_generation(
    residual_raw: np.ndarray,
    x_base: np.ndarray,
    day_mask: np.ndarray,
    event_mask: np.ndarray | None,
    residual_real: np.ndarray,
    residual_abs_reference: np.ndarray,
    out_dir: Path,
    generated_name: str,
    residual_name: str,
    residual_scale: float = 1.0,
    clip_quantile: float | None = None,
    background_residual_scale: float = 1.0,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated, residual = _protected_generation_from_residual(
        residual_raw=residual_raw,
        x_base=x_base,
        day_mask=day_mask,
        event_mask=event_mask,
        residual_abs_reference=residual_abs_reference,
        residual_scale=residual_scale,
        clip_quantile=clip_quantile,
        background_residual_scale=background_residual_scale,
    )
    gen_path = out_dir / "generated_samples_copula_guided.npy"
    residual_path = out_dir / "residual_generated.npy"
    gen_path = out_dir / generated_name
    residual_path = out_dir / residual_name
    np.save(gen_path, generated)
    np.save(residual_path, residual)
    np.save(out_dir / "residual_real.npy", residual_real.astype(np.float32))
    return gen_path, residual_path


def generate_residual(trained: dict, out_dir: Path) -> tuple[Path, Path]:
    residual_raw, x_base, day_mask, event_mask, residual_real = _sample_residual_raw(trained, split="test")
    return _save_protected_generation(
        residual_raw=residual_raw,
        x_base=x_base,
        day_mask=day_mask,
        event_mask=event_mask,
        residual_real=residual_real,
        residual_abs_reference=trained["bundle"]["residual_train_abs"],
        out_dir=out_dir,
        generated_name="generated_samples_copula_guided.npy",
        residual_name="residual_generated.npy",
        residual_scale=1.0,
        clip_quantile=None,
        background_residual_scale=1.0,
    )


def _run_simple(data_dir: Path, out_dir: Path, cfg: ResidualRunConfig) -> Path:
    method_out = out_dir / "simple_evt_risk_diffusion"
    args = SimpleNamespace(
        data_dir=str(data_dir),
        out_dir=str(method_out),
        method_name="Simple_EVT_Risk_Diffusion",
        device=cfg.device,
        seed=cfg.seed,
        stage1_epochs=cfg.stage1_epochs,
        stage2_epochs=cfg.stage2_epochs,
        batch_size=cfg.batch_size,
        diffusion_steps=cfg.diffusion_steps,
        base_channels=cfg.base_channels,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        ema_decay=cfg.ema_decay,
        cond_dropout=cfg.cond_dropout,
        duration_temp=cfg.duration_temp,
        lambda_cum=1.0,
        lambda_ramp=0.4,
        lambda_ramp_curve=0.0,
        use_core_ramp_curve_loss=False,
        lambda_core_ramp_curve=0.10,
        lambda_dur=0.30,
        lambda_phy=0.02,
        use_ramp_level_condition=False,
        use_ramp_sampler=False,
        sampler_alpha_cum=0.3,
        sampler_alpha_ramp=0.0,
        sampler_alpha_dur=0.2,
        use_temporal_attention=False,
        temporal_hidden_dim=64,
        temporal_layers=1,
        temporal_heads=4,
        temporal_dropout=0.10,
        temporal_feedforward_dim=128,
    )
    trained = train_simple_model(args)
    generated = generate_simple(trained, args)
    target = out_dir / "generated_samples_Simple_EVT_Risk_Diffusion.npy"
    shutil.copy2(generated, target)
    return target


def _evaluate_row(method: str, generated: Path, data_dir: Path, out_dir: Path) -> dict:
    row = evaluate_method(method, generated, data_dir, out_dir)
    return row


def _evaluate_split_row(method: str, generated: Path, data_dir: Path, out_dir: Path, split: str) -> dict:
    eval_dir = out_dir / "evaluations" / f"{method}_{split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    mask_path = data_dir / f"event_mask_{split}.npy"
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / f"X_{split}.npy"),
            generated=str(generated),
            cond=str(data_dir / f"cond_{split}.csv"),
            meta=str(data_dir / f"meta_{split}.csv"),
            event_mask=str(mask_path) if mask_path.exists() else None,
            out_dir=str(eval_dir),
            model_name=method,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = {"method": method}
    row.update({metric: summary["metrics"].get(metric, np.nan) for metric in FULL_COMPARE_METRICS})
    return row


def _select_residual_protection_on_val(
    trained: dict,
    data_dir: Path,
    out_dir: Path,
    cfg: ResidualRunConfig,
) -> dict:
    """Select residual protection parameters on validation split only.

    验证集只用于选择残差保护强度；测试集不参与残差缩放、裁剪或背景保护参数选择。
    """
    val_raw, val_base, val_day, val_event, val_real_residual = _sample_residual_raw(trained, split="val")
    scales = _parse_float_candidates(cfg.residual_scale_candidates)
    clips = _parse_clip_candidates(cfg.residual_clip_quantile_candidates)
    bg_scales = _parse_float_candidates(cfg.background_residual_scale_candidates)
    rows: list[dict] = []
    candidates_dir = out_dir / "val_residual_protection_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for scale in scales:
        for clip in clips:
            for bg_scale in bg_scales:
                tag = f"s{scale:g}_clip{'none' if clip is None else str(clip).replace('.', 'p')}_bg{bg_scale:g}".replace(".", "p")
                generated, _ = _protected_generation_from_residual(
                    residual_raw=val_raw,
                    x_base=val_base,
                    day_mask=val_day,
                    event_mask=val_event,
                    residual_abs_reference=trained["bundle"]["residual_train_abs"],
                    residual_scale=scale,
                    clip_quantile=clip,
                    background_residual_scale=bg_scale,
                )
                path = candidates_dir / f"generated_val_{tag}.npy"
                np.save(path, generated)
                method = f"{PROTECTED_METHOD_NAME}_{tag}"
                row = _evaluate_split_row(method, path, data_dir, out_dir, split="val")
                row.update(
                    {
                        "residual_scale": float(scale),
                        "residual_clip_quantile": np.nan if clip is None else float(clip),
                        "background_residual_scale": float(bg_scale),
                    }
                )
                rows.append(row)
    val_df = add_risk_score(pd.DataFrame(rows))
    val_df.to_csv(out_dir / "residual_protection_val_selection.csv", index=False, encoding="utf-8-sig")
    return _choose_residual_protection_from_val_table(
        out_dir=out_dir,
        output_name="selected_residual_protection.json",
        max_residual_scale=None,
    )


def _choose_residual_protection_from_val_table(
    out_dir: Path,
    output_name: str,
    max_residual_scale: float | None = None,
) -> dict:
    val_df = pd.read_csv(out_dir / "residual_protection_val_selection.csv")
    pool = val_df.copy()
    if max_residual_scale is not None and "residual_scale" in pool.columns:
        pool = pool[pd.to_numeric(pool["residual_scale"], errors="coerce") <= float(max_residual_scale)].copy()
        if pool.empty:
            pool = val_df.copy()
    best = pool.sort_values(["risk_score", "risk_rank"], na_position="last").iloc[0].to_dict()
    selected = {
        "method": best.get("method"),
        "residual_scale": float(best.get("residual_scale", 1.0)),
        "residual_clip_quantile": None
        if pd.isna(best.get("residual_clip_quantile"))
        else float(best.get("residual_clip_quantile")),
        "background_residual_scale": float(best.get("background_residual_scale", 1.0)),
        "validation_risk_score": float(best.get("risk_score", np.nan)),
        "validation_risk_rank": float(best.get("risk_rank", np.nan)),
        "selection_rule": "best validation risk_score"
        if max_residual_scale is None
        else f"best validation risk_score with residual_scale <= {float(max_residual_scale):g}",
    }
    (out_dir / output_name).write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return selected


def generate_val_protected_test(trained: dict, out_dir: Path, selected: dict) -> tuple[Path, Path]:
    test_raw, test_base, test_day, test_event, test_real_residual = _sample_residual_raw(trained, split="test")
    return _save_protected_generation(
        residual_raw=test_raw,
        x_base=test_base,
        day_mask=test_day,
        event_mask=test_event,
        residual_real=test_real_residual,
        residual_abs_reference=trained["bundle"]["residual_train_abs"],
        out_dir=out_dir,
        generated_name="generated_samples_copula_guided_valprotected.npy",
        residual_name="residual_generated_valprotected.npy",
        residual_scale=float(selected["residual_scale"]),
        clip_quantile=selected["residual_clip_quantile"],
        background_residual_scale=float(selected["background_residual_scale"]),
    )


def generate_val_protected_conservative_test(trained: dict, out_dir: Path, selected: dict) -> tuple[Path, Path]:
    test_raw, test_base, test_day, test_event, test_real_residual = _sample_residual_raw(trained, split="test")
    return _save_protected_generation(
        residual_raw=test_raw,
        x_base=test_base,
        day_mask=test_day,
        event_mask=test_event,
        residual_real=test_real_residual,
        residual_abs_reference=trained["bundle"]["residual_train_abs"],
        out_dir=out_dir,
        generated_name="generated_samples_copula_guided_valprotected_conservative.npy",
        residual_name="residual_generated_valprotected_conservative.npy",
        residual_scale=float(selected["residual_scale"]),
        clip_quantile=selected["residual_clip_quantile"],
        background_residual_scale=float(selected["background_residual_scale"]),
    )


def _existing_sources_for_dataset(spec: DatasetSpec) -> dict[str, Path]:
    if spec.existing_sources:
        return spec.existing_sources
    if spec.existing_results_dir and spec.existing_results_dir.exists():
        names = {
            "enhanced_gan": "generated_samples_enhanced_gan.npy",
            "plain_diffusion_baseline": "generated_samples_plain_diffusion_baseline.npy",
            "improved_diffusion": "generated_samples_improved_diffusion.npy",
            "proposed_E0": "generated_samples_proposed_E0.npy",
            "JRPD_best_3h": "generated_samples_JRPD_best_3h.npy",
        }
        return {name: spec.existing_results_dir / filename for name, filename in names.items()}
    return {}


def run_one_dataset(spec: DatasetSpec, cfg: ResidualRunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = cfg.out_dir / spec.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = spec.data_dir
    rows: list[dict] = []

    trained = train_residual_model(data_dir, out_dir, cfg)
    guided_path, _ = generate_residual(trained, out_dir)
    selected_protection = _select_residual_protection_on_val(trained, data_dir, out_dir, cfg)
    protected_path, _ = generate_val_protected_test(trained, out_dir, selected_protection)
    conservative_protection = _choose_residual_protection_from_val_table(
        out_dir=out_dir,
        output_name="selected_residual_protection_conservative.json",
        max_residual_scale=0.5,
    )
    conservative_path, _ = generate_val_protected_conservative_test(trained, out_dir, conservative_protection)
    copula_base_path = out_dir / "x_base_copula.npy"
    rows.append(_evaluate_row("traditional_gaussian_copula", copula_base_path, data_dir, out_dir))
    rows.append(_evaluate_row("Simple_EVT_Risk_Diffusion", _run_simple(data_dir, out_dir, cfg), data_dir, out_dir))
    rows.append(_evaluate_row(METHOD_NAME, guided_path, data_dir, out_dir))
    protected_row = _evaluate_row(PROTECTED_METHOD_NAME, protected_path, data_dir, out_dir)
    protected_row.update(
        {
            "selected_residual_scale": selected_protection["residual_scale"],
            "selected_residual_clip_quantile": np.nan
            if selected_protection["residual_clip_quantile"] is None
            else selected_protection["residual_clip_quantile"],
            "selected_background_residual_scale": selected_protection["background_residual_scale"],
            "selected_val_risk_score": selected_protection["validation_risk_score"],
        }
    )
    rows.append(protected_row)
    conservative_row = _evaluate_row(CONSERVATIVE_METHOD_NAME, conservative_path, data_dir, out_dir)
    conservative_row.update(
        {
            "selected_residual_scale": conservative_protection["residual_scale"],
            "selected_residual_clip_quantile": np.nan
            if conservative_protection["residual_clip_quantile"] is None
            else conservative_protection["residual_clip_quantile"],
            "selected_background_residual_scale": conservative_protection["background_residual_scale"],
            "selected_val_risk_score": conservative_protection["validation_risk_score"],
        }
    )
    rows.append(conservative_row)
    pd.DataFrame([rows[-3], rows[-2], rows[-1]]).to_csv(out_dir / "copula_guided_metrics.csv", index=False, encoding="utf-8-sig")

    for method, source in _existing_sources_for_dataset(spec).items():
        if not source.exists():
            continue
        target = out_dir / f"generated_samples_{method}.npy"
        shutil.copy2(source, target)
        if method in {row["method"] for row in rows}:
            continue
        rows.append(_evaluate_row(method, target, data_dir, out_dir))

    df = add_risk_score(pd.DataFrame(rows))
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir, method_col="method")
    _write_method_report(spec, out_dir, risk_main, aux, cfg)
    return risk_main, aux


def _write_method_report(spec: DatasetSpec, out_dir: Path, risk_main: pd.DataFrame, aux: pd.DataFrame, cfg: ResidualRunConfig) -> None:
    data_dir = spec.data_dir
    x_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    x_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
    selected_path = out_dir / "selected_residual_protection.json"
    conservative_selected_path = out_dir / "selected_residual_protection_conservative.json"
    selected_text = selected_path.read_text(encoding="utf-8") if selected_path.exists() else "{}"
    conservative_selected_text = conservative_selected_path.read_text(encoding="utf-8") if conservative_selected_path.exists() else "{}"
    lines = [
        f"# {spec.name} - Copula-guided EVT Risk Residual Diffusion",
        "",
        "## Method",
        "",
        "This method first fits a Gaussian Copula baseline on the train split only, generates x_base for each condition, and then trains a conditional diffusion model on residual_real = x_real - x_base. Risk losses are computed on x_gen = x_base + residual_gen.",
        "",
        "The `ValProtected` variant selects residual scale, residual clipping, and background residual shrinkage on the validation split only, then freezes those parameters for test generation.",
        "",
        "## Dataset",
        "",
        f"- data_dir: `{data_dir}`",
        f"- train samples: {int(x_train.shape[0])}",
        f"- test samples: {int(x_test.shape[0])}",
        "",
        "## Main Risk Table",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Auxiliary Realism Table",
        "",
        aux[[col for col in ["method", *AUXILIARY_REALISM_METRICS, "realism_check_pass"] if col in aux.columns]].to_markdown(index=False),
        "",
        "## Weights",
        "",
        f"- lambda_cum={cfg.lambda_cum}, lambda_ramp={cfg.lambda_ramp}, lambda_dur={cfg.lambda_dur}",
        f"- lambda_anchor={cfg.lambda_anchor}, lambda_res={cfg.lambda_res}, lambda_phy={cfg.lambda_phy}",
        "",
        "## Selected Residual Protection",
        "",
        "```json",
        selected_text,
        "```",
        "",
        "## Selected Conservative Residual Protection",
        "",
        "```json",
        conservative_selected_text,
        "```",
    ]
    (out_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8")


def _dataset_specs() -> list[DatasetSpec]:
    singleton_sources = {
        "enhanced_gan": BASE_DIR / "outputs" / "main_compare_e0_fixed" / "models" / "enhanced_gan" / "generation" / "generated_samples.npy",
        "plain_diffusion_baseline": BASE_DIR / "outputs" / "main_compare_e0_fixed" / "models" / "plain_diffusion_baseline" / "generation" / "generated_samples.npy",
        "improved_diffusion": BASE_DIR / "outputs" / "main_compare_e0_fixed" / "models" / "improved_diffusion" / "generated_samples.npy",
        "proposed_E0": BASE_DIR / "outputs" / "main_compare_e0_fixed" / "models" / "proposed" / "generated_samples.npy",
        "JRPD_best_3h": BASE_DIR / "outputs" / "ramp_window_retrain" / "models" / "RAMPDIAG4_JRPD_ramp3h" / "generated_samples.npy",
    }
    return [
        DatasetSpec(
            name="Singleton / Singleton North",
            out_name="singleton",
            data_dir=BASE_DIR / "outputs" / "ramp_window_retrain" / "datasets" / "dataset_window_3h",
            existing_sources=singleton_sources,
        ),
        DatasetSpec(
            name="Muswellbrook",
            out_name="muswellbrook",
            data_dir=BASE_DIR / "outputs" / "muswellbrook_ramp_window_retrain" / "datasets" / "dataset_window_3h",
            existing_results_dir=BASE_DIR / "results" / "muswellbrook_all_methods_compare",
        ),
        DatasetSpec(
            name="Cessnock South / New area",
            out_name="cessnock_or_newarea",
            data_dir=BASE_DIR / "outputs" / "cessnock_south_ramp_window_retrain" / "datasets" / "dataset_window_3h",
            existing_results_dir=BASE_DIR / "results" / "cessnock_south_all_methods_compare",
        ),
    ]


def _write_fairness_report(root: Path, specs: list[DatasetSpec]) -> None:
    lines = [
        "# Copula Fairness Check",
        "",
        "## Code Path Checked",
        "",
        "- The Copula baseline implementation is `traditional_statistical_extreme_baseline/traditional_copula_baseline.py`.",
        "- In this experiment, Copula is fitted by calling `TraditionalCopulaBaseline.fit(X_train, cond_train)` for each dataset.",
        "- Test generation calls `sample_conditions(cond_test, n_per_condition=1, rng=seeded_rng)` and does not read `X_test.npy`.",
        "",
        "## Leakage Assessment",
        "",
        "- Marginal empirical distributions: train split only.",
        "- Gaussian copula covariance/correlation: train split only.",
        "- Group fallback mapping: train `cond_train.csv` only.",
        "- Test split usage: only `cond_test.csv` is used as generation conditions; `X_test.npy` is used only by evaluation.",
        "- Residual protection selection: residual scale/clipping/background shrinkage are selected on validation split only; test split is not used for hyperparameter selection.",
        "- Sample count: one generated scenario per test condition for every method.",
        "",
        "## Dataset Runs",
        "",
    ]
    for spec in specs:
        lines.append(f"- {spec.out_name}: `{spec.data_dir}`")
    lines.extend(
        [
            "",
            "Conclusion: this Copula-guided run is fair under the requested rules. No test curve is used for Copula fitting, residual training, or candidate selection.",
        ]
    )
    (root / "copula_fairness_check.md").write_text("\n".join(lines), encoding="utf-8")


def _write_global_summaries(root: Path, dataset_tables: dict[str, pd.DataFrame], aux_tables: dict[str, pd.DataFrame]) -> None:
    all_rows = []
    for dataset, table in dataset_tables.items():
        tmp = table.copy()
        tmp.insert(0, "dataset", dataset)
        all_rows.append(tmp)
    risk_summary = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    risk_summary.to_csv(root / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")

    rank_rows = []
    methods = sorted(risk_summary["method"].astype(str).unique()) if len(risk_summary) else []
    for method in methods:
        row = {"method": method}
        ranks = []
        scores = []
        for key in ["singleton", "muswellbrook", "cessnock_or_newarea"]:
            sub = risk_summary[(risk_summary["dataset"] == key) & (risk_summary["method"] == method)]
            rank = float(sub["risk_rank"].iloc[0]) if len(sub) and "risk_rank" in sub.columns else np.nan
            score = float(sub["risk_score"].iloc[0]) if len(sub) and "risk_score" in sub.columns else np.nan
            row[f"{key}_risk_rank"] = rank
            row[f"{key}_risk_score"] = score
            if np.isfinite(rank):
                ranks.append(rank)
            if np.isfinite(score):
                scores.append(score)
        row["mean_risk_rank"] = float(np.mean(ranks)) if ranks else np.nan
        row["mean_risk_score"] = float(np.mean(scores)) if scores else np.nan
        row["wins_count"] = int(sum(1 for r in ranks if int(r) == 1))
        row["top3_count"] = int(sum(1 for r in ranks if r <= 3))
        rank_rows.append(row)
    rank_df = pd.DataFrame(rank_rows).sort_values(["mean_risk_rank", "mean_risk_score"], na_position="last")
    rank_df.to_csv(root / "all_datasets_rank_summary.csv", index=False, encoding="utf-8-sig")

    aux_rows = []
    for dataset, table in aux_tables.items():
        tmp = table.copy()
        tmp.insert(0, "dataset", dataset)
        aux_rows.append(tmp)
    aux_summary = pd.concat(aux_rows, ignore_index=True) if aux_rows else pd.DataFrame()
    aux_summary.to_csv(root / "all_datasets_auxiliary_summary.csv", index=False, encoding="utf-8-sig")
    _write_final_report(root, risk_summary, rank_df)


def _write_final_report(root: Path, risk_summary: pd.DataFrame, rank_df: pd.DataFrame) -> None:
    guided = rank_df[rank_df["method"].eq(METHOD_NAME)]
    protected = rank_df[rank_df["method"].eq(PROTECTED_METHOD_NAME)]
    conservative = rank_df[rank_df["method"].eq(CONSERVATIVE_METHOD_NAME)]
    simple = rank_df[rank_df["method"].eq("Simple_EVT_Risk_Diffusion")]
    copula = rank_df[rank_df["method"].eq("traditional_gaussian_copula")]
    lines = [
        "# Final Experiment Report",
        "",
        "## Method Summary",
        "",
        "Copula_Guided_EVT_Risk_Residual_Diffusion fits a train-only Gaussian Copula prior, then learns an EVT risk residual with a two-stage conditional diffusion model. The final sample is `x_gen = x_base_copula + residual_gen`.",
        "",
        "`Copula_Guided_EVT_Risk_Residual_Diffusion_ValProtected` additionally selects residual protection parameters on validation data only: residual scale, residual clipping quantile, and background residual shrinkage.",
        "`Copula_Guided_EVT_Risk_Residual_Diffusion_ValProtectedConservative` uses the same validation selection but caps residual_scale at 0.5 to avoid aggressive residual overcorrection.",
        "",
        "## Ranking Logic",
        "",
        RISK_RANKING_EXPLANATION,
        RISK_EVALUATION_EXPLANATION,
        "",
        "## All Dataset Risk Summary",
        "",
        risk_summary.to_markdown(index=False) if len(risk_summary) else "No rows.",
        "",
        "## Cross-Dataset Rank Summary",
        "",
        rank_df.to_markdown(index=False) if len(rank_df) else "No rows.",
        "",
        "## Key Comparison",
        "",
    ]
    if len(guided):
        g = guided.iloc[0].to_dict()
        lines.append(f"- Copula_Guided mean_risk_rank: {g.get('mean_risk_rank')}, mean_risk_score: {g.get('mean_risk_score')}, wins_count: {g.get('wins_count')}, top3_count: {g.get('top3_count')}")
    if len(protected):
        p = protected.iloc[0].to_dict()
        lines.append(f"- ValProtected mean_risk_rank: {p.get('mean_risk_rank')}, mean_risk_score: {p.get('mean_risk_score')}, wins_count: {p.get('wins_count')}, top3_count: {p.get('top3_count')}")
    if len(conservative):
        cns = conservative.iloc[0].to_dict()
        lines.append(f"- ValProtectedConservative mean_risk_rank: {cns.get('mean_risk_rank')}, mean_risk_score: {cns.get('mean_risk_score')}, wins_count: {cns.get('wins_count')}, top3_count: {cns.get('top3_count')}")
    if len(simple):
        s = simple.iloc[0].to_dict()
        lines.append(f"- Simple mean_risk_rank: {s.get('mean_risk_rank')}, mean_risk_score: {s.get('mean_risk_score')}")
    if len(copula):
        c = copula.iloc[0].to_dict()
        lines.append(f"- Copula mean_risk_rank: {c.get('mean_risk_rank')}, mean_risk_score: {c.get('mean_risk_score')}")
    lines.extend(
        [
            "",
            "## Recommendation Logic",
            "",
            "Adopt the ValProtected residual method as the final main method only if it is more stable than Simple and reaches top-3 in at least two datasets without losing Copula's 3h ramp advantage. If validation selects residual_scale=0 frequently, the residual diffusion branch is not adding robust value beyond Copula.",
        ]
    )
    (root / "final_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(cfg: ResidualRunConfig) -> None:
    specs = [spec for spec in _dataset_specs() if spec.data_dir.exists()]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _write_fairness_report(cfg.out_dir, specs)
    dataset_tables: dict[str, pd.DataFrame] = {}
    aux_tables: dict[str, pd.DataFrame] = {}
    for spec in specs:
        print(f"\n=== Dataset: {spec.out_name} ===")
        risk_main, aux = run_one_dataset(spec, cfg)
        dataset_tables[spec.out_name] = risk_main
        aux_tables[spec.out_name] = aux
    _write_global_summaries(cfg.out_dir, dataset_tables, aux_tables)


def parse_args() -> ResidualRunConfig:
    parser = argparse.ArgumentParser(description="Run Copula-guided EVT risk residual diffusion on three regional datasets.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "copula_guided_residual_diffusion")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--cond-dropout", type=float, default=0.10)
    parser.add_argument("--duration-temp", type=float, default=12.0)
    parser.add_argument("--lambda-cum", type=float, default=0.8)
    parser.add_argument("--lambda-ramp", type=float, default=0.20)
    parser.add_argument("--lambda-dur", type=float, default=0.25)
    parser.add_argument("--lambda-anchor", type=float, default=0.08)
    parser.add_argument("--lambda-res", type=float, default=0.10)
    parser.add_argument("--lambda-phy", type=float, default=0.02)
    parser.add_argument("--residual-scale-candidates", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--residual-clip-quantile-candidates", type=str, default="none,0.95,0.90")
    parser.add_argument("--background-residual-scale-candidates", type=str, default="1.0,0.5,0.25")
    return ResidualRunConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_all(parse_args())
