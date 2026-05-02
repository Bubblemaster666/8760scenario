from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHANNELS = ["load", "wind_power", "solar_power"]
EPS = 1e-8


@dataclass
class CopulaConfig:
    data_dir: str = "outputs/dataset"
    output_dir: str = "outputs/models/traditional_copula"
    seed: int = 42
    test_ratio: float = 0.2
    val_ratio: float = 0.0
    group_cols: str = "event_type_code,severity_level,low_wind_flag,low_irradiance_flag"
    fallback_group_cols: str = "event_type_code,severity_level;event_type_code;global"
    min_group_size: int = 8
    covariance_shrinkage: float = 0.08
    quantile_grid_size: int = 401
    n_per_condition: int = 1
    seq_len: int = -1
    delta_t: float = 1.0
    imbalance_tau: float = 0.0
    acf_max_lag: int = 12
    js_bins: int = 80
    solar_zero_before_hour: int = 6
    solar_zero_after_hour: int = 20
    use_time_index_for_solar_mask: bool = True
    ramp_clip_quantile: float = 0.995
    temporal_smooth_strength: float = 0.15
    temporal_smooth_window: int = 3
    make_plots: bool = True


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_nct(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got shape={x.shape}")
    if x.shape[1] == 3:
        return x
    if x.shape[2] == 3:
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"Cannot infer channel dimension from shape={x.shape}; expected [N,3,T] or [N,T,3].")


def flatten_nct(x: np.ndarray) -> np.ndarray:
    # [N, C, T] -> [N, C*T] with channel-major layout.
    return x.reshape(x.shape[0], -1)


def unflatten_nct(v: np.ndarray, c: int, t: int) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(-1, c, t)


def load_dataset(data_dir: str) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.DataFrame]]:
    root = Path(data_dir)
    x_path = root / "X.npy"
    cond_path = root / "cond.csv"
    meta_path = root / "meta.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"Cannot find {x_path}")
    if not cond_path.exists():
        raise FileNotFoundError(f"Cannot find {cond_path}")
    x = ensure_nct(np.load(x_path))
    cond = pd.read_csv(cond_path)
    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    if len(cond) != x.shape[0]:
        raise ValueError(f"X and cond.csv sample count mismatch: {x.shape[0]} vs {len(cond)}")
    if meta is not None and len(meta) != x.shape[0]:
        raise ValueError(f"X and meta.csv sample count mismatch: {x.shape[0]} vs {len(meta)}")
    return x, cond, meta


def resolve_existing_split(data_dir: str, out_dir: Path) -> Optional[Dict[str, str]]:
    root = Path(data_dir)
    required = ["X_train.npy", "X_test.npy", "cond_train.csv", "cond_test.csv"]
    if not all((root / name).exists() for name in required):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in required + ["X_val.npy", "cond_val.csv", "meta_train.csv", "meta_test.csv", "meta_val.csv"]:
        src = root / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    return {
        "X_train": str(out_dir / "X_train.npy"),
        "X_test": str(out_dir / "X_test.npy"),
        "cond_train": str(out_dir / "cond_train.csv"),
        "cond_test": str(out_dir / "cond_test.csv"),
    }


def save_split_dataset(
    x: np.ndarray,
    cond: pd.DataFrame,
    meta: Optional[pd.DataFrame],
    out_dir: Path,
    cfg: CopulaConfig,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    if "sample_id" in cond.columns:
        ids = cond["sample_id"].astype(str).values
    else:
        ids = np.arange(len(cond)).astype(str)

    strat_cols = [c for c in ["event_type_code", "severity_level"] if c in cond.columns]
    if strat_cols:
        labels = cond[strat_cols].astype(str).agg("_".join, axis=1).values
    elif "event_type" in cond.columns:
        labels = cond["event_type"].astype(str).values
    else:
        labels = np.array(["global"] * len(cond))

    train_idx: List[int] = []
    test_idx: List[int] = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * cfg.test_ratio))) if len(idx) >= 4 else max(0, int(round(len(idx) * cfg.test_ratio)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    if not train_idx:
        raise ValueError("Empty training split. Reduce test_ratio or provide more samples.")
    if not test_idx:
        # fallback: one random test sample
        all_idx = np.arange(len(cond))
        rng.shuffle(all_idx)
        test_idx = [int(all_idx[0])]
        train_idx = [int(i) for i in all_idx[1:]]

    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    np.save(out_dir / "X_train.npy", x[train_idx].astype(np.float32))
    np.save(out_dir / "X_test.npy", x[test_idx].astype(np.float32))
    cond.iloc[train_idx].reset_index(drop=True).to_csv(out_dir / "cond_train.csv", index=False, encoding="utf-8-sig")
    cond.iloc[test_idx].reset_index(drop=True).to_csv(out_dir / "cond_test.csv", index=False, encoding="utf-8-sig")
    if meta is not None:
        meta.iloc[train_idx].reset_index(drop=True).to_csv(out_dir / "meta_train.csv", index=False, encoding="utf-8-sig")
        meta.iloc[test_idx].reset_index(drop=True).to_csv(out_dir / "meta_test.csv", index=False, encoding="utf-8-sig")
    return {
        "X_train": str(out_dir / "X_train.npy"),
        "X_test": str(out_dir / "X_test.npy"),
        "cond_train": str(out_dir / "cond_train.csv"),
        "cond_test": str(out_dir / "cond_test.csv"),
    }


class EmpiricalMarginal:
    """Monotone empirical inverse-CDF for one scalar dimension."""

    def __init__(self, values: np.ndarray, grid_size: int = 401):
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            v = np.array([0.0], dtype=np.float64)
        self.n = int(v.size)
        self.constant = bool(np.nanstd(v) < 1e-10)
        self.const_value = float(np.nanmean(v))
        # Use a fixed quantile grid to keep model compact and stable.
        q = np.linspace(0.0, 1.0, int(grid_size))
        if self.constant:
            xq = np.full_like(q, self.const_value, dtype=np.float64)
        else:
            xq = np.quantile(v, q)
            xq = np.maximum.accumulate(xq)
        self.q = q.astype(np.float64)
        self.xq = xq.astype(np.float64)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.constant:
            return np.full_like(np.asarray(x, dtype=np.float64), 0.5, dtype=np.float64)
        return np.interp(np.asarray(x, dtype=np.float64), self.xq, self.q, left=0.0, right=1.0)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64), 1e-5, 1 - 1e-5)
        if self.constant:
            return np.full_like(u, self.const_value, dtype=np.float64)
        return np.interp(u, self.q, self.xq)

    def to_dict(self) -> dict:
        return {"n": self.n, "constant": self.constant, "const_value": self.const_value, "q": self.q, "xq": self.xq}

    @classmethod
    def from_dict(cls, d: dict) -> "EmpiricalMarginal":
        obj = cls(np.array([0.0]))
        obj.n = int(d["n"])
        obj.constant = bool(d["constant"])
        obj.const_value = float(d["const_value"])
        obj.q = np.asarray(d["q"], dtype=np.float64)
        obj.xq = np.asarray(d["xq"], dtype=np.float64)
        return obj


class GaussianCopulaGroupModel:
    def __init__(
        self,
        group_name: str,
        x_group: np.ndarray,
        cfg: CopulaConfig,
    ):
        self.group_name = group_name
        self.n_samples = int(x_group.shape[0])
        self.n_channels = int(x_group.shape[1])
        self.seq_len = int(x_group.shape[2])
        self.dim = int(self.n_channels * self.seq_len)
        self.cfg_dict = asdict(cfg)
        x_flat = flatten_nct(x_group)
        self.marginals: List[EmpiricalMarginal] = [EmpiricalMarginal(x_flat[:, j], cfg.quantile_grid_size) for j in range(self.dim)]

        # Transform each scalar dimension into Gaussian score using empirical CDF.
        z = np.zeros_like(x_flat, dtype=np.float64)
        for j, m in enumerate(self.marginals):
            u = m.cdf(x_flat[:, j])
            u = np.clip(u, 1.0 / (self.n_samples + 2), 1.0 - 1.0 / (self.n_samples + 2))
            z[:, j] = stats.norm.ppf(u)
        z = np.nan_to_num(z, nan=0.0, posinf=4.75, neginf=-4.75)
        if self.n_samples >= 2:
            cov = np.cov(z, rowvar=False, bias=False)
        else:
            cov = np.eye(self.dim)
        cov = np.atleast_2d(cov)
        if cov.shape != (self.dim, self.dim):
            cov = np.eye(self.dim)
        # Shrink covariance towards diagonal to avoid singularity in small extreme groups.
        shrink = float(np.clip(cfg.covariance_shrinkage, 0.0, 0.95))
        diag = np.diag(np.maximum(np.diag(cov), 1e-5))
        cov = (1.0 - shrink) * cov + shrink * diag
        cov = 0.5 * (cov + cov.T)
        # PSD repair.
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-5, None)
        self.cov = (vecs * vals[None, :]) @ vecs.T
        self.cov_sqrt = vecs * np.sqrt(vals)[None, :]
        self.mean = np.mean(z, axis=0) if self.n_samples >= 2 else np.zeros(self.dim, dtype=np.float64)

        # Time-series references for postprocess.
        self.channel_min = np.quantile(x_group, 0.001, axis=(0, 2))
        self.channel_max = np.quantile(x_group, 0.999, axis=(0, 2))
        self.ramp_abs_q = self._estimate_ramp_quantile(x_group, cfg.ramp_clip_quantile)
        self.mean_profile = np.mean(x_group, axis=0)

    def _estimate_ramp_quantile(self, x: np.ndarray, q: float) -> np.ndarray:
        out = np.ones(self.n_channels, dtype=np.float64)
        for c in range(self.n_channels):
            r = np.abs(np.diff(x[:, c, :], axis=1)).reshape(-1)
            out[c] = float(np.quantile(r, np.clip(q, 0.5, 0.9999))) if r.size else np.inf
            out[c] = max(out[c], 1e-6)
        return out

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((int(n), self.dim)) @ self.cov_sqrt.T + self.mean[None, :]
        u = stats.norm.cdf(z)
        x_flat = np.zeros_like(u, dtype=np.float64)
        for j, m in enumerate(self.marginals):
            x_flat[:, j] = m.ppf(u[:, j])
        return unflatten_nct(x_flat, self.n_channels, self.seq_len)

    def to_dict(self) -> dict:
        return {
            "group_name": self.group_name,
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "seq_len": self.seq_len,
            "dim": self.dim,
            "cfg_dict": self.cfg_dict,
            "marginals": [m.to_dict() for m in self.marginals],
            "cov": self.cov,
            "cov_sqrt": self.cov_sqrt,
            "mean": self.mean,
            "channel_min": self.channel_min,
            "channel_max": self.channel_max,
            "ramp_abs_q": self.ramp_abs_q,
            "mean_profile": self.mean_profile,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianCopulaGroupModel":
        obj = cls.__new__(cls)
        obj.group_name = d["group_name"]
        obj.n_samples = int(d["n_samples"])
        obj.n_channels = int(d["n_channels"])
        obj.seq_len = int(d["seq_len"])
        obj.dim = int(d["dim"])
        obj.cfg_dict = dict(d.get("cfg_dict", {}))
        obj.marginals = [EmpiricalMarginal.from_dict(m) for m in d["marginals"]]
        obj.cov = np.asarray(d["cov"], dtype=np.float64)
        obj.cov_sqrt = np.asarray(d.get("cov_sqrt", np.linalg.cholesky(obj.cov + 1e-6*np.eye(obj.cov.shape[0]))), dtype=np.float64)
        obj.mean = np.asarray(d["mean"], dtype=np.float64)
        obj.channel_min = np.asarray(d["channel_min"], dtype=np.float64)
        obj.channel_max = np.asarray(d["channel_max"], dtype=np.float64)
        obj.ramp_abs_q = np.asarray(d["ramp_abs_q"], dtype=np.float64)
        obj.mean_profile = np.asarray(d["mean_profile"], dtype=np.float64)
        return obj


def parse_group_cols(s: str) -> List[str]:
    return [c.strip() for c in str(s).split(",") if c.strip()]


def group_key_from_row(row: pd.Series, cols: List[str]) -> str:
    if not cols or cols == ["global"] or "global" in cols:
        return "global"
    vals = []
    for c in cols:
        if c not in row.index:
            vals.append("NA")
        else:
            v = row[c]
            if pd.isna(v):
                vals.append("NA")
            else:
                vals.append(str(v))
    return "|".join(f"{c}={v}" for c, v in zip(cols, vals))


def group_key_series(cond: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols or cols == ["global"] or "global" in cols:
        return np.array(["global"] * len(cond), dtype=object)
    use = cond.copy()
    for c in cols:
        if c not in use.columns:
            use[c] = "NA"
    return use[cols].astype(str).agg(lambda r: "|".join(f"{c}={r[c]}" for c in cols), axis=1).values


class TraditionalCopulaBaseline:
    def __init__(self, cfg: CopulaConfig):
        self.cfg = cfg
        self.models: Dict[str, GaussianCopulaGroupModel] = {}
        self.group_columns = parse_group_cols(cfg.group_cols)
        self.fallback_columns = [parse_group_cols(x) for x in cfg.fallback_group_cols.split(";") if x.strip()]
        if not any("global" in cols for cols in self.fallback_columns):
            self.fallback_columns.append(["global"])
        self.exact_group_to_model: Dict[str, str] = {}
        self.global_model_name = "global"
        self.seq_len = None

    def fit(self, x: np.ndarray, cond: pd.DataFrame) -> dict:
        x = ensure_nct(x)
        self.seq_len = int(x.shape[2])
        summary_rows = []

        # Build models for exact groups and fallback levels.
        all_specs: List[Tuple[str, List[str]]] = [("exact", self.group_columns)]
        for i, cols in enumerate(self.fallback_columns):
            all_specs.append((f"fallback_{i}", cols))

        built = set()
        for spec_name, cols in all_specs:
            keys = group_key_series(cond, cols)
            for key in sorted(set(keys)):
                idx = np.where(keys == key)[0]
                if key != "global" and len(idx) < self.cfg.min_group_size:
                    continue
                model_name = f"{spec_name}::{key}"
                if key == "global":
                    model_name = "global"
                if model_name in built:
                    continue
                self.models[model_name] = GaussianCopulaGroupModel(model_name, x[idx], self.cfg)
                built.add(model_name)
                summary_rows.append({"model_name": model_name, "cols": ",".join(cols), "n_samples": int(len(idx))})

        if "global" not in self.models:
            self.models["global"] = GaussianCopulaGroupModel("global", x, self.cfg)
            summary_rows.append({"model_name": "global", "cols": "global", "n_samples": int(x.shape[0])})
        self.global_model_name = "global"

        # Map each exact condition group to the most specific available model.
        exact_keys = group_key_series(cond, self.group_columns)
        for key in sorted(set(exact_keys)):
            chosen = None
            exact_name = f"exact::{key}"
            if exact_name in self.models:
                chosen = exact_name
            else:
                # Use the first fallback whose key has a fitted model.
                probe_row = cond.iloc[np.where(exact_keys == key)[0][0]]
                for i, cols in enumerate(self.fallback_columns):
                    k = group_key_from_row(probe_row, cols)
                    name = "global" if k == "global" else f"fallback_{i}::{k}"
                    if name in self.models:
                        chosen = name
                        break
            self.exact_group_to_model[key] = chosen or self.global_model_name
        return {"group_summary": summary_rows, "exact_group_to_model": self.exact_group_to_model}

    def _choose_model_for_row(self, row: pd.Series) -> GaussianCopulaGroupModel:
        exact_key = group_key_from_row(row, self.group_columns)
        name = self.exact_group_to_model.get(exact_key)
        if name is None:
            for i, cols in enumerate(self.fallback_columns):
                k = group_key_from_row(row, cols)
                cand = "global" if k == "global" else f"fallback_{i}::{k}"
                if cand in self.models:
                    name = cand
                    break
        name = name or self.global_model_name
        return self.models[name]

    def sample_conditions(self, cond: pd.DataFrame, n_per_condition: int, rng: np.random.Generator) -> Tuple[np.ndarray, pd.DataFrame]:
        samples = []
        rows = []
        for i, row in cond.reset_index(drop=True).iterrows():
            model = self._choose_model_for_row(row)
            x = model.sample_raw(n_per_condition, rng)
            x = temporal_correction(x, model, self.cfg)
            x = physical_projection_np(x, cond_rows=pd.DataFrame([row] * n_per_condition), cfg=self.cfg)
            samples.append(x)
            for k in range(n_per_condition):
                r = row.copy()
                r["source_condition_index"] = int(i)
                r["generated_id"] = f"g{i:05d}_{k:03d}"
                r["copula_model_group"] = model.group_name
                rows.append(r)
        return np.concatenate(samples, axis=0).astype(np.float32), pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        payload = {
            "cfg": asdict(self.cfg),
            "group_columns": self.group_columns,
            "fallback_columns": self.fallback_columns,
            "exact_group_to_model": self.exact_group_to_model,
            "global_model_name": self.global_model_name,
            "seq_len": self.seq_len,
            "models": {k: v.to_dict() for k, v in self.models.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "TraditionalCopulaBaseline":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        cfg = CopulaConfig(**payload["cfg"])
        obj = cls(cfg)
        obj.group_columns = payload["group_columns"]
        obj.fallback_columns = payload["fallback_columns"]
        obj.exact_group_to_model = payload["exact_group_to_model"]
        obj.global_model_name = payload["global_model_name"]
        obj.seq_len = payload.get("seq_len")
        obj.models = {k: GaussianCopulaGroupModel.from_dict(v) for k, v in payload["models"].items()}
        return obj


def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: x.size]


def temporal_correction(x: np.ndarray, model: GaussianCopulaGroupModel, cfg: CopulaConfig) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    n, c, t = out.shape
    # Clip channel ranges by learned group quantiles.
    for ch in range(c):
        lo = min(0.0, float(model.channel_min[ch])) if ch in (1, 2) else max(0.0, float(model.channel_min[ch]) * 0.9)
        hi = max(float(model.channel_max[ch]) * 1.1, lo + 1e-6)
        out[:, ch, :] = np.clip(out[:, ch, :], lo, hi)

    # Mild smoothing only; the copula already models intra-window temporal dependence.
    smooth = float(np.clip(cfg.temporal_smooth_strength, 0.0, 1.0))
    if smooth > 0 and cfg.temporal_smooth_window > 1:
        for i in range(n):
            for ch in range(c):
                ma = moving_average_1d(out[i, ch], cfg.temporal_smooth_window)
                out[i, ch] = (1.0 - smooth) * out[i, ch] + smooth * ma

    # Ramp clipping to avoid physically implausible jumps from finite-sample copula sampling.
    for i in range(n):
        for ch in range(c):
            lim = float(model.ramp_abs_q[ch]) * 1.15
            if not np.isfinite(lim) or lim <= 0:
                continue
            for tt in range(1, t):
                diff = out[i, ch, tt] - out[i, ch, tt - 1]
                if diff > lim:
                    out[i, ch, tt] = out[i, ch, tt - 1] + lim
                elif diff < -lim:
                    out[i, ch, tt] = out[i, ch, tt - 1] - lim
    return np.clip(out, 0.0, None)


def infer_day_mask_from_meta_or_length(n: int, t: int, cfg: CopulaConfig, meta_or_cond: Optional[pd.DataFrame] = None) -> np.ndarray:
    h0 = int(np.clip(cfg.solar_zero_before_hour, 0, 23))
    h1 = int(np.clip(cfg.solar_zero_after_hour, 1, 24))
    if h1 <= h0:
        h1 = min(24, h0 + 1)
    masks = np.ones((n, t), dtype=np.float64)

    starts = None
    if meta_or_cond is not None:
        for col in ["window_start_time", "start_time", "core_start_time", "time"]:
            if col in meta_or_cond.columns:
                starts = pd.to_datetime(meta_or_cond[col], errors="coerce")
                break
    for i in range(n):
        if cfg.use_time_index_for_solar_mask and starts is not None and i < len(starts) and pd.notna(starts.iloc[i]):
            start = starts.iloc[i]
            hours = np.array([(start + pd.Timedelta(hours=int(k))).hour for k in range(t)], dtype=np.int32)
        else:
            hours = np.arange(t, dtype=np.int32) % 24
        day = (hours >= h0) & (hours < h1)
        masks[i] = day.astype(np.float64)
    return masks


def physical_projection_np(x: np.ndarray, cond_rows: Optional[pd.DataFrame], cfg: CopulaConfig) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    out[:, 0, :] = np.clip(out[:, 0, :], 0.0, None)
    out[:, 1, :] = np.clip(out[:, 1, :], 0.0, None)
    out[:, 2, :] = np.clip(out[:, 2, :], 0.0, None)
    mask = infer_day_mask_from_meta_or_length(out.shape[0], out.shape[2], cfg, cond_rows)
    out[:, 2, :] *= mask
    return out


def compute_risk_metrics_np(x: np.ndarray, delta_t: float = 1.0, tau: float = 0.0) -> pd.DataFrame:
    x = ensure_nct(x)
    load = x[:, 0, :]
    wind = x[:, 1, :]
    solar = x[:, 2, :]
    net = load - wind - solar
    cum = np.maximum(net - tau, 0.0).sum(axis=1) * delta_t
    ramp = np.diff(net, axis=1)
    ramp_max = ramp.max(axis=1) if ramp.shape[1] else np.zeros(x.shape[0])
    dur = (net > tau).sum(axis=1) * delta_t
    return pd.DataFrame({"cum_deficit": cum, "netload_ramp_max": ramp_max, "imbalance_duration": dur})


def js_divergence(x: np.ndarray, y: np.ndarray, bins: int = 80) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if hi <= lo:
        return 0.0
    px, edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    py, _ = np.histogram(y, bins=edges, density=False)
    px = px.astype(np.float64) + 1e-12
    py = py.astype(np.float64) + 1e-12
    px /= px.sum()
    py /= py.sum()
    m = 0.5 * (px + py)
    return float(0.5 * np.sum(px * np.log(px / m)) + 0.5 * np.sum(py * np.log(py / m)))


def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.ones(max_lag + 1)
    x = x - x.mean()
    denom = float(np.dot(x, x)) + 1e-12
    out = np.ones(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        if lag >= x.size:
            out[lag] = np.nan
        else:
            out[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return out


def mean_acf_for_samples(x: np.ndarray, channel: int, max_lag: int) -> np.ndarray:
    vals = []
    for i in range(x.shape[0]):
        vals.append(acf_1d(x[i, channel, :], max_lag))
    return np.nanmean(np.vstack(vals), axis=0)


def corr_matrix(x: np.ndarray) -> np.ndarray:
    # Aggregate all time points and samples for [load, wind, solar].
    xt = np.transpose(x, (0, 2, 1)).reshape(-1, 3)
    if xt.shape[0] < 3:
        return np.eye(3)
    c = np.corrcoef(xt.T)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def severity_from_quantiles(cum: np.ndarray, train_ref: Optional[np.ndarray] = None) -> np.ndarray:
    ref = np.asarray(train_ref if train_ref is not None else cum, dtype=np.float64)
    q50, q75, q90 = np.quantile(ref, [0.50, 0.75, 0.90])
    out = np.zeros_like(cum, dtype=np.int32)
    out[cum > q50] = 1
    out[cum > q75] = 2
    out[cum > q90] = 3
    return out


def metric_errors(real_v: np.ndarray, gen_v: np.ndarray, prefix: str) -> dict:
    real_v = np.asarray(real_v, dtype=np.float64)
    gen_v = np.asarray(gen_v, dtype=np.float64)
    return {
        f"{prefix}_mae": float(abs(np.mean(gen_v) - np.mean(real_v))),
        f"{prefix}_relative_error": float(abs(np.mean(gen_v) - np.mean(real_v)) / (abs(np.mean(real_v)) + 1e-8)),
        f"{prefix}_q95_error": float(abs(np.quantile(gen_v, 0.95) - np.quantile(real_v, 0.95))),
        f"{prefix}_q99_error": float(abs(np.quantile(gen_v, 0.99) - np.quantile(real_v, 0.99))),
    }


def evaluate_generation(
    real: np.ndarray,
    gen: np.ndarray,
    cond: Optional[pd.DataFrame],
    out_dir: Path,
    model_name: str,
    cfg: CopulaConfig,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    real = ensure_nct(real)
    gen = ensure_nct(gen)

    # Align if generated has n_per_condition > 1.
    if gen.shape[0] != real.shape[0] and gen.shape[0] % real.shape[0] == 0:
        rep = gen.shape[0] // real.shape[0]
        real_eval = np.repeat(real, rep, axis=0)
    else:
        real_eval = real[: gen.shape[0]] if real.shape[0] >= gen.shape[0] else np.resize(real, (gen.shape[0], real.shape[1], real.shape[2]))

    metrics = {"model_name": model_name, "n_real": int(real.shape[0]), "n_generated": int(gen.shape[0])}
    w_vals = []
    js_vals = []
    acf_vals = []
    for ci, ch in enumerate(CHANNELS):
        rv = real_eval[:, ci, :].reshape(-1)
        gv = gen[:, ci, :].reshape(-1)
        w = float(wasserstein_distance(rv, gv))
        js = js_divergence(rv, gv, bins=cfg.js_bins)
        acf_real = mean_acf_for_samples(real_eval, ci, min(cfg.acf_max_lag, real.shape[2] - 1))
        acf_gen = mean_acf_for_samples(gen, ci, min(cfg.acf_max_lag, gen.shape[2] - 1))
        acf_mae = float(np.nanmean(np.abs(acf_real - acf_gen)))
        metrics[f"wasserstein_{ch}"] = w
        metrics[f"js_{ch}"] = js
        metrics[f"acf_mae_{ch}"] = acf_mae
        w_vals.append(w)
        js_vals.append(js)
        acf_vals.append(acf_mae)
    metrics["wasserstein_mean"] = float(np.mean(w_vals))
    metrics["js_mean"] = float(np.mean(js_vals))
    metrics["acf_mae_mean"] = float(np.mean(acf_vals))

    corr_r = corr_matrix(real_eval)
    corr_g = corr_matrix(gen)
    metrics["corr_matrix_error_fro"] = float(np.linalg.norm(corr_r - corr_g, ord="fro"))
    metrics["corr_matrix_error_mae"] = float(np.mean(np.abs(corr_r - corr_g)))

    risk_real = compute_risk_metrics_np(real_eval, cfg.delta_t, cfg.imbalance_tau)
    risk_gen = compute_risk_metrics_np(gen, cfg.delta_t, cfg.imbalance_tau)
    for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
        metrics.update(metric_errors(risk_real[col].values, risk_gen[col].values, col))
    # legacy short names for paper table convenience
    metrics["cum_deficit_error"] = metrics["cum_deficit_relative_error"]
    metrics["netload_ramp_max_error"] = metrics["netload_ramp_max_relative_error"]
    metrics["imbalance_duration_error"] = metrics["imbalance_duration_relative_error"]

    # Extreme degree match rate if severity labels are provided.
    if cond is not None and "severity_level" in cond.columns:
        cond_eval = cond.copy().reset_index(drop=True)
        if len(cond_eval) != gen.shape[0] and gen.shape[0] % len(cond_eval) == 0:
            cond_eval = pd.concat([cond_eval] * (gen.shape[0] // len(cond_eval)), ignore_index=True)
        cond_eval = cond_eval.iloc[: gen.shape[0]].reset_index(drop=True)
        true_sev = pd.to_numeric(cond_eval["severity_level"], errors="coerce").fillna(0).astype(int).values
        pred_sev = severity_from_quantiles(risk_gen["cum_deficit"].values, risk_real["cum_deficit"].values)
        metrics["extreme_degree_match_rate_strict"] = float(np.mean(pred_sev == true_sev))
        metrics["extreme_degree_match_rate_adjacent"] = float(np.mean(np.abs(pred_sev - true_sev) <= 1))
        risk_out = pd.concat([cond_eval[[c for c in cond_eval.columns if c in ["sample_id", "event_type", "event_type_code", "severity_level"]]].reset_index(drop=True), risk_gen.add_prefix("gen_")], axis=1)
    else:
        risk_out = risk_gen.add_prefix("gen_")

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([risk_real.add_prefix("real_"), risk_gen.add_prefix("gen_")], axis=1).to_csv(
        out_dir / "risk_metrics_real_vs_generated.csv", index=False, encoding="utf-8-sig"
    )
    risk_out.to_csv(out_dir / "generated_risk_metrics_with_condition.csv", index=False, encoding="utf-8-sig")
    (out_dir / "evaluation_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savetxt(out_dir / "corr_matrix_real.csv", corr_r, delimiter=",")
    np.savetxt(out_dir / "corr_matrix_generated.csv", corr_g, delimiter=",")

    if cond is not None:
        by_cols = []
        if "event_type" in cond.columns:
            by_cols.append("event_type")
        elif "event_type_code" in cond.columns:
            by_cols.append("event_type_code")
        if by_cols:
            rows = []
            cond_short = cond.reset_index(drop=True)
            if len(cond_short) != gen.shape[0] and gen.shape[0] % len(cond_short) == 0:
                cond_short = pd.concat([cond_short] * (gen.shape[0] // len(cond_short)), ignore_index=True)
            cond_short = cond_short.iloc[: gen.shape[0]].reset_index(drop=True)
            for val, idx in cond_short.groupby(by_cols[0]).groups.items():
                idx = np.asarray(list(idx), dtype=np.int64)
                if len(idx) < 1:
                    continue
                sub_real = real_eval[idx]
                sub_gen = gen[idx]
                rr = compute_risk_metrics_np(sub_real, cfg.delta_t, cfg.imbalance_tau)
                rg = compute_risk_metrics_np(sub_gen, cfg.delta_t, cfg.imbalance_tau)
                row = {by_cols[0]: val, "n": int(len(idx))}
                for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
                    row.update(metric_errors(rr[col].values, rg[col].values, col))
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "metrics_by_event_type.csv", index=False, encoding="utf-8-sig")
        if "severity_level" in cond.columns:
            rows = []
            cond_short = cond.reset_index(drop=True)
            if len(cond_short) != gen.shape[0] and gen.shape[0] % len(cond_short) == 0:
                cond_short = pd.concat([cond_short] * (gen.shape[0] // len(cond_short)), ignore_index=True)
            cond_short = cond_short.iloc[: gen.shape[0]].reset_index(drop=True)
            for val, idx in cond_short.groupby("severity_level").groups.items():
                idx = np.asarray(list(idx), dtype=np.int64)
                rr = compute_risk_metrics_np(real_eval[idx], cfg.delta_t, cfg.imbalance_tau)
                rg = compute_risk_metrics_np(gen[idx], cfg.delta_t, cfg.imbalance_tau)
                row = {"severity_level": val, "n": int(len(idx))}
                for col in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
                    row.update(metric_errors(rr[col].values, rg[col].values, col))
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "metrics_by_severity_level.csv", index=False, encoding="utf-8-sig")

    if cfg.make_plots:
        make_plots(real_eval, gen, risk_real, risk_gen, corr_r, corr_g, fig_dir, cfg)
    return metrics


def make_plots(real: np.ndarray, gen: np.ndarray, risk_real: pd.DataFrame, risk_gen: pd.DataFrame, corr_r: np.ndarray, corr_g: np.ndarray, fig_dir: Path, cfg: CopulaConfig) -> None:
    t = np.arange(real.shape[2])
    idx = 0
    plt.figure(figsize=(12, 7))
    for c, name in enumerate(CHANNELS):
        ax = plt.subplot(3, 1, c + 1)
        ax.plot(t, real[idx, c], label="Real", linewidth=1.5)
        ax.plot(t, gen[idx, c], label="Generated", linewidth=1.2, alpha=0.85)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        if c == 0:
            ax.legend(loc="upper right")
    plt.xlabel("time step")
    plt.tight_layout()
    plt.savefig(fig_dir / "typical_generated_curve.png", dpi=180)
    plt.close()

    net_r = real[idx, 0] - real[idx, 1] - real[idx, 2]
    net_g = gen[idx, 0] - gen[idx, 1] - gen[idx, 2]
    plt.figure(figsize=(10, 4))
    plt.plot(t, net_r, label="Real net load", linewidth=1.6)
    plt.plot(t, net_g, label="Generated net load", linewidth=1.3)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "netload_curve.png", dpi=180)
    plt.close()

    max_lag = min(cfg.acf_max_lag, real.shape[2] - 1)
    lags = np.arange(max_lag + 1)
    plt.figure(figsize=(10, 5))
    for c, name in enumerate(CHANNELS):
        plt.plot(lags, mean_acf_for_samples(real, c, max_lag), label=f"Real {name}", linestyle="-")
        plt.plot(lags, mean_acf_for_samples(gen, c, max_lag), label=f"Gen {name}", linestyle="--")
    plt.grid(alpha=0.25)
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "acf_comparison.png", dpi=180)
    plt.close()

    for mat, name in [(corr_r, "corr_matrix_real.png"), (corr_g, "corr_matrix_generated.png")]:
        plt.figure(figsize=(4.8, 4))
        plt.imshow(mat, vmin=-1, vmax=1)
        plt.xticks(np.arange(3), CHANNELS, rotation=30)
        plt.yticks(np.arange(3), CHANNELS)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=180)
        plt.close()

    plt.figure(figsize=(10, 5))
    data = [risk_real["cum_deficit"], risk_gen["cum_deficit"], risk_real["netload_ramp_max"], risk_gen["netload_ramp_max"], risk_real["imbalance_duration"], risk_gen["imbalance_duration"]]
    plt.boxplot(data, labels=["cum R", "cum G", "ramp R", "ramp G", "dur R", "dur G"], showfliers=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "risk_metric_boxplot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    rr = np.sort(risk_real["cum_deficit"].values)
    gg = np.sort(risk_gen["cum_deficit"].values)
    plt.plot(np.linspace(0, 1, rr.size), rr, label="Real")
    plt.plot(np.linspace(0, 1, gg.size), gg, label="Generated")
    plt.xlabel("empirical probability")
    plt.ylabel("cum_deficit")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "tail_distribution_comparison.png", dpi=180)
    plt.close()


def save_generated_long(x: np.ndarray, cond: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for i in range(x.shape[0]):
        base = {}
        if cond is not None and i < len(cond):
            for col in ["sample_id", "generated_id", "event_type", "event_type_code", "month", "severity_level", "extreme_prob", "copula_model_group"]:
                if col in cond.columns:
                    base[col] = cond.iloc[i][col]
        for tt in range(x.shape[2]):
            rows.append({**base, "generated_index": i, "t": tt, "load": x[i, 0, tt], "wind_power": x[i, 1, tt], "solar_power": x[i, 2, tt]})
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def fit_command(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, cond, meta = load_dataset(cfg.data_dir)
    if cfg.seq_len > 0 and x.shape[2] != cfg.seq_len:
        raise ValueError(f"seq_len mismatch: data has {x.shape[2]}, expected {cfg.seq_len}")
    split_paths = resolve_existing_split(cfg.data_dir, out_dir / "dataset_split")
    if split_paths is None:
        split_paths = save_split_dataset(x, cond, meta, out_dir / "dataset_split", cfg)
    x_train = np.load(split_paths["X_train"])
    cond_train = pd.read_csv(split_paths["cond_train"])
    model = TraditionalCopulaBaseline(cfg)
    fit_info = model.fit(x_train, cond_train)
    model.save(out_dir / "traditional_copula_model.pkl")
    pd.DataFrame(fit_info["group_summary"]).to_csv(out_dir / "group_fit_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fit_summary.json").write_text(json.dumps(fit_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Traditional Gaussian Copula baseline fitted.")
    print(f"Saved to: {out_dir.resolve()}")


def generate_command(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    rng = set_seed(cfg.seed)
    model_dir = Path(args.model_dir or cfg.output_dir)
    out_dir = Path(args.output_dir or (model_dir / "generation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model = TraditionalCopulaBaseline.load(model_dir / "traditional_copula_model.pkl")
    cond_csv = Path(args.cond_csv) if args.cond_csv else (model_dir / "dataset_split" / "cond_test.csv")
    cond = pd.read_csv(cond_csv)
    x_gen, cond_gen = model.sample_conditions(cond, int(args.n_per_condition or cfg.n_per_condition), rng)
    np.save(out_dir / "generated_samples.npy", x_gen.astype(np.float32))
    cond_gen.to_csv(out_dir / "generated_conditions.csv", index=False, encoding="utf-8-sig")
    save_generated_long(x_gen, cond_gen, out_dir / "generated_samples_long.csv")
    summary = {"n_generated": int(x_gen.shape[0]), "shape": list(x_gen.shape), "cond_csv": str(cond_csv), "model_dir": str(model_dir)}
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated samples saved.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_command(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    real = ensure_nct(np.load(args.real))
    gen = ensure_nct(np.load(args.generated))
    cond = pd.read_csv(args.cond) if args.cond else None
    metrics = evaluate_generation(real, gen, cond, Path(args.output_dir), args.model_name, cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def pipeline_command(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = set_seed(cfg.seed)

    x, cond, meta = load_dataset(cfg.data_dir)
    split_paths = resolve_existing_split(cfg.data_dir, out_dir / "dataset_split")
    if split_paths is None:
        split_paths = save_split_dataset(x, cond, meta, out_dir / "dataset_split", cfg)
    x_train = np.load(split_paths["X_train"])
    x_test = np.load(split_paths["X_test"])
    cond_train = pd.read_csv(split_paths["cond_train"])
    cond_test = pd.read_csv(split_paths["cond_test"])

    model = TraditionalCopulaBaseline(cfg)
    fit_info = model.fit(x_train, cond_train)
    model.save(out_dir / "traditional_copula_model.pkl")
    pd.DataFrame(fit_info["group_summary"]).to_csv(out_dir / "group_fit_summary.csv", index=False, encoding="utf-8-sig")

    x_gen, cond_gen = model.sample_conditions(cond_test, cfg.n_per_condition, rng)
    gen_dir = out_dir / "generation"
    gen_dir.mkdir(exist_ok=True)
    np.save(gen_dir / "generated_samples.npy", x_gen.astype(np.float32))
    cond_gen.to_csv(gen_dir / "generated_conditions.csv", index=False, encoding="utf-8-sig")
    save_generated_long(x_gen, cond_gen, gen_dir / "generated_samples_long.csv")

    eval_dir = out_dir / "evaluation"
    metrics = evaluate_generation(x_test, x_gen, cond_gen, eval_dir, "traditional_gaussian_copula", cfg)
    summary = {
        "method": "traditional_gaussian_copula",
        "role": "传统统计基线：极端样本分组-边缘分布拟合-Gaussian Copula联合建模-抽样生成联合场景-时序修正-反归一化与物理约束",
        "output_dir": str(out_dir.resolve()),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_generated": int(x_gen.shape[0]),
        "shape_generated": list(x_gen.shape),
        "metrics": metrics,
    }
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== Traditional Gaussian Copula baseline pipeline finished ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def config_from_args(args: argparse.Namespace) -> CopulaConfig:
    cfg = CopulaConfig()
    for k in asdict(cfg).keys():
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    return cfg


def add_common_args(p: argparse.ArgumentParser) -> None:
    defaults = CopulaConfig()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--test-ratio", type=float, default=None)
    p.add_argument("--group-cols", default=None)
    p.add_argument("--fallback-group-cols", default=None)
    p.add_argument("--min-group-size", type=int, default=None)
    p.add_argument("--covariance-shrinkage", type=float, default=None)
    p.add_argument("--quantile-grid-size", type=int, default=None)
    p.add_argument("--n-per-condition", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--delta-t", type=float, default=None)
    p.add_argument("--imbalance-tau", type=float, default=None)
    p.add_argument("--acf-max-lag", type=int, default=None)
    p.add_argument("--js-bins", type=int, default=None)
    p.add_argument("--solar-zero-before-hour", type=int, default=None)
    p.add_argument("--solar-zero-after-hour", type=int, default=None)
    p.add_argument("--ramp-clip-quantile", type=float, default=None)
    p.add_argument("--temporal-smooth-strength", type=float, default=None)
    p.add_argument("--temporal-smooth-window", type=int, default=None)
    p.add_argument("--no-plots", dest="make_plots", action="store_false", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traditional statistical baseline: grouped empirical marginals + Gaussian Copula + temporal/physical correction.")
    sub = p.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit")
    add_common_args(p_fit)
    p_fit.set_defaults(func=fit_command)

    p_gen = sub.add_parser("generate")
    add_common_args(p_gen)
    p_gen.add_argument("--model-dir", default=None)
    p_gen.add_argument("--cond-csv", default=None)
    p_gen.set_defaults(func=generate_command)

    p_eval = sub.add_parser("evaluate")
    add_common_args(p_eval)
    p_eval.add_argument("--real", required=True)
    p_eval.add_argument("--generated", required=True)
    p_eval.add_argument("--cond", default="")
    p_eval.add_argument("--model-name", default="traditional_gaussian_copula")
    p_eval.set_defaults(func=evaluate_command)

    p_pipe = sub.add_parser("pipeline")
    add_common_args(p_pipe)
    p_pipe.set_defaults(func=pipeline_command)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
