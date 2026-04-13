from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass
class Build8760Config:
    data_dir: str = ""
    output_dir: str = "background_8760_outputs"
    start_time: str = "2025-01-01 00:00:00"
    random_seed: int = 42
    n_states: int = 6
    kmeans_n_init: int = 12
    kmeans_max_iter: int = 80
    transition_weight: float = 0.75
    continuity_top_k: int = 12
    continuity_temp: float = 0.25
    shape_score_weight: float = 0.70
    shape_weight_p95: float = 1.00
    shape_weight_p05: float = 1.00
    shape_weight_ramp_std: float = 1.20
    smooth_hours: int = 4
    solar_zero_before_hour: int = 6
    solar_zero_after_hour: int = 20
    boundary_target_quantile: float = 0.60
    boundary_smooth_strength: float = 0.75
    onshore_weight: float = 1.0
    offshore_weight: float = 0.6
    utility_pv_weight: float = 1.0
    distributed_pv_weight: float = 0.7
    csp_weight: float = 0.0
    wind_monthly_rebalance_strength: float = 0.85
    wind_scale_clip_min: float = 0.70
    wind_scale_clip_max: float = 1.30
    solar_monthly_rebalance_strength: float = 0.75
    solar_scale_clip_min: float = 0.75
    solar_scale_clip_max: float = 1.25
    net_monthly_rebalance_strength: float = 0.60
    net_rebalance_weight_component: float = 1.0
    net_rebalance_weight_net: float = 2.0
    net_rebalance_weight_reg: float = 0.35
    net_load_scale_clip_min: float = 0.93
    net_load_scale_clip_max: float = 1.07
    net_wind_scale_clip_min: float = 0.85
    net_wind_scale_clip_max: float = 1.15
    monthly_qmap_strength: float = 0.35
    monthly_qmap_load_mult: float = 1.00
    monthly_qmap_wind_mult: float = 0.80
    monthly_qmap_solar_mult: float = 0.00
    monthly_qmap_preserve_month_sum: bool = True
    monthly_qmap_low_q: float = 0.005
    monthly_qmap_high_q: float = 0.995
    make_plots: bool = True


REQUIRED_FILES = {
    "load": ("Load_data_8760.mat", "load_data"),
    "wind_onshore": ("Onshore_wind_data_8760.mat", "wind_data"),
    "wind_offshore": ("Offshore_wind_data_8760.mat", "wind_data_ofs"),
    "solar_utility": ("Utility_PV_data_8760.mat", "solar_data"),
    "solar_distributed": ("Distributed_PV_data_8760.mat", "solar_data_dis"),
    "solar_csp": ("CSP_data_8760.mat", "solar_data"),
}


def _month_vector_365() -> np.ndarray:
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_vec = np.concatenate(
        [np.full(days, month, dtype=np.int32) for month, days in enumerate(month_days, start=1)]
    )
    if month_vec.size != 365:
        raise ValueError("Month calendar must be 365 days.")
    return month_vec


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + 1e-8)


def _find_mat_file(file_name: str, data_dir: str) -> Path:
    if data_dir:
        p = Path(data_dir) / file_name
        if p.exists():
            return p
        raise FileNotFoundError(f"Cannot find {file_name} in {Path(data_dir).resolve()}")

    root = Path.cwd()
    for p in root.rglob(file_name):
        if ".venv" in p.parts:
            continue
        return p
    raise FileNotFoundError(f"Cannot find {file_name} under {root}")


def _load_numeric_array(mat_path: Path, preferred_key: str) -> tuple[np.ndarray, str]:
    d = loadmat(mat_path)
    keys = [k for k in d.keys() if not k.startswith("__")]
    if preferred_key in d:
        keys = [preferred_key] + [k for k in keys if k != preferred_key]

    for key in keys:
        arr = np.asarray(d[key])
        if np.issubdtype(arr.dtype, np.number) and arr.ndim == 3:
            return arr.astype(np.float64), key
    raise ValueError(f"No 3D numeric array found in {mat_path}")


def _as_year_day_hour(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")

    hour_axes = [i for i, s in enumerate(arr.shape) if s == 24]
    day_axes = [i for i, s in enumerate(arr.shape) if s == 365]
    if not hour_axes or not day_axes:
        raise ValueError(f"Cannot infer hour/day axes from shape {arr.shape}")

    h_axis = hour_axes[0]
    d_axis = next((ax for ax in day_axes if ax != h_axis), day_axes[0])
    if d_axis == h_axis:
        raise ValueError(f"Hour axis and day axis collide for shape {arr.shape}")

    y_axis = [i for i in range(3) if i not in (h_axis, d_axis)]
    if len(y_axis) != 1:
        raise ValueError(f"Cannot infer year axis from shape {arr.shape}")
    y_axis = y_axis[0]

    out = np.transpose(arr, (y_axis, d_axis, h_axis))
    if out.shape[1] != 365 or out.shape[2] != 24:
        raise ValueError(f"Unexpected reordered shape {out.shape} from {arr.shape}")
    return out


def _load_profiles(cfg: Build8760Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    loaded = {}
    meta = {}
    for name, (file_name, key_name) in REQUIRED_FILES.items():
        mat_path = _find_mat_file(file_name, cfg.data_dir)
        raw, used_key = _load_numeric_array(mat_path, key_name)
        arr = _as_year_day_hour(raw)
        loaded[name] = arr
        meta[name] = {"path": str(mat_path), "key": used_key, "shape": list(arr.shape)}

    ref_shape = loaded["load"].shape
    for name, arr in loaded.items():
        if arr.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {name}: {arr.shape} vs {ref_shape}")

    wind = (
        cfg.onshore_weight * loaded["wind_onshore"]
        + cfg.offshore_weight * loaded["wind_offshore"]
    )
    solar = (
        cfg.utility_pv_weight * loaded["solar_utility"]
        + cfg.distributed_pv_weight * loaded["solar_distributed"]
        + cfg.csp_weight * loaded["solar_csp"]
    )
    load = loaded["load"]
    return load, wind, solar, meta


def _daily_features(day_profiles: np.ndarray) -> np.ndarray:
    load = day_profiles[:, :, 0]
    wind = day_profiles[:, :, 1]
    solar = day_profiles[:, :, 2]
    net = load - wind - solar

    feat = np.column_stack(
        [
            load.sum(axis=1),
            wind.sum(axis=1),
            solar.sum(axis=1),
            net.sum(axis=1),
            np.maximum(net, 0.0).sum(axis=1),
            np.abs(np.diff(net, axis=1)).mean(axis=1),
            net.max(axis=1),
        ]
    )
    return feat


def _shape_features(day_profiles: np.ndarray) -> np.ndarray:
    net = day_profiles[:, :, 0] - day_profiles[:, :, 1] - day_profiles[:, :, 2]
    ramp = np.diff(net, axis=1)
    feat = np.column_stack(
        [
            np.quantile(net, 0.95, axis=1),
            np.quantile(net, 0.05, axis=1),
            np.std(ramp, axis=1),
        ]
    )
    return feat


def _kmeans_numpy(
    x: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("kmeans input must be 2D")
    n_samples = x.shape[0]
    if n_clusters < 2 or n_clusters > n_samples:
        raise ValueError("Invalid n_clusters")

    rng = np.random.default_rng(random_seed)
    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for _ in range(n_init):
        idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = x[idx].copy()

        for _ in range(max_iter):
            dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist2.argmin(axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = x[mask].mean(axis=0)
                else:
                    far_idx = np.argmax(dist2.min(axis=1))
                    new_centers[k] = x[far_idx]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < 1e-6:
                break

        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1)
        inertia = ((x - centers[labels]) ** 2).sum()

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("KMeans failed to produce a solution")
    return best_labels.astype(np.int32), best_centers


def _build_month_freq(month_flat: np.ndarray, state_flat: np.ndarray, n_states: int) -> np.ndarray:
    freq = np.zeros((12, n_states), dtype=np.float64)
    for m in range(1, 13):
        mask = month_flat == m
        counts = np.bincount(state_flat[mask], minlength=n_states).astype(np.float64)
        s = counts.sum()
        if s > 0:
            freq[m - 1] = counts / s
        else:
            freq[m - 1] = 1.0 / n_states
    return freq


def _build_month_transition(
    month_doy: np.ndarray,
    state_yd: np.ndarray,
    month_freq: np.ndarray,
    n_states: int,
) -> np.ndarray:
    n_years = state_yd.shape[0]
    trans = np.zeros((12, n_states, n_states), dtype=np.float64)
    month_next = month_doy[1:]

    for y in range(n_years):
        s_prev = state_yd[y, :-1]
        s_next = state_yd[y, 1:]
        for m in range(1, 13):
            idx = np.where(month_next == m)[0]
            if idx.size == 0:
                continue
            np.add.at(trans[m - 1], (s_prev[idx], s_next[idx]), 1.0)

    for m in range(12):
        for s in range(n_states):
            row_sum = trans[m, s].sum()
            if row_sum > 0:
                trans[m, s] /= row_sum
            else:
                trans[m, s] = month_freq[m]
    return trans


def _sample_state_path(
    month_doy: np.ndarray,
    month_freq: np.ndarray,
    month_trans: np.ndarray,
    transition_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_days = month_doy.size
    n_states = month_freq.shape[1]
    out = np.zeros(n_days, dtype=np.int32)

    for d in range(n_days):
        m_idx = month_doy[d] - 1
        if d == 0:
            p = month_freq[m_idx].copy()
        else:
            p = transition_weight * month_trans[m_idx, out[d - 1]] + (1.0 - transition_weight) * month_freq[m_idx]

        p = np.clip(p, 0.0, None)
        ps = p.sum()
        if ps <= 0:
            p = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        else:
            p /= ps
        out[d] = int(rng.choice(n_states, p=p))

    return out


def _pick_profile_indices(
    day_profiles: np.ndarray,
    month_flat: np.ndarray,
    state_flat: np.ndarray,
    gen_month: np.ndarray,
    gen_state: np.ndarray,
    top_k: int,
    temp: float,
    shape_score_weight: float,
    shape_weight_p95: float,
    shape_weight_p05: float,
    shape_weight_ramp_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_hist = day_profiles.shape[0]
    if n_hist == 0:
        raise ValueError("No historical day profiles available")

    n_days = gen_month.size
    chosen = np.zeros(n_days, dtype=np.int32)
    pools_month = {m: np.where(month_flat == m)[0] for m in range(1, 13)}
    pools_month_state = {
        (m, s): np.where((month_flat == m) & (state_flat == s))[0]
        for m in range(1, 13)
        for s in range(int(state_flat.max()) + 1)
    }

    scale = np.std(day_profiles[:, 0, :], axis=0) + 1e-6
    shape_feat = _shape_features(day_profiles)
    shape_scale = np.std(shape_feat, axis=0) + 1e-6
    shape_weight = np.array([shape_weight_p95, shape_weight_p05, shape_weight_ramp_std], dtype=np.float64)
    shape_weight = np.maximum(shape_weight, 0.0)
    if shape_weight.sum() <= 0:
        shape_weight = np.ones_like(shape_weight)
    shape_weight = shape_weight / shape_weight.sum()

    global_shape_target = np.median(shape_feat, axis=0)
    month_shape_target = {}
    month_state_shape_target = {}
    for m in range(1, 13):
        pool_m = pools_month.get(m, np.array([], dtype=np.int32))
        if pool_m.size > 0:
            month_shape_target[m] = np.median(shape_feat[pool_m], axis=0)
        else:
            month_shape_target[m] = global_shape_target

    n_states = int(state_flat.max()) + 1
    for m in range(1, 13):
        for s in range(n_states):
            pool_ms = pools_month_state.get((m, s), np.array([], dtype=np.int32))
            if pool_ms.size > 0:
                month_state_shape_target[(m, s)] = np.median(shape_feat[pool_ms], axis=0)
            else:
                month_state_shape_target[(m, s)] = month_shape_target[m]

    top_k = max(1, int(top_k))
    temp = max(1e-6, float(temp))
    shape_score_weight = max(0.0, float(shape_score_weight))

    selected_profiles = np.zeros((n_days, 24, 3), dtype=np.float64)
    for d in range(n_days):
        m = int(gen_month[d])
        s = int(gen_state[d])
        cands = pools_month_state.get((m, s), np.array([], dtype=np.int32))
        if cands.size == 0:
            cands = pools_month.get(m, np.array([], dtype=np.int32))
        if cands.size == 0:
            cands = np.arange(n_hist, dtype=np.int32)

        if d == 0:
            continuity_score = np.zeros(cands.size, dtype=np.float64)
        else:
            prev_last = selected_profiles[d - 1, -1, :]
            start_vals = day_profiles[cands, 0, :]
            continuity_score = np.sum(np.abs(start_vals - prev_last[None, :]) / scale[None, :], axis=1)

        target_shape = month_state_shape_target.get((m, s), month_shape_target.get(m, global_shape_target))
        shape_score = np.sum(
            np.abs(shape_feat[cands] - target_shape[None, :]) / shape_scale[None, :] * shape_weight[None, :],
            axis=1,
        )
        score = continuity_score + shape_score_weight * shape_score

        if cands.size > top_k:
            keep = np.argpartition(score, top_k - 1)[:top_k]
            cands = cands[keep]
            score = score[keep]

        score = score - score.min()
        w = np.exp(-score / temp)
        if np.isfinite(w).all() and w.sum() > 0:
            w /= w.sum()
            idx = int(rng.choice(cands, p=w))
        else:
            idx = int(rng.choice(cands))

        chosen[d] = idx
        selected_profiles[d] = day_profiles[idx]

    return chosen


def _rebalance_wind_monthly(
    hourly: np.ndarray,
    hist_wind_hourly: np.ndarray,
    month_doy: np.ndarray,
    strength: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = hourly.copy()
    if strength <= 0:
        return out, np.ones(12, dtype=np.float64)

    month_hour = np.repeat(month_doy, 24)
    scales = np.ones(12, dtype=np.float64)
    hist_monthly = np.zeros((hist_wind_hourly.shape[0], 12), dtype=np.float64)
    gen_monthly = np.zeros(12, dtype=np.float64)

    for m in range(1, 13):
        mask = month_hour == m
        hist_monthly[:, m - 1] = hist_wind_hourly[:, mask].sum(axis=1)
        gen_monthly[m - 1] = out[mask, 1].sum()

    target = np.median(hist_monthly, axis=0)
    ratio = target / (gen_monthly + 1e-12)
    ratio = np.clip(ratio, clip_min, clip_max)
    scales = 1.0 + strength * (ratio - 1.0)

    for m in range(1, 13):
        mask = month_hour == m
        out[mask, 1] *= scales[m - 1]

    out[:, 1] = np.clip(out[:, 1], 0.0, None)
    return out, scales


def _rebalance_solar_monthly(
    hourly: np.ndarray,
    hist_solar_hourly: np.ndarray,
    month_doy: np.ndarray,
    strength: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = hourly.copy()
    if strength <= 0:
        return out, np.ones(12, dtype=np.float64)

    month_hour = np.repeat(month_doy, 24)
    scales = np.ones(12, dtype=np.float64)
    hist_monthly = np.zeros((hist_solar_hourly.shape[0], 12), dtype=np.float64)
    gen_monthly = np.zeros(12, dtype=np.float64)

    for m in range(1, 13):
        mask = month_hour == m
        hist_monthly[:, m - 1] = hist_solar_hourly[:, mask].sum(axis=1)
        gen_monthly[m - 1] = out[mask, 2].sum()

    target = np.median(hist_monthly, axis=0)
    ratio = target / (gen_monthly + 1e-12)
    ratio = np.clip(ratio, clip_min, clip_max)
    scales = 1.0 + strength * (ratio - 1.0)

    for m in range(1, 13):
        mask = month_hour == m
        out[mask, 2] *= scales[m - 1]

    out[:, 2] = np.clip(out[:, 2], 0.0, None)
    return out, scales


def _rebalance_netload_monthly_load_wind(
    hourly: np.ndarray,
    hist_hourly: np.ndarray,
    month_doy: np.ndarray,
    strength: float,
    weight_component: float,
    weight_net: float,
    weight_reg: float,
    load_clip_min: float,
    load_clip_max: float,
    wind_clip_min: float,
    wind_clip_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = hourly.copy()
    load_scales = np.ones(12, dtype=np.float64)
    wind_scales = np.ones(12, dtype=np.float64)
    if strength <= 0:
        return out, load_scales, wind_scales

    month_hour = np.repeat(month_doy, 24)
    w_c = max(0.0, float(weight_component))
    w_n = max(0.0, float(weight_net))
    w_r = max(0.0, float(weight_reg))

    hist_load_med = np.zeros(12, dtype=np.float64)
    hist_wind_med = np.zeros(12, dtype=np.float64)
    hist_net_med = np.zeros(12, dtype=np.float64)

    for m in range(1, 13):
        mask = month_hour == m
        h_load = hist_hourly[:, mask, 0].sum(axis=1)
        h_wind = hist_hourly[:, mask, 1].sum(axis=1)
        h_solar = hist_hourly[:, mask, 2].sum(axis=1)
        hist_load_med[m - 1] = np.median(h_load)
        hist_wind_med[m - 1] = np.median(h_wind)
        hist_net_med[m - 1] = np.median(h_load - h_wind - h_solar)

    eps = 1e-8
    for m in range(1, 13):
        mask = month_hour == m
        l = float(out[mask, 0].sum())
        w = float(out[mask, 1].sum())
        s = float(out[mask, 2].sum())
        lt = float(hist_load_med[m - 1])
        wt = float(hist_wind_med[m - 1])
        nt = float(hist_net_med[m - 1])

        # Solve monthly scales (a for load, b for wind):
        # min w_c*(a*l-lt)^2 + w_c*(b*w-wt)^2 + w_n*(a*l-b*w-(nt+s))^2 + w_r*((a-1)^2+(b-1)^2)
        a_l = max(abs(l), eps)
        b_w = max(abs(w), eps)
        A = np.array(
            [
                [np.sqrt(w_c) * a_l, 0.0],
                [0.0, np.sqrt(w_c) * b_w],
                [np.sqrt(w_n) * a_l, -np.sqrt(w_n) * b_w],
                [np.sqrt(w_r), 0.0],
                [0.0, np.sqrt(w_r)],
            ],
            dtype=np.float64,
        )
        y = np.array(
            [
                np.sqrt(w_c) * lt,
                np.sqrt(w_c) * wt,
                np.sqrt(w_n) * (nt + s),
                np.sqrt(w_r) * 1.0,
                np.sqrt(w_r) * 1.0,
            ],
            dtype=np.float64,
        )

        x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a_opt = float(x[0])
        b_opt = float(x[1])

        a = 1.0 + float(strength) * (a_opt - 1.0)
        b = 1.0 + float(strength) * (b_opt - 1.0)
        a = float(np.clip(a, load_clip_min, load_clip_max))
        b = float(np.clip(b, wind_clip_min, wind_clip_max))

        out[mask, 0] *= a
        out[mask, 1] *= b
        load_scales[m - 1] = a
        wind_scales[m - 1] = b

    out[:, 0] = np.clip(out[:, 0], 0.0, None)
    out[:, 1] = np.clip(out[:, 1], 0.0, None)
    return out, load_scales, wind_scales


def _monthly_quantile_mapping(
    hourly: np.ndarray,
    hist_hourly: np.ndarray,
    month_doy: np.ndarray,
    strength: float,
    load_mult: float,
    wind_mult: float,
    solar_mult: float,
    preserve_month_sum: bool,
    low_q: float,
    high_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = hourly.copy()
    shift_abs = np.zeros((12, 3), dtype=np.float64)
    if strength <= 0:
        return out, shift_abs

    low_q = float(np.clip(low_q, 0.0, 0.49))
    high_q = float(np.clip(high_q, 0.51, 1.0))
    if high_q <= low_q:
        high_q = min(1.0, low_q + 0.1)
    strength = float(np.clip(strength, 0.0, 1.0))
    channel_strength = np.array(
        [
            strength * max(0.0, float(load_mult)),
            strength * max(0.0, float(wind_mult)),
            strength * max(0.0, float(solar_mult)),
        ],
        dtype=np.float64,
    )
    channel_strength = np.clip(channel_strength, 0.0, 1.0)

    month_hour = np.repeat(month_doy, 24)
    for m in range(1, 13):
        mask = month_hour == m
        if mask.sum() < 8:
            continue

        for i in range(3):
            if channel_strength[i] <= 0:
                continue
            g = out[mask, i]
            h = hist_hourly[:, mask, i].reshape(-1)
            if g.size < 8 or h.size < 32:
                continue

            before_sum = float(g.sum())
            g_sorted = np.sort(g)
            cdf = np.searchsorted(g_sorted, g, side="right").astype(np.float64) / max(len(g_sorted), 1)
            cdf = np.clip(cdf, low_q, high_q)

            target = np.quantile(h, cdf)
            mapped = (1.0 - channel_strength[i]) * g + channel_strength[i] * target
            if preserve_month_sum and before_sum > 0:
                scale = before_sum / max(float(mapped.sum()), 1e-12)
                mapped = mapped * scale
            shift_abs[m - 1, i] = float(np.mean(np.abs(mapped - g)))
            out[mask, i] = mapped

    out = np.clip(out, 0.0, None)
    return out, shift_abs


def _smooth_day_boundaries(
    hourly: np.ndarray,
    smooth_hours: int,
    hist_boundary_jumps: Optional[np.ndarray] = None,
    target_quantile: float = 0.60,
    smooth_strength: float = 0.75,
) -> np.ndarray:
    if smooth_hours <= 0 or smooth_strength <= 0:
        return hourly

    target_quantile = float(np.clip(target_quantile, 0.05, 0.95))
    if hist_boundary_jumps is not None and hist_boundary_jumps.size > 0:
        target_jump = float(np.quantile(hist_boundary_jumps, target_quantile))
    else:
        target_jump = 0.0

    out = hourly.copy()
    n = out.shape[0]
    for b in range(24, n, 24):
        k = min(int(smooth_hours), n - b)
        if k <= 0:
            continue

        delta = out[b, :] - out[b - 1, :]
        jump = float(np.linalg.norm(delta))
        if target_jump > 0:
            excess = max(0.0, jump - target_jump)
            if excess <= 0:
                continue
            alpha = smooth_strength * (excess / (jump + 1e-12))
        else:
            alpha = smooth_strength

        taper = np.linspace(1.0, 0.0, num=k, endpoint=False)
        out[b : b + k, :] -= taper[:, None] * alpha * delta[None, :]

    return np.clip(out, 0.0, None)


def _enforce_solar_night_zero(hourly: np.ndarray, cfg: Build8760Config) -> np.ndarray:
    out = np.asarray(hourly, dtype=np.float64).copy()
    h0 = int(np.clip(cfg.solar_zero_before_hour, 0, 23))
    h1 = int(np.clip(cfg.solar_zero_after_hour, 1, 24))
    if h1 <= h0:
        h1 = min(24, h0 + 1)
    hh = np.arange(out.shape[0], dtype=np.int32) % 24
    night = (hh < h0) | (hh >= h1)
    out[night, 2] = 0.0
    return np.clip(out, 0.0, None)


def _plot_8760_envelope(
    hist_hourly: np.ndarray,
    gen_hourly: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(gen_hourly.shape[0])
    names = ["Load", "Wind", "Solar", "Net Load"]
    hist_net = hist_hourly[:, :, 0] - hist_hourly[:, :, 1] - hist_hourly[:, :, 2]
    gen_net = gen_hourly[:, 0] - gen_hourly[:, 1] - gen_hourly[:, 2]

    plt.figure(figsize=(14, 12))
    for i, name in enumerate(names):
        ax = plt.subplot(4, 1, i + 1)
        if i < 3:
            hist = hist_hourly[:, :, i]
            gen = gen_hourly[:, i]
        else:
            hist = hist_net
            gen = gen_net

        p10 = np.quantile(hist, 0.10, axis=0)
        p50 = np.quantile(hist, 0.50, axis=0)
        p90 = np.quantile(hist, 0.90, axis=0)

        ax.fill_between(x, p10, p90, alpha=0.22, color="#4c78a8", label="Hist P10-P90")
        ax.plot(x, p50, color="#1f4e79", linewidth=1.2, label="Hist P50")
        ax.plot(x, gen, color="#d62728", linewidth=1.0, alpha=0.9, label="Generated 8760")
        ax.set_xlim(0, len(x) - 1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title("Generated vs Historical Envelope by Hour Index (8760)")
            ax.legend(loc="upper right", ncol=3, fontsize=9)
    plt.xlabel("Hour Index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _monthly_sum(hourly: np.ndarray, month_hour: np.ndarray) -> np.ndarray:
    out = np.zeros((12, hourly.shape[1]), dtype=np.float64)
    for m in range(1, 13):
        mask = month_hour == m
        out[m - 1] = hourly[mask].sum(axis=0)
    return out


def _plot_monthly_energy(
    hist_hourly: np.ndarray,
    gen_hourly: np.ndarray,
    month_doy: np.ndarray,
    out_path: Path,
) -> None:
    month_hour = np.repeat(month_doy, 24)
    hist_net = hist_hourly[:, :, 0] - hist_hourly[:, :, 1] - hist_hourly[:, :, 2]
    gen_net = gen_hourly[:, 0] - gen_hourly[:, 1] - gen_hourly[:, 2]

    hist_stack = np.concatenate([hist_hourly, hist_net[:, :, None]], axis=2)
    gen_stack = np.concatenate([gen_hourly, gen_net[:, None]], axis=1)

    hist_monthly = np.stack([_monthly_sum(hist_stack[y], month_hour) for y in range(hist_stack.shape[0])], axis=0)
    gen_monthly = _monthly_sum(gen_stack, month_hour)

    names = ["Load", "Wind", "Solar", "Net Load"]
    x = np.arange(1, 13)

    plt.figure(figsize=(14, 10))
    for i, name in enumerate(names):
        ax = plt.subplot(2, 2, i + 1)
        p10 = np.quantile(hist_monthly[:, :, i], 0.10, axis=0)
        p50 = np.quantile(hist_monthly[:, :, i], 0.50, axis=0)
        p90 = np.quantile(hist_monthly[:, :, i], 0.90, axis=0)

        ax.fill_between(x, p10, p90, color="#9ecae1", alpha=0.35, label="Hist P10-P90")
        ax.plot(x, p50, color="#1f77b4", linewidth=2.0, marker="o", label="Hist P50")
        ax.plot(x, gen_monthly[:, i], color="#d62728", linewidth=2.0, marker="s", label="Generated")
        ax.set_xticks(x)
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Sum")
        ax.set_title(name)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
    plt.suptitle("Generated vs Historical Monthly Energy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_duration_curve(
    hist_hourly: np.ndarray,
    gen_hourly: np.ndarray,
    out_path: Path,
) -> None:
    hist_net = hist_hourly[:, :, 0] - hist_hourly[:, :, 1] - hist_hourly[:, :, 2]
    gen_net = gen_hourly[:, 0] - gen_hourly[:, 1] - gen_hourly[:, 2]

    hist_stack = np.concatenate([hist_hourly, hist_net[:, :, None]], axis=2)
    gen_stack = np.concatenate([gen_hourly, gen_net[:, None]], axis=1)

    names = ["Load", "Wind", "Solar", "Net Load"]
    ranks = np.arange(1, gen_hourly.shape[0] + 1)

    plt.figure(figsize=(14, 10))
    for i, name in enumerate(names):
        ax = plt.subplot(2, 2, i + 1)
        hist_sorted = -np.sort(-hist_stack[:, :, i], axis=1)
        gen_sorted = -np.sort(-gen_stack[:, i], axis=0)

        p10 = np.quantile(hist_sorted, 0.10, axis=0)
        p50 = np.quantile(hist_sorted, 0.50, axis=0)
        p90 = np.quantile(hist_sorted, 0.90, axis=0)

        ax.fill_between(ranks, p10, p90, color="#74c476", alpha=0.28, label="Hist P10-P90")
        ax.plot(ranks, p50, color="#238b45", linewidth=1.8, label="Hist P50")
        ax.plot(ranks, gen_sorted, color="#d62728", linewidth=1.4, label="Generated")
        ax.set_xlabel("Rank (descending)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} Duration Curve")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_comparison_plots(
    load_y_d_h: np.ndarray,
    wind_y_d_h: np.ndarray,
    solar_y_d_h: np.ndarray,
    gen_hourly: np.ndarray,
    month_doy: np.ndarray,
    out_dir: Path,
) -> list[str]:
    hist_hourly = np.stack([load_y_d_h, wind_y_d_h, solar_y_d_h], axis=-1).reshape(load_y_d_h.shape[0], 365 * 24, 3)

    p1 = out_dir / "compare_8760_envelope.png"
    p2 = out_dir / "compare_monthly_energy.png"
    p3 = out_dir / "compare_duration_curve.png"

    _plot_8760_envelope(hist_hourly, gen_hourly, p1)
    _plot_monthly_energy(hist_hourly, gen_hourly, month_doy, p2)
    _plot_duration_curve(hist_hourly, gen_hourly, p3)
    return [p1.name, p2.name, p3.name]


def build_background_8760(cfg: Build8760Config) -> dict:
    rng = np.random.default_rng(cfg.random_seed)
    month_doy = _month_vector_365()

    load_y_d_h, wind_y_d_h, solar_y_d_h, source_meta = _load_profiles(cfg)
    n_years, n_days, n_hours = load_y_d_h.shape
    if n_days != 365 or n_hours != 24:
        raise ValueError(f"Expected [years, 365, 24], got {load_y_d_h.shape}")

    # [years, days, hours, channels] -> [hist_days, 24, 3]
    hist_profiles = np.stack([load_y_d_h, wind_y_d_h, solar_y_d_h], axis=-1).reshape(-1, 24, 3)
    hist_month = np.tile(month_doy, n_years)
    hist_year_idx = np.repeat(np.arange(n_years), n_days)
    hist_doy = np.tile(np.arange(n_days), n_years)

    feat = _daily_features(hist_profiles)
    feat_z = _zscore(feat)
    state_flat, state_centers = _kmeans_numpy(
        feat_z,
        n_clusters=cfg.n_states,
        n_init=cfg.kmeans_n_init,
        max_iter=cfg.kmeans_max_iter,
        random_seed=cfg.random_seed,
    )
    state_yd = state_flat.reshape(n_years, n_days)

    month_freq = _build_month_freq(hist_month, state_flat, cfg.n_states)
    month_trans = _build_month_transition(month_doy, state_yd, month_freq, cfg.n_states)

    gen_state = _sample_state_path(month_doy, month_freq, month_trans, cfg.transition_weight, rng)
    chosen_hist_idx = _pick_profile_indices(
        day_profiles=hist_profiles,
        month_flat=hist_month,
        state_flat=state_flat,
        gen_month=month_doy,
        gen_state=gen_state,
        top_k=cfg.continuity_top_k,
        temp=cfg.continuity_temp,
        shape_score_weight=cfg.shape_score_weight,
        shape_weight_p95=cfg.shape_weight_p95,
        shape_weight_p05=cfg.shape_weight_p05,
        shape_weight_ramp_std=cfg.shape_weight_ramp_std,
        rng=rng,
    )

    selected_daily = hist_profiles[chosen_hist_idx]
    hourly_raw = selected_daily.reshape(365 * 24, 3)
    hist_hourly = np.stack([load_y_d_h, wind_y_d_h, solar_y_d_h], axis=-1).reshape(n_years, 365 * 24, 3)
    boundary = np.arange(1, 365) * 24
    hist_jump = np.linalg.norm(hist_hourly[:, boundary, :] - hist_hourly[:, boundary - 1, :], axis=2).reshape(-1)

    hourly_rebalanced, wind_month_scales = _rebalance_wind_monthly(
        hourly=hourly_raw,
        hist_wind_hourly=hist_hourly[:, :, 1],
        month_doy=month_doy,
        strength=cfg.wind_monthly_rebalance_strength,
        clip_min=cfg.wind_scale_clip_min,
        clip_max=cfg.wind_scale_clip_max,
    )
    hourly_rebalanced, solar_month_scales = _rebalance_solar_monthly(
        hourly=hourly_rebalanced,
        hist_solar_hourly=hist_hourly[:, :, 2],
        month_doy=month_doy,
        strength=cfg.solar_monthly_rebalance_strength,
        clip_min=cfg.solar_scale_clip_min,
        clip_max=cfg.solar_scale_clip_max,
    )
    hourly_rebalanced, net_load_month_scales, net_wind_month_scales = _rebalance_netload_monthly_load_wind(
        hourly=hourly_rebalanced,
        hist_hourly=hist_hourly,
        month_doy=month_doy,
        strength=cfg.net_monthly_rebalance_strength,
        weight_component=cfg.net_rebalance_weight_component,
        weight_net=cfg.net_rebalance_weight_net,
        weight_reg=cfg.net_rebalance_weight_reg,
        load_clip_min=cfg.net_load_scale_clip_min,
        load_clip_max=cfg.net_load_scale_clip_max,
        wind_clip_min=cfg.net_wind_scale_clip_min,
        wind_clip_max=cfg.net_wind_scale_clip_max,
    )
    hourly_rebalanced, qmap_abs_shift = _monthly_quantile_mapping(
        hourly=hourly_rebalanced,
        hist_hourly=hist_hourly,
        month_doy=month_doy,
        strength=cfg.monthly_qmap_strength,
        load_mult=cfg.monthly_qmap_load_mult,
        wind_mult=cfg.monthly_qmap_wind_mult,
        solar_mult=cfg.monthly_qmap_solar_mult,
        preserve_month_sum=cfg.monthly_qmap_preserve_month_sum,
        low_q=cfg.monthly_qmap_low_q,
        high_q=cfg.monthly_qmap_high_q,
    )
    hourly = _smooth_day_boundaries(
        hourly_rebalanced,
        cfg.smooth_hours,
        hist_boundary_jumps=hist_jump,
        target_quantile=cfg.boundary_target_quantile,
        smooth_strength=cfg.boundary_smooth_strength,
    )
    hourly = _enforce_solar_night_zero(hourly, cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    time = pd.date_range(pd.Timestamp(cfg.start_time), periods=365 * 24, freq="1h")
    hourly_df = pd.DataFrame(
        {
            "time": time,
            "load": hourly[:, 0],
            "wind_power": hourly[:, 1],
            "solar_power": hourly[:, 2],
        }
    )
    hourly_df["net_load"] = hourly_df["load"] - hourly_df["wind_power"] - hourly_df["solar_power"]
    hourly_df["month"] = np.repeat(month_doy, 24)
    hourly_df.to_csv(out_dir / "background_8760.csv", index=False, encoding="utf-8-sig")

    day_jump = np.zeros(365, dtype=np.float64)
    day_jump[1:] = np.linalg.norm(hourly_rebalanced[boundary] - hourly_rebalanced[boundary - 1], axis=1)
    day_jump_after = np.zeros(365, dtype=np.float64)
    day_jump_after[1:] = np.linalg.norm(hourly[boundary] - hourly[boundary - 1], axis=1)

    day_df = pd.DataFrame(
        {
            "day_index": np.arange(1, 366, dtype=np.int32),
            "month": month_doy,
            "state_id": gen_state,
            "source_hist_index": chosen_hist_idx,
            "source_year_index": hist_year_idx[chosen_hist_idx],
            "source_day_of_year": hist_doy[chosen_hist_idx] + 1,
            "boundary_jump_before_smooth": day_jump,
            "boundary_jump_after_smooth": day_jump_after,
        }
    )
    day_df.to_csv(out_dir / "background_day_meta.csv", index=False, encoding="utf-8-sig")

    freq_df = pd.DataFrame(month_freq, columns=[f"state_{i}" for i in range(cfg.n_states)])
    freq_df.insert(0, "month", np.arange(1, 13))
    freq_df.to_csv(out_dir / "month_state_frequency.csv", index=False, encoding="utf-8-sig")

    trans_rows = []
    for m in range(12):
        for s_from in range(cfg.n_states):
            for s_to in range(cfg.n_states):
                trans_rows.append(
                    {
                        "month": m + 1,
                        "state_from": s_from,
                        "state_to": s_to,
                        "prob": month_trans[m, s_from, s_to],
                    }
                )
    pd.DataFrame(trans_rows).to_csv(out_dir / "month_state_transition.csv", index=False, encoding="utf-8-sig")

    np.savez(
        out_dir / "background_state_artifacts.npz",
        month_doy=month_doy.astype(np.int32),
        hist_month=hist_month.astype(np.int32),
        hist_state=state_flat.astype(np.int32),
        gen_state=gen_state.astype(np.int32),
        month_freq=month_freq.astype(np.float32),
        month_trans=month_trans.astype(np.float32),
        state_centers=state_centers.astype(np.float32),
        chosen_hist_idx=chosen_hist_idx.astype(np.int32),
    )

    (out_dir / "build_background_8760_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "build_background_8760_source_meta.json").write_text(
        json.dumps(source_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_files: list[str] = []
    if cfg.make_plots:
        plot_files = _save_comparison_plots(
            load_y_d_h=load_y_d_h,
            wind_y_d_h=wind_y_d_h,
            solar_y_d_h=solar_y_d_h,
            gen_hourly=hourly,
            month_doy=month_doy,
            out_dir=out_dir,
        )

    summary = {
        "output_dir": str(out_dir.resolve()),
        "n_years_history": int(n_years),
        "n_hist_days": int(hist_profiles.shape[0]),
        "n_states": int(cfg.n_states),
        "shape_hourly": [int(v) for v in hourly.shape],
        "load_mean": float(hourly_df["load"].mean()),
        "wind_mean": float(hourly_df["wind_power"].mean()),
        "solar_mean": float(hourly_df["solar_power"].mean()),
        "netload_mean": float(hourly_df["net_load"].mean()),
        "boundary_jump_p50_before_smooth": float(np.median(day_jump[1:])),
        "boundary_jump_p95_before_smooth": float(np.quantile(day_jump[1:], 0.95)),
        "boundary_jump_p50_after_smooth": float(np.median(day_jump_after[1:])),
        "boundary_jump_p95_after_smooth": float(np.quantile(day_jump_after[1:], 0.95)),
        "hist_boundary_jump_p50": float(np.median(hist_jump)),
        "hist_boundary_jump_p95": float(np.quantile(hist_jump, 0.95)),
        "wind_month_scales": [float(v) for v in wind_month_scales],
        "solar_month_scales": [float(v) for v in solar_month_scales],
        "net_rebalance_load_scales": [float(v) for v in net_load_month_scales],
        "net_rebalance_wind_scales": [float(v) for v in net_wind_month_scales],
        "monthly_qmap_strength": float(cfg.monthly_qmap_strength),
        "monthly_qmap_channel_strengths": [
            float(cfg.monthly_qmap_strength * max(0.0, cfg.monthly_qmap_load_mult)),
            float(cfg.monthly_qmap_strength * max(0.0, cfg.monthly_qmap_wind_mult)),
            float(cfg.monthly_qmap_strength * max(0.0, cfg.monthly_qmap_solar_mult)),
        ],
        "monthly_qmap_preserve_month_sum": bool(cfg.monthly_qmap_preserve_month_sum),
        "monthly_qmap_mean_abs_shift": [float(v) for v in qmap_abs_shift.mean(axis=0)],
        "plot_files": plot_files,
    }

    print("=== 8760 background series generated ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved files:")
    print("- background_8760.csv")
    print("- background_day_meta.csv")
    print("- month_state_frequency.csv")
    print("- month_state_transition.csv")
    print("- background_state_artifacts.npz")
    print("- build_background_8760_config.json")
    print("- build_background_8760_source_meta.json")
    for file_name in plot_files:
        print(f"- {file_name}")
    return summary


def parse_args() -> Build8760Config:
    p = argparse.ArgumentParser(description="Build one 8760 background sequence from 24x365xN wind/solar/load datasets.")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--output-dir", type=str, default="background_8760_outputs")
    p.add_argument("--start-time", type=str, default="2025-01-01 00:00:00")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-states", type=int, default=6)
    p.add_argument("--kmeans-n-init", type=int, default=12)
    p.add_argument("--kmeans-max-iter", type=int, default=80)
    p.add_argument("--transition-weight", type=float, default=0.75)
    p.add_argument("--continuity-top-k", type=int, default=12)
    p.add_argument("--continuity-temp", type=float, default=0.25)
    p.add_argument("--shape-score-weight", type=float, default=0.70)
    p.add_argument("--shape-weight-p95", type=float, default=1.00)
    p.add_argument("--shape-weight-p05", type=float, default=1.00)
    p.add_argument("--shape-weight-ramp-std", type=float, default=1.20)
    p.add_argument("--smooth-hours", type=int, default=4)
    p.add_argument("--solar-zero-before-hour", type=int, default=6)
    p.add_argument("--solar-zero-after-hour", type=int, default=20)
    p.add_argument("--boundary-target-quantile", type=float, default=0.60)
    p.add_argument("--boundary-smooth-strength", type=float, default=0.75)
    p.add_argument("--onshore-weight", type=float, default=1.0)
    p.add_argument("--offshore-weight", type=float, default=0.6)
    p.add_argument("--utility-pv-weight", type=float, default=1.0)
    p.add_argument("--distributed-pv-weight", type=float, default=0.7)
    p.add_argument("--csp-weight", type=float, default=0.0)
    p.add_argument("--wind-monthly-rebalance-strength", type=float, default=0.85)
    p.add_argument("--wind-scale-clip-min", type=float, default=0.70)
    p.add_argument("--wind-scale-clip-max", type=float, default=1.30)
    p.add_argument("--solar-monthly-rebalance-strength", type=float, default=0.75)
    p.add_argument("--solar-scale-clip-min", type=float, default=0.75)
    p.add_argument("--solar-scale-clip-max", type=float, default=1.25)
    p.add_argument("--net-monthly-rebalance-strength", type=float, default=0.60)
    p.add_argument("--net-rebalance-weight-component", type=float, default=1.0)
    p.add_argument("--net-rebalance-weight-net", type=float, default=2.0)
    p.add_argument("--net-rebalance-weight-reg", type=float, default=0.35)
    p.add_argument("--net-load-scale-clip-min", type=float, default=0.93)
    p.add_argument("--net-load-scale-clip-max", type=float, default=1.07)
    p.add_argument("--net-wind-scale-clip-min", type=float, default=0.85)
    p.add_argument("--net-wind-scale-clip-max", type=float, default=1.15)
    p.add_argument("--monthly-qmap-strength", type=float, default=0.35)
    p.add_argument("--monthly-qmap-load-mult", type=float, default=1.00)
    p.add_argument("--monthly-qmap-wind-mult", type=float, default=0.80)
    p.add_argument("--monthly-qmap-solar-mult", type=float, default=0.00)
    p.add_argument("--monthly-qmap-no-preserve-sum", action="store_true")
    p.add_argument("--monthly-qmap-low-q", type=float, default=0.005)
    p.add_argument("--monthly-qmap-high-q", type=float, default=0.995)
    p.add_argument("--no-plots", action="store_true", help="Disable comparison figure output.")
    a = p.parse_args()
    return Build8760Config(
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        start_time=a.start_time,
        random_seed=a.random_seed,
        n_states=a.n_states,
        kmeans_n_init=a.kmeans_n_init,
        kmeans_max_iter=a.kmeans_max_iter,
        transition_weight=a.transition_weight,
        continuity_top_k=a.continuity_top_k,
        continuity_temp=a.continuity_temp,
        shape_score_weight=a.shape_score_weight,
        shape_weight_p95=a.shape_weight_p95,
        shape_weight_p05=a.shape_weight_p05,
        shape_weight_ramp_std=a.shape_weight_ramp_std,
        smooth_hours=a.smooth_hours,
        solar_zero_before_hour=a.solar_zero_before_hour,
        solar_zero_after_hour=a.solar_zero_after_hour,
        boundary_target_quantile=a.boundary_target_quantile,
        boundary_smooth_strength=a.boundary_smooth_strength,
        onshore_weight=a.onshore_weight,
        offshore_weight=a.offshore_weight,
        utility_pv_weight=a.utility_pv_weight,
        distributed_pv_weight=a.distributed_pv_weight,
        csp_weight=a.csp_weight,
        wind_monthly_rebalance_strength=a.wind_monthly_rebalance_strength,
        wind_scale_clip_min=a.wind_scale_clip_min,
        wind_scale_clip_max=a.wind_scale_clip_max,
        solar_monthly_rebalance_strength=a.solar_monthly_rebalance_strength,
        solar_scale_clip_min=a.solar_scale_clip_min,
        solar_scale_clip_max=a.solar_scale_clip_max,
        net_monthly_rebalance_strength=a.net_monthly_rebalance_strength,
        net_rebalance_weight_component=a.net_rebalance_weight_component,
        net_rebalance_weight_net=a.net_rebalance_weight_net,
        net_rebalance_weight_reg=a.net_rebalance_weight_reg,
        net_load_scale_clip_min=a.net_load_scale_clip_min,
        net_load_scale_clip_max=a.net_load_scale_clip_max,
        net_wind_scale_clip_min=a.net_wind_scale_clip_min,
        net_wind_scale_clip_max=a.net_wind_scale_clip_max,
        monthly_qmap_strength=a.monthly_qmap_strength,
        monthly_qmap_load_mult=a.monthly_qmap_load_mult,
        monthly_qmap_wind_mult=a.monthly_qmap_wind_mult,
        monthly_qmap_solar_mult=a.monthly_qmap_solar_mult,
        monthly_qmap_preserve_month_sum=not a.monthly_qmap_no_preserve_sum,
        monthly_qmap_low_q=a.monthly_qmap_low_q,
        monthly_qmap_high_q=a.monthly_qmap_high_q,
        make_plots=not a.no_plots,
    )


def main() -> None:
    cfg = parse_args()
    build_background_8760(cfg)


if __name__ == "__main__":
    main()
