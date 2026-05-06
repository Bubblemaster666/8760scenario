from __future__ import annotations

import numpy as np


def moving_average_same(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Channel-wise centered moving average that preserves [C, T] shape."""
    window = max(1, int(window))
    if window == 1:
        return x.astype(np.float32, copy=True)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, ((0, 0), (pad_left, pad_right)), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    smooth = np.vstack([np.convolve(padded[c], kernel, mode="valid") for c in range(x.shape[0])])
    return smooth.astype(np.float32)


def hard_risk_metrics(window: np.ndarray, tau: float, delta_t_hours: float = 1.0) -> dict[str, float]:
    load = window[0].astype(float)
    wind = window[1].astype(float)
    solar = window[2].astype(float)
    net = load - wind - solar
    excess = np.maximum(0.0, net - float(tau))
    ramp = np.diff(net, prepend=net[0])
    return {
        "cum_deficit": float(excess.sum() * float(delta_t_hours)),
        "netload_ramp_max": float(ramp.max()),
        "imbalance_duration": float((net > float(tau)).sum() * float(delta_t_hours)),
    }


def relative_or_absolute_within(new_value: float, old_value: float, tolerance: float, floor: float = 1.0) -> bool:
    scale = max(abs(float(old_value)), float(floor))
    return abs(float(new_value) - float(old_value)) / scale <= float(tolerance)


def apply_physical_bounds(window: np.ndarray, channel_max: np.ndarray, day_mask: np.ndarray | None = None) -> np.ndarray:
    out = np.maximum(window.astype(np.float32), 0.0)
    upper = np.asarray(channel_max, dtype=np.float32).reshape(3, 1)
    out = np.minimum(out, upper)
    if day_mask is not None:
        mask = np.asarray(day_mask, dtype=np.float32).reshape(-1)
        if mask.size == out.shape[1]:
            out[2, mask <= 0.0] = 0.0
    return out.astype(np.float32)


def synthesize_shapelet_preserving_window(
    source: np.ndarray,
    donor: np.ndarray,
    event_mask: np.ndarray,
    day_mask: np.ndarray | None,
    channel_std: np.ndarray,
    channel_max: np.ndarray,
    rng: np.random.Generator,
    core_jitter_scale: float = 0.01,
    buffer_jitter_scale: float = 0.03,
    residual_bootstrap: bool = True,
) -> np.ndarray:
    core = np.asarray(event_mask, dtype=bool).reshape(-1)
    if core.size != source.shape[1]:
        core = np.ones((source.shape[1],), dtype=bool)
    buffer = ~core
    if not buffer.any():
        buffer = np.ones_like(core, dtype=bool)
        core = ~buffer

    smooth_source = moving_average_same(source, window=3)
    smooth_donor = moving_average_same(donor, window=3)
    residual_source = source - smooth_source
    residual_donor = donor - smooth_donor
    out = source.astype(np.float32, copy=True)
    if residual_bootstrap:
        # Keep the source shape dominant; donor residuals only enrich buffer texture.
        out[:, buffer] = smooth_source[:, buffer] + 0.70 * residual_source[:, buffer] + 0.30 * residual_donor[:, buffer]

    std = np.asarray(channel_std, dtype=np.float32).reshape(3, 1)
    core_noise = rng.normal(0.0, float(core_jitter_scale), size=source.shape).astype(np.float32) * std
    buffer_noise = rng.normal(0.0, float(buffer_jitter_scale), size=source.shape).astype(np.float32) * std
    if core.any():
        out[:, core] = out[:, core] + core_noise[:, core]
    if buffer.any():
        out[:, buffer] = out[:, buffer] + buffer_noise[:, buffer]
    return apply_physical_bounds(out, channel_max=channel_max, day_mask=day_mask)
