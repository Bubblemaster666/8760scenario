from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from evt_fit import EVTConfig, fit_evt_and_label
from risk_metrics import batch_hard_risk_metrics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class EvalConfig:
    real: str
    generated: str
    cond: str
    meta: str
    out_dir: str
    model_name: str
    max_lag: int = 12
    event_mask: str | None = None
    ramp_metric_mode: str = "one_step"
    ramp_window_hours: float = 1.0
    multiscale_ramp_windows: str = "1.0,2.0,3.0"


EXTREME_MAIN_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "extreme_degree_match_rate",
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

AUXILIARY_GLOBAL_STAT_METRICS = [
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
]

HIGHRISK_METRIC_MAP = {
    "mean_wasserstein": "highrisk_wasserstein",
    "mean_js": "highrisk_js",
    "acf_mae": "highrisk_acf_mae",
    "corr_matrix_error": "highrisk_corr_matrix_error",
    "cum_deficit_mae": "highrisk_cum_deficit_mae",
    "q99_cum_deficit_error": "highrisk_q99_cum_deficit_error",
    "netload_ramp_max_mae": "highrisk_netload_ramp_max_mae",
    "imbalance_duration_mae": "highrisk_imbalance_duration_mae",
}


def _load_inputs(cfg: EvalConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    real = np.load(cfg.real).astype(np.float32)
    gen = np.load(cfg.generated).astype(np.float32)
    cond = pd.read_csv(cfg.cond)
    meta = pd.read_csv(cfg.meta)
    n = min(len(real), len(gen), len(cond), len(meta))
    event_mask = None
    mask_path = None
    if cfg.event_mask:
        mask_path = Path(cfg.event_mask)
    else:
        candidate = Path(cfg.cond).parent / "event_mask_test.npy"
        if candidate.exists():
            mask_path = candidate
    if mask_path is not None and mask_path.exists():
        event_mask = np.load(mask_path).astype(np.float32)[:n]
    return real[:n], gen[:n], cond.iloc[:n].reset_index(drop=True), meta.iloc[:n].reset_index(drop=True), event_mask


def _js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    lo = float(min(float(np.min(a)), float(np.min(b))))
    hi = float(max(float(np.max(a)), float(np.max(b))))
    if hi <= lo:
        hi = lo + 1.0
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    hist_a = hist_a + 1e-12
    hist_b = hist_b + 1e-12
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()
    return float(jensenshannon(hist_a, hist_b, base=2.0) ** 2)


def _acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 1e-12:
        return np.zeros((max_lag,), dtype=float)
    out = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            out.append(0.0)
        else:
            out.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(out, dtype=float)


def _mean_acf(samples: np.ndarray, channel: int, max_lag: int) -> np.ndarray:
    return np.mean([_acf(seq[channel], max_lag) for seq in samples], axis=0)


def _mean_corr_matrix(samples: np.ndarray) -> np.ndarray:
    mats = []
    for seq in samples:
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.corrcoef(seq)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        mats.append(mat)
    return np.mean(mats, axis=0)


def _severity_match_rate(cond: pd.DataFrame, generated_cum: np.ndarray) -> tuple[float, float]:
    gen_df = pd.DataFrame({"cum_deficit": generated_cum})
    labeled, _ = fit_evt_and_label(gen_df, EVTConfig(metric_col="cum_deficit"))
    gen_level = labeled["severity_level"].fillna(0).astype(int).to_numpy()
    target = cond["severity_level"].fillna(0).astype(int).to_numpy()
    exact = float(np.mean(gen_level == target))
    adjacent = float(np.mean(np.abs(gen_level - target) <= 1))
    return exact, adjacent



def _batch_core_risk_metrics(x: np.ndarray, event_mask: np.ndarray, tau: np.ndarray | float, delta_t_hours: float = 1.0) -> dict[str, np.ndarray]:
    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.ndim == 0:
        tau_arr = np.full((x.shape[0],), float(tau_arr), dtype=float)
    cum, ramp_max, duration = [], [], []
    for i in range(x.shape[0]):
        mask = np.asarray(event_mask[i], dtype=float) > 0.5
        if not np.any(mask):
            mask = np.ones((x.shape[2],), dtype=bool)
        net = x[i, 0, mask] - x[i, 1, mask] - x[i, 2, mask]
        if net.size == 0:
            cum.append(0.0); ramp_max.append(0.0); duration.append(0.0); continue
        excess = np.maximum(0.0, net - tau_arr[i])
        cum.append(float(excess.sum() * delta_t_hours))
        ramp = np.diff(net, prepend=net[0])
        ramp_max.append(float(np.max(ramp)))
        duration.append(float((net > tau_arr[i]).sum() * delta_t_hours))
    return {
        "cum_deficit": np.asarray(cum, dtype=float),
        "netload_ramp_max": np.asarray(ramp_max, dtype=float),
        "imbalance_duration": np.asarray(duration, dtype=float),
    }

def compute_metrics(
    real: np.ndarray,
    gen: np.ndarray,
    cond: pd.DataFrame,
    max_lag: int,
    event_mask: np.ndarray | None = None,
    ramp_metric_mode: str = "one_step",
    ramp_window_hours: float = 1.0,
    multiscale_ramp_windows: str = "1.0,2.0,3.0",
) -> dict[str, float | str]:
    names = ["load", "wind_power", "solar_power"]
    metrics: dict[str, float] = {}
    wasserstein_scores = []
    js_scores = []
    acf_scores = []

    for ch, name in enumerate(names):
        real_flat = real[:, ch, :].reshape(-1)
        gen_flat = gen[:, ch, :].reshape(-1)
        w = float(wasserstein_distance(real_flat, gen_flat))
        j = _js_divergence(real_flat, gen_flat)
        acf_mae = float(np.mean(np.abs(_mean_acf(real, ch, max_lag) - _mean_acf(gen, ch, max_lag))))
        metrics[f"{name}_wasserstein"] = w
        metrics[f"{name}_js"] = j
        metrics[f"{name}_acf_mae"] = acf_mae
        wasserstein_scores.append(w)
        js_scores.append(j)
        acf_scores.append(acf_mae)

    metrics["mean_wasserstein"] = float(np.mean(wasserstein_scores))
    metrics["mean_js"] = float(np.mean(js_scores))
    metrics["acf_mae"] = float(np.mean(acf_scores))

    corr_real = _mean_corr_matrix(real)
    corr_gen = _mean_corr_matrix(gen)
    metrics["corr_matrix_error"] = float(np.mean(np.abs(corr_real - corr_gen)))

    tau = cond["imbalance_tau"].astype(float).to_numpy() if "imbalance_tau" in cond.columns else np.zeros((len(cond),), dtype=float)
    delta_t = float(cond["delta_t_hours"].astype(float).iloc[0]) if "delta_t_hours" in cond.columns else 1.0
    risk_real = batch_hard_risk_metrics(
        real,
        tau=tau,
        delta_t_hours=delta_t,
        ramp_metric_mode=ramp_metric_mode,
        ramp_window_hours=ramp_window_hours,
        multiscale_ramp_windows=multiscale_ramp_windows,
    )
    risk_gen = batch_hard_risk_metrics(
        gen,
        tau=tau,
        delta_t_hours=delta_t,
        ramp_metric_mode=ramp_metric_mode,
        ramp_window_hours=ramp_window_hours,
        multiscale_ramp_windows=multiscale_ramp_windows,
    )

    risk_rows = []
    for key in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
        real_arr = np.asarray(risk_real[key], dtype=float)
        gen_arr = np.asarray(risk_gen[key], dtype=float)
        mae = float(np.mean(np.abs(real_arr - gen_arr)))
        rel = float(abs(gen_arr.mean() - real_arr.mean()) / (abs(real_arr.mean()) + 1e-8))
        q95_err = float(abs(np.quantile(gen_arr, 0.95) - np.quantile(real_arr, 0.95)))
        q99_err = float(abs(np.quantile(gen_arr, 0.99) - np.quantile(real_arr, 0.99)))
        metrics[f"{key}_mae"] = mae
        metrics[f"{key}_relative_error"] = rel
        metrics[f"q95_{key}_error"] = q95_err
        metrics[f"q99_{key}_error"] = q99_err
        risk_rows.append((key, real_arr, gen_arr))

    if event_mask is not None:
        core_real = _batch_core_risk_metrics(real, event_mask, tau=tau, delta_t_hours=delta_t)
        core_gen = _batch_core_risk_metrics(gen, event_mask, tau=tau, delta_t_hours=delta_t)
        for key in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
            real_arr = np.asarray(core_real[key], dtype=float)
            gen_arr = np.asarray(core_gen[key], dtype=float)
            metrics[f"core_{key}_mae"] = float(np.mean(np.abs(real_arr - gen_arr)))
            metrics[f"core_q95_{key}_error"] = float(abs(np.quantile(gen_arr, 0.95) - np.quantile(real_arr, 0.95)))
            metrics[f"core_q99_{key}_error"] = float(abs(np.quantile(gen_arr, 0.99) - np.quantile(real_arr, 0.99)))

    exact_match, adjacent_match = _severity_match_rate(cond, np.asarray(risk_gen["cum_deficit"], dtype=float))
    metrics["extreme_degree_match_rate"] = exact_match
    metrics["extreme_degree_adjacent_match_rate"] = adjacent_match
    return metrics


def compute_highrisk_metrics(
    real: np.ndarray,
    gen: np.ndarray,
    cond: pd.DataFrame,
    max_lag: int,
    event_mask: np.ndarray | None = None,
    ramp_metric_mode: str = "one_step",
    ramp_window_hours: float = 1.0,
    multiscale_ramp_windows: str = "1.0,2.0,3.0",
) -> tuple[dict[str, float], str | None]:
    """Compute high-risk conditional metrics on samples with severity_level >= 2."""
    out = {target: float("nan") for target in HIGHRISK_METRIC_MAP.values()}
    if "severity_level" not in cond.columns:
        return out, "severity_level is missing; highrisk metrics are NaN."
    severity = pd.to_numeric(cond["severity_level"], errors="coerce").fillna(0).astype(int)
    idx = np.where(severity.to_numpy() >= 2)[0]
    if idx.size < 3:
        return out, "highrisk sample count is less than 3; highrisk metrics are NaN."
    sub_metrics = compute_metrics(
        real[idx],
        gen[idx],
        cond.iloc[idx].reset_index(drop=True),
        max_lag=max_lag,
        event_mask=None if event_mask is None else event_mask[idx],
        ramp_metric_mode=ramp_metric_mode,
        ramp_window_hours=ramp_window_hours,
        multiscale_ramp_windows=multiscale_ramp_windows,
    )
    for source, target in HIGHRISK_METRIC_MAP.items():
        out[target] = float(sub_metrics.get(source, np.nan))
    return out, None


def _group_metrics(real: np.ndarray, gen: np.ndarray, cond: pd.DataFrame, group_col: str, max_lag: int, event_mask: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for value, sub_idx in cond.groupby(group_col).groups.items():
        idx = np.asarray(list(sub_idx), dtype=int)
        sub_metrics = compute_metrics(real[idx], gen[idx], cond.iloc[idx].reset_index(drop=True), max_lag=max_lag, event_mask=None if event_mask is None else event_mask[idx])
        sub_metrics[group_col] = value
        rows.append(sub_metrics)
    return pd.DataFrame(rows)


def _plot_typical_curves(real: np.ndarray, gen: np.ndarray, cond: pd.DataFrame, out_dir: Path) -> None:
    idx = int(cond["severity_level"].astype(int).sort_values(ascending=False).index[0])
    names = ["Load", "Wind", "Solar"]
    x = np.arange(real.shape[2])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    for ch, ax in enumerate(axes):
        ax.plot(x, real[idx, ch], label="Real", linewidth=2.0)
        ax.plot(x, gen[idx, ch], label="Generated", linestyle="--", linewidth=2.0)
        ax.set_ylabel(names[ch])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Hour index")
    fig.suptitle(f"Typical Event: {cond.loc[idx, 'event_type']} | severity={int(cond.loc[idx, 'severity_level'])}")
    fig.tight_layout()
    fig.savefig(out_dir / "typical_generated_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    net_real = real[idx, 0] - real[idx, 1] - real[idx, 2]
    net_gen = gen[idx, 0] - gen[idx, 1] - gen[idx, 2]
    plt.figure(figsize=(10.5, 4.8))
    plt.plot(x, net_real, label="Real net load", linewidth=2.0)
    plt.plot(x, net_gen, label="Generated net load", linestyle="--", linewidth=2.0)
    plt.xlabel("Hour index")
    plt.ylabel("Net load")
    plt.title("Net-load comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "netload_curve.png", dpi=220, bbox_inches="tight")
    plt.close()


def _plot_acf(real: np.ndarray, gen: np.ndarray, max_lag: int, out_dir: Path) -> None:
    names = ["Load", "Wind", "Solar"]
    lags = np.arange(1, max_lag + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True)
    for ch, ax in enumerate(axes):
        acf_real = _mean_acf(real, ch, max_lag)
        acf_gen = _mean_acf(gen, ch, max_lag)
        ax.plot(lags, acf_real, label="Real", linewidth=2.0)
        ax.plot(lags, acf_gen, label="Generated", linestyle="--", linewidth=2.0)
        ax.set_ylabel(f"{names[ch]} ACF")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Lag")
    fig.tight_layout()
    fig.savefig(out_dir / "acf_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_corr(real: np.ndarray, gen: np.ndarray, out_dir: Path) -> None:
    labels = ["Load", "Wind", "Solar"]
    for name, mat in [("corr_matrix_real.png", _mean_corr_matrix(real)), ("corr_matrix_generated.png", _mean_corr_matrix(gen))]:
        plt.figure(figsize=(4.8, 4.2))
        plt.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(3), labels)
        plt.yticks(range(3), labels)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=220, bbox_inches="tight")
        plt.close()


def _plot_risk_boxplot(real: np.ndarray, gen: np.ndarray, cond: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    tau = cond["imbalance_tau"].astype(float).to_numpy() if "imbalance_tau" in cond.columns else np.zeros((len(cond),), dtype=float)
    delta_t = float(cond["delta_t_hours"].astype(float).iloc[0]) if "delta_t_hours" in cond.columns else 1.0
    risk_real = batch_hard_risk_metrics(real, tau=tau, delta_t_hours=delta_t)
    risk_gen = batch_hard_risk_metrics(gen, tau=tau, delta_t_hours=delta_t)
    rows = []
    for metric_name in ["cum_deficit", "netload_ramp_max", "imbalance_duration"]:
        for i, sample_id in enumerate(cond["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "event_type": cond.loc[i, "event_type"],
                    "severity_level": int(cond.loc[i, "severity_level"]),
                    "metric_name": metric_name,
                    "real_value": float(risk_real[metric_name][i]),
                    "generated_value": float(risk_gen[metric_name][i]),
                }
            )
    risk_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    for ax, metric_name in zip(axes, ["cum_deficit", "netload_ramp_max", "imbalance_duration"]):
        data_real = risk_df.loc[risk_df["metric_name"] == metric_name, "real_value"]
        data_gen = risk_df.loc[risk_df["metric_name"] == metric_name, "generated_value"]
        ax.boxplot([data_real, data_gen], tick_labels=["Real", "Generated"])
        ax.set_title(metric_name)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_metric_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return risk_df


def evaluate_generation(cfg: EvalConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    real, gen, cond, meta, event_mask = _load_inputs(cfg)
    warnings: list[str] = []
    if event_mask is None:
        warnings.append("event_mask_test.npy is missing; core metrics are unavailable.")
    metrics = compute_metrics(
        real,
        gen,
        cond,
        cfg.max_lag,
        event_mask=event_mask,
        ramp_metric_mode=cfg.ramp_metric_mode,
        ramp_window_hours=cfg.ramp_window_hours,
        multiscale_ramp_windows=cfg.multiscale_ramp_windows,
    )
    if event_mask is None:
        metrics["core_q99_cum_deficit_error"] = float("nan")
    highrisk_metrics, highrisk_warning = compute_highrisk_metrics(
        real,
        gen,
        cond,
        cfg.max_lag,
        event_mask=event_mask,
        ramp_metric_mode=cfg.ramp_metric_mode,
        ramp_window_hours=cfg.ramp_window_hours,
        multiscale_ramp_windows=cfg.multiscale_ramp_windows,
    )
    metrics.update(highrisk_metrics)
    if highrisk_warning:
        warnings.append(highrisk_warning)
    metrics["model_name"] = cfg.model_name

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    extreme_columns = ["model_name", *EXTREME_MAIN_METRICS]
    pd.DataFrame([{col: metrics.get(col, np.nan) for col in extreme_columns}]).to_csv(
        out_dir / "extreme_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    auxiliary_columns = ["model_name", *AUXILIARY_GLOBAL_STAT_METRICS]
    pd.DataFrame([{col: metrics.get(col, np.nan) for col in auxiliary_columns}]).to_csv(
        out_dir / "auxiliary_global_stat_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _group_metrics(real, gen, cond, "event_type", cfg.max_lag, event_mask=event_mask).to_csv(out_dir / "metrics_by_event_type.csv", index=False, encoding="utf-8-sig")
    _group_metrics(real, gen, cond, "severity_level", cfg.max_lag, event_mask=event_mask).to_csv(out_dir / "metrics_by_severity.csv", index=False, encoding="utf-8-sig")

    risk_df = _plot_risk_boxplot(real, gen, cond, figures_dir)
    risk_df.to_csv(out_dir / "risk_metrics_real_vs_generated.csv", index=False, encoding="utf-8-sig")
    _plot_typical_curves(real, gen, cond, figures_dir)
    _plot_acf(real, gen, cfg.max_lag, figures_dir)
    _plot_corr(real, gen, figures_dir)

    summary = {
        "model_name": cfg.model_name,
        "num_samples": int(len(cond)),
        "risk_metric_scope": "both" if event_mask is not None else "full_window",
        "event_mask_used": bool(event_mask is not None),
        "warnings": warnings,
        "paper_main_metrics": {metric: metrics.get(metric) for metric in EXTREME_MAIN_METRICS},
        "auxiliary_global_stat_metrics": {metric: metrics.get(metric) for metric in AUXILIARY_GLOBAL_STAT_METRICS},
        "ramp_metric_mode": cfg.ramp_metric_mode,
        "ramp_window_hours": cfg.ramp_window_hours,
        "multiscale_ramp_windows": cfg.multiscale_ramp_windows,
        "metrics": metrics,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate generated scenarios against real samples.")
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--generated", type=str, required=True)
    parser.add_argument("--cond", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--event-mask", type=str, default=None)
    parser.add_argument("--ramp-metric-mode", type=str, default="one_step", choices=["one_step", "window_1h", "window_2h", "window_3h", "multiscale"])
    parser.add_argument("--ramp-window-hours", type=float, default=1.0)
    parser.add_argument("--multiscale-ramp-windows", type=str, default="1.0,2.0,3.0")
    args = parser.parse_args()
    return EvalConfig(
        real=args.real,
        generated=args.generated,
        cond=args.cond,
        meta=args.meta,
        out_dir=args.out_dir,
        model_name=args.model_name,
        max_lag=args.max_lag,
        event_mask=args.event_mask,
        ramp_metric_mode=args.ramp_metric_mode,
        ramp_window_hours=args.ramp_window_hours,
        multiscale_ramp_windows=args.multiscale_ramp_windows,
    )


if __name__ == "__main__":
    evaluate_generation(parse_args())
