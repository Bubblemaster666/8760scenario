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
from scipy.stats import genpareto

from evt_fit import prob_to_level
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
    severity_info: str | None = None


def _load_inputs(cfg: EvalConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    real = np.load(cfg.real).astype(np.float32)
    gen = np.load(cfg.generated).astype(np.float32)
    cond = pd.read_csv(cfg.cond)
    meta = pd.read_csv(cfg.meta)
    n = min(len(real), len(gen), len(cond), len(meta))
    return real[:n], gen[:n], cond.iloc[:n].reset_index(drop=True), meta.iloc[:n].reset_index(drop=True)


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


def _load_severity_info(cfg: EvalConfig) -> dict:
    if cfg.severity_info:
        path = Path(cfg.severity_info)
    else:
        path = Path(cfg.cond).parent / "dataset_summary.json"
    if not path.exists():
        return {"method": "condition_severity_thresholds_fallback", "source": "cond.csv"}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"method": "condition_severity_thresholds_fallback", "source": str(path)}
    evt_info = summary.get("evt_info", {})
    return {"method": "dataset_evt_info", "source": str(path), "evt_info": evt_info}


def _levels_from_evt_info(generated_cum: np.ndarray, severity_info: dict) -> tuple[np.ndarray | None, str]:
    evt_info = severity_info.get("evt_info", {})
    severity_mode = str(evt_info.get("severity_mode", "")).lower()
    severity_quantiles = evt_info.get("severity_quantiles", {})
    if severity_mode in {"quantile", "hybrid"} and all(k in severity_quantiles for k in ["q1", "q2", "q3"]):
        q1 = float(severity_quantiles["q1"])
        q2 = float(severity_quantiles["q2"])
        q3 = float(severity_quantiles["q3"])
        if np.isfinite(q1) and np.isfinite(q2) and np.isfinite(q3):
            values = np.asarray(generated_cum, dtype=float)
            levels = np.zeros((len(values),), dtype=int)
            levels = np.where(values >= q1, 1, levels)
            levels = np.where(values >= q2, 2, levels)
            levels = np.where(values >= q3, 3, levels)
            if bool(severity_quantiles.get("positive_only", False)):
                levels = np.where(values <= 0, 0, levels)
            return levels.astype(int), f"dataset_evt_info_{severity_mode}_quantiles"

    thresholds = evt_info.get("severity_thresholds", {})
    required = {"threshold_u", "shape_c", "scale", "tail_prob_at_u"}
    if evt_info.get("method") != "pot_gpd" or not required.issubset(evt_info):
        return None, "condition_severity_thresholds_fallback"

    threshold_u = float(evt_info["threshold_u"])
    shape_c = float(evt_info["shape_c"])
    scale = float(evt_info["scale"])
    tail_prob_at_u = float(evt_info["tail_prob_at_u"])
    severe_prob = float(thresholds.get("severe_prob", 0.01))
    moderate_prob = float(thresholds.get("moderate_prob", 0.05))
    mild_prob = float(thresholds.get("mild_prob", 0.10))

    levels = []
    for value in generated_cum:
        if not np.isfinite(value):
            levels.append(0)
            continue
        if value <= threshold_u:
            prob = 1.0
        else:
            exceedance = float(value - threshold_u)
            tail_cond = 1.0 - genpareto.cdf(exceedance, c=shape_c, loc=0.0, scale=scale)
            prob = float(np.clip(tail_prob_at_u * tail_cond, 1e-8, 1.0))
        levels.append(prob_to_level(prob, severe_prob, moderate_prob, mild_prob))
    return np.asarray(levels, dtype=int), "dataset_evt_info_pot_gpd"


def _levels_from_condition_thresholds(cond: pd.DataFrame, generated_cum: np.ndarray) -> np.ndarray:
    target_level = cond["severity_level"].fillna(0).astype(int).to_numpy()
    target_cum = cond["cum_deficit"].fillna(0).astype(float).to_numpy()
    thresholds: dict[int, float] = {}
    for level in [1, 2, 3]:
        mask = target_level >= level
        if np.any(mask):
            thresholds[level] = float(np.min(target_cum[mask]))

    levels = np.zeros((len(generated_cum),), dtype=int)
    for level, threshold in thresholds.items():
        levels = np.where(generated_cum >= threshold, level, levels)
    return levels


def _severity_match_rate(cond: pd.DataFrame, generated_cum: np.ndarray, severity_info: dict) -> tuple[float, float, str]:
    gen_level, method = _levels_from_evt_info(generated_cum, severity_info)
    if gen_level is None:
        gen_level = _levels_from_condition_thresholds(cond, generated_cum)
    target = cond["severity_level"].fillna(0).astype(int).to_numpy()
    exact = float(np.mean(gen_level == target))
    adjacent = float(np.mean(np.abs(gen_level - target) <= 1))
    return exact, adjacent, method


def compute_metrics(real: np.ndarray, gen: np.ndarray, cond: pd.DataFrame, max_lag: int, severity_info: dict | None = None) -> dict[str, float | str]:
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
    risk_real = batch_hard_risk_metrics(real, tau=tau, delta_t_hours=delta_t)
    risk_gen = batch_hard_risk_metrics(gen, tau=tau, delta_t_hours=delta_t)

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

    exact_match, adjacent_match, severity_method = _severity_match_rate(
        cond,
        np.asarray(risk_gen["cum_deficit"], dtype=float),
        severity_info or {"method": "condition_severity_thresholds_fallback"},
    )
    metrics["extreme_degree_match_rate"] = exact_match
    metrics["extreme_degree_adjacent_match_rate"] = adjacent_match
    metrics["severity_classification_method"] = severity_method
    return metrics


def _group_metrics(real: np.ndarray, gen: np.ndarray, cond: pd.DataFrame, group_col: str, max_lag: int, severity_info: dict) -> pd.DataFrame:
    rows = []
    for value, sub_idx in cond.groupby(group_col).groups.items():
        idx = np.asarray(list(sub_idx), dtype=int)
        sub_metrics = compute_metrics(real[idx], gen[idx], cond.iloc[idx].reset_index(drop=True), max_lag=max_lag, severity_info=severity_info)
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

    real, gen, cond, meta = _load_inputs(cfg)
    severity_info = _load_severity_info(cfg)
    metrics = compute_metrics(real, gen, cond, cfg.max_lag, severity_info=severity_info)
    metrics["model_name"] = cfg.model_name

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    _group_metrics(real, gen, cond, "event_type", cfg.max_lag, severity_info).to_csv(out_dir / "metrics_by_event_type.csv", index=False, encoding="utf-8-sig")
    _group_metrics(real, gen, cond, "severity_level", cfg.max_lag, severity_info).to_csv(out_dir / "metrics_by_severity.csv", index=False, encoding="utf-8-sig")

    risk_df = _plot_risk_boxplot(real, gen, cond, figures_dir)
    risk_df.to_csv(out_dir / "risk_metrics_real_vs_generated.csv", index=False, encoding="utf-8-sig")
    _plot_typical_curves(real, gen, cond, figures_dir)
    _plot_acf(real, gen, cfg.max_lag, figures_dir)
    _plot_corr(real, gen, figures_dir)

    summary = {
        "model_name": cfg.model_name,
        "num_samples": int(len(cond)),
        "severity_classification": {
            "method": metrics.get("severity_classification_method"),
            "source": severity_info.get("source"),
            "note": "Generated samples are classified with fixed dataset EVT information when available; otherwise condition severity thresholds are used. The evaluator no longer re-fits EVT on generated samples.",
        },
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
    parser.add_argument("--severity-info", type=str, default=None)
    args = parser.parse_args()
    return EvalConfig(
        real=args.real,
        generated=args.generated,
        cond=args.cond,
        meta=args.meta,
        out_dir=args.out_dir,
        model_name=args.model_name,
        max_lag=args.max_lag,
        severity_info=args.severity_info,
    )


if __name__ == "__main__":
    evaluate_generation(parse_args())
