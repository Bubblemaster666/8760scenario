from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evt_fit import compute_tail_score, metric_to_quantile_level
from risk_metrics import batch_hard_risk_metrics


def _night_solar_zero(seq: np.ndarray, window_start_time: str | pd.Timestamp) -> np.ndarray:
    out = seq.copy()
    start = pd.to_datetime(window_start_time)
    hours = pd.date_range(start, periods=out.shape[-1], freq="1h").hour.to_numpy()
    night = (hours < 6) | (hours > 18)
    out[2, night] = 0.0
    return out


def _amplitude_augment(x: np.ndarray, meta: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    out[:, 0, :] *= rng.uniform(0.97, 1.03, size=(len(out), 1))
    out[:, 1, :] *= rng.uniform(0.90, 1.10, size=(len(out), 1))
    out[:, 2, :] *= rng.uniform(0.90, 1.10, size=(len(out), 1))
    out = np.clip(out, 0.0, None)
    for i in range(len(out)):
        out[i] = _night_solar_zero(out[i], meta.loc[i, "window_start_time"])
    return out.astype(np.float32)


def _apply_risk_labels(x_aug: np.ndarray, cond_aug: pd.DataFrame, dataset_summary: dict) -> pd.DataFrame:
    tau = cond_aug["imbalance_tau"].astype(float).to_numpy() if "imbalance_tau" in cond_aug.columns else np.zeros(len(cond_aug))
    delta_t = float(cond_aug["delta_t_hours"].astype(float).iloc[0]) if "delta_t_hours" in cond_aug.columns else 1.0
    risk = batch_hard_risk_metrics(x_aug, tau=tau, delta_t_hours=delta_t)
    for key, value in risk.items():
        cond_aug[key] = value

    evt_info = dataset_summary.get("evt_info", {})
    severity_quantiles = evt_info.get("severity_quantiles", {})
    if evt_info.get("severity_mode") in {"quantile", "hybrid"} and all(k in severity_quantiles for k in ["q1", "q2", "q3"]):
        levels = np.zeros(len(cond_aug), dtype=int)
        cum = cond_aug["cum_deficit"].astype(float).to_numpy()
        levels = np.where(cum >= float(severity_quantiles["q1"]), 1, levels)
        levels = np.where(cum >= float(severity_quantiles["q2"]), 2, levels)
        levels = np.where(cum >= float(severity_quantiles["q3"]), 3, levels)
        if bool(severity_quantiles.get("positive_only", False)):
            levels = np.where(cum <= 0, 0, levels)
        cond_aug["severity_level"] = levels
    else:
        levels, _ = metric_to_quantile_level(cond_aug["cum_deficit"], positive_only=True)
        cond_aug["severity_level"] = levels

    prob_rank = cond_aug["cum_deficit"].rank(method="average", ascending=False) / (len(cond_aug) + 1)
    tail_score, tail_score_z, _ = compute_tail_score(prob_rank.to_numpy(dtype=float))
    cond_aug["extreme_prob"] = prob_rank.to_numpy(dtype=float)
    cond_aug["tail_score"] = tail_score
    cond_aug["tail_score_zscore"] = tail_score_z
    return cond_aug


def augment_trainset(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    cond_train = pd.read_csv(data_dir / "cond_train.csv")
    meta_train = pd.read_csv(data_dir / "meta_train.csv")
    summary_path = data_dir / "dataset_summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    rng = np.random.default_rng(args.seed)
    x_parts = [x_train]
    cond_parts = [cond_train.copy()]
    meta_parts = [meta_train.copy()]
    methods = ["original"]

    for rep in range(args.amplitude_repeats):
        x_aug = _amplitude_augment(x_train, meta_train, rng)
        cond_aug = cond_train.copy()
        meta_aug = meta_train.copy()
        cond_aug["sample_id"] = cond_aug["sample_id"].astype(str) + f"_amp{rep + 1}"
        meta_aug["sample_id"] = meta_aug["sample_id"].astype(str) + f"_amp{rep + 1}"
        cond_aug["augmentation_type"] = "amplitude"
        meta_aug["augmentation_type"] = "amplitude"
        cond_aug = _apply_risk_labels(x_aug, cond_aug, dataset_summary)
        x_parts.append(x_aug)
        cond_parts.append(cond_aug)
        meta_parts.append(meta_aug)
        methods.append("amplitude")

    x_all = np.concatenate(x_parts, axis=0).astype(np.float32)
    cond_all = pd.concat(cond_parts, ignore_index=True)
    meta_all = pd.concat(meta_parts, ignore_index=True)

    np.save(out_dir / "X_train_aug.npy", x_all)
    cond_all.to_csv(out_dir / "cond_train_aug.csv", index=False, encoding="utf-8-sig")
    meta_all.to_csv(out_dir / "meta_train_aug.csv", index=False, encoding="utf-8-sig")
    summary = {
        "enabled_by_default": False,
        "original_count": int(len(x_train)),
        "augmented_count": int(len(x_all)),
        "amplitude_repeats": int(args.amplitude_repeats),
        "methods": methods,
        "note": "Only the training split is augmented. Validation and test splits are untouched.",
    }
    (out_dir / "augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional train-only augmentation for extreme samples.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--amplitude-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(augment_trainset(parse_args()), ensure_ascii=False, indent=2))
