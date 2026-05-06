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


def _amplitude_augment_one(x: np.ndarray, meta_row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    out[0, :] *= rng.uniform(0.97, 1.03)
    out[1, :] *= rng.uniform(0.90, 1.10)
    out[2, :] *= rng.uniform(0.90, 1.10)
    out = np.clip(out, 0.0, None)
    out = _night_solar_zero(out, meta_row.get("window_start_time", "2024-01-01 00:00:00"))
    return out.astype(np.float32)


def _repeat_for_severity(level: int, args: argparse.Namespace) -> int:
    if level <= 0:
        return int(args.repeat_sev0)
    if level == 1:
        return int(args.repeat_sev1)
    if level == 2:
        return int(args.repeat_sev2)
    return int(args.repeat_sev3)


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
    mask_path = data_dir / "event_mask_train.npy"
    event_mask_train = np.load(mask_path).astype(np.float32) if mask_path.exists() else None
    summary_path = data_dir / "dataset_summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    rng = np.random.default_rng(args.seed)
    x_parts = [x_train]
    cond_parts = [cond_train.assign(augmentation_type="original")]
    meta_parts = [meta_train.assign(augmentation_type="original")]
    mask_parts = [event_mask_train] if event_mask_train is not None else []

    new_x, new_cond_rows, new_meta_rows, new_masks = [], [], [], []
    for i in range(len(x_train)):
        level = int(cond_train.loc[i, "severity_level"]) if "severity_level" in cond_train.columns else 0
        repeats = _repeat_for_severity(level, args)
        for rep in range(repeats):
            x_aug = _amplitude_augment_one(x_train[i], meta_train.loc[i], rng)
            cond_row = cond_train.iloc[i].copy()
            meta_row = meta_train.iloc[i].copy()
            suffix = f"_aug{rep + 1}"
            cond_row["sample_id"] = str(cond_row["sample_id"]) + suffix
            meta_row["sample_id"] = str(meta_row["sample_id"]) + suffix
            cond_row["augmentation_type"] = "severity_weighted_amplitude"
            meta_row["augmentation_type"] = "severity_weighted_amplitude"
            new_x.append(x_aug)
            new_cond_rows.append(cond_row)
            new_meta_rows.append(meta_row)
            if event_mask_train is not None:
                new_masks.append(event_mask_train[i].copy())

    if new_x:
        x_new = np.stack(new_x, axis=0).astype(np.float32)
        cond_new = pd.DataFrame(new_cond_rows).reset_index(drop=True)
        meta_new = pd.DataFrame(new_meta_rows).reset_index(drop=True)
        cond_new = _apply_risk_labels(x_new, cond_new, dataset_summary)
        x_parts.append(x_new)
        cond_parts.append(cond_new)
        meta_parts.append(meta_new)
        if event_mask_train is not None:
            mask_parts.append(np.stack(new_masks, axis=0).astype(np.float32))

    x_all = np.concatenate(x_parts, axis=0).astype(np.float32)
    cond_all = pd.concat(cond_parts, ignore_index=True)
    meta_all = pd.concat(meta_parts, ignore_index=True)

    np.save(out_dir / "X_train_aug.npy", x_all)
    cond_all.to_csv(out_dir / "cond_train_aug.csv", index=False, encoding="utf-8-sig")
    meta_all.to_csv(out_dir / "meta_train_aug.csv", index=False, encoding="utf-8-sig")
    if event_mask_train is not None:
        mask_all = np.concatenate(mask_parts, axis=0).astype(np.float32)
        np.save(out_dir / "event_mask_train_aug.npy", mask_all)

    summary = {
        "enabled_by_default": False,
        "original_count": int(len(x_train)),
        "augmented_count": int(len(x_all)),
        "new_augmented_only_count": int(len(x_all) - len(x_train)),
        "repeat_by_severity": {"0": args.repeat_sev0, "1": args.repeat_sev1, "2": args.repeat_sev2, "3": args.repeat_sev3},
        "methods": ["severity_weighted_amplitude"],
        "event_mask_augmented": bool(event_mask_train is not None),
        "note": "Only the training split is augmented. Validation and test splits are untouched.",
    }
    (out_dir / "augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional train-only augmentation for extreme samples.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--repeat-sev0", type=int, default=0)
    parser.add_argument("--repeat-sev1", type=int, default=1)
    parser.add_argument("--repeat-sev2", type=int, default=3)
    parser.add_argument("--repeat-sev3", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(augment_trainset(parse_args()), ensure_ascii=False, indent=2))
