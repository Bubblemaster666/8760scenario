from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_diffusion import infer_day_mask
from risk_shapelet_augmentation_utils import (
    hard_risk_metrics,
    relative_or_absolute_within,
    synthesize_shapelet_preserving_window,
)


@dataclass
class RiskShapeletAugConfig:
    data_dir: str
    out_dir: str
    aug_ratio_sev0: int = 0
    aug_ratio_sev1: int = 1
    aug_ratio_sev2: int = 2
    aug_ratio_sev3: int = 3
    risk_tolerance_cum: float = 0.20
    risk_tolerance_ramp: float = 0.30
    risk_tolerance_dur: float = 0.30
    core_jitter_scale: float = 0.01
    buffer_jitter_scale: float = 0.03
    residual_bootstrap: bool = True
    seed: int = 42
    daylight_start_hour: int = 6
    daylight_end_hour: int = 18


def _severity_ratio(cfg: RiskShapeletAugConfig, severity: int) -> int:
    return {
        0: int(cfg.aug_ratio_sev0),
        1: int(cfg.aug_ratio_sev1),
        2: int(cfg.aug_ratio_sev2),
        3: int(cfg.aug_ratio_sev3),
    }.get(int(severity), 0)


def _candidate_donors(cond: pd.DataFrame, idx: int) -> np.ndarray:
    row = cond.iloc[idx]
    event_type = row.get("event_type", "")
    severity = int(row.get("severity_level", 0))
    same = cond.index[(cond["event_type"] == event_type) & (cond["severity_level"].astype(int) == severity)].to_numpy()
    same = same[same != idx]
    if same.size:
        return same
    adjacent = cond.index[
        (cond["event_type"] == event_type)
        & (cond["severity_level"].astype(int).sub(severity).abs() <= 1)
    ].to_numpy()
    adjacent = adjacent[adjacent != idx]
    if adjacent.size:
        return adjacent
    fallback = cond.index[cond["event_type"] == event_type].to_numpy()
    fallback = fallback[fallback != idx]
    return fallback if fallback.size else np.array([idx], dtype=int)


def _load_day_mask(data_dir: Path, meta_train: pd.DataFrame, seq_len: int, cfg: RiskShapeletAugConfig) -> np.ndarray:
    path = data_dir / "day_mask_train.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    return infer_day_mask(meta_train, seq_len, cfg.daylight_start_hour, cfg.daylight_end_hour)


def augment_risk_shapelet_trainset(cfg: RiskShapeletAugConfig) -> dict:
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    cond_train = pd.read_csv(data_dir / "cond_train.csv")
    meta_train = pd.read_csv(data_dir / "meta_train.csv")
    mask_path = data_dir / "event_mask_train.npy"
    if mask_path.exists():
        event_mask = np.load(mask_path).astype(np.float32)
    else:
        event_mask = np.ones((len(x_train), x_train.shape[2]), dtype=np.float32)
    day_mask = _load_day_mask(data_dir, meta_train, x_train.shape[2], cfg)

    cond_train = cond_train.reset_index(drop=True).copy()
    meta_train = meta_train.reset_index(drop=True).copy()
    cond_train["severity_level"] = pd.to_numeric(cond_train["severity_level"], errors="coerce").fillna(0).astype(int)
    channel_std = x_train.std(axis=(0, 2)).astype(np.float32)
    channel_std = np.maximum(channel_std, 1e-6)
    channel_max = (x_train.max(axis=(0, 2)) * 1.05 + 1e-6).astype(np.float32)
    rng = np.random.default_rng(int(cfg.seed))

    aug_x: list[np.ndarray] = []
    aug_cond: list[dict] = []
    aug_meta: list[dict] = []
    aug_mask: list[np.ndarray] = []
    rejected: list[dict] = []

    for idx in range(len(x_train)):
        severity = int(cond_train.loc[idx, "severity_level"])
        n_aug = _severity_ratio(cfg, severity)
        if n_aug <= 0:
            continue
        donors = _candidate_donors(cond_train, idx)
        tau = float(cond_train.loc[idx, "imbalance_tau"])
        delta_t = float(cond_train.loc[idx, "delta_t_hours"]) if "delta_t_hours" in cond_train.columns else 1.0
        old_metrics = hard_risk_metrics(x_train[idx], tau=tau, delta_t_hours=delta_t)
        old_cum = old_metrics["cum_deficit"]
        old_ramp = old_metrics["netload_ramp_max"]
        old_dur = old_metrics["imbalance_duration"]

        for aug_no in range(n_aug):
            donor_idx = int(rng.choice(donors))
            source_net = x_train[idx, 0] - x_train[idx, 1] - x_train[idx, 2]
            protected_mask = np.maximum(event_mask[idx], (source_net > tau).astype(np.float32))
            new_x = synthesize_shapelet_preserving_window(
                source=x_train[idx],
                donor=x_train[donor_idx],
                event_mask=protected_mask,
                day_mask=day_mask[idx],
                channel_std=channel_std,
                channel_max=channel_max,
                rng=rng,
                core_jitter_scale=cfg.core_jitter_scale,
                buffer_jitter_scale=cfg.buffer_jitter_scale,
                residual_bootstrap=cfg.residual_bootstrap,
            )
            non_risk_buffer = protected_mask <= 0.0
            if non_risk_buffer.any():
                new_net = new_x[0] - new_x[1] - new_x[2]
                excess = np.maximum(0.0, new_net - tau)
                new_x[0, non_risk_buffer] = np.maximum(0.0, new_x[0, non_risk_buffer] - excess[non_risk_buffer])
            metrics = hard_risk_metrics(new_x, tau=tau, delta_t_hours=delta_t)
            checks = {
                "cum": relative_or_absolute_within(metrics["cum_deficit"], old_cum, cfg.risk_tolerance_cum, floor=1.0),
                "ramp": relative_or_absolute_within(metrics["netload_ramp_max"], old_ramp, cfg.risk_tolerance_ramp, floor=0.5),
                "dur": relative_or_absolute_within(metrics["imbalance_duration"], old_dur, cfg.risk_tolerance_dur, floor=1.0),
            }
            if not all(checks.values()):
                rejected.append(
                    {
                        "source_idx": int(idx),
                        "source_sample_id": cond_train.loc[idx, "sample_id"],
                        "donor_idx": donor_idx,
                        "severity_level": severity,
                        "reject_reason": ",".join([key for key, ok in checks.items() if not ok]),
                        "old_cum_deficit": old_cum,
                        "new_cum_deficit": metrics["cum_deficit"],
                        "old_netload_ramp_max": old_ramp,
                        "new_netload_ramp_max": metrics["netload_ramp_max"],
                        "old_imbalance_duration": old_dur,
                        "new_imbalance_duration": metrics["imbalance_duration"],
                    }
                )
                continue

            source_sample_id = str(cond_train.loc[idx, "sample_id"])
            new_sample_id = f"{source_sample_id}_RS{aug_no + 1}"
            cond_row = cond_train.iloc[idx].to_dict()
            cond_row.update(
                {
                    "sample_id": new_sample_id,
                    "cum_deficit": metrics["cum_deficit"],
                    "netload_ramp_max": metrics["netload_ramp_max"],
                    "imbalance_duration": metrics["imbalance_duration"],
                    "augmented": 1,
                    "source_sample_id": source_sample_id,
                    "augmentation_type": "risk_shapelet_preserving",
                }
            )
            meta_row = meta_train.iloc[idx].to_dict()
            meta_row.update(
                {
                    "sample_id": new_sample_id,
                    "source_sample_id": source_sample_id,
                    "augmentation_type": "risk_shapelet_preserving",
                }
            )
            aug_x.append(new_x.astype(np.float32))
            aug_cond.append(cond_row)
            aug_meta.append(meta_row)
            aug_mask.append(event_mask[idx].astype(np.float32))

    original_cond = cond_train.copy()
    original_meta = meta_train.copy()
    original_cond["augmented"] = 0
    original_cond["source_sample_id"] = original_cond["sample_id"].astype(str)
    original_cond["augmentation_type"] = "original"
    original_meta["source_sample_id"] = original_meta["sample_id"].astype(str)
    original_meta["augmentation_type"] = "original"

    if aug_x:
        x_out = np.concatenate([x_train, np.stack(aug_x).astype(np.float32)], axis=0)
        cond_out = pd.concat([original_cond, pd.DataFrame(aug_cond)], ignore_index=True)
        meta_out = pd.concat([original_meta, pd.DataFrame(aug_meta)], ignore_index=True)
        mask_out = np.concatenate([event_mask, np.stack(aug_mask).astype(np.float32)], axis=0)
    else:
        x_out = x_train.copy()
        cond_out = original_cond
        meta_out = original_meta
        mask_out = event_mask.copy()

    np.save(out_dir / "X_train_riskshape_aug.npy", x_out.astype(np.float32))
    np.save(out_dir / "event_mask_train_riskshape_aug.npy", mask_out.astype(np.float32))
    cond_out.to_csv(out_dir / "cond_train_riskshape_aug.csv", index=False, encoding="utf-8-sig")
    meta_out.to_csv(out_dir / "meta_train_riskshape_aug.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rejected).to_csv(out_dir / "rejected_augmentation_log.csv", index=False, encoding="utf-8-sig")

    accepted = int(len(aug_x))
    rejected_count = int(len(rejected))
    if aug_cond:
        severity_counts = pd.DataFrame(aug_cond)["severity_level"].value_counts().sort_index().to_dict()
        accepted_by_severity = {str(int(k)): int(v) for k, v in severity_counts.items()}
    else:
        accepted_by_severity = {}
    summary = {
        "config": asdict(cfg),
        "original_train_size": int(len(x_train)),
        "accepted_aug_count": accepted,
        "rejected_aug_count": rejected_count,
        "augmented_train_size": int(len(x_out)),
        "augmentation_ratio": float(len(x_out) / max(len(x_train), 1)),
        "accepted_by_severity": accepted_by_severity,
        "note": "Only train split was augmented. Original X_train/val/test files were not modified.",
    }
    (out_dir / "augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> RiskShapeletAugConfig:
    parser = argparse.ArgumentParser(description="Build risk-shapelet-preserving train-only augmentation.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--aug-ratio-sev0", type=int, default=0)
    parser.add_argument("--aug-ratio-sev1", type=int, default=1)
    parser.add_argument("--aug-ratio-sev2", type=int, default=2)
    parser.add_argument("--aug-ratio-sev3", type=int, default=3)
    parser.add_argument("--risk-tolerance-cum", type=float, default=0.20)
    parser.add_argument("--risk-tolerance-ramp", type=float, default=0.30)
    parser.add_argument("--risk-tolerance-dur", type=float, default=0.30)
    parser.add_argument("--core-jitter-scale", type=float, default=0.01)
    parser.add_argument("--buffer-jitter-scale", type=float, default=0.03)
    parser.add_argument("--residual-bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return RiskShapeletAugConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    augment_risk_shapelet_trainset(parse_args())
