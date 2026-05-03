from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RiskScreenConfig:
    enabled: bool = True
    mode: str = "medium"
    min_cum_deficit: float = 0.0
    min_imbalance_duration: float = 1.0
    ramp_quantile: float = 0.70
    keep_top_risk_ratio: Optional[float] = None
    min_samples_after_screen: int = 50


def _counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].value_counts().to_dict().items()}


def _mode_thresholds(samples: pd.DataFrame, cfg: RiskScreenConfig, mode: str) -> tuple[float, float, float]:
    mode = mode.strip().lower()
    if mode == "hybrid":
        mode = "medium"

    ramp_q = float(cfg.ramp_quantile)
    min_duration = float(cfg.min_imbalance_duration)
    min_cum = float(cfg.min_cum_deficit)

    if mode == "loose":
        ramp_q = 0.60
        min_duration = min(min_duration, 1.0)
    elif mode == "medium":
        ramp_q = ramp_q if np.isfinite(ramp_q) else 0.70
        min_duration = max(min_duration, 2.0)
    elif mode == "strict":
        ramp_q = 0.80
        min_duration = max(min_duration, 3.0)
        if "cum_deficit" in samples.columns:
            valid = samples["cum_deficit"].astype(float).dropna()
            if not valid.empty:
                min_cum = max(min_cum, float(valid.quantile(0.50)))
    else:
        raise ValueError("risk_screen_mode must be one of {'loose', 'medium', 'strict', 'hybrid'}.")

    ramp_values = samples["netload_ramp_max"].astype(float).dropna()
    ramp_threshold = float(ramp_values.quantile(ramp_q)) if not ramp_values.empty else float("inf")
    return min_cum, min_duration, ramp_threshold


def _apply_screen_once(samples: pd.DataFrame, cfg: RiskScreenConfig, mode: str) -> tuple[pd.DataFrame, dict]:
    if samples.empty:
        return samples.copy(), {"mode_used": mode, "warning": "empty_samples"}

    required = {"cum_deficit", "imbalance_duration", "netload_ramp_max"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples is missing risk metric columns: {sorted(missing)}")

    min_cum, min_duration, ramp_threshold = _mode_thresholds(samples, cfg, mode)
    cum = samples["cum_deficit"].astype(float).fillna(0.0)
    duration = samples["imbalance_duration"].astype(float).fillna(0.0)
    ramp = samples["netload_ramp_max"].astype(float).fillna(-np.inf)

    keep = (cum > min_cum) | (duration >= min_duration) | (ramp >= ramp_threshold)

    if cfg.keep_top_risk_ratio is not None:
        ratio = float(cfg.keep_top_risk_ratio)
        if 0.0 < ratio < 1.0:
            risk_score = cum.rank(pct=True) + duration.rank(pct=True) + ramp.rank(pct=True)
            n_top = max(1, int(np.ceil(len(samples) * ratio)))
            top_idx = risk_score.sort_values(ascending=False).head(n_top).index
            keep.loc[top_idx] = True

    screened = samples.loc[keep].copy().reset_index(drop=True)
    info = {
        "mode_used": mode,
        "min_cum_deficit_used": float(min_cum),
        "min_imbalance_duration_used": float(min_duration),
        "ramp_threshold_used": float(ramp_threshold),
        "keep_top_risk_ratio": cfg.keep_top_risk_ratio,
    }
    return screened, info


def screen_risk_samples(
    samples: pd.DataFrame,
    cfg: RiskScreenConfig | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or RiskScreenConfig()
    before = samples.copy().reset_index(drop=True)

    if not cfg.enabled:
        summary = {
            "enabled": False,
            "before_count": int(len(before)),
            "after_count": int(len(before)),
            "removed_count": 0,
            "removed_ratio": 0.0,
            "event_type_counts_before": _counts(before, "event_type"),
            "event_type_counts_after": _counts(before, "event_type"),
            "severity_counts_after": _counts(before, "severity_level"),
            "risk_screen_mode_used": "disabled",
            "warnings": [],
        }
        screened = before
    else:
        warnings: list[str] = []
        mode = cfg.mode.strip().lower()
        screened, screen_info = _apply_screen_once(before, cfg, mode)
        mode_used = screen_info["mode_used"]
        if len(screened) < int(cfg.min_samples_after_screen) and mode_used != "loose":
            warnings.append(
                f"screened sample count {len(screened)} is below {cfg.min_samples_after_screen}; fallback to loose mode"
            )
            screened, screen_info = _apply_screen_once(before, cfg, "loose")
            mode_used = "loose"
        if len(screened) < int(cfg.min_samples_after_screen):
            warnings.append(
                f"sample count remains below {cfg.min_samples_after_screen} after loose screening"
            )

        removed_count = int(len(before) - len(screened))
        summary = {
            "enabled": True,
            "config": asdict(cfg),
            "before_count": int(len(before)),
            "after_count": int(len(screened)),
            "removed_count": removed_count,
            "removed_ratio": float(removed_count / max(len(before), 1)),
            "event_type_counts_before": _counts(before, "event_type"),
            "event_type_counts_after": _counts(screened, "event_type"),
            "severity_counts_after": _counts(screened, "severity_level"),
            "risk_screen_mode_used": mode_used,
            "screen_info": screen_info,
            "warnings": warnings,
        }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        before.to_csv(out_dir / "samples_before_risk_screen.csv", index=False, encoding="utf-8-sig")
        screened.to_csv(out_dir / "samples_after_risk_screen.csv", index=False, encoding="utf-8-sig")
        (out_dir / "risk_screen_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    return screened, summary
