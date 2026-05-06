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
    rain_require_power_impact: bool = True


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
        # Medium mode keeps the user-configured minimum duration.
        # Default is 1h so short strong-rain events are not removed.
        min_duration = max(min_duration, 1.0)
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


def _heavy_rain_power_impact_screen(
    samples: pd.DataFrame,
    cfg: RiskScreenConfig,
    ramp_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    if not cfg.rain_require_power_impact or samples.empty or "event_type" not in samples.columns:
        return samples, {
            "enabled": bool(cfg.rain_require_power_impact),
            "before_heavy_rain_count": 0,
            "after_heavy_rain_count": 0,
            "removed_heavy_rain_count": 0,
            "removed_heavy_rain_ratio": 0.0,
        }

    event = samples["event_type"].astype(str)
    rain_mask = event.eq("暴雨/强降水") | event.str.contains("鏆撮洦/寮洪檷姘?", regex=False, na=False)
    before_count = int(rain_mask.sum())
    if before_count == 0:
        return samples, {
            "enabled": True,
            "before_heavy_rain_count": 0,
            "after_heavy_rain_count": 0,
            "removed_heavy_rain_count": 0,
            "removed_heavy_rain_ratio": 0.0,
        }

    low_irr = pd.to_numeric(samples.get("low_irradiance_flag", 0), errors="coerce").fillna(0.0)
    cum = pd.to_numeric(samples.get("cum_deficit", 0), errors="coerce").fillna(0.0)
    dur = pd.to_numeric(samples.get("imbalance_duration", 0), errors="coerce").fillna(0.0)
    ramp = pd.to_numeric(samples.get("netload_ramp_max", -np.inf), errors="coerce").fillna(-np.inf)

    impact_keep = (low_irr >= 1.0) | (cum > 0.0) | (dur >= float(cfg.min_imbalance_duration)) | (ramp >= float(ramp_threshold))
    keep_mask = (~rain_mask) | (rain_mask & impact_keep)
    screened = samples.loc[keep_mask].copy().reset_index(drop=True)

    rain_after = screened["event_type"].astype(str).eq("暴雨/强降水") | screened["event_type"].astype(str).str.contains("鏆撮洦/寮洪檷姘?", regex=False, na=False)
    after_count = int(rain_after.sum())
    removed_count = int(before_count - after_count)

    rain_before_df = samples.loc[rain_mask]
    rain_after_df = screened.loc[rain_after]
    ramp_before = pd.to_numeric(rain_before_df.get("netload_ramp_max", pd.Series(dtype=float)), errors="coerce").fillna(-np.inf)
    ramp_after_series = pd.to_numeric(rain_after_df.get("netload_ramp_max", pd.Series(dtype=float)), errors="coerce").fillna(-np.inf)

    summary = {
        "enabled": True,
        "before_heavy_rain_count": before_count,
        "after_heavy_rain_count": after_count,
        "removed_heavy_rain_count": removed_count,
        "removed_heavy_rain_ratio": float(removed_count / max(before_count, 1)),
        "heavy_rain_low_irr_count_before": int((pd.to_numeric(rain_before_df.get("low_irradiance_flag", 0), errors="coerce").fillna(0) >= 1).sum()),
        "heavy_rain_low_irr_count_after": int((pd.to_numeric(rain_after_df.get("low_irradiance_flag", 0), errors="coerce").fillna(0) >= 1).sum()),
        "heavy_rain_cum_deficit_positive_count_before": int((pd.to_numeric(rain_before_df.get("cum_deficit", 0), errors="coerce").fillna(0) > 0).sum()),
        "heavy_rain_cum_deficit_positive_count_after": int((pd.to_numeric(rain_after_df.get("cum_deficit", 0), errors="coerce").fillna(0) > 0).sum()),
        "heavy_rain_high_ramp_count_before": int((ramp_before >= float(ramp_threshold)).sum()),
        "heavy_rain_high_ramp_count_after": int((ramp_after_series >= float(ramp_threshold)).sum()),
        "ramp_threshold_used": float(ramp_threshold),
        "min_imbalance_duration_used": float(cfg.min_imbalance_duration),
    }
    return screened, summary


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

        ramp_threshold = float(screen_info.get("ramp_threshold_used", float("inf")))
        screened, heavy_rain_summary = _heavy_rain_power_impact_screen(screened, cfg, ramp_threshold)
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
            "heavy_rain_power_impact_screen": heavy_rain_summary,
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
        if summary.get("heavy_rain_power_impact_screen", {}).get("enabled", False):
            (out_dir / "rain_power_impact_screen_summary.json").write_text(
                json.dumps(summary["heavy_rain_power_impact_screen"], ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    return screened, summary
