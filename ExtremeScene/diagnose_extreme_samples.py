from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


RISK_COLS = ["cum_deficit", "netload_ramp_max", "imbalance_duration"]


def _json_counts(series: pd.Series) -> str:
    if series.empty:
        return "{}"
    return json.dumps({str(k): int(v) for k, v in series.value_counts().to_dict().items()}, ensure_ascii=False)


def _q(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(q))


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _load_samples(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        for name in ["samples_evt_labeled.csv", "cond.csv"]:
            candidate = p / name
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        raise FileNotFoundError(f"Cannot find sample file: {path}")
    return pd.read_csv(p)


def _overview(df: pd.DataFrame) -> pd.DataFrame:
    cum = pd.to_numeric(df.get("cum_deficit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return pd.DataFrame(
        [
            {
                "n_samples": int(len(df)),
                "event_type_counts": _json_counts(df.get("event_type", pd.Series(dtype=object))),
                "severity_level_counts": _json_counts(df.get("severity_level", pd.Series(dtype=object))),
                "low_wind_flag_count": int(pd.to_numeric(df.get("low_wind_flag", 0), errors="coerce").fillna(0).sum()),
                "low_irradiance_flag_count": int(pd.to_numeric(df.get("low_irradiance_flag", 0), errors="coerce").fillna(0).sum()),
                "cum_deficit_zero_count": int((cum <= 1e-12).sum()),
                "cum_deficit_positive_count": int((cum > 1e-12).sum()),
            }
        ]
    )


def _event_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "event_type" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)
    for event_type, sub in df.groupby("event_type"):
        cum = pd.to_numeric(sub.get("cum_deficit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        ramp = pd.to_numeric(sub.get("netload_ramp_max", pd.Series(dtype=float)), errors="coerce")
        dur = pd.to_numeric(sub.get("imbalance_duration", pd.Series(dtype=float)), errors="coerce")
        low_wind = pd.to_numeric(sub.get("low_wind_flag", 0), errors="coerce").fillna(0.0)
        low_irr = pd.to_numeric(sub.get("low_irradiance_flag", 0), errors="coerce").fillna(0.0)
        rows.append(
            {
                "event_type": event_type,
                "count": int(len(sub)),
                "count_ratio": float(len(sub) / total),
                "cum_deficit_mean": float(cum.mean()),
                "cum_deficit_median": float(cum.median()),
                "cum_deficit_q75": _q(cum, 0.75),
                "cum_deficit_q90": _q(cum, 0.90),
                "cum_deficit_q95": _q(cum, 0.95),
                "cum_deficit_zero_ratio": float((cum <= 1e-12).mean()),
                "netload_ramp_max_mean": _safe_mean(ramp),
                "netload_ramp_max_q90": _q(ramp, 0.90),
                "imbalance_duration_mean": _safe_mean(dur),
                "imbalance_duration_q90": _q(dur, 0.90),
                "low_wind_ratio": float(low_wind.mean()) if len(low_wind) else 0.0,
                "low_irradiance_ratio": float(low_irr.mean()) if len(low_irr) else 0.0,
                "severity_distribution": _json_counts(
                    pd.to_numeric(sub.get("severity_level", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def _severity_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "severity_level" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    levels = pd.to_numeric(df["severity_level"], errors="coerce").fillna(0).astype(int)
    for level, idx in levels.groupby(levels).groups.items():
        sub = df.loc[idx]
        rows.append(
            {
                "severity_level": int(level),
                "count": int(len(sub)),
                "cum_deficit_mean": _safe_mean(sub.get("cum_deficit", pd.Series(dtype=float))),
                "cum_deficit_q90": _q(sub.get("cum_deficit", pd.Series(dtype=float)), 0.90),
                "netload_ramp_max_mean": _safe_mean(sub.get("netload_ramp_max", pd.Series(dtype=float))),
                "imbalance_duration_mean": _safe_mean(sub.get("imbalance_duration", pd.Series(dtype=float))),
                "event_type_distribution": _json_counts(sub.get("event_type", pd.Series(dtype=object))),
            }
        )
    return pd.DataFrame(rows).sort_values("severity_level")


def _split_diagnostics(base_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        path = base_dir / f"cond_{split}.csv"
        if not path.exists():
            continue
        sub = pd.read_csv(path)
        cum = pd.to_numeric(sub.get("cum_deficit", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "split": split,
                "count": int(len(sub)),
                "event_type_counts": _json_counts(sub.get("event_type", pd.Series(dtype=object))),
                "severity_counts": _json_counts(sub.get("severity_level", pd.Series(dtype=object))),
                "cum_deficit_mean": _safe_mean(cum),
                "cum_deficit_q90": _q(cum, 0.90),
                "cum_deficit_q95": _q(cum, 0.95),
                "low_wind_ratio": float(pd.to_numeric(sub.get("low_wind_flag", 0), errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                "low_irradiance_ratio": float(pd.to_numeric(sub.get("low_irradiance_flag", 0), errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _diagnostic_summary(df: pd.DataFrame, event_summary: pd.DataFrame) -> dict[str, Any]:
    n = max(len(df), 1)
    severity = pd.to_numeric(df.get("severity_level", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    high_risk = df.loc[severity > 0]
    event_counts = df.get("event_type", pd.Series(dtype=object)).value_counts()
    cum = pd.to_numeric(df.get("cum_deficit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    low_wind_count = int(pd.to_numeric(df.get("low_wind_flag", 0), errors="coerce").fillna(0).sum())
    low_irr_count = int(pd.to_numeric(df.get("low_irradiance_flag", 0), errors="coerce").fillna(0).sum())
    snow_count = 0
    heavy_rain_mask = pd.Series(False, index=df.index)
    if "event_type" in df.columns:
        snow_mask = df["event_type"].astype(str).str.contains("雪|snow|blizzard", case=False, regex=True, na=False)
        snow_count = int(snow_mask.sum())
        heavy_rain_mask = df["event_type"].astype(str).eq("暴雨/强降水")

    heavy_rain = df.loc[heavy_rain_mask].copy()
    heavy_rain_n = int(len(heavy_rain))
    heavy_rain_ratio = float(heavy_rain_n / n)
    heavy_severity = pd.to_numeric(heavy_rain.get("severity_level", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    heavy_cum = pd.to_numeric(heavy_rain.get("cum_deficit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    heavy_low_irr = pd.to_numeric(heavy_rain.get("low_irradiance_flag", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    heavy_sev0_ratio = float((heavy_severity == 0).mean()) if heavy_rain_n > 0 else 0.0
    heavy_low_power_ratio = float(((heavy_cum <= 1e-12) & (heavy_low_irr <= 0)).mean()) if heavy_rain_n > 0 else 0.0

    warnings: list[str] = []
    flags = {
        "severity_high_too_few": int((severity > 0).sum()) < max(5, int(0.10 * n)),
        "event_type_imbalanced": bool((not event_counts.empty) and (event_counts.iloc[0] / n > 0.65)),
        "low_irradiance_missing": low_irr_count == 0,
        "low_wind_too_few": low_wind_count < max(10, int(0.08 * n)),
        "many_zero_cum_deficit": float((cum <= 1e-12).mean()) > 0.35,
        "snow_samples_too_few": snow_count < 5,
        "high_risk_only_one_event_type": bool(len(high_risk) > 0 and high_risk.get("event_type", pd.Series(dtype=object)).nunique() <= 1),
        "heavy_rain_too_many": heavy_rain_ratio > 0.50,
        "heavy_rain_mostly_low_risk": heavy_sev0_ratio > 0.75,
        "heavy_rain_low_power_impact": heavy_low_power_ratio > 0.50,
    }
    for key, value in flags.items():
        if value:
            warnings.append(key)

    return {
        **flags,
        "n_samples": int(len(df)),
        "event_type_counts": {str(k): int(v) for k, v in event_counts.to_dict().items()},
        "severity_level_counts": {str(k): int(v) for k, v in severity.value_counts().sort_index().to_dict().items()},
        "low_wind_flag_count": low_wind_count,
        "low_irradiance_flag_count": low_irr_count,
        "cum_deficit_zero_ratio": float((cum <= 1e-12).mean()) if len(cum) else 0.0,
        "heavy_rain_count": heavy_rain_n,
        "heavy_rain_ratio": heavy_rain_ratio,
        "heavy_rain_severity0_ratio": heavy_sev0_ratio,
        "heavy_rain_low_power_impact_ratio": heavy_low_power_ratio,
        "snow_sample_count": snow_count,
        "warnings": warnings,
        "rare_event_type_suggestion": "Keep rare weather events for total-data training or case analysis, but avoid separate per-event claims when count < 5.",
    }


def _plot_figures(df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if "event_type" in df.columns and "cum_deficit" in df.columns:
        plt.figure(figsize=(9, 5))
        data = [pd.to_numeric(sub["cum_deficit"], errors="coerce").dropna().to_numpy() for _, sub in df.groupby("event_type")]
        labels = [str(k) for k in df.groupby("event_type").groups.keys()]
        if data:
            plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("cum_deficit")
        plt.title("Cumulative Deficit by Event Type")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "cum_deficit_by_event_type.png", dpi=220, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7, 4.5))
        df["event_type"].value_counts().plot(kind="bar")
        plt.ylabel("count")
        plt.title("Event Type Distribution")
        plt.tight_layout()
        plt.savefig(fig_dir / "event_type_distribution.png", dpi=220, bbox_inches="tight")
        plt.close()

    if "severity_level" in df.columns:
        plt.figure(figsize=(6, 4.2))
        pd.to_numeric(df["severity_level"], errors="coerce").fillna(0).astype(int).value_counts().sort_index().plot(kind="bar")
        plt.xlabel("severity_level")
        plt.ylabel("count")
        plt.title("Severity Distribution")
        plt.tight_layout()
        plt.savefig(fig_dir / "severity_distribution.png", dpi=220, bbox_inches="tight")
        plt.close()

    if "cum_deficit" in df.columns:
        plt.figure(figsize=(7, 4.5))
        pd.to_numeric(df["cum_deficit"], errors="coerce").fillna(0.0).plot(kind="hist", bins=30)
        plt.xlabel("cum_deficit")
        plt.title("Cumulative Deficit Histogram")
        plt.tight_layout()
        plt.savefig(fig_dir / "cum_deficit_hist.png", dpi=220, bbox_inches="tight")
        plt.close()

    if {"cum_deficit", "imbalance_duration"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        colors = pd.to_numeric(df.get("severity_level", 0), errors="coerce").fillna(0)
        plt.scatter(
            pd.to_numeric(df["cum_deficit"], errors="coerce"),
            pd.to_numeric(df["imbalance_duration"], errors="coerce"),
            c=colors,
            cmap="viridis",
            alpha=0.75,
        )
        plt.xlabel("cum_deficit")
        plt.ylabel("imbalance_duration")
        plt.title("Risk Scatter: Cum Deficit vs Duration")
        plt.colorbar(label="severity_level")
        plt.tight_layout()
        plt.savefig(fig_dir / "risk_scatter_cum_duration.png", dpi=220, bbox_inches="tight")
        plt.close()


def run_diagnostics(samples: str | Path, out_dir: str | Path) -> dict[str, Any]:
    sample_path = Path(samples)
    df = _load_samples(sample_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    overview = _overview(df)
    event_summary = _event_type_summary(df)
    severity_summary = _severity_summary(df)
    split_summary = _split_diagnostics(sample_path if sample_path.is_dir() else sample_path.parent)
    diagnostic_summary = _diagnostic_summary(df, event_summary)

    overview.to_csv(out / "sample_overview.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(out / "event_type_risk_summary.csv", index=False, encoding="utf-8-sig")
    severity_summary.to_csv(out / "severity_risk_summary.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(out / "split_diagnostic.csv", index=False, encoding="utf-8-sig")
    (out / "diagnostic_summary.json").write_text(
        json.dumps(diagnostic_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _plot_figures(df, out)
    return diagnostic_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose extracted extreme-event samples.")
    parser.add_argument("--samples", type=str, required=True, help="samples_evt_labeled.csv, cond.csv, or a dataset directory.")
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_diagnostics(args.samples, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
