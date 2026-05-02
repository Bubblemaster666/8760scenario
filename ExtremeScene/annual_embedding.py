from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from risk_metrics import compute_net_load


@dataclass
class AnnualEmbeddingConfig:
    background: Optional[str]
    segments: str
    cond: str
    out_dir: str
    meta: Optional[str] = None
    monthly_event_prob: Optional[str] = None
    monthly_severity_prob: Optional[str] = None
    smooth_width: int = 4
    use_mock_background: bool = False
    annual_hours: int = 8760


def _mock_background(hours: int = 8760) -> pd.DataFrame:
    time = pd.date_range("2024-01-01 00:00:00", periods=hours, freq="1h")
    hour = time.hour.to_numpy()
    day = np.arange(hours) / 24.0
    load = 600 + 70 * np.sin((hour - 8) / 24 * 2 * np.pi) + 30 * np.sin(2 * np.pi * day / 365)
    wind = np.clip(180 + 40 * np.sin(2 * np.pi * day / 7) + 15 * np.cos(2 * np.pi * hour / 24), 0, None)
    solar = np.clip(220 * np.sin((hour - 6) / 12 * np.pi), 0, None)
    return pd.DataFrame({"time": time, "load": load, "wind_power": wind, "solar_power": solar})


def _load_background(cfg: AnnualEmbeddingConfig) -> pd.DataFrame:
    if cfg.use_mock_background or not cfg.background:
        return _mock_background(cfg.annual_hours)
    path = Path(cfg.background)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Background .npy must have shape [T, 3].")
        time = pd.date_range("2024-01-01 00:00:00", periods=arr.shape[0], freq="1h")
        return pd.DataFrame({"time": time, "load": arr[:, 0], "wind_power": arr[:, 1], "solar_power": arr[:, 2]})
    bg_df = pd.read_csv(path)
    if "time" not in bg_df.columns:
        bg_df.insert(0, "time", pd.date_range("2024-01-01 00:00:00", periods=len(bg_df), freq="1h"))
    bg_df["time"] = pd.to_datetime(bg_df["time"])
    return bg_df[["time", "load", "wind_power", "solar_power"]].copy()


def _monthly_prob_from_cond(cond: pd.DataFrame) -> pd.DataFrame:
    out = cond["month"].value_counts(normalize=True).sort_index().rename_axis("month").reset_index(name="probability")
    return out


def _monthly_severity_prob_from_cond(cond: pd.DataFrame) -> pd.DataFrame:
    grouped = cond.groupby(["month", "severity_level"]).size().rename("count").reset_index()
    grouped["probability"] = grouped["count"] / grouped.groupby("month")["count"].transform("sum")
    return grouped[["month", "severity_level", "probability"]]


def _load_probabilities(cfg: AnnualEmbeddingConfig, cond: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.monthly_event_prob:
        month_prob = pd.read_csv(cfg.monthly_event_prob)
    else:
        month_prob = _monthly_prob_from_cond(cond)
    if cfg.monthly_severity_prob:
        severity_prob = pd.read_csv(cfg.monthly_severity_prob)
    else:
        severity_prob = _monthly_severity_prob_from_cond(cond)
    return month_prob, severity_prob


def _candidate_positions(time_index: pd.Series, month: int, seg_len: int) -> np.ndarray:
    hours = time_index.dt.month.to_numpy(dtype=int)
    valid = np.where(hours == int(month))[0]
    if valid.size == 0:
        return np.asarray([], dtype=int)
    starts = valid[valid <= len(time_index) - seg_len]
    return starts


def _cosine_blend(existing: np.ndarray, segment: np.ndarray, smooth_width: int) -> np.ndarray:
    seg = segment.copy()
    width = min(smooth_width, seg.shape[0] // 2)
    if width <= 0:
        return seg
    weights = 0.5 * (1 - np.cos(np.linspace(0, np.pi, width)))
    for i in range(width):
        seg[i] = (1 - weights[i]) * existing[i] + weights[i] * seg[i]
        seg[-width + i] = weights[i] * existing[-width + i] + (1 - weights[i]) * seg[-width + i]
    return seg


def _embed_segments(
    background_df: pd.DataFrame,
    segments: np.ndarray,
    cond: pd.DataFrame,
    smooth_width: int,
    use_smoothing: bool,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bg = background_df.copy()
    values = bg[["load", "wind_power", "solar_power"]].to_numpy(dtype=np.float32, copy=True)
    insert_rows = []
    for i, row in cond.iterrows():
        segment = segments[i].T.copy()
        seg_len = segment.shape[0]
        starts = _candidate_positions(bg["time"], int(row["month"]), seg_len)
        if starts.size == 0:
            continue
        start = int(rng.choice(starts))
        end = start + seg_len
        original = values[start:end].copy()
        boundary_before = float(np.abs(values[start] - values[start - 1]).mean()) if start > 0 else 0.0
        if use_smoothing:
            segment = _cosine_blend(original, segment, smooth_width=smooth_width)
        values[start:end] = segment
        boundary_after = float(np.abs(values[start] - values[start - 1]).mean()) if start > 0 else 0.0
        insert_rows.append(
            {
                "generated_id": f"G{i:05d}",
                "sample_id": row["sample_id"],
                "month": int(row["month"]),
                "severity_level": int(row["severity_level"]),
                "start_idx": start,
                "end_idx": end - 1,
                "boundary_jump_before": boundary_before,
                "boundary_jump_after": boundary_after,
            }
        )
    out_df = bg.copy()
    out_df[["load", "wind_power", "solar_power"]] = values
    return out_df, pd.DataFrame(insert_rows)


def _evaluate_embedding(df: pd.DataFrame, inserts: pd.DataFrame, cond: pd.DataFrame, month_prob: pd.DataFrame, severity_prob: pd.DataFrame) -> dict[str, float]:
    inserted_month = inserts["month"].value_counts(normalize=True).sort_index()
    target_month = month_prob.set_index("month")["probability"]
    month_error = float(np.abs(inserted_month.reindex(target_month.index, fill_value=0.0) - target_month).mean())

    inserted_severity = inserts["severity_level"].value_counts(normalize=True).sort_index()
    target_severity = cond["severity_level"].value_counts(normalize=True).sort_index()
    severity_error = float(np.abs(inserted_severity.reindex(target_severity.index, fill_value=0.0) - target_severity).mean())

    net = compute_net_load(df["load"], df["wind_power"], df["solar_power"])
    tau = float(np.quantile(net, 0.75))
    high_risk_hours = float((net > tau).sum())
    cumulative_deficit = float(np.maximum(0.0, net - tau).sum())

    return {
        "monthly_event_freq_error": month_error,
        "severity_ratio_error": severity_error,
        "mean_boundary_jump_before": float(inserts["boundary_jump_before"].mean() if not inserts.empty else 0.0),
        "mean_boundary_jump_after": float(inserts["boundary_jump_after"].mean() if not inserts.empty else 0.0),
        "extreme_month_high_risk_hours": high_risk_hours,
        "extreme_month_cumulative_deficit": cumulative_deficit,
    }


def run_annual_embedding(cfg: AnnualEmbeddingConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    background_df = _load_background(cfg)
    segments = np.load(cfg.segments).astype(np.float32)
    cond = pd.read_csv(cfg.cond).iloc[: len(segments)].reset_index(drop=True)
    month_prob, severity_prob = _load_probabilities(cfg, cond)

    baseline_df, baseline_inserts = _embed_segments(background_df, segments, cond, smooth_width=0, use_smoothing=False, seed=42)
    proposed_df, proposed_inserts = _embed_segments(background_df, segments, cond, smooth_width=cfg.smooth_width, use_smoothing=True, seed=42)

    baseline_df.to_csv(out_dir / "annual_scenario_baseline.csv", index=False, encoding="utf-8-sig")
    proposed_df.to_csv(out_dir / "annual_scenario_proposed.csv", index=False, encoding="utf-8-sig")
    boundary_df = pd.concat(
        [
            baseline_inserts.assign(method="baseline"),
            proposed_inserts.assign(method="proposed"),
        ],
        ignore_index=True,
    )
    boundary_df.to_csv(out_dir / "boundary_smoothing_metrics.csv", index=False, encoding="utf-8-sig")

    baseline_eval = _evaluate_embedding(baseline_df, baseline_inserts, cond, month_prob, severity_prob)
    proposed_eval = _evaluate_embedding(proposed_df, proposed_inserts, cond, month_prob, severity_prob)
    summary_df = pd.DataFrame(
        [
            {"method": "baseline_embedding", **baseline_eval},
            {"method": "proposed_embedding", **proposed_eval},
        ]
    )
    summary_df.to_csv(out_dir / "annual_embedding_summary.csv", index=False, encoding="utf-8-sig")

    if not proposed_inserts.empty:
        row = proposed_inserts.iloc[0]
        start = max(0, int(row["start_idx"]) - cfg.smooth_width - 6)
        end = min(len(proposed_df), int(row["end_idx"]) + cfg.smooth_width + 6)
        x = np.arange(start, end)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10.5, 4.8))
        plt.plot(x, baseline_df["load"].iloc[start:end].to_numpy(), label="Baseline load")
        plt.plot(x, proposed_df["load"].iloc[start:end].to_numpy(), label="Proposed load", linestyle="--")
        plt.xlabel("Hour index")
        plt.ylabel("Load")
        plt.title("Boundary smoothing comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "annual_embedding_boundary.png", dpi=220, bbox_inches="tight")
        plt.close()

    summary = {
        "baseline": baseline_eval,
        "proposed": proposed_eval,
        "smooth_width": cfg.smooth_width,
    }
    (out_dir / "annual_embedding_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> AnnualEmbeddingConfig:
    parser = argparse.ArgumentParser(description="Lightweight annual embedding for generated extreme segments.")
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument("--segments", type=str, required=True)
    parser.add_argument("--cond", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--monthly-event-prob", type=str, default=None)
    parser.add_argument("--monthly-severity-prob", type=str, default=None)
    parser.add_argument("--smooth-width", type=int, default=4)
    parser.add_argument("--use-mock-background", action="store_true")
    parser.add_argument("--annual-hours", type=int, default=8760)
    args = parser.parse_args()
    return AnnualEmbeddingConfig(
        background=args.background,
        segments=args.segments,
        cond=args.cond,
        out_dir=args.out_dir,
        meta=args.meta,
        monthly_event_prob=args.monthly_event_prob,
        monthly_severity_prob=args.monthly_severity_prob,
        smooth_width=args.smooth_width,
        use_mock_background=args.use_mock_background,
        annual_hours=args.annual_hours,
    )


if __name__ == "__main__":
    run_annual_embedding(parse_args())
