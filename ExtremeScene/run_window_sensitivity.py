from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from Extreme_Extract import DetectConfig, detect_extreme_samples
from evt_fit import EVTConfig, fit_evt_and_label
from risk_screening import RiskScreenConfig, screen_risk_samples
from sample_metrics import MetricConfig, compute_metrics_for_samples

try:
    from build_mock_dataset import make_random_mock_data
except ModuleNotFoundError:
    from legacy.build_mock_dataset import make_random_mock_data


def _load_real(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        one = pd.read_csv(path)
        one["time"] = pd.to_datetime(one["time"])
        one["source_file"] = Path(path).name
        frames.append(one)
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _mock_data() -> pd.DataFrame:
    dfs = []
    for i in range(8):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 21)
        one = make_random_mock_data(seed=1000 + i, start_time=start, days=18)
        one["scenario_id"] = i
        dfs.append(one)
    return pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)


def _fixed_window_samples(samples: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    out = samples.copy()
    starts = []
    ends = []
    durations = []
    for _, row in out.iterrows():
        core_start = pd.to_datetime(row["core_start_time"])
        core_end = pd.to_datetime(row["core_end_time"])
        center = pd.Timestamp(core_start + (core_end - core_start) / 2).round("1h")
        start = center - pd.Timedelta(hours=seq_len // 2)
        end = start + pd.Timedelta(hours=seq_len - 1)
        starts.append(start)
        ends.append(end)
        durations.append(seq_len)
    out["start_time"] = starts
    out["end_time"] = ends
    out["duration_hours"] = durations
    return out


def _counts(series: pd.Series) -> str:
    return json.dumps({str(k): int(v) for k, v in series.value_counts().to_dict().items()}, ensure_ascii=False)


def run_window_sensitivity(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _mock_data() if args.use_mock or not args.real_data_csv else _load_real(args.real_data_csv)
    detect_cfg = DetectConfig(
        use_adaptive_thresholds=not bool(args.use_mock),
        low_irr_quantile=args.low_irr_quantile,
        low_wind_quantile=args.low_wind_quantile,
        low_resource_min_hours=args.low_resource_min_hours,
        daylight_irradiance_min=args.daylight_irradiance_min,
        cold_temp_quantile=0.10,
        cold_drop_24h_quantile=0.95,
        heat_temp_quantile=0.98,
        high_wind_speed_quantile=0.99,
        snowfall_quantile=0.97,
    )
    base_samples = detect_extreme_samples(df, cfg=detect_cfg, add_buffer_hours=0, merge_overlap=False)
    base_samples.to_csv(out_dir / "detected_weather_candidates.csv", index=False, encoding="utf-8-sig")

    rows = []
    for seq_len in args.seq_len:
        fixed_samples = _fixed_window_samples(base_samples, seq_len)
        metric_samples = compute_metrics_for_samples(
            df,
            fixed_samples,
            MetricConfig(
                imbalance_tau_mode=args.imbalance_tau_mode,
                imbalance_tau_quantile=args.imbalance_tau_quantile,
            ),
        )
        screened, screen_summary = screen_risk_samples(
            metric_samples,
            RiskScreenConfig(
                enabled=args.risk_screen_enabled,
                mode=args.risk_screen_mode,
                min_cum_deficit=args.min_cum_deficit,
                min_imbalance_duration=args.min_imbalance_duration,
                ramp_quantile=args.ramp_quantile,
            ),
            output_dir=out_dir / f"seq_len_{seq_len}",
        )
        labeled, evt_info = fit_evt_and_label(
            screened,
            EVTConfig(
                metric_col="cum_deficit",
                threshold_quantile=0.90,
                severity_mode=args.severity_mode,
                severity_q1=args.severity_q1,
                severity_q2=args.severity_q2,
                severity_q3=args.severity_q3,
            ),
        )
        labeled.to_csv(out_dir / f"seq_len_{seq_len}" / "samples_evt_labeled.csv", index=False, encoding="utf-8-sig")
        cum = pd.to_numeric(labeled.get("cum_deficit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        rows.append(
            {
                "seq_len": int(seq_len),
                "n_samples": int(len(labeled)),
                "event_type_counts": _counts(labeled.get("event_type", pd.Series(dtype=object))),
                "severity_counts": _counts(labeled.get("severity_level", pd.Series(dtype=object))),
                "cum_deficit_mean": float(cum.mean()) if len(cum) else 0.0,
                "cum_deficit_q90": float(cum.quantile(0.90)) if len(cum) else 0.0,
                "cum_deficit_q95": float(cum.quantile(0.95)) if len(cum) else 0.0,
                "high_risk_sample_count": int((pd.to_numeric(labeled.get("severity_level", 0), errors="coerce").fillna(0).astype(int) > 0).sum()),
                "low_wind_ratio": float(pd.to_numeric(labeled.get("low_wind_flag", 0), errors="coerce").fillna(0).mean()) if len(labeled) else 0.0,
                "low_irradiance_ratio": float(pd.to_numeric(labeled.get("low_irradiance_flag", 0), errors="coerce").fillna(0).mean()) if len(labeled) else 0.0,
                "risk_screen_mode_used": screen_summary.get("risk_screen_mode_used"),
                "evt_method": evt_info.get("method"),
                "severity_mode": evt_info.get("severity_mode"),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "window_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "window_sensitivity_config.json").write_text(
        json.dumps(vars(args) | {"detect_cfg": asdict(detect_cfg)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare extracted risk-sample libraries across sequence lengths.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--real-data-csv", type=str, action="append", default=None)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[24, 36, 48])
    parser.add_argument("--low-irr-quantile", type=float, default=0.30)
    parser.add_argument("--low-wind-quantile", type=float, default=0.30)
    parser.add_argument("--low-resource-min-hours", type=int, default=3)
    parser.add_argument("--daylight-irradiance-min", type=float, default=30.0)
    parser.add_argument("--risk-screen-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--risk-screen-mode", type=str, default="medium", choices=["loose", "medium", "strict", "hybrid"])
    parser.add_argument("--min-cum-deficit", type=float, default=0.0)
    parser.add_argument("--min-imbalance-duration", type=float, default=1.0)
    parser.add_argument("--ramp-quantile", type=float, default=0.70)
    parser.add_argument("--imbalance-tau-mode", type=str, default="monthly_quantile", choices=["global_quantile", "monthly_quantile", "seasonal_quantile", "fixed", "quantile"])
    parser.add_argument("--imbalance-tau-quantile", type=float, default=0.75)
    parser.add_argument("--severity-mode", type=str, default="hybrid", choices=["evt_prob", "quantile", "hybrid"])
    parser.add_argument("--severity-q1", type=float, default=0.60)
    parser.add_argument("--severity-q2", type=float, default=0.80)
    parser.add_argument("--severity-q3", type=float, default=0.92)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_window_sensitivity(parse_args())
    print(result.to_string(index=False))
