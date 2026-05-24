from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from run_paper_pipeline import PaperPipelineConfig, run_pipeline


BASE_DIR = Path(__file__).resolve().parent


def _base_cfg(real_csv: Path, out_dir: Path) -> PaperPipelineConfig:
    return PaperPipelineConfig(
        out_dir=str(out_dir),
        use_mock=False,
        real_data_csv=(str(real_csv),),
        seq_len=36,
        seed=42,
        skip_model_experiments=True,
    )


def _event_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


def _summarize_run(name: str, run_dir: Path) -> dict:
    summary_path = run_dir / "pipeline_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_dir = run_dir / "dataset"
    cond_train = pd.read_csv(dataset_dir / "cond_train.csv")
    cond_val = pd.read_csv(dataset_dir / "cond_val.csv")
    cond_test = pd.read_csv(dataset_dir / "cond_test.csv")
    all_cond = pd.concat([cond_train, cond_val, cond_test], ignore_index=True)

    event_counts = _event_counts(all_cond["event_type"]) if "event_type" in all_cond.columns else {}
    severity_counts = _event_counts(all_cond["severity_level"]) if "severity_level" in all_cond.columns else {}
    dominant_ratio = 0.0
    if event_counts:
        dominant_ratio = max(event_counts.values()) / max(sum(event_counts.values()), 1)
    highrisk_total = int(pd.to_numeric(all_cond.get("severity_level", 0), errors="coerce").fillna(0).ge(2).sum())
    highrisk_test = int(pd.to_numeric(cond_test.get("severity_level", 0), errors="coerce").fillna(0).ge(2).sum())
    zero_cum_ratio = float(pd.to_numeric(all_cond.get("cum_deficit", 0), errors="coerce").fillna(0.0).le(0.0).mean())
    n_event_types = int(len(event_counts))
    n_samples = int(len(all_cond))
    expansion_score = float(
        n_samples
        + 3.0 * highrisk_total
        + 5.0 * highrisk_test
        + 4.0 * n_event_types
        - 15.0 * max(dominant_ratio - 0.75, 0.0)
    )

    row = {
        "experiment_name": name,
        "out_dir": str(run_dir),
        "n_samples": n_samples,
        "train_count": int(len(cond_train)),
        "val_count": int(len(cond_val)),
        "test_count": int(len(cond_test)),
        "n_event_types": n_event_types,
        "dominant_event_ratio": dominant_ratio,
        "highrisk_total_count": highrisk_total,
        "highrisk_test_count": highrisk_test,
        "zero_cum_deficit_ratio": zero_cum_ratio,
        "severity_0_count": int(severity_counts.get("0", 0)),
        "severity_1_count": int(severity_counts.get("1", 0)),
        "severity_2_count": int(severity_counts.get("2", 0)),
        "severity_3_count": int(severity_counts.get("3", 0)),
        "heat_count": int(event_counts.get("高温", 0)),
        "wind_count": int(event_counts.get("大风/沙尘暴", 0)),
        "cold_count": int(event_counts.get("寒潮", 0)),
        "rain_count": int(event_counts.get("暴雨/强降水", 0)),
        "expansion_score": expansion_score,
        "warnings": "; ".join(summary.get("diagnostic_summary", {}).get("warnings", [])),
        "event_type_counts": json.dumps(event_counts, ensure_ascii=False),
        "severity_counts": json.dumps(severity_counts, ensure_ascii=False),
    }
    return row


def _report_line(cfg: PaperPipelineConfig) -> dict:
    return {
        "min_event_hours": cfg.min_event_hours,
        "heat_temp_quantile": cfg.heat_temp_quantile,
        "high_wind_speed_quantile": cfg.high_wind_speed_quantile,
        "cold_temp_quantile": cfg.cold_temp_quantile,
        "cold_drop_24h_quantile": cfg.cold_drop_24h_quantile,
        "low_irr_quantile": cfg.low_irr_quantile,
        "low_wind_quantile": cfg.low_wind_quantile,
        "low_resource_min_hours": cfg.low_resource_min_hours,
        "risk_screen_mode": cfg.risk_screen_mode,
        "min_imbalance_duration": cfg.min_imbalance_duration,
        "ramp_quantile": cfg.ramp_quantile,
        "min_samples_after_screen": cfg.min_samples_after_screen,
    }


def run_sweep(real_csv: Path, out_root: Path, result_root: Path) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    base = _base_cfg(real_csv, out_root / "baseline_current")
    configs: list[tuple[str, PaperPipelineConfig]] = [
        ("baseline_current", base),
        (
            "relaxed_event_q95",
            replace(
                base,
                out_dir=str(out_root / "relaxed_event_q95"),
                min_event_hours=4,
                heat_temp_quantile=0.95,
                high_wind_speed_quantile=0.97,
                cold_temp_quantile=0.15,
                cold_drop_24h_quantile=0.90,
                low_irr_quantile=0.40,
                low_wind_quantile=0.35,
                low_resource_min_hours=2,
                ramp_quantile=0.65,
                min_samples_after_screen=40,
            ),
        ),
        (
            "balanced_relaxed",
            replace(
                base,
                out_dir=str(out_root / "balanced_relaxed"),
                min_event_hours=4,
                heat_temp_quantile=0.96,
                high_wind_speed_quantile=0.96,
                cold_temp_quantile=0.15,
                cold_drop_24h_quantile=0.90,
                low_irr_quantile=0.40,
                low_wind_quantile=0.35,
                low_resource_min_hours=2,
                ramp_quantile=0.60,
                min_samples_after_screen=40,
            ),
        ),
        (
            "relaxed_loose_screen",
            replace(
                base,
                out_dir=str(out_root / "relaxed_loose_screen"),
                min_event_hours=4,
                heat_temp_quantile=0.95,
                high_wind_speed_quantile=0.97,
                cold_temp_quantile=0.15,
                cold_drop_24h_quantile=0.90,
                low_irr_quantile=0.40,
                low_wind_quantile=0.35,
                low_resource_min_hours=2,
                risk_screen_mode="loose",
                min_imbalance_duration=0.0,
                ramp_quantile=0.60,
                min_samples_after_screen=35,
            ),
        ),
        (
            "more_relaxed",
            replace(
                base,
                out_dir=str(out_root / "more_relaxed"),
                min_event_hours=3,
                heat_temp_quantile=0.93,
                high_wind_speed_quantile=0.95,
                cold_temp_quantile=0.20,
                cold_drop_24h_quantile=0.85,
                low_irr_quantile=0.45,
                low_wind_quantile=0.40,
                low_resource_min_hours=2,
                risk_screen_mode="loose",
                min_imbalance_duration=0.0,
                ramp_quantile=0.55,
                min_samples_after_screen=30,
            ),
        ),
    ]

    rows: list[dict] = []
    detail_rows: list[dict] = []
    for name, cfg in configs:
        print(f"\n=== threshold sweep: {name} ===")
        run_pipeline(cfg)
        run_dir = Path(cfg.out_dir)
        row = _summarize_run(name, run_dir)
        row.update(_report_line(cfg))
        rows.append(row)
        detail_rows.append({"experiment_name": name, **_report_line(cfg)})

    df = pd.DataFrame(rows).sort_values(["expansion_score", "n_samples"], ascending=[False, False]).reset_index(drop=True)
    if len(df):
        df["rank"] = np.arange(1, len(df) + 1)
    else:
        df["rank"] = pd.Series(dtype=int)
    df.to_csv(result_root / "openenergyhub_threshold_sweep_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# OpenEnergyHub CAISO threshold sweep",
        "",
        "This sweep adjusts event-extraction and risk-screen thresholds only. No model training is run here.",
        "",
        "## Ranking rule",
        "",
        "- Prefer more retained extreme windows.",
        "- Prefer more severity>=2 windows, especially in test split.",
        "- Prefer more event-type coverage.",
        "- Penalize severe one-event dominance.",
        "",
        "## Summary",
        "",
        df.to_markdown(index=False) if len(df) else "No runs.",
        "",
        "## Recommendation",
        "",
    ]
    if len(df):
        best = df.iloc[0]
        lines.extend(
            [
                f"- Recommended threshold set: `{best['experiment_name']}`",
                f"- Retained samples: {int(best['n_samples'])}",
                f"- Event types covered: {int(best['n_event_types'])}",
                f"- High-risk samples total/test: {int(best['highrisk_total_count'])}/{int(best['highrisk_test_count'])}",
                f"- Dominant event ratio: {float(best['dominant_event_ratio']):.3f}",
                "",
                "Use this configuration for the next external-validation rerun if we want a larger CAISO extreme-event library.",
            ]
        )
    (result_root / "openenergyhub_threshold_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep OpenEnergyHub CAISO event-extraction thresholds without training models.")
    parser.add_argument(
        "--real-data-csv",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "openenergyhub_caiso_hourly_pipeline_input.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=BASE_DIR / "outputs" / "openenergyhub_caiso_threshold_sweep",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=BASE_DIR / "results" / "openenergyhub_caiso_threshold_sweep",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args.real_data_csv, args.out_root, args.result_root)
