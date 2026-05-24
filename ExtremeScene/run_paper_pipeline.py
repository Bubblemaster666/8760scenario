from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Extreme_Extract import DetectConfig
from annual_embedding import AnnualEmbeddingConfig, run_annual_embedding, _mock_background
from data_interface import ColumnMapping, DatasetBuildConfig, build_dataset_artifacts, build_event_samples
from diagnose_extreme_samples import run_diagnostics
from evt_fit import EVTConfig
from risk_screening import RiskScreenConfig
from run_experiments import ExperimentConfig, run_experiments
from sample_metrics import MetricConfig

try:
    from build_mock_dataset import make_random_mock_data
except ModuleNotFoundError:
    from legacy.build_mock_dataset import make_random_mock_data


@dataclass
class PaperPipelineConfig:
    out_dir: str
    use_mock: bool = True
    real_data_csv: tuple[str, ...] | None = None
    seq_len: int = 36
    n_scenarios: int = 12
    days_per_scenario: int = 18
    gap_days: int = 3
    run_annual_embedding: bool = False
    stage1_epochs: int | None = None
    stage2_epochs: int | None = None
    stage3_epochs: int | None = None
    batch_size: int = 32
    diffusion_steps: int = 100
    base_channels: int = 64
    plain_epochs: int | None = None
    gan_epochs: int | None = None
    seed: int = 42

    replace_snow_with_heavy_rain: bool = True
    heavy_rain_quantile: float = 0.98
    heavy_rain_rolling_quantile: float = 0.97
    rain_rolling_window_hours: int = 3
    rain_min_event_hours: int = 2
    rain_min_total_precip: float | None = None
    rain_require_power_impact: bool = True
    low_irr_quantile: float = 0.35
    low_wind_quantile: float = 0.30
    low_resource_min_hours: int = 3
    daylight_irradiance_min: float = 30.0
    min_event_hours: int = 6
    use_adaptive_thresholds: bool = True
    cold_temp_quantile: float = 0.10
    cold_drop_24h_quantile: float = 0.95
    adaptive_cold_drop_min: float = 4.5
    heat_temp_quantile: float = 0.98
    high_wind_speed_quantile: float = 0.99
    snowfall_quantile: float = 0.97

    risk_screen_enabled: bool = True
    risk_screen_mode: str = "medium"
    min_cum_deficit: float = 0.0
    min_imbalance_duration: float = 1.0
    ramp_quantile: float = 0.70
    min_samples_after_screen: int = 50

    imbalance_tau_mode: str = "monthly_quantile"
    imbalance_tau_quantile: float = 0.75
    imbalance_tau_fixed: float = 0.0

    severity_mode: str = "hybrid"
    severity_q1: float = 0.60
    severity_q2: float = 0.80
    severity_q3: float = 0.92

    drop_rare_event_types: bool = False
    min_event_type_count: int = 5
    use_augmented_train: bool = False
    checkpoint_type: str = "best-risk"
    skip_model_experiments: bool = False


def _build_mock_timeseries(cfg: PaperPipelineConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    all_df = []
    for scenario_id in range(cfg.n_scenarios):
        seed = int(rng.integers(0, 1_000_000_000))
        start = pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(days=scenario_id * (cfg.days_per_scenario + cfg.gap_days))
        df_one = make_random_mock_data(seed=seed, start_time=start, days=cfg.days_per_scenario)
        df_one["scenario_id"] = scenario_id
        df_one["scenario_seed"] = seed
        all_df.append(df_one)
    return pd.concat(all_df, ignore_index=True).sort_values("time").reset_index(drop=True)


def _load_real_timeseries(cfg: PaperPipelineConfig) -> pd.DataFrame:
    if not cfg.real_data_csv:
        raise ValueError("real_data_csv is required when use_mock is False.")
    csv_paths = [cfg.real_data_csv] if isinstance(cfg.real_data_csv, str) else list(cfg.real_data_csv)
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df_one = pd.read_csv(csv_path)
        if "time" not in df_one.columns:
            raise ValueError(f"The real-data CSV must contain a 'time' column: {csv_path}")
        df_one["time"] = pd.to_datetime(df_one["time"])
        df_one["source_file"] = Path(csv_path).name
        frames.append(df_one)
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)


def _build_detect_config(cfg: PaperPipelineConfig) -> DetectConfig:
    common = dict(
        replace_snow_with_heavy_rain=cfg.replace_snow_with_heavy_rain,
        heavy_rain_quantile=cfg.heavy_rain_quantile,
        heavy_rain_rolling_quantile=cfg.heavy_rain_rolling_quantile,
        rain_rolling_window_hours=cfg.rain_rolling_window_hours,
        rain_min_event_hours=cfg.rain_min_event_hours,
        rain_min_total_precip=cfg.rain_min_total_precip,
        rain_require_power_impact=cfg.rain_require_power_impact,
        low_irr_quantile=cfg.low_irr_quantile,
        low_wind_quantile=cfg.low_wind_quantile,
        low_resource_min_hours=cfg.low_resource_min_hours,
        daylight_irradiance_min=cfg.daylight_irradiance_min,
        min_event_hours=cfg.min_event_hours,
        use_adaptive_thresholds=cfg.use_adaptive_thresholds,
        cold_temp_quantile=cfg.cold_temp_quantile,
        cold_drop_24h_quantile=cfg.cold_drop_24h_quantile,
        adaptive_cold_drop_min=cfg.adaptive_cold_drop_min,
        heat_temp_quantile=cfg.heat_temp_quantile,
        high_wind_speed_quantile=cfg.high_wind_speed_quantile,
        snowfall_quantile=cfg.snowfall_quantile,
    )
    return DetectConfig(**common)


def run_pipeline(cfg: PaperPipelineConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    dataset_dir = out_dir / "dataset"
    annual_dir = out_dir / "annual"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    annual_dir.mkdir(parents=True, exist_ok=True)
    _mock_background().to_csv(annual_dir / "regular_background.csv", index=False, encoding="utf-8-sig")

    timeseries_df = _build_mock_timeseries(cfg) if cfg.use_mock else _load_real_timeseries(cfg)
    timeseries_df.to_csv(dataset_dir / "timeseries_input.csv", index=False, encoding="utf-8-sig")

    column_mapping = ColumnMapping()
    detect_cfg = _build_detect_config(cfg)
    metric_cfg = MetricConfig(
        imbalance_tau_mode=cfg.imbalance_tau_mode,
        imbalance_tau_quantile=cfg.imbalance_tau_quantile,
        imbalance_tau_fixed=cfg.imbalance_tau_fixed,
    )
    risk_screen_cfg = RiskScreenConfig(
        enabled=cfg.risk_screen_enabled,
        mode=cfg.risk_screen_mode,
        min_cum_deficit=cfg.min_cum_deficit,
        min_imbalance_duration=cfg.min_imbalance_duration,
        ramp_quantile=cfg.ramp_quantile,
        min_samples_after_screen=cfg.min_samples_after_screen,
        rain_require_power_impact=cfg.rain_require_power_impact,
    )
    evt_cfg = EVTConfig(
        metric_col="cum_deficit",
        threshold_quantile=0.90,
        severity_mode=cfg.severity_mode,
        severity_q1=cfg.severity_q1,
        severity_q2=cfg.severity_q2,
        severity_q3=cfg.severity_q3,
    )

    samples_labeled, evt_info = build_event_samples(
        timeseries_df,
        column_mapping=column_mapping,
        detect_cfg=detect_cfg,
        metric_cfg=metric_cfg,
        evt_cfg=evt_cfg,
        risk_screen_cfg=risk_screen_cfg,
        intermediate_output_dir=dataset_dir,
        add_buffer_hours=2,
        merge_overlap=False,
    )

    rare_event_summary = {"enabled": False, "min_event_type_count": int(cfg.min_event_type_count)}
    if cfg.drop_rare_event_types and not samples_labeled.empty:
        counts = samples_labeled["event_type"].value_counts()
        keep_types = counts[counts >= int(cfg.min_event_type_count)].index
        before_count = int(len(samples_labeled))
        samples_labeled = samples_labeled[samples_labeled["event_type"].isin(keep_types)].reset_index(drop=True)
        samples_labeled["sample_id"] = [f"S{i:04d}" for i in range(1, len(samples_labeled) + 1)]
        rare_event_summary = {
            "enabled": True,
            "min_event_type_count": int(cfg.min_event_type_count),
            "before_count": before_count,
            "after_count": int(len(samples_labeled)),
            "kept_event_types": [str(x) for x in keep_types],
            "dropped_event_type_counts": {str(k): int(v) for k, v in counts[counts < int(cfg.min_event_type_count)].to_dict().items()},
        }
    samples_labeled.to_csv(dataset_dir / "samples_evt_labeled.csv", index=False, encoding="utf-8-sig")

    dataset_result = build_dataset_artifacts(
        df=timeseries_df,
        labeled_samples=samples_labeled,
        output_dir=dataset_dir,
        build_cfg=DatasetBuildConfig(
            seq_len=cfg.seq_len,
            output_dir=str(dataset_dir),
            split_group_col="scenario_id" if "scenario_id" in samples_labeled.columns else "sample_id",
        ),
        column_mapping=column_mapping,
        evt_info=evt_info,
        extra_summary={
            "source": "mock" if cfg.use_mock else "real",
            "detect_cfg": asdict(detect_cfg),
            "metric_cfg": asdict(metric_cfg),
            "risk_screen_cfg": asdict(risk_screen_cfg),
            "rare_event_filter": rare_event_summary,
        },
    )
    diagnostic_summary = run_diagnostics(dataset_dir / "samples_evt_labeled.csv", out_dir / "diagnostics")

    stage1_epochs = cfg.stage1_epochs if cfg.stage1_epochs is not None else 12
    stage2_epochs = cfg.stage2_epochs if cfg.stage2_epochs is not None else 10
    stage3_epochs = cfg.stage3_epochs if cfg.stage3_epochs is not None else 2
    plain_epochs = cfg.plain_epochs if cfg.plain_epochs is not None else (12 if cfg.use_mock else 40)
    gan_epochs = cfg.gan_epochs if cfg.gan_epochs is not None else (12 if cfg.use_mock else 40)

    if cfg.skip_model_experiments:
        all_metrics = pd.DataFrame()
    else:
        experiment_cfg = ExperimentConfig(
            data_dir=str(dataset_dir),
            out_dir=str(out_dir),
            methods=[
                "traditional_gaussian_copula",
                "plain_diffusion_baseline",
                "enhanced_gan",
                "proposed",
                "no_evt_strict",
                "no_evt",
                "no_risk_loss",
                "no_month",
                "flat_condition",
            ],
            seq_len=cfg.seq_len,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            stage3_epochs=stage3_epochs,
            batch_size=cfg.batch_size,
            diffusion_steps=cfg.diffusion_steps,
            base_channels=cfg.base_channels,
            plain_epochs=plain_epochs,
            gan_epochs=gan_epochs,
            device="cpu",
            guidance_scale=1.0,
            checkpoint_type=cfg.checkpoint_type,
            seed=cfg.seed,
        )
        all_metrics = run_experiments(experiment_cfg)

    annual_summary = None
    if cfg.run_annual_embedding:
        proposed_segments = out_dir / "models" / "proposed" / "generated_samples.npy"
        proposed_cond = dataset_dir / "cond_test.csv"
        annual_summary = run_annual_embedding(
            AnnualEmbeddingConfig(
                background=None,
                segments=str(proposed_segments),
                cond=str(proposed_cond),
                out_dir=str(out_dir / "annual_embedding"),
                use_mock_background=True,
            )
        )

    summary = {
        "config": asdict(cfg),
        "real_data_files": list(cfg.real_data_csv or ()),
        "dataset_summary": dataset_result["summary"],
        "diagnostic_summary": diagnostic_summary,
        "all_model_metrics_path": str(out_dir / "evaluations" / "all_model_metrics.csv"),
        "annual_embedding_ran": bool(cfg.run_annual_embedding),
        "annual_embedding_summary": annual_summary,
        "model_names": all_metrics["model_name"].tolist() if "model_name" in all_metrics.columns else [],
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> PaperPipelineConfig:
    parser = argparse.ArgumentParser(description="Run the paper-level extreme-scenario generation pipeline.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--real-data-csv", type=str, action="append", default=None)
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--n-scenarios", type=int, default=12)
    parser.add_argument("--days-per-scenario", type=int, default=18)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--run-annual-embedding", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage3-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--plain-epochs", type=int, default=None)
    parser.add_argument("--gan-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--replace-snow-with-heavy-rain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heavy-rain-quantile", type=float, default=0.98)
    parser.add_argument("--heavy-rain-rolling-quantile", type=float, default=0.97)
    parser.add_argument("--rain-rolling-window-hours", type=int, default=3)
    parser.add_argument("--rain-min-event-hours", type=int, default=2)
    parser.add_argument("--rain-min-total-precip", type=float, default=None)
    parser.add_argument("--rain-require-power-impact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-irr-quantile", type=float, default=0.35)
    parser.add_argument("--low-wind-quantile", type=float, default=0.30)
    parser.add_argument("--low-resource-min-hours", type=int, default=3)
    parser.add_argument("--daylight-irradiance-min", type=float, default=30.0)
    parser.add_argument("--min-event-hours", type=int, default=6)
    parser.add_argument("--use-adaptive-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cold-temp-quantile", type=float, default=0.10)
    parser.add_argument("--cold-drop-24h-quantile", type=float, default=0.95)
    parser.add_argument("--adaptive-cold-drop-min", type=float, default=4.5)
    parser.add_argument("--heat-temp-quantile", type=float, default=0.98)
    parser.add_argument("--high-wind-speed-quantile", type=float, default=0.99)
    parser.add_argument("--snowfall-quantile", type=float, default=0.97)

    parser.add_argument("--risk-screen-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--risk-screen-mode", type=str, default="medium", choices=["loose", "medium", "strict", "hybrid"])
    parser.add_argument("--min-cum-deficit", type=float, default=0.0)
    parser.add_argument("--min-imbalance-duration", type=float, default=1.0)
    parser.add_argument("--ramp-quantile", type=float, default=0.70)
    parser.add_argument("--min-samples-after-screen", type=int, default=50)

    parser.add_argument("--imbalance-tau-mode", type=str, default="monthly_quantile", choices=["global_quantile", "monthly_quantile", "seasonal_quantile", "fixed", "quantile"])
    parser.add_argument("--imbalance-tau-quantile", type=float, default=0.75)
    parser.add_argument("--imbalance-tau-fixed", type=float, default=0.0)

    parser.add_argument("--severity-mode", type=str, default="hybrid", choices=["evt_prob", "quantile", "hybrid"])
    parser.add_argument("--severity-q1", type=float, default=0.60)
    parser.add_argument("--severity-q2", type=float, default=0.80)
    parser.add_argument("--severity-q3", type=float, default=0.92)

    parser.add_argument("--drop-rare-event-types", action="store_true")
    parser.add_argument("--min-event-type-count", type=int, default=5)
    parser.add_argument("--use-augmented-train", action="store_true")
    parser.add_argument("--checkpoint-type", type=str, default="best-risk", choices=["best", "best-risk", "final"])
    parser.add_argument("--skip-model-experiments", action="store_true")
    args = parser.parse_args()
    return PaperPipelineConfig(
        out_dir=args.out_dir,
        use_mock=bool(args.use_mock) or not args.real_data_csv,
        real_data_csv=tuple(args.real_data_csv) if args.real_data_csv else None,
        seq_len=args.seq_len,
        n_scenarios=args.n_scenarios,
        days_per_scenario=args.days_per_scenario,
        gap_days=args.gap_days,
        run_annual_embedding=args.run_annual_embedding,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        plain_epochs=args.plain_epochs,
        gan_epochs=args.gan_epochs,
        seed=args.seed,
        replace_snow_with_heavy_rain=args.replace_snow_with_heavy_rain,
        heavy_rain_quantile=args.heavy_rain_quantile,
        heavy_rain_rolling_quantile=args.heavy_rain_rolling_quantile,
        rain_rolling_window_hours=args.rain_rolling_window_hours,
        rain_min_event_hours=args.rain_min_event_hours,
        rain_min_total_precip=args.rain_min_total_precip,
        rain_require_power_impact=args.rain_require_power_impact,
        low_irr_quantile=args.low_irr_quantile,
        low_wind_quantile=args.low_wind_quantile,
        low_resource_min_hours=args.low_resource_min_hours,
        daylight_irradiance_min=args.daylight_irradiance_min,
        min_event_hours=args.min_event_hours,
        use_adaptive_thresholds=args.use_adaptive_thresholds,
        cold_temp_quantile=args.cold_temp_quantile,
        cold_drop_24h_quantile=args.cold_drop_24h_quantile,
        adaptive_cold_drop_min=args.adaptive_cold_drop_min,
        heat_temp_quantile=args.heat_temp_quantile,
        high_wind_speed_quantile=args.high_wind_speed_quantile,
        snowfall_quantile=args.snowfall_quantile,
        risk_screen_enabled=args.risk_screen_enabled,
        risk_screen_mode=args.risk_screen_mode,
        min_cum_deficit=args.min_cum_deficit,
        min_imbalance_duration=args.min_imbalance_duration,
        ramp_quantile=args.ramp_quantile,
        min_samples_after_screen=args.min_samples_after_screen,
        imbalance_tau_mode=args.imbalance_tau_mode,
        imbalance_tau_quantile=args.imbalance_tau_quantile,
        imbalance_tau_fixed=args.imbalance_tau_fixed,
        severity_mode=args.severity_mode,
        severity_q1=args.severity_q1,
        severity_q2=args.severity_q2,
        severity_q3=args.severity_q3,
        drop_rare_event_types=args.drop_rare_event_types,
        min_event_type_count=args.min_event_type_count,
        use_augmented_train=args.use_augmented_train,
        checkpoint_type=args.checkpoint_type,
        skip_model_experiments=args.skip_model_experiments,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
