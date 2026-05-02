from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Extreme_Extract import DetectConfig
from annual_embedding import AnnualEmbeddingConfig, run_annual_embedding, _mock_background
from data_interface import (
    ColumnMapping,
    DatasetBuildConfig,
    build_dataset_artifacts,
    build_event_samples,
)
from evt_fit import EVTConfig
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
    seq_len: int = 24
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

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    return df


def _build_detect_config(cfg: PaperPipelineConfig) -> DetectConfig | None:
    if cfg.use_mock:
        return None
    # Real singleton data has much milder temperature and wind ranges than the
    # original mock assumptions, so we switch to quantile-adaptive thresholds.
    return DetectConfig(
        use_adaptive_thresholds=True,
        min_event_hours=6,
        low_resource_min_hours=6,
        cold_temp_quantile=0.10,
        cold_drop_24h_quantile=0.95,
        adaptive_cold_drop_min=4.5,
        heat_temp_quantile=0.98,
        high_wind_speed_quantile=0.99,
        snowfall_quantile=0.97,
    )


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
    samples_labeled, evt_info = build_event_samples(
        timeseries_df,
        column_mapping=column_mapping,
        detect_cfg=detect_cfg,
        metric_cfg=MetricConfig(),
        evt_cfg=EVTConfig(metric_col="cum_deficit", threshold_quantile=0.90),
        add_buffer_hours=2,
        merge_overlap=False,
    )
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
            "detect_cfg": asdict(detect_cfg) if detect_cfg is not None else "default_mock_thresholds",
        },
    )

    stage1_epochs = cfg.stage1_epochs if cfg.stage1_epochs is not None else 12
    stage2_epochs = cfg.stage2_epochs if cfg.stage2_epochs is not None else 10
    stage3_epochs = cfg.stage3_epochs if cfg.stage3_epochs is not None else 2
    plain_epochs = cfg.plain_epochs if cfg.plain_epochs is not None else (12 if cfg.use_mock else 40)
    gan_epochs = cfg.gan_epochs if cfg.gan_epochs is not None else (12 if cfg.use_mock else 40)

    experiment_cfg = ExperimentConfig(
        data_dir=str(dataset_dir),
        out_dir=str(out_dir),
        methods=[
            "traditional_gaussian_copula",
            "plain_diffusion_baseline",
            "enhanced_gan",
            "proposed",
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
        "all_model_metrics_path": str(out_dir / "evaluations" / "all_model_metrics.csv"),
        "annual_embedding_ran": bool(cfg.run_annual_embedding),
        "annual_embedding_summary": annual_summary,
        "model_names": all_metrics["model_name"].tolist(),
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> PaperPipelineConfig:
    parser = argparse.ArgumentParser(description="Run the paper-level extreme-scenario generation pipeline.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--real-data-csv", type=str, action="append", default=None)
    parser.add_argument("--seq-len", type=int, default=24)
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
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
