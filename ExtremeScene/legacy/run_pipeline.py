from __future__ import annotations
from pathlib import Path
import pandas as pd

# 1) 极端样本截取
from Extreme_Extract import (
    DetectConfig,
    detect_extreme_samples,
    make_mock_data,
)

# 2) 指标计算
from sample_metrics import (
    MetricConfig,
    compute_metrics_for_samples,
)

# 3) EVT 分级
from evt_fit import (
    EVTConfig,
    fit_evt_and_label,
)

BASE_DIR = Path(__file__).resolve().parent


def main():
    # =========================
    # Step 0: 准备数据
    # =========================
    # 当前先用模拟数据跑通流程
    df = make_mock_data()

    print("=== 原始数据预览 ===")
    print(df.head())
    print(f"\n原始数据条数: {len(df)}")

    # =========================
    # Step 1: 极端样本截取
    # =========================
    detect_cfg = DetectConfig(
        time_col="time",
        temp_col="temp",
        wind_speed_col="wind_speed",
        irradiance_col="irradiance",
        snowfall_col="snowfall",
        visibility_col="visibility",
        freq_hours=1,
        min_event_hours=6,
        low_resource_min_hours=6,
        daylight_irradiance_min=50.0,
    )

    samples = detect_extreme_samples(
        df=df,
        cfg=detect_cfg,
        add_buffer_hours=2,
        merge_overlap=False,
    )

    print("\n=== 极端样本窗口 ===")
    print(samples)
    samples.to_csv(BASE_DIR / "extreme_samples.csv", index=False, encoding="utf-8-sig")
    print("\n已保存: extreme_samples.csv")

    if samples.empty:
        print("\n未检测到极端样本，流程结束。")
        return

    # =========================
    # Step 2: 指标计算
    # =========================
    metric_cfg = MetricConfig(
        time_col="time",
        load_col="load",
        wind_power_col="wind_power",
        solar_power_col="solar_power",
        imbalance_tau_mode="quantile",
        imbalance_tau_quantile=0.75,
    )

    samples_with_metrics = compute_metrics_for_samples(
        df=df,
        samples=samples,
        cfg=metric_cfg,
    )

    print("\n=== 样本指标结果 ===")
    print(samples_with_metrics)
    samples_with_metrics.to_csv(
        BASE_DIR / "samples_with_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print("\n已保存: samples_with_metrics.csv")

    # =========================
    # Step 3: EVT 分级
    # =========================
    evt_cfg = EVTConfig(
        metric_col="cum_deficit",
        threshold_quantile=0.90,
        severe_prob=0.01,
        moderate_prob=0.05,
        mild_prob=0.10,
        min_exceedances=3,
    )

    samples_evt, evt_info = fit_evt_and_label(
        samples=samples_with_metrics,
        cfg=evt_cfg,
    )

    print("\n=== EVT 参数信息 ===")
    print(evt_info)

    print("\n=== EVT 分级结果 ===")
    print(samples_evt)
    samples_evt.to_csv(
        BASE_DIR / "samples_evt_labeled.csv", index=False, encoding="utf-8-sig"
    )
    print("\n已保存: samples_evt_labeled.csv")

    # =========================
    # Step 4: 输出一个精简结果表
    # =========================
    final_cols = [
        "sample_id",
        "event_type",
        "start_time",
        "end_time",
        "month",
        "low_irradiance_flag",
        "low_wind_flag",
        "imbalance_tau",
        "cum_deficit",
        "netload_ramp_max",
        "imbalance_duration",
        "extreme_prob",
        "severity_level",
    ]

    final_cols = [c for c in final_cols if c in samples_evt.columns]
    final_result = samples_evt[final_cols].copy()

    print("\n=== 最终结果表 ===")
    print(final_result)
    final_result.to_csv(
        BASE_DIR / "final_extreme_samples.csv", index=False, encoding="utf-8-sig"
    )
    print("\n已保存: final_extreme_samples.csv")


if __name__ == "__main__":
    main()
