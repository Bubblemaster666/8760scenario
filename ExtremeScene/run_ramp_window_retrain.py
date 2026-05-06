from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_generation import EXTREME_MAIN_METRICS, EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from risk_metrics import batch_hard_risk_metrics
from train_hierarchical_evt_diffusion import TrainConfig, train_model


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class RampRetrainSpec:
    experiment_name: str
    method: str
    ramp_metric_mode: str
    ramp_window_hours: float = 1.0
    multiscale_ramp_windows: str = "1.0,2.0,3.0"


SPECS = [
    RampRetrainSpec("RAMPDIAG0_E0_original_ramp", "E0", "one_step"),
    RampRetrainSpec("RAMPDIAG1_E0_ramp2h", "E0", "window_2h", ramp_window_hours=2.0),
    RampRetrainSpec("RAMPDIAG2_E0_ramp3h", "E0", "window_3h", ramp_window_hours=3.0),
    RampRetrainSpec("RAMPDIAG3_JRPD_ramp2h", "JRPD", "window_2h", ramp_window_hours=2.0),
    RampRetrainSpec("RAMPDIAG4_JRPD_ramp3h", "JRPD", "window_3h", ramp_window_hours=3.0),
    RampRetrainSpec("RAMPDIAG5_JRPD_multiscale", "JRPD", "multiscale", ramp_window_hours=1.0),
]


def _copy_dataset(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.suffix.lower() in {".npy", ".csv", ".json"}:
            shutil.copy2(path, dst / path.name)


def _assign_levels(values: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return pd.Series(
        np.select([arr <= thresholds["q50"], arr <= thresholds["q75"], arr <= thresholds["q90"]], [0, 1, 2], default=3),
        index=values.index,
        dtype="int64",
    )


def prepare_ramp_dataset(base_data_dir: Path, out_dir: Path, mode: str, ramp_window_hours: float = 1.0, multiscale_windows: str = "1.0,2.0,3.0") -> Path:
    """Create a dataset copy with netload_ramp_max/ramp_level recalculated."""

    data_dir = out_dir / f"dataset_{mode}"
    if data_dir.exists():
        return data_dir
    _copy_dataset(base_data_dir, data_dir)

    cond_train = pd.read_csv(data_dir / "cond_train.csv")
    metric_to_level = {
        "cum_deficit": "cum_level",
        "netload_ramp_max": "ramp_level",
        "imbalance_duration": "duration_level",
    }
    # First ensure cumulative/duration levels exist using train-only thresholds.
    thresholds: dict[str, dict[str, float]] = {}
    for metric, level_col in metric_to_level.items():
        if metric == "netload_ramp_max":
            continue
        values = pd.to_numeric(cond_train[metric], errors="coerce").fillna(0.0)
        thresholds[level_col] = {
            "q50": float(values.quantile(0.50)),
            "q75": float(values.quantile(0.75)),
            "q90": float(values.quantile(0.90)),
        }

    ramp_values_train = None
    for split in ["train", "val", "test"]:
        cond_path = data_dir / f"cond_{split}.csv"
        cond = pd.read_csv(cond_path)
        x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        delta_t = float(cond["delta_t_hours"].iloc[0]) if "delta_t_hours" in cond.columns else 1.0
        tau = pd.to_numeric(cond.get("imbalance_tau", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ramp = batch_hard_risk_metrics(
            x,
            tau=tau,
            delta_t_hours=delta_t,
            ramp_metric_mode=mode,
            ramp_window_hours=ramp_window_hours,
            multiscale_ramp_windows=multiscale_windows,
        )["netload_ramp_max"]
        cond["netload_ramp_max"] = ramp
        for metric, level_col in metric_to_level.items():
            if metric == "netload_ramp_max":
                continue
            cond[level_col] = _assign_levels(cond[metric], thresholds[level_col]).astype(int)
        cond.to_csv(cond_path, index=False, encoding="utf-8-sig")
        if split == "train":
            ramp_values_train = pd.Series(ramp)

    if ramp_values_train is None:
        raise RuntimeError("Failed to build train ramp values.")
    ramp_thresholds = {
        "q50": float(ramp_values_train.quantile(0.50)),
        "q75": float(ramp_values_train.quantile(0.75)),
        "q90": float(ramp_values_train.quantile(0.90)),
    }
    thresholds["ramp_level"] = ramp_thresholds
    for split in ["train", "val", "test"]:
        cond_path = data_dir / f"cond_{split}.csv"
        cond = pd.read_csv(cond_path)
        cond["ramp_level"] = _assign_levels(cond["netload_ramp_max"], ramp_thresholds).astype(int)
        cond["risk_profile_id"] = cond["cum_level"].astype(int) * 16 + cond["ramp_level"].astype(int) * 4 + cond["duration_level"].astype(int)
        cond.to_csv(cond_path, index=False, encoding="utf-8-sig")

    config = {
        "ramp_metric_mode": mode,
        "ramp_window_hours": float(ramp_window_hours),
        "multiscale_ramp_windows": multiscale_windows,
        "risk_profile_thresholds": thresholds,
    }
    (data_dir / "ramp_metric_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return data_dir


def _train_generate_eval(spec: RampRetrainSpec, data_dir: Path, out_dir: Path, device: str) -> dict:
    model_dir = out_dir / "models" / spec.experiment_name
    eval_dir = out_dir / "evaluations" / spec.experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    is_jrpd = spec.method == "JRPD"
    train_model(
        TrainConfig(
            data_dir=str(data_dir),
            out_dir=str(model_dir),
            ablation="full",
            seq_len=36,
            batch_size=32,
            lr=1e-4,
            weight_decay=1e-5,
            diffusion_steps=100,
            base_channels=64,
            guidance_scale=1.0,
            cond_dropout=0.10,
            ema_decay=0.995,
            stage1_epochs=12,
            stage2_epochs=10,
            stage3_epochs=2,
            lambda_tail=0.25,
            lambda_risk=0.04,
            lambda_cum=1.0,
            lambda_ramp=0.25,
            lambda_dur=0.35,
            lambda_recon=0.05,
            lambda_physics=0.02,
            lambda_resource=0.02,
            sampler_mode="risk_profile_balanced" if is_jrpd else "none",
            use_risk_profile_condition=is_jrpd,
            use_profile_loss=is_jrpd,
            lambda_profile=0.05 if is_jrpd else 0.0,
            ramp_metric_mode=spec.ramp_metric_mode,
            ramp_window_hours=spec.ramp_window_hours,
            multiscale_ramp_windows=spec.multiscale_ramp_windows,
            device=device,
            seed=42,
        )
    )
    generate_from_checkpoint(
        GenerationConfig(
            checkpoint=None,
            data_dir=str(data_dir),
            out_dir=str(model_dir),
            split="test",
            guidance_scale=1.0,
            checkpoint_type="best-risk",
        )
    )
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(model_dir / "generated_samples.npy"),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=spec.experiment_name,
            event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
            ramp_metric_mode=spec.ramp_metric_mode,
            ramp_window_hours=spec.ramp_window_hours,
            multiscale_ramp_windows=spec.multiscale_ramp_windows,
        )
    )
    row = asdict(spec)
    row.update(summary["metrics"])
    row["status"] = "ok"
    return row


def _recommend(row: pd.Series, t4: pd.Series | None) -> str:
    if row["experiment_name"] == "RAMPDIAG0_E0_original_ramp":
        return "baseline E0 original ramp"
    if t4 is None:
        return "diagnostic"
    q99_ok = float(row["q99_cum_deficit_error"]) <= 1.2 * float(t4["q99_cum_deficit_error"])
    degree_ok = float(row["extreme_degree_match_rate"]) >= float(t4["extreme_degree_match_rate"])
    core_ok = float(row["core_q99_cum_deficit_error"]) <= 1.2 * float(t4["core_q99_cum_deficit_error"])
    duration_ok = float(row["imbalance_duration_mae"]) <= 1.1 * float(t4["imbalance_duration_mae"])
    if q99_ok and degree_ok and core_ok and duration_ok:
        return "candidate: preserves JRPD tail/profile under windowed ramp"
    if not q99_ok:
        return "not recommended: q99 worsens"
    if not degree_ok:
        return "not recommended: extreme-degree match drops"
    return "mixed diagnostic"


def summarize(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # Reference T4 from previous screening uses original one-step ramp.
    t4_path = BASE_DIR / "outputs" / "three_method_trials" / "three_method_trials_summary.csv"
    t4 = None
    if t4_path.exists():
        ref = pd.read_csv(t4_path)
        sub = ref[ref["experiment_name"] == "T4_JRPD_sampler_loss"]
        if not sub.empty:
            t4 = sub.iloc[0]
    for metric in EXTREME_MAIN_METRICS:
        if t4 is not None:
            df[f"delta_{metric}_vs_T4"] = pd.to_numeric(df[metric], errors="coerce") - float(t4[metric])
        else:
            df[f"delta_{metric}_vs_T4"] = np.nan
    df["recommendation_reason"] = df.apply(lambda row: _recommend(row, t4), axis=1)
    cols = [
        "experiment_name",
        "method",
        "ramp_metric_mode",
        "ramp_window_hours",
        "multiscale_ramp_windows",
        *EXTREME_MAIN_METRICS,
        *[f"delta_{metric}_vs_T4" for metric in EXTREME_MAIN_METRICS],
        "recommendation_reason",
        "status",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols]
    df.to_csv(out_dir / "ramp_window_retrain_summary.csv", index=False, encoding="utf-8-sig")
    candidates = df[df["recommendation_reason"].str.startswith("candidate", na=False)]
    best = candidates.iloc[0] if not candidates.empty else df.iloc[0]
    lines = [
        "# Ramp Window Retrain Report",
        "",
        "本阶段使用新的 ramp_metric_mode 重算训练标签、JRPD ramp_level/profile、risk loss 和评价 ramp 指标。",
        "",
        f"## Recommended Config: {best['experiment_name']}",
        f"- ramp_metric_mode: {best['ramp_metric_mode']}",
        f"- ramp_window_hours: {best['ramp_window_hours']}",
        f"- reason: {best['recommendation_reason']}",
        "",
        "## Required Answers",
        f"- 是否建议用 2h/3h 替代 one-step ramp: {bool(not candidates.empty)}",
        f"- 最推荐 ramp_window_hours: {best['ramp_window_hours']}",
        "- 是否需要进入下一步联合失衡风险定义重构: " + ("no, windowed ramp has a candidate" if not candidates.empty else "yes, windowed ramp retraining did not preserve JRPD tradeoff"),
    ]
    (out_dir / "ramp_window_retrain_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return df


def run_retrain(data_dir: str, out_dir: str, device: str = "cpu") -> pd.DataFrame:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_cache = root / "datasets"
    rows = []
    for spec in SPECS:
        mode_key = spec.ramp_metric_mode
        dataset = prepare_ramp_dataset(Path(data_dir), data_cache, mode_key, spec.ramp_window_hours, spec.multiscale_ramp_windows)
        rows.append(_train_generate_eval(spec, dataset, root, device))
    return summarize(rows, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain E0/JRPD with windowed net-load ramp metrics.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "ramp_window_retrain"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_retrain(args.data_dir, args.out_dir, device=args.device)
