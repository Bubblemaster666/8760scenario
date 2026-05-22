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
from risk_metrics import batch_hard_risk_metrics, batch_joint_risk_profile
from train_hierarchical_evt_diffusion import TrainConfig, train_model


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class JointProfileSpec:
    experiment_name: str
    main_technique: str
    use_joint_profile_condition: bool
    use_joint_profile_loss: bool
    lambda_joint_profile_stage2: float
    lambda_joint_profile_stage3: float
    lambda_core_share_stage3: float
    lambda_tail_dist: float = 0.0
    tail_dist_topk_ratio: float = 0.10


SPECS = [
    JointProfileSpec("JP0_JRPD_3h_baseline", "JRPD + 3h ramp baseline", False, False, 0.0, 0.0, 0.0),
    JointProfileSpec("JP1_profile_condition", "JRPD + joint risk profile condition", True, False, 0.0, 0.0, 0.0),
    JointProfileSpec("JP2_profile_stage2_loss", "profile condition + Stage2 profile loss", True, True, 0.03, 0.0, 0.0),
    JointProfileSpec("JP3_profile_stage2_stage3_core", "profile condition + Stage2/Stage3 profile/core loss", True, True, 0.03, 0.03, 0.02),
    JointProfileSpec("JP4_profile_q99_protect_003", "profile condition + q99 cumulative tail protection", True, True, 0.03, 0.03, 0.02, 0.03, 0.10),
    JointProfileSpec("JP5_profile_q99_protect_006", "profile condition + stronger q99 cumulative tail protection", True, True, 0.03, 0.03, 0.02, 0.06, 0.10),
]


def _copy_dataset(src: Path, dst: Path, force: bool = False) -> None:
    if dst.exists() and force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.suffix.lower() in {".npy", ".csv", ".json"}:
            shutil.copy2(path, dst / path.name)


def _assign_levels(values: pd.Series | np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.select([arr <= thresholds["q50"], arr <= thresholds["q75"], arr <= thresholds["q90"]], [0, 1, 2], default=3).astype(int)


def _thresholds(values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    return {"q50": float(arr.quantile(0.50)), "q75": float(arr.quantile(0.75)), "q90": float(arr.quantile(0.90))}


def _resource_thresholds(x_train: np.ndarray) -> dict[str, float]:
    solar = x_train[:, 2, :].reshape(-1)
    solar_day = solar[solar > 1e-6]
    return {
        "load_high": float(np.quantile(x_train[:, 0, :], 0.75)),
        "wind_low": float(np.quantile(x_train[:, 1, :], 0.25)),
        "solar_low": float(np.quantile(solar_day if solar_day.size else solar, 0.25)),
    }


def prepare_joint_profile_dataset(base_data_dir: Path, out_dir: Path, force: bool = False) -> Path:
    """Create a dataset copy with G(t) = [D(t), R3h+(t), C(t), event_mask(t)]."""

    data_dir = out_dir / "dataset_joint_profile"
    if data_dir.exists() and not force:
        return data_dir
    _copy_dataset(base_data_dir, data_dir, force=force)

    x_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    train_thresholds = _resource_thresholds(x_train)
    raw_profiles: dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        cond = pd.read_csv(data_dir / f"cond_{split}.csv")
        mask_path = data_dir / f"event_mask_{split}.npy"
        event_mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else np.ones((len(x), x.shape[2]), dtype=np.float32)
        tau = pd.to_numeric(cond.get("imbalance_tau", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        delta_t = float(cond["delta_t_hours"].iloc[0]) if "delta_t_hours" in cond.columns else 1.0
        raw_profile = batch_joint_risk_profile(
            x,
            tau=tau,
            event_mask=event_mask,
            delta_t_hours=delta_t,
            ramp_window_hours=3.0,
            resource_thresholds=train_thresholds,
        )
        raw_profiles[split] = raw_profile
        np.save(data_dir / f"risk_profile_raw_{split}.npy", raw_profile.astype(np.float32))

        ramp_3h = batch_hard_risk_metrics(
            x,
            tau=tau,
            delta_t_hours=delta_t,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )["netload_ramp_max"]
        cond["netload_ramp_3h"] = ramp_3h
        cond["netload_ramp_max"] = ramp_3h
        cond.to_csv(data_dir / f"cond_{split}.csv", index=False, encoding="utf-8-sig")

    train_raw = raw_profiles["train"]
    profile_norm = {
        "profile_d_mean": float(train_raw[:, 0, :].mean()),
        "profile_d_std": float(train_raw[:, 0, :].std() + 1e-6),
        "profile_r_mean": float(train_raw[:, 1, :].mean()),
        "profile_r_std": float(train_raw[:, 1, :].std() + 1e-6),
        "profile_c_mean": float(train_raw[:, 2, :].mean()),
        "profile_c_std": float(train_raw[:, 2, :].std() + 1e-6),
    }

    for split, raw_profile in raw_profiles.items():
        profile = raw_profile.copy()
        profile[:, 0, :] = (profile[:, 0, :] - profile_norm["profile_d_mean"]) / profile_norm["profile_d_std"]
        profile[:, 1, :] = (profile[:, 1, :] - profile_norm["profile_r_mean"]) / profile_norm["profile_r_std"]
        profile[:, 2, :] = (profile[:, 2, :] - profile_norm["profile_c_mean"]) / profile_norm["profile_c_std"]
        profile[:, 3, :] = raw_profile[:, 3, :]
        np.save(data_dir / f"risk_profile_{split}.npy", profile.astype(np.float32))

    cond_train = pd.read_csv(data_dir / "cond_train.csv")
    thresholds = {
        "cum_level": _thresholds(cond_train["cum_deficit"]),
        "ramp_level": _thresholds(cond_train["netload_ramp_max"]),
        "duration_level": _thresholds(cond_train["imbalance_duration"]),
    }
    for split in ["train", "val", "test"]:
        cond_path = data_dir / f"cond_{split}.csv"
        cond = pd.read_csv(cond_path)
        cond["cum_level"] = _assign_levels(cond["cum_deficit"], thresholds["cum_level"])
        cond["ramp_level"] = _assign_levels(cond["netload_ramp_max"], thresholds["ramp_level"])
        cond["duration_level"] = _assign_levels(cond["imbalance_duration"], thresholds["duration_level"])
        cond["risk_profile_id"] = cond["cum_level"].astype(int) * 16 + cond["ramp_level"].astype(int) * 4 + cond["duration_level"].astype(int)
        cond.to_csv(cond_path, index=False, encoding="utf-8-sig")

    payload = {
        "definition": "G(t) = [D(t), R3h_plus(t), C(t), event_mask(t)]",
        "resource_thresholds": train_thresholds,
        "profile_normalization": profile_norm,
        "risk_profile_level_thresholds": thresholds,
        "ramp_window_hours": 3.0,
        "note": "Thresholds and normalizers are fitted on train split only.",
    }
    (data_dir / "joint_risk_profile_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return data_dir


def _profile_diagnostics(data_dir: Path, model_dir: Path, model_name: str) -> dict:
    real = np.load(data_dir / "X_test.npy").astype(np.float32)
    gen = np.load(model_dir / "generated_samples.npy").astype(np.float32)
    cond = pd.read_csv(data_dir / "cond_test.csv")
    config = json.loads((data_dir / "joint_risk_profile_config.json").read_text(encoding="utf-8-sig"))
    event_mask_path = data_dir / "event_mask_test.npy"
    event_mask = np.load(event_mask_path).astype(np.float32) if event_mask_path.exists() else np.ones((len(real), real.shape[2]), dtype=np.float32)
    tau = pd.to_numeric(cond.get("imbalance_tau", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta_t = float(cond["delta_t_hours"].iloc[0]) if "delta_t_hours" in cond.columns else 1.0
    real_profile = batch_joint_risk_profile(real, tau, event_mask, delta_t, 3.0, config["resource_thresholds"])
    gen_profile = batch_joint_risk_profile(gen, tau, event_mask, delta_t, 3.0, config["resource_thresholds"])
    depth_real = real_profile[:, 0, :]
    depth_gen = gen_profile[:, 0, :]
    p_core_real = (depth_real * event_mask).sum(axis=1) / (depth_real.sum(axis=1) + 1e-6)
    p_core_gen = (depth_gen * event_mask).sum(axis=1) / (depth_gen.sum(axis=1) + 1e-6)
    return {
        "model_name": model_name,
        "profile_depth_mae": float(np.mean(np.abs(real_profile[:, 0, :] - gen_profile[:, 0, :]))),
        "profile_ramp3h_mae": float(np.mean(np.abs(real_profile[:, 1, :] - gen_profile[:, 1, :]))),
        "profile_sync_mae": float(np.mean(np.abs(real_profile[:, 2, :] - gen_profile[:, 2, :]))),
        "profile_total_mae": float(np.mean(np.abs(real_profile[:, :3, :] - gen_profile[:, :3, :]))),
        "p_core_mae": float(np.mean(np.abs(p_core_real - p_core_gen))),
    }


def _train_generate_eval(spec: JointProfileSpec, data_dir: Path, out_dir: Path, device: str, quick: bool) -> dict:
    model_dir = out_dir / "models" / spec.experiment_name
    eval_dir = out_dir / "evaluations" / spec.experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    stage1, stage2, stage3 = (4, 3, 1) if quick else (12, 10, 2)
    diffusion_steps = 50 if quick else 100
    base_channels = 32 if quick else 64
    metrics_path = eval_dir / "metrics_summary.csv"
    generated_path = model_dir / "generated_samples.npy"
    if metrics_path.exists() and generated_path.exists():
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        row = asdict(spec)
        row.update({"stage1_epochs": stage1, "stage2_epochs": stage2, "stage3_epochs": stage3, "diffusion_steps": diffusion_steps, "base_channels": base_channels})
        row.update(metrics)
        row.update(_profile_diagnostics(data_dir, model_dir, spec.experiment_name))
        return row

    train_model(
        TrainConfig(
            data_dir=str(data_dir),
            out_dir=str(model_dir),
            ablation="full",
            seq_len=36,
            batch_size=32,
            lr=1e-4,
            weight_decay=1e-5,
            diffusion_steps=diffusion_steps,
            base_channels=base_channels,
            guidance_scale=1.0,
            cond_dropout=0.10,
            ema_decay=0.995,
            stage1_epochs=stage1,
            stage2_epochs=stage2,
            stage3_epochs=stage3,
            lambda_tail=0.25,
            lambda_tail_dist=spec.lambda_tail_dist,
            tail_dist_topk_ratio=spec.tail_dist_topk_ratio,
            lambda_risk=0.04,
            lambda_cum=1.0,
            lambda_ramp=0.25,
            lambda_dur=0.35,
            lambda_recon=0.05,
            lambda_physics=0.02,
            lambda_resource=0.02,
            sampler_mode="risk_profile_balanced",
            use_risk_profile_condition=True,
            use_profile_loss=True,
            lambda_profile=0.05,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
            use_joint_profile_condition=spec.use_joint_profile_condition,
            use_joint_profile_loss=spec.use_joint_profile_loss,
            lambda_joint_profile_stage2=spec.lambda_joint_profile_stage2,
            lambda_joint_profile_stage3=spec.lambda_joint_profile_stage3,
            lambda_core_share_stage3=spec.lambda_core_share_stage3,
            joint_profile_ramp_window_hours=3.0,
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
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = asdict(spec)
    row.update({"stage1_epochs": stage1, "stage2_epochs": stage2, "stage3_epochs": stage3, "diffusion_steps": diffusion_steps, "base_channels": base_channels})
    row.update(summary["metrics"])
    row.update(_profile_diagnostics(data_dir, model_dir, spec.experiment_name))
    return row


def _recommend(row: pd.Series, baseline: pd.Series) -> str:
    if row["experiment_name"] == baseline["experiment_name"]:
        return "debug baseline"
    q99_ok = float(row["q99_cum_deficit_error"]) <= 1.25 * float(baseline["q99_cum_deficit_error"])
    profile_better = float(row["profile_total_mae"]) < float(baseline["profile_total_mae"])
    ramp_better = float(row["netload_ramp_max_mae"]) < float(baseline["netload_ramp_max_mae"])
    if q99_ok and profile_better and ramp_better:
        return "promising: profile/ramp improves while q99 is acceptable"
    if not q99_ok:
        return "not recommended yet: q99 tail worsens too much"
    if profile_better:
        return "mixed: profile process improves but main risk gains are incomplete"
    return "not recommended: no clear profile-process gain"


def run(data_dir: Path, out_dir: Path, device: str, quick: bool, force_dataset: bool) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_data_dir = prepare_joint_profile_dataset(data_dir, out_dir, force=force_dataset)
    rows = [_train_generate_eval(spec, profile_data_dir, out_dir, device=device, quick=quick) for spec in SPECS]
    df = pd.DataFrame(rows)
    baseline = df[df["experiment_name"] == "JP0_JRPD_3h_baseline"].iloc[0]
    for metric in [*EXTREME_MAIN_METRICS, "profile_total_mae", "p_core_mae"]:
        df[f"delta_{metric}_vs_JP0"] = pd.to_numeric(df[metric], errors="coerce") - float(baseline[metric])
    df["recommendation_reason"] = df.apply(lambda row: _recommend(row, baseline), axis=1)
    cols = [
        "experiment_name",
        "main_technique",
        "use_joint_profile_condition",
        "use_joint_profile_loss",
        "lambda_joint_profile_stage2",
        "lambda_joint_profile_stage3",
        "lambda_core_share_stage3",
        "lambda_tail_dist",
        "tail_dist_topk_ratio",
        "stage1_epochs",
        "stage2_epochs",
        "stage3_epochs",
        *EXTREME_MAIN_METRICS,
        "profile_depth_mae",
        "profile_ramp3h_mae",
        "profile_sync_mae",
        "profile_total_mae",
        "p_core_mae",
        "recommendation_reason",
    ]
    df[cols].to_csv(out_dir / "joint_profile_diffusion_summary.csv", index=False, encoding="utf-8-sig")
    report_lines = [
        "# Joint Risk Profile Diffusion Debug Report",
        "",
        f"- Mode: {'quick debug' if quick else 'full'}",
        "- Ramp metric: 3h window ramp",
        "- Baseline: JP0_JRPD_3h_baseline",
        "",
        "## Best Rows",
        df[cols].to_markdown(index=False),
        "",
        "## Verdict",
    ]
    best = df.sort_values(["q99_cum_deficit_error", "profile_total_mae"]).iloc[0]
    report_lines.append(f"- Best by q99/profile ordering: {best['experiment_name']}.")
    report_lines.append(f"- Recommendation: {best['recommendation_reason']}.")
    (out_dir / "joint_profile_diffusion_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run joint risk-profile conditional diffusion debug experiments.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "joint_profile_diffusion"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-dataset", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.data_dir), Path(args.out_dir), device=args.device, quick=bool(args.quick), force_dataset=bool(args.force_dataset))
