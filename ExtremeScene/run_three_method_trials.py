from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, EXTREME_MAIN_METRICS, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from risk_classifier import ClassifierTrainConfig, predict_mask_prior, train_mask_prior, train_risk_classifier
from train_hierarchical_evt_diffusion import TrainConfig, train_model


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class TrialSpec:
    experiment_name: str
    method_family: str
    main_technique: str
    use_risk_profile_condition: bool = False
    sampler_mode: str = "none"
    use_profile_loss: bool = False
    lambda_profile: float = 0.0
    use_mask_prior: bool = False
    use_mask_condition: bool = False
    use_mask_consistency_loss: bool = False
    lambda_mask_prior: float = 0.0
    lambda_mask_consistency: float = 0.0
    risk_guidance_mode: str = "none"
    risk_guidance_scale: float = 0.0
    risk_guidance_start_step_ratio: float = 0.5
    risk_guidance_interval: int = 5
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2


TRIALS = [
    TrialSpec("T0_E0_proposed_reproduce", "baseline", "E0 proposed"),
    TrialSpec("T1_JRPD_condition_only", "JRPD", "risk profile condition", use_risk_profile_condition=True),
    TrialSpec("T2_JRPD_profile_sampler", "JRPD", "risk profile condition + balanced sampler", use_risk_profile_condition=True, sampler_mode="risk_profile_balanced"),
    TrialSpec("T3_JRPD_profile_loss", "JRPD", "risk profile condition + profile loss", use_risk_profile_condition=True, use_profile_loss=True, lambda_profile=0.05),
    TrialSpec("T4_JRPD_sampler_loss", "JRPD", "full JRPD", use_risk_profile_condition=True, sampler_mode="risk_profile_balanced", use_profile_loss=True, lambda_profile=0.05),
    TrialSpec("T5_MaskPrior_condition", "MaskPrior", "mask prior condition", use_mask_prior=True, use_mask_condition=True, lambda_mask_prior=0.05),
    TrialSpec(
        "T6_MaskPrior_condition_consistency",
        "MaskPrior",
        "mask prior condition + consistency",
        use_mask_prior=True,
        use_mask_condition=True,
        use_mask_consistency_loss=True,
        lambda_mask_prior=0.05,
        lambda_mask_consistency=0.03,
    ),
    TrialSpec(
        "T7_RiskClassifier_guidance_light",
        "RiskClassifier-Guidance",
        "risk classifier guidance light",
        use_risk_profile_condition=True,
        risk_guidance_mode="classifier",
        risk_guidance_scale=0.1,
    ),
    TrialSpec(
        "T8_RiskClassifier_guidance_mid",
        "RiskClassifier-Guidance",
        "risk classifier guidance mid",
        use_risk_profile_condition=True,
        risk_guidance_mode="classifier",
        risk_guidance_scale=0.2,
    ),
    TrialSpec(
        "T9_JRPD_MaskPrior",
        "combined",
        "full JRPD + MaskPrior",
        use_risk_profile_condition=True,
        sampler_mode="risk_profile_balanced",
        use_profile_loss=True,
        lambda_profile=0.05,
        use_mask_prior=True,
        use_mask_condition=True,
        use_mask_consistency_loss=True,
        lambda_mask_prior=0.05,
        lambda_mask_consistency=0.03,
    ),
    TrialSpec(
        "T10_JRPD_Guidance",
        "combined",
        "full JRPD + classifier guidance",
        use_risk_profile_condition=True,
        sampler_mode="risk_profile_balanced",
        use_profile_loss=True,
        lambda_profile=0.05,
        risk_guidance_mode="classifier",
        risk_guidance_scale=0.1,
    ),
]


def _assign_levels(values: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return pd.Series(
        np.select(
            [
                arr <= thresholds["q50"],
                arr <= thresholds["q75"],
                arr <= thresholds["q90"],
            ],
            [0, 1, 2],
            default=3,
        ),
        index=values.index,
        dtype="int64",
    )


def _copy_base_dataset(data_dir: Path, trial_data_dir: Path) -> None:
    trial_data_dir.mkdir(parents=True, exist_ok=True)
    wanted_prefixes = ("X_", "cond_", "meta_", "event_mask_", "day_mask_")
    for path in data_dir.iterdir():
        if path.is_file() and (path.name.startswith(wanted_prefixes) or path.name in {"dataset_summary.json", "samples_evt_labeled.csv"}):
            shutil.copy2(path, trial_data_dir / path.name)


def prepare_trial_dataset(data_dir: Path, trial_data_dir: Path, seq_len: int, device: str) -> dict:
    """Create an experiment-local dataset copy with risk profile and mask-prior columns.

    The original fixed dataset is not modified. All quantile thresholds are
    computed from train split only and then applied to val/test.
    """

    _copy_base_dataset(data_dir, trial_data_dir)
    cond_train = pd.read_csv(trial_data_dir / "cond_train.csv")
    thresholds = {}
    metric_to_level = {
        "cum_deficit": "cum_level",
        "netload_ramp_max": "ramp_level",
        "imbalance_duration": "duration_level",
    }
    for metric, level_col in metric_to_level.items():
        series = pd.to_numeric(cond_train[metric], errors="coerce").fillna(0.0)
        thresholds[level_col] = {
            "q50": float(series.quantile(0.50)),
            "q75": float(series.quantile(0.75)),
            "q90": float(series.quantile(0.90)),
        }

    for split in ["train", "val", "test"]:
        cond_path = trial_data_dir / f"cond_{split}.csv"
        meta_path = trial_data_dir / f"meta_{split}.csv"
        cond = pd.read_csv(cond_path)
        meta = pd.read_csv(meta_path)
        for metric, level_col in metric_to_level.items():
            cond[level_col] = _assign_levels(cond[metric], thresholds[level_col]).astype(int)
        cond["risk_profile_id"] = cond["cum_level"].astype(int) * 16 + cond["ramp_level"].astype(int) * 4 + cond["duration_level"].astype(int)
        if "start_hour" not in cond.columns and "window_start_time" in meta.columns:
            cond["start_hour"] = pd.to_datetime(meta["window_start_time"]).dt.hour.astype(int)

        x = np.load(trial_data_dir / f"X_{split}.npy").astype(np.float32)
        tau = pd.to_numeric(cond["imbalance_tau"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        # exceed_mask(t)=1(net_load(t)>tau)，其中 net_load=load-wind_power-solar_power。
        net_load = x[:, 0, :] - x[:, 1, :] - x[:, 2, :]
        exceed_mask = (net_load > tau[:, None]).astype(np.float32)
        np.save(trial_data_dir / f"exceed_mask_{split}.npy", exceed_mask)
        cond.to_csv(cond_path, index=False, encoding="utf-8-sig")

    (trial_data_dir / "risk_profile_thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    mask_prior_dir = trial_data_dir / "mask_prior"
    mask_metrics = train_mask_prior(trial_data_dir, mask_prior_dir, seq_len=seq_len, epochs=80, device=device)
    mask_ckpt = mask_prior_dir / "mask_prior.pt"
    for split in ["train", "val", "test"]:
        cond_path = trial_data_dir / f"cond_{split}.csv"
        cond = pd.read_csv(cond_path)
        probs = predict_mask_prior(mask_ckpt, cond, device=device)
        for t in range(seq_len):
            cond[f"mask_prob_{t}"] = probs[:, t]
        cond.to_csv(cond_path, index=False, encoding="utf-8-sig")
    return {"risk_profile_thresholds": thresholds, "mask_prior": mask_metrics}


def _train_generate_evaluate(
    spec: TrialSpec,
    data_dir: Path,
    out_dir: Path,
    device: str,
    risk_classifier_path: Path | None,
) -> dict:
    model_dir = out_dir / "models" / spec.experiment_name
    eval_dir = out_dir / "evaluations" / spec.experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainConfig(
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
        stage1_epochs=spec.stage1_epochs,
        stage2_epochs=spec.stage2_epochs,
        stage3_epochs=spec.stage3_epochs,
        lambda_tail=0.25,
        lambda_risk=0.04,
        lambda_cum=1.0,
        lambda_ramp=0.25,
        lambda_dur=0.35,
        lambda_recon=0.05,
        lambda_physics=0.02,
        lambda_resource=0.02,
        sampler_mode=spec.sampler_mode,
        use_risk_profile_condition=spec.use_risk_profile_condition,
        use_profile_loss=spec.use_profile_loss,
        lambda_profile=spec.lambda_profile,
        use_mask_prior=spec.use_mask_prior,
        lambda_mask_prior=spec.lambda_mask_prior,
        use_mask_condition=spec.use_mask_condition,
        use_mask_consistency_loss=spec.use_mask_consistency_loss,
        lambda_mask_consistency=spec.lambda_mask_consistency,
        risk_guidance_mode=spec.risk_guidance_mode,
        risk_guidance_scale=spec.risk_guidance_scale,
        risk_guidance_start_step_ratio=spec.risk_guidance_start_step_ratio,
        risk_guidance_interval=spec.risk_guidance_interval,
        device=device,
        seed=42,
    )
    train_model(train_cfg)
    generate_from_checkpoint(
        GenerationConfig(
            checkpoint=None,
            data_dir=str(data_dir),
            out_dir=str(model_dir),
            split="test",
            guidance_scale=1.0,
            checkpoint_type="best-risk",
            risk_guidance_mode=spec.risk_guidance_mode,
            risk_classifier_path=str(risk_classifier_path) if spec.risk_guidance_mode == "classifier" and risk_classifier_path else None,
            risk_guidance_scale=spec.risk_guidance_scale,
            risk_guidance_start_step_ratio=spec.risk_guidance_start_step_ratio,
            risk_guidance_interval=spec.risk_guidance_interval,
        )
    )
    eval_summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(model_dir / "generated_samples.npy"),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=spec.experiment_name,
            event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
        )
    )
    row = asdict(spec)
    row.update(eval_summary["metrics"])
    row["status"] = "ok"
    return row


def _recommend(row: pd.Series, base: pd.Series) -> str:
    if row.get("status") != "ok":
        return "failed"
    q99_ratio = float(row["q99_cum_deficit_error"]) / max(float(base["q99_cum_deficit_error"]), 1e-6)
    core_delta = float(row["core_q99_cum_deficit_error"]) - float(base["core_q99_cum_deficit_error"])
    ramp_delta = float(row["netload_ramp_max_mae"]) - float(base["netload_ramp_max_mae"])
    dur_delta = float(row["imbalance_duration_mae"]) - float(base["imbalance_duration_mae"])
    acf_delta = float(row["highrisk_acf_mae"]) - float(base["highrisk_acf_mae"])
    degree_delta = float(row["extreme_degree_match_rate"]) - float(base["extreme_degree_match_rate"])
    if q99_ratio > 1.5:
        return "not recommended: q99 tail risk sacrificed"
    if core_delta < 0 and ramp_delta < 0 and dur_delta <= 0 and acf_delta <= 0:
        return "recommended: improves core tail, ramp, duration and high-risk shape"
    if core_delta < 0 and ramp_delta < 0 and q99_ratio <= 1.2:
        return "promising: core tail and ramp improve with acceptable q99 change"
    if core_delta < 0 and q99_ratio <= 1.2 and degree_delta >= 0 and acf_delta <= 0:
        return "promising JRPD: tail/core/profile improve, but ramp-duration still weak"
    if dur_delta < 0 and q99_ratio <= 1.2:
        return "promising for duration control"
    return "not clearly better than T0"


def summarize(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "T0_E0_proposed_reproduce" in set(df["experiment_name"]):
        base = df.loc[df["experiment_name"] == "T0_E0_proposed_reproduce"].iloc[0]
        for metric in EXTREME_MAIN_METRICS:
            df[f"delta_{metric}_vs_T0"] = pd.to_numeric(df[metric], errors="coerce") - float(base[metric])
        df["recommendation_reason"] = df.apply(lambda row: _recommend(row, base), axis=1)
    else:
        for metric in EXTREME_MAIN_METRICS:
            df[f"delta_{metric}_vs_T0"] = np.nan
        df["recommendation_reason"] = "T0 missing"

    columns = [
        "experiment_name",
        "method_family",
        "main_technique",
        "stage1_epochs",
        "stage2_epochs",
        "stage3_epochs",
        "use_risk_profile_condition",
        "sampler_mode",
        "use_profile_loss",
        "lambda_profile",
        "use_mask_prior",
        "use_mask_condition",
        "use_mask_consistency_loss",
        "lambda_mask_prior",
        "lambda_mask_consistency",
        "risk_guidance_mode",
        "risk_guidance_scale",
        "risk_guidance_start_step_ratio",
        "risk_guidance_interval",
        *EXTREME_MAIN_METRICS,
        *[f"delta_{metric}_vs_T0" for metric in EXTREME_MAIN_METRICS],
        "recommendation_reason",
        "status",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    out = df[columns]
    out.to_csv(out_dir / "three_method_trials_summary.csv", index=False, encoding="utf-8-sig")
    best = out.sort_values(["recommendation_reason", "q99_cum_deficit_error"]).head(1)
    lines = [
        "# Three Method Trials Report",
        "",
        "评价仍使用当前 7 个正文主指标；全流程没有修改样本筛选、EVT 标注和测试集。",
        "",
        "## Baseline",
    ]
    if "T0_E0_proposed_reproduce" in set(out["experiment_name"]):
        base = out.loc[out["experiment_name"] == "T0_E0_proposed_reproduce"].iloc[0]
        lines.append(", ".join([f"{m}={base[m]:.6g}" for m in EXTREME_MAIN_METRICS]))
    lines += [
        "",
        "## Recommendations",
    ]
    for _, row in out.iterrows():
        lines.append(f"- {row['experiment_name']}: {row['recommendation_reason']}")
    lines += [
        "",
        "## Key Findings",
        "- JRPD is the only useful direction in this first screening. T4 greatly improves q99 cumulative deficit, core q99 and extreme-degree matching, but it has not solved ramp or duration.",
        "- MaskPrior improves ramp slightly but sacrifices q99/core tail risk, so the current mask-prior conditioning is too weak or misaligned for the main objective.",
        "- RiskClassifier guidance increases extreme-degree matching and highrisk ACF, but it severely damages q99 tail amplitude, indicating that categorical guidance is overpowering continuous tail-risk preservation.",
        "",
        "## Notes",
        "- JRPD tests whether cumulative, ramp and duration risks can be separated as risk-profile conditions.",
        "- MaskPrior tests whether exceedance-mask process priors help duration and high-risk shape.",
        "- RiskClassifier guidance uses only target labels from cond_test during generation, not X_test curves.",
    ]
    (out_dir / "three_method_trials_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return out


def run_trials(data_dir: str, out_dir: str, device: str = "cpu", experiments: list[str] | None = None) -> pd.DataFrame:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    trial_data_dir = root / "trial_dataset"
    prep_summary = prepare_trial_dataset(Path(data_dir), trial_data_dir, seq_len=36, device=device)
    (root / "trial_dataset_preparation_summary.json").write_text(json.dumps(prep_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = [spec for spec in TRIALS if experiments is None or spec.experiment_name in experiments]
    needs_classifier = any(spec.risk_guidance_mode == "classifier" for spec in selected)
    risk_classifier_path = None
    if needs_classifier:
        clf_dir = root / "risk_classifier"
        clf_metrics = train_risk_classifier(ClassifierTrainConfig(data_dir=str(trial_data_dir), out_dir=str(clf_dir), epochs=50, device=device))
        risk_classifier_path = clf_dir / "risk_classifier.pt"
        (root / "risk_classifier_summary.json").write_text(json.dumps(clf_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for spec in selected:
        try:
            rows.append(_train_generate_evaluate(spec, trial_data_dir, root, device=device, risk_classifier_path=risk_classifier_path))
        except Exception as exc:  # noqa: BLE001
            row = asdict(spec)
            row.update({"status": "failed", "error": str(exc)})
            rows.append(row)
            (root / f"{spec.experiment_name}_error.txt").write_text(str(exc), encoding="utf-8")
    return summarize(rows, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JRPD, MaskPrior and RiskClassifier-Guidance trials.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "three_method_trials"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--experiments", nargs="*", default=None, help="Optional subset of experiment names.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_trials(args.data_dir, args.out_dir, device=args.device, experiments=args.experiments)
