from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_generation import EXTREME_MAIN_METRICS, EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from run_three_method_trials import prepare_trial_dataset
from train_hierarchical_evt_diffusion import TrainConfig, train_model


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class MaskPriorVariant:
    experiment_name: str
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2
    use_mask_prior: bool = False
    use_mask_condition: bool = False
    use_mask_consistency_loss: bool = False
    lambda_mask_consistency: float = 0.0
    mask_condition_dropout: float = 0.2
    note: str = ""


BASE_VARIANTS = [
    MaskPriorVariant("M0_T4_JRPD_baseline", note="T4 JRPD baseline"),
    MaskPriorVariant(
        "M1_JRPD_mask_condition",
        use_mask_prior=True,
        use_mask_condition=True,
        use_mask_consistency_loss=False,
        note="JRPD + mask prior condition",
    ),
    MaskPriorVariant(
        "M2_JRPD_mask_condition_consistency",
        use_mask_prior=True,
        use_mask_condition=True,
        use_mask_consistency_loss=True,
        lambda_mask_consistency=0.03,
        note="JRPD + mask condition + consistency 0.03",
    ),
    MaskPriorVariant(
        "M3_JRPD_mask_condition_stronger_consistency",
        use_mask_prior=True,
        use_mask_condition=True,
        use_mask_consistency_loss=True,
        lambda_mask_consistency=0.05,
        note="JRPD + mask condition + consistency 0.05",
    ),
]

M4_VARIANT = MaskPriorVariant(
    "M4_JRPD_mask_condition_light_long",
    stage1_epochs=16,
    stage2_epochs=12,
    stage3_epochs=3,
    use_mask_prior=True,
    use_mask_condition=True,
    use_mask_consistency_loss=True,
    lambda_mask_consistency=0.03,
    note="light-long JRPD + mask prior, only run if M1/M2 is promising",
)


def _train_generate_evaluate(variant: MaskPriorVariant, data_dir: Path, out_dir: Path, device: str) -> dict:
    model_dir = out_dir / "models" / variant.experiment_name
    eval_dir = out_dir / "evaluations" / variant.experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

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
            stage1_epochs=variant.stage1_epochs,
            stage2_epochs=variant.stage2_epochs,
            stage3_epochs=variant.stage3_epochs,
            lambda_tail=0.25,
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
            use_mask_prior=variant.use_mask_prior,
            lambda_mask_prior=0.05 if variant.use_mask_prior else 0.0,
            use_mask_condition=variant.use_mask_condition,
            mask_condition_dropout=variant.mask_condition_dropout,
            use_mask_consistency_loss=variant.use_mask_consistency_loss,
            lambda_mask_consistency=variant.lambda_mask_consistency,
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
            model_name=variant.experiment_name,
            event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
        )
    )
    row = asdict(variant)
    row.update(summary["metrics"])
    row["status"] = "ok"
    return row


def _is_promising(row: pd.Series, baseline: pd.Series) -> bool:
    q99_ok = float(row["q99_cum_deficit_error"]) <= 1.2 * float(baseline["q99_cum_deficit_error"])
    degree_ok = float(row["extreme_degree_match_rate"]) >= float(baseline["extreme_degree_match_rate"])
    core_ok = float(row["core_q99_cum_deficit_error"]) <= 1.15 * float(baseline["core_q99_cum_deficit_error"])
    acf_or_duration = (
        float(row["highrisk_acf_mae"]) < float(baseline["highrisk_acf_mae"])
        or float(row["imbalance_duration_mae"]) < float(baseline["imbalance_duration_mae"])
    )
    ramp_ok = float(row["netload_ramp_max_mae"]) <= 1.10 * float(baseline["netload_ramp_max_mae"])
    return bool(q99_ok and degree_ok and core_ok and acf_or_duration and ramp_ok)


def _recommend(row: pd.Series, baseline: pd.Series) -> str:
    if str(row["experiment_name"]) == "M0_T4_JRPD_baseline":
        return "baseline reference"
    q99_ok = float(row["q99_cum_deficit_error"]) <= 1.2 * float(baseline["q99_cum_deficit_error"])
    degree_ok = float(row["extreme_degree_match_rate"]) >= float(baseline["extreme_degree_match_rate"])
    core_ok = float(row["core_q99_cum_deficit_error"]) <= 1.15 * float(baseline["core_q99_cum_deficit_error"])
    acf_better = float(row["highrisk_acf_mae"]) < float(baseline["highrisk_acf_mae"])
    duration_better = float(row["imbalance_duration_mae"]) < float(baseline["imbalance_duration_mae"])
    ramp_ok = float(row["netload_ramp_max_mae"]) <= 1.10 * float(baseline["netload_ramp_max_mae"])
    if q99_ok and degree_ok and core_ok and ramp_ok and (acf_better or duration_better):
        return "recommended: keeps JRPD tail/profile and improves mask-related process metric"
    if not q99_ok:
        return "not recommended: q99 tail advantage is not preserved"
    if not degree_ok:
        return "not recommended: extreme-degree control weakens"
    if not core_ok:
        return "not recommended: core q99 worsens"
    return "mixed: mask prior signal is insufficient"


def _score(df: pd.DataFrame) -> pd.Series:
    weights = {
        "q99_cum_deficit_error": 2.5,
        "core_q99_cum_deficit_error": 2.0,
        "extreme_degree_match_rate": 2.0,
        "highrisk_acf_mae": 1.2,
        "imbalance_duration_mae": 1.2,
        "netload_ramp_max_mae": 1.0,
        "highrisk_wasserstein": 0.6,
    }
    out = pd.Series(0.0, index=df.index)
    for metric, weight in weights.items():
        values = pd.to_numeric(df[metric], errors="coerce")
        ascending = metric != "extreme_degree_match_rate"
        out += float(weight) * values.rank(ascending=ascending, na_option="bottom")
    return out / sum(weights.values())


def summarize(rows: list[dict], out_dir: Path, m4_run: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    baseline = df.loc[df["experiment_name"].eq("M0_T4_JRPD_baseline")].iloc[0]
    for metric in EXTREME_MAIN_METRICS:
        df[f"delta_{metric}_vs_M0"] = pd.to_numeric(df[metric], errors="coerce") - float(baseline[metric])
    df["maskprior_score"] = _score(df)
    df["recommendation_reason"] = df.apply(lambda row: _recommend(row, baseline), axis=1)
    cols = [
        "experiment_name",
        "stage1_epochs",
        "stage2_epochs",
        "stage3_epochs",
        "use_mask_prior",
        "use_mask_condition",
        "use_mask_consistency_loss",
        "lambda_mask_consistency",
        "mask_condition_dropout",
        *EXTREME_MAIN_METRICS,
        *[f"delta_{metric}_vs_M0" for metric in EXTREME_MAIN_METRICS],
        "maskprior_score",
        "recommendation_reason",
        "note",
        "status",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols].sort_values("maskprior_score", na_position="last")
    df.to_csv(out_dir / "jrpd_maskprior_summary.csv", index=False, encoding="utf-8-sig")

    best = df.iloc[0]
    lines = [
        "# JRPD + MaskPrior Report",
        "",
        "本轮只在 T4_JRPD_sampler_loss 基础上尝试 MaskPrior，不继续 JRPD 长训练，也不启用 classifier guidance。",
        "",
        f"## Recommended Config: {best['experiment_name']}",
        f"- reason: {best['recommendation_reason']}",
        f"- M4 light-long was run: {bool(m4_run)}",
        "",
        "## Best Metrics",
    ]
    for metric in EXTREME_MAIN_METRICS:
        lines.append(f"- {metric}: {float(best[metric]):.6g}")
    mask_rows = df[df["use_mask_condition"].astype(bool)]
    duration_improved = bool((mask_rows["imbalance_duration_mae"] < float(baseline["imbalance_duration_mae"])).any()) if not mask_rows.empty else False
    acf_improved = bool((mask_rows["highrisk_acf_mae"] < float(baseline["highrisk_acf_mae"])).any()) if not mask_rows.empty else False
    q99_kept = bool((mask_rows["q99_cum_deficit_error"] <= 1.2 * float(baseline["q99_cum_deficit_error"])).any()) if not mask_rows.empty else False
    degree_kept = bool((mask_rows["extreme_degree_match_rate"] >= float(baseline["extreme_degree_match_rate"])).any()) if not mask_rows.empty else False
    lines += [
        "",
        "## Required Questions",
        f"- MaskPrior 是否改善 duration: {duration_improved}",
        f"- MaskPrior 是否改善 highrisk_acf_mae: {acf_improved}",
        f"- 是否保持 q99: {q99_kept}",
        f"- 是否保持 extreme_degree_match_rate: {degree_kept}",
        "- 是否值得进入下一轮 RiskClassifier-Guidance: " + ("yes, if the recommended config is a mask-prior variant" if str(best["experiment_name"]) != "M0_T4_JRPD_baseline" else "no, current mask-prior variants do not beat T4 enough to justify classifier guidance yet"),
    ]
    (out_dir / "jrpd_maskprior_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return df


def run_maskprior(data_dir: str, out_dir: str, device: str = "cpu") -> pd.DataFrame:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    trial_data_dir = root / "trial_dataset"
    prep = prepare_trial_dataset(Path(data_dir), trial_data_dir, seq_len=36, device=device)
    (root / "trial_dataset_preparation_summary.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict] = []
    for variant in BASE_VARIANTS:
        rows.append(_train_generate_evaluate(variant, trial_data_dir, root, device=device))
    interim = pd.DataFrame(rows)
    baseline = interim.loc[interim["experiment_name"].eq("M0_T4_JRPD_baseline")].iloc[0]
    m4_run = any(
        _is_promising(interim.loc[interim["experiment_name"].eq(name)].iloc[0], baseline)
        for name in ["M1_JRPD_mask_condition", "M2_JRPD_mask_condition_consistency"]
    )
    if m4_run:
        rows.append(_train_generate_evaluate(M4_VARIANT, trial_data_dir, root, device=device))
    return summarize(rows, root, m4_run=m4_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JRPD + MaskPrior focused tuning.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "jrpd_maskprior"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_maskprior(args.data_dir, args.out_dir, device=args.device)
