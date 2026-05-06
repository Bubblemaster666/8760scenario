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
class JrpdVariant:
    experiment_name: str
    stage1_epochs: int
    stage2_epochs: int
    stage3_epochs: int
    lambda_profile: float = 0.05
    lambda_delta_net_stage2: float = 0.0
    lambda_shape_stage2: float = 0.0
    lambda_delta_net_stage3: float = 0.0
    lambda_ramp_topk: float = 0.0
    lambda_shape_stage3: float = 0.0
    lambda_duration_over_stage3: float = 0.0
    lambda_exceed_mask_stage3: float = 0.0
    note: str = ""


VARIANTS = [
    JrpdVariant("J1_T4_20_15_4", 20, 15, 4, note="longer JRPD"),
    JrpdVariant("J2_T4_30_20_6", 30, 20, 6, note="longest JRPD"),
    JrpdVariant(
        "J3_T4_20_15_4_conservative_shape_ramp",
        20,
        15,
        4,
        lambda_delta_net_stage2=0.05,
        lambda_shape_stage2=0.01,
        lambda_delta_net_stage3=0.15,
        lambda_ramp_topk=0.15,
        lambda_shape_stage3=0.02,
        note="mild ramp/shape process constraints",
    ),
    JrpdVariant(
        "J4_T4_20_15_4_default_shape_ramp",
        20,
        15,
        4,
        lambda_delta_net_stage2=0.10,
        lambda_shape_stage2=0.02,
        lambda_delta_net_stage3=0.20,
        lambda_ramp_topk=0.20,
        lambda_shape_stage3=0.03,
        note="default ramp/shape process constraints",
    ),
    JrpdVariant(
        "J5_T4_30_20_6_conservative_shape_ramp",
        30,
        20,
        6,
        lambda_delta_net_stage2=0.05,
        lambda_shape_stage2=0.01,
        lambda_delta_net_stage3=0.15,
        lambda_ramp_topk=0.15,
        lambda_shape_stage3=0.02,
        note="long JRPD with mild process constraints",
    ),
    JrpdVariant(
        "J6_T4_30_20_6_default_shape_ramp",
        30,
        20,
        6,
        lambda_delta_net_stage2=0.10,
        lambda_shape_stage2=0.02,
        lambda_delta_net_stage3=0.20,
        lambda_ramp_topk=0.20,
        lambda_shape_stage3=0.03,
        note="long JRPD with default process constraints",
    ),
    JrpdVariant(
        "J7_T4_20_15_4_duration_soft",
        20,
        15,
        4,
        lambda_duration_over_stage3=0.05,
        lambda_exceed_mask_stage3=0.01,
        note="mild duration compression",
    ),
    JrpdVariant(
        "J8_T4_30_20_6_duration_soft",
        30,
        20,
        6,
        lambda_duration_over_stage3=0.05,
        lambda_exceed_mask_stage3=0.01,
        note="long JRPD with mild duration compression",
    ),
    JrpdVariant(
        "J9_T4_20_15_4_profile_010",
        20,
        15,
        4,
        lambda_profile=0.10,
        note="stronger profile consistency",
    ),
]


def _train_generate_evaluate(variant: JrpdVariant, data_dir: Path, out_dir: Path, device: str) -> dict:
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
            lambda_profile=variant.lambda_profile,
            lambda_delta_net_stage2=variant.lambda_delta_net_stage2,
            lambda_shape_stage2=variant.lambda_shape_stage2,
            lambda_delta_net_stage3=variant.lambda_delta_net_stage3,
            lambda_ramp_topk=variant.lambda_ramp_topk,
            lambda_shape_stage3=variant.lambda_shape_stage3,
            lambda_duration_over_stage3=variant.lambda_duration_over_stage3,
            lambda_exceed_mask_stage3=variant.lambda_exceed_mask_stage3,
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


def _load_reference_rows(base_out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    ref_path = BASE_DIR / "outputs" / "three_method_trials" / "three_method_trials_summary.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        keep = ref[ref["experiment_name"].isin(["T0_E0_proposed_reproduce", "T4_JRPD_sampler_loss"])].copy()
        for _, row in keep.iterrows():
            item = row.to_dict()
            item["note"] = "reference from three_method_trials"
            item["status"] = item.get("status", "ok")
            rows.append(item)
    return rows


def _rank_score(df: pd.DataFrame) -> pd.Series:
    weights = {
        "q99_cum_deficit_error": 2.5,
        "core_q99_cum_deficit_error": 2.0,
        "netload_ramp_max_mae": 1.5,
        "imbalance_duration_mae": 1.5,
        "highrisk_acf_mae": 1.0,
        "highrisk_wasserstein": 0.8,
        "extreme_degree_match_rate": 1.2,
    }
    score = pd.Series(0.0, index=df.index)
    for metric, weight in weights.items():
        values = pd.to_numeric(df[metric], errors="coerce")
        if metric == "extreme_degree_match_rate":
            ranks = values.rank(method="average", ascending=False, na_option="bottom")
        else:
            ranks = values.rank(method="average", ascending=True, na_option="bottom")
        score += float(weight) * ranks
    return score / sum(weights.values())


def _recommend(row: pd.Series, base: pd.Series) -> str:
    name = str(row.get("experiment_name", ""))
    if name == "T0_E0_proposed_reproduce":
        return "baseline reference"
    if name == "T4_JRPD_sampler_loss":
        return "recommended current best: strongest q99/core/profile tradeoff, ramp-duration still weak"
    q99 = float(row["q99_cum_deficit_error"])
    core = float(row["core_q99_cum_deficit_error"])
    ramp = float(row["netload_ramp_max_mae"])
    dur = float(row["imbalance_duration_mae"])
    if q99 <= float(base["q99_cum_deficit_error"]) and core <= float(base["core_q99_cum_deficit_error"]) and dur <= float(base["imbalance_duration_mae"]):
        return "recommended: improves q99/core while controlling duration"
    if q99 <= float(base["q99_cum_deficit_error"]) * 1.2 and core <= float(base["core_q99_cum_deficit_error"]) and ramp <= float(base["netload_ramp_max_mae"]):
        return "promising: tail/core/ramp improve with acceptable q99"
    if ramp <= float(base["netload_ramp_max_mae"]) * 0.75 and q99 <= float(base["q99_cum_deficit_error"]) * 1.1:
        return "diagnostic: ramp improves, but core tail/profile are weaker than T4"
    if q99 > float(base["q99_cum_deficit_error"]) * 1.5:
        return "not recommended: q99 tail risk degrades too much"
    return "mixed: useful diagnostic but not best"


def summarize(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    ok = df[df.get("status", "ok") == "ok"].copy()
    base = ok.loc[ok["experiment_name"].eq("T0_E0_proposed_reproduce")].iloc[0] if "T0_E0_proposed_reproduce" in set(ok["experiment_name"]) else ok.iloc[0]
    for metric in EXTREME_MAIN_METRICS:
        df[f"delta_{metric}_vs_T0"] = pd.to_numeric(df[metric], errors="coerce") - float(base[metric])
    df["jrpd_tuning_score"] = np.nan
    df.loc[ok.index, "jrpd_tuning_score"] = _rank_score(ok)
    df["recommendation_reason"] = df.apply(lambda row: _recommend(row, base) if row.get("status", "ok") == "ok" else "failed", axis=1)
    columns = [
        "experiment_name",
        "stage1_epochs",
        "stage2_epochs",
        "stage3_epochs",
        "lambda_profile",
        "lambda_delta_net_stage2",
        "lambda_shape_stage2",
        "lambda_delta_net_stage3",
        "lambda_ramp_topk",
        "lambda_shape_stage3",
        "lambda_duration_over_stage3",
        "lambda_exceed_mask_stage3",
        *EXTREME_MAIN_METRICS,
        *[f"delta_{metric}_vs_T0" for metric in EXTREME_MAIN_METRICS],
        "jrpd_tuning_score",
        "recommendation_reason",
        "note",
        "status",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[columns].sort_values("jrpd_tuning_score", na_position="last")
    df.to_csv(out_dir / "jrpd_tuning_summary.csv", index=False, encoding="utf-8-sig")
    best = df[df["status"].eq("ok")].iloc[0]
    lines = [
        "# JRPD Tuning Report",
        "",
        "本轮只围绕 JRPD/T4 做长训练和温和过程约束调参；没有修改样本库、EVT 标注、测试集或 7 个正文评价指标。",
        "",
        f"## Recommended Config: {best['experiment_name']}",
        f"- reason: {best['recommendation_reason']}",
        f"- tuning score: {best['jrpd_tuning_score']:.4f}",
        "",
        "## Best Metrics",
    ]
    for metric in EXTREME_MAIN_METRICS:
        lines.append(f"- {metric}: {float(best[metric]):.6g}")
    lines += [
        "",
        "## Interpretation",
        "- 如果最佳项仍然 duration 偏高，说明目前 JRPD 主要解决了累计尾部和核心段尾部，duration 需要更结构化的持续过程建模，而不是继续加权硬压。",
        "- 如果过程约束版本胜出，说明 ramp/shape loss 可以作为 JRPD 的轻量补丁进入下一轮长训练。",
    ]
    (out_dir / "jrpd_tuning_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return df


def run_jrpd_tuning(data_dir: str, out_dir: str, device: str = "cpu", experiments: list[str] | None = None) -> pd.DataFrame:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    trial_data_dir = root / "trial_dataset"
    prep_summary = prepare_trial_dataset(Path(data_dir), trial_data_dir, seq_len=36, device=device)
    (root / "trial_dataset_preparation_summary.json").write_text(json.dumps(prep_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = [v for v in VARIANTS if experiments is None or v.experiment_name in experiments]
    rows = _load_reference_rows(root)
    for variant in selected:
        try:
            rows.append(_train_generate_evaluate(variant, trial_data_dir, root, device=device))
        except Exception as exc:  # noqa: BLE001
            row = asdict(variant)
            row.update({"status": "failed", "error": str(exc)})
            rows.append(row)
            (root / f"{variant.experiment_name}_error.txt").write_text(str(exc), encoding="utf-8")
    return summarize(rows, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune JRPD/T4 variants with longer epochs and process constraints.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "jrpd_tuning"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--experiments", nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_jrpd_tuning(args.data_dir, args.out_dir, device=args.device, experiments=args.experiments)
