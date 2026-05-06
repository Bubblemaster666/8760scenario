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
class RampEventVariant:
    experiment_name: str
    use_ramp_event_condition: bool = False
    ramp_event_condition_scale: float = 1.0
    use_ramp_event_loss: bool = False
    lambda_ramp_event: float = 0.0
    note: str = ""


VARIANTS = [
    RampEventVariant("R0_T4_JRPD_baseline", note="T4 JRPD baseline"),
    RampEventVariant("R1_ramp_event_condition", use_ramp_event_condition=True, note="ramp event process condition"),
    RampEventVariant("R2_ramp_event_condition_scale2", use_ramp_event_condition=True, ramp_event_condition_scale=2.0, note="stronger ramp event condition"),
    RampEventVariant("R3_ramp_event_loss005", use_ramp_event_condition=True, use_ramp_event_loss=True, lambda_ramp_event=0.05, note="ramp event loss 0.05"),
    RampEventVariant("R4_ramp_event_loss010", use_ramp_event_condition=True, use_ramp_event_loss=True, lambda_ramp_event=0.10, note="ramp event loss 0.10"),
    RampEventVariant("R5_ramp_event_loss020", use_ramp_event_condition=True, use_ramp_event_loss=True, lambda_ramp_event=0.20, note="ramp event loss 0.20"),
]


def compute_ramp_event_labels(x: np.ndarray) -> pd.DataFrame:
    """Compute ramp event labels from each sample itself.

    X shape is [N, 3, T], channels are load, wind_power, solar_power.
    net_load = load - wind_power - solar_power, and delta_net describes adjacent
    net-load ramp events.
    """

    load = x[:, 0, :]
    wind = x[:, 1, :]
    solar = x[:, 2, :]
    net_load = load - wind - solar
    delta_net = net_load[:, 1:] - net_load[:, :-1]
    peak_time = delta_net.argmax(axis=1).astype(int)
    peak_value = delta_net[np.arange(len(x)), peak_time].astype(float)
    widths: list[int] = []
    source_types: list[int] = []
    for i, t in enumerate(peak_time):
        if peak_value[i] <= 0:
            widths.append(0)
        else:
            threshold = 0.5 * peak_value[i]
            left = int(t)
            right = int(t)
            while left > 0 and delta_net[i, left - 1] >= threshold:
                left -= 1
            while right < delta_net.shape[1] - 1 and delta_net[i, right + 1] >= threshold:
                right += 1
            widths.append(int(right - left + 1))

        delta_load = load[i, t + 1] - load[i, t]
        delta_wind = wind[i, t + 1] - wind[i, t]
        delta_solar = solar[i, t + 1] - solar[i, t]
        contrib = np.array([max(delta_load, 0.0), max(-delta_wind, 0.0), max(-delta_solar, 0.0)], dtype=float)
        total = float(contrib.sum())
        if total <= 1e-8 or float(contrib.max() / total) <= 0.5:
            source_types.append(3)
        else:
            source_types.append(int(contrib.argmax()))

    denom = max(delta_net.shape[1], 1)
    return pd.DataFrame(
        {
            "ramp_peak_time": peak_time,
            "ramp_peak_time_sin": np.sin(2 * np.pi * peak_time / denom),
            "ramp_peak_time_cos": np.cos(2 * np.pi * peak_time / denom),
            "ramp_peak_value": peak_value,
            "ramp_width": np.asarray(widths, dtype=int),
            "ramp_source_type": np.asarray(source_types, dtype=int),
        }
    )


def add_ramp_event_columns(data_dir: Path) -> dict:
    summary = {}
    for split in ["train", "val", "test"]:
        x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        cond_path = data_dir / f"cond_{split}.csv"
        cond = pd.read_csv(cond_path)
        labels = compute_ramp_event_labels(x)
        for col in labels.columns:
            cond[col] = labels[col].to_numpy()
        cond.to_csv(cond_path, index=False, encoding="utf-8-sig")
        summary[split] = {
            "ramp_peak_value_mean": float(labels["ramp_peak_value"].mean()),
            "ramp_peak_value_q99": float(labels["ramp_peak_value"].quantile(0.99)),
            "ramp_width_mean": float(labels["ramp_width"].mean()),
            "ramp_source_type_counts": {str(k): int(v) for k, v in labels["ramp_source_type"].value_counts().sort_index().items()},
        }
    (data_dir / "ramp_event_label_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def ramp_diagnostics(real: np.ndarray, gen: np.ndarray, cond: pd.DataFrame) -> dict:
    real_labels = compute_ramp_event_labels(real)
    gen_labels = compute_ramp_event_labels(gen)
    net_real = real[:, 0, :] - real[:, 1, :] - real[:, 2, :]
    net_gen = gen[:, 0, :] - gen[:, 1, :] - gen[:, 2, :]
    delta_real = net_real[:, 1:] - net_real[:, :-1]
    delta_gen = net_gen[:, 1:] - net_gen[:, :-1]
    return {
        "ramp_peak_time_mae": float(np.abs(gen_labels["ramp_peak_time"].to_numpy() - cond["ramp_peak_time"].to_numpy()).mean()),
        "ramp_peak_value_mae": float(np.abs(gen_labels["ramp_peak_value"].to_numpy() - cond["ramp_peak_value"].to_numpy()).mean()),
        "ramp_source_match_rate": float((gen_labels["ramp_source_type"].to_numpy() == cond["ramp_source_type"].to_numpy()).mean()),
        "delta_net_mae": float(np.abs(delta_gen - delta_real).mean()),
        "q99_netload_ramp_max_error": float(abs(np.quantile(gen_labels["ramp_peak_value"], 0.99) - np.quantile(real_labels["ramp_peak_value"], 0.99))),
    }


def _train_generate_evaluate(variant: RampEventVariant, data_dir: Path, out_dir: Path, device: str) -> dict:
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
            sampler_mode="risk_profile_balanced",
            use_risk_profile_condition=True,
            use_profile_loss=True,
            lambda_profile=0.05,
            use_ramp_event_condition=variant.use_ramp_event_condition,
            ramp_event_condition_scale=variant.ramp_event_condition_scale,
            use_ramp_event_loss=variant.use_ramp_event_loss,
            lambda_ramp_event=variant.lambda_ramp_event,
            ramp_peak_softmax_temp=0.2,
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
    eval_summary = evaluate_generation(
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
    real = np.load(data_dir / "X_test.npy").astype(np.float32)
    gen = np.load(model_dir / "generated_samples.npy").astype(np.float32)
    cond = pd.read_csv(data_dir / "cond_test.csv")
    row = asdict(variant)
    row.update(eval_summary["metrics"])
    row.update(ramp_diagnostics(real, gen, cond))
    row["status"] = "ok"
    return row


def _recommend(row: pd.Series, base: pd.Series) -> str:
    if row["experiment_name"] == "R0_T4_JRPD_baseline":
        return "baseline reference"
    ramp_improved = float(row["netload_ramp_max_mae"]) <= 0.85 * float(base["netload_ramp_max_mae"])
    q99_ok = float(row["q99_cum_deficit_error"]) <= 1.2 * float(base["q99_cum_deficit_error"])
    degree_ok = float(row["extreme_degree_match_rate"]) >= float(base["extreme_degree_match_rate"])
    duration_ok = float(row["imbalance_duration_mae"]) <= 1.10 * float(base["imbalance_duration_mae"])
    acf_ok = float(row["highrisk_acf_mae"]) <= 1.10 * float(base["highrisk_acf_mae"])
    if ramp_improved and q99_ok and degree_ok and duration_ok and acf_ok:
        return "recommended: ramp event improves ramp while preserving JRPD tail/profile"
    if ramp_improved and not q99_ok:
        return "not recommended: ramp improves but q99 tail is sacrificed"
    if not ramp_improved and q99_ok and degree_ok:
        return "mixed: JRPD tail/profile preserved but ramp not improved enough"
    return "not recommended for next JRPD main version"


def summarize(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    base = df.loc[df["experiment_name"].eq("R0_T4_JRPD_baseline")].iloc[0]
    for metric in EXTREME_MAIN_METRICS:
        df[f"delta_{metric}_vs_R0"] = pd.to_numeric(df[metric], errors="coerce") - float(base[metric])
    diag_cols = ["ramp_peak_time_mae", "ramp_peak_value_mae", "ramp_source_match_rate", "delta_net_mae", "q99_netload_ramp_max_error"]
    df["recommendation_reason"] = df.apply(lambda row: _recommend(row, base), axis=1)
    cols = [
        "experiment_name",
        "use_ramp_event_condition",
        "ramp_event_condition_scale",
        "use_ramp_event_loss",
        "lambda_ramp_event",
        *EXTREME_MAIN_METRICS,
        *diag_cols,
        *[f"delta_{metric}_vs_R0" for metric in EXTREME_MAIN_METRICS],
        "recommendation_reason",
        "note",
        "status",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols]
    df.to_csv(out_dir / "jrpd_ramp_event_summary.csv", index=False, encoding="utf-8-sig")
    recommended = df[df["recommendation_reason"].str.startswith("recommended", na=False)]
    best = recommended.iloc[0] if not recommended.empty else df.iloc[0]
    cond_rows = df[df["use_ramp_event_condition"].astype(bool)]
    lines = [
        "# JRPD Ramp Event Report",
        "",
        "本轮只在 T4_JRPD_sampler_loss 基础上新增 ramp event process condition 和 ramp event consistency loss；未修改样本筛选、EVT 标注、测试集或 7 个正文指标。",
        "",
        f"## Recommended Config: {best['experiment_name']}",
        f"- reason: {best['recommendation_reason']}",
        "",
        "## Required Questions",
        f"- ramp event condition 是否改善爬坡: {bool((cond_rows['netload_ramp_max_mae'] <= 0.85 * float(base['netload_ramp_max_mae'])).any())}",
        f"- ramp event loss 是否有效: {bool((df[df['use_ramp_event_loss'].astype(bool)]['netload_ramp_max_mae'] <= 0.85 * float(base['netload_ramp_max_mae'])).any())}",
        f"- 是否保住 q99: {bool((cond_rows['q99_cum_deficit_error'] <= 1.2 * float(base['q99_cum_deficit_error'])).any())}",
        f"- 是否保住 extreme_degree_match_rate: {bool((cond_rows['extreme_degree_match_rate'] >= float(base['extreme_degree_match_rate'])).any())}",
        f"- 是否推荐作为 JRPD 下一版主方法: {bool(not recommended.empty)}",
        "",
        "## Best Metrics",
    ]
    for metric in EXTREME_MAIN_METRICS:
        lines.append(f"- {metric}: {float(best[metric]):.6g}")
    (out_dir / "jrpd_ramp_event_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return df


def run_ramp_event(data_dir: str, out_dir: str, device: str = "cpu") -> pd.DataFrame:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    trial_data_dir = root / "trial_dataset"
    prep = prepare_trial_dataset(Path(data_dir), trial_data_dir, seq_len=36, device=device)
    prep["ramp_event_labels"] = add_ramp_event_columns(trial_data_dir)
    (root / "trial_dataset_preparation_summary.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for variant in VARIANTS:
        rows.append(_train_generate_evaluate(variant, trial_data_dir, root, device=device))
    return summarize(rows, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune JRPD with ramp event process condition.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "jrpd_ramp_event"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ramp_event(args.data_dir, args.out_dir, device=args.device)
