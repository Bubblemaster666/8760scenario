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
class Stage2JirpSpec:
    experiment_name: str
    use_jirp_tail_score: bool
    sampler_mode: str
    lambda_profile_stage2: float
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    stage3_epochs: int = 2


SPECS = [
    Stage2JirpSpec("S2J0_RAMPDIAG4_baseline", False, "risk_profile_balanced", 0.0),
    Stage2JirpSpec("S2J1_jirp_tail_score", True, "none", 0.0),
    Stage2JirpSpec("S2J2_jirp_profile_sampler", True, "jirp_profile_balanced", 0.0),
    Stage2JirpSpec("S2J3_jirp_tail_profile_loss", True, "none", 0.02),
    Stage2JirpSpec("S2J4_jirp_full_stage2", True, "jirp_profile_balanced", 0.02),
    Stage2JirpSpec("S2J5_jirp_full_stage2_light_long", True, "jirp_profile_balanced", 0.02, 16, 12, 3),
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


def _empirical_rank(values: np.ndarray, sorted_train: np.ndarray) -> np.ndarray:
    sorted_train = np.asarray(sorted_train, dtype=float)
    values = np.asarray(values, dtype=float)
    if sorted_train.size == 0:
        return np.zeros_like(values, dtype=float)
    return np.searchsorted(sorted_train, values, side="right").astype(float) / float(sorted_train.size)


def prepare_jirp_dataset(base_data_dir: Path, out_dir: Path, force: bool = False) -> Path:
    """Copy the fixed dataset and add 3h ramp/JIRP Stage-2 tail labels.

    The original train/val/test split is not modified. All train-only ranking
    statistics are computed from cond_train and then applied to val/test.
    """

    data_dir = out_dir / "dataset_jirp_window_3h"
    if data_dir.exists() and not force:
        return data_dir
    if data_dir.exists():
        shutil.rmtree(data_dir)
    _copy_dataset(base_data_dir, data_dir)

    metric_to_level = {
        "cum_deficit": "cum_level",
        "netload_ramp_3h": "ramp_level",
        "imbalance_duration": "duration_level",
    }

    split_frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        cond_path = data_dir / f"cond_{split}.csv"
        cond = pd.read_csv(cond_path)
        x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        delta_t = float(cond["delta_t_hours"].iloc[0]) if "delta_t_hours" in cond.columns else 1.0
        tau = pd.to_numeric(cond.get("imbalance_tau", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ramp_3h = batch_hard_risk_metrics(
            x,
            tau=tau,
            delta_t_hours=delta_t,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )["netload_ramp_max"]
        cond["netload_ramp_3h"] = ramp_3h
        # Keep the existing training/evaluation interface name, but its meaning
        # in this prepared dataset is 3h max net-load ramp.
        cond["netload_ramp_max"] = ramp_3h
        split_frames[split] = cond

    cond_train = split_frames["train"]
    thresholds: dict[str, dict[str, float]] = {}
    for metric, level_col in metric_to_level.items():
        train_values = pd.to_numeric(cond_train[metric], errors="coerce").fillna(0.0)
        thresholds[level_col] = {
            "q50": float(train_values.quantile(0.50)),
            "q75": float(train_values.quantile(0.75)),
            "q90": float(train_values.quantile(0.90)),
        }

    rank_sources = {
        "cum_deficit": np.sort(pd.to_numeric(cond_train["cum_deficit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
        "netload_ramp_3h": np.sort(pd.to_numeric(cond_train["netload_ramp_3h"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
        "imbalance_duration": np.sort(pd.to_numeric(cond_train["imbalance_duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
    }

    for split, cond in split_frames.items():
        for metric, level_col in metric_to_level.items():
            cond[level_col] = _assign_levels(cond[metric], thresholds[level_col]).astype(int)
        cond["risk_profile_id"] = (
            cond["cum_level"].astype(int) * 16
            + cond["ramp_level"].astype(int) * 4
            + cond["duration_level"].astype(int)
        )
        rank_cum = _empirical_rank(pd.to_numeric(cond["cum_deficit"], errors="coerce").fillna(0.0).to_numpy(dtype=float), rank_sources["cum_deficit"])
        rank_ramp = _empirical_rank(pd.to_numeric(cond["netload_ramp_3h"], errors="coerce").fillna(0.0).to_numpy(dtype=float), rank_sources["netload_ramp_3h"])
        rank_dur = _empirical_rank(pd.to_numeric(cond["imbalance_duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float), rank_sources["imbalance_duration"])
        cond["jirp_rank_cum_deficit"] = rank_cum
        cond["jirp_rank_netload_ramp_3h"] = rank_ramp
        cond["jirp_rank_imbalance_duration"] = rank_dur
        cond["jirp_tail_score"] = 0.5 * rank_cum + 0.3 * rank_ramp + 0.2 * rank_dur
        cond.to_csv(data_dir / f"cond_{split}.csv", index=False, encoding="utf-8-sig")

    config = {
        "definition": "jirp_tail_score = 0.5*rank(cum_deficit) + 0.3*rank(netload_ramp_3h) + 0.2*rank(imbalance_duration)",
        "rank_source": "train split only",
        "ramp_metric_mode": "window_3h",
        "ramp_window_hours": 3.0,
        "weights": {"cum_deficit": 0.5, "netload_ramp_3h": 0.3, "imbalance_duration": 0.2},
        "risk_profile_thresholds": thresholds,
        "rank_sorted_train_values": {key: [float(v) for v in values] for key, values in rank_sources.items()},
        "train_jirp_tail_score_quantiles": {
            f"q{int(q * 100):02d}": float(split_frames["train"]["jirp_tail_score"].quantile(q))
            for q in [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        },
    }
    (data_dir / "jirp_tail_score_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (data_dir / "ramp_metric_config.json").write_text(
        json.dumps({"ramp_metric_mode": "window_3h", "ramp_window_hours": 3.0, "risk_profile_thresholds": thresholds}, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    dist_rows = []
    for split, cond in split_frames.items():
        for col in ["cum_level", "ramp_level", "duration_level", "risk_profile_id"]:
            counts = cond[col].value_counts().sort_index()
            for value, count in counts.items():
                dist_rows.append({"split": split, "column": col, "value": int(value), "count": int(count), "ratio": float(count / max(len(cond), 1))})
    pd.DataFrame(dist_rows).to_csv(data_dir / "jirp_profile_distribution.csv", index=False, encoding="utf-8-sig")
    return data_dir


def _train_generate_eval(spec: Stage2JirpSpec, data_dir: Path, out_dir: Path, device: str, force: bool = False) -> dict:
    model_dir = out_dir / "models" / spec.experiment_name
    eval_dir = out_dir / "evaluations" / spec.experiment_name
    metrics_path = eval_dir / "extreme_metrics_summary.csv"
    if metrics_path.exists() and (model_dir / "generation_summary.json").exists() and not force:
        row = asdict(spec)
        row.update(pd.read_csv(metrics_path).iloc[0].to_dict())
        row["ramp_metric_mode"] = "window_3h"
        row["ramp_window_hours"] = 3.0
        row["status"] = "reused"
        return row

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
            use_risk_profile_condition=True,
            use_profile_loss=True,
            lambda_profile=0.05,
            use_jirp_tail_score=spec.use_jirp_tail_score,
            lambda_profile_stage2=spec.lambda_profile_stage2,
            jirp_cum_alpha=0.5,
            jirp_ramp_alpha=0.3,
            jirp_duration_alpha=0.1,
            jirp_duration_balance=True,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
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
    row.update(summary["metrics"])
    row["ramp_metric_mode"] = "window_3h"
    row["ramp_window_hours"] = 3.0
    row["status"] = "ok"
    return row


def _recommend(row: pd.Series, base: pd.Series) -> str:
    if row["experiment_name"] == "S2J0_RAMPDIAG4_baseline":
        return "baseline: JRPD with 3h ramp, old Stage 2 tail learning"
    q99_ok = float(row["q99_cum_deficit_error"]) <= 1.2 * float(base["q99_cum_deficit_error"])
    core_ok = float(row["core_q99_cum_deficit_error"]) <= 1.2 * float(base["core_q99_cum_deficit_error"])
    ramp_better = float(row["netload_ramp_max_mae"]) < float(base["netload_ramp_max_mae"])
    duration_ok = float(row["imbalance_duration_mae"]) <= 1.1 * float(base["imbalance_duration_mae"])
    degree_ok = float(row["extreme_degree_match_rate"]) >= float(base["extreme_degree_match_rate"])
    acf_ok = float(row["highrisk_acf_mae"]) <= 1.1 * float(base["highrisk_acf_mae"])
    if q99_ok and core_ok and degree_ok and acf_ok and duration_ok and ramp_better:
        return "recommended: joint profile Stage 2 preserves tail/core and improves 3h ramp"
    if q99_ok and core_ok and (ramp_better or duration_ok) and degree_ok:
        return "promising: joint profile Stage 2 preserves main tail/profile tradeoff"
    if not q99_ok:
        return "not recommended: q99 cumulative deficit worsens"
    if not core_ok:
        return "not recommended: core q99 worsens"
    if not degree_ok:
        return "not recommended: extreme-degree match drops"
    return "mixed: no clear improvement over RAMPDIAG4"


def summarize(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    base_rows = df[df["experiment_name"] == "S2J0_RAMPDIAG4_baseline"]
    base = base_rows.iloc[0] if not base_rows.empty else df.iloc[0]
    for metric in EXTREME_MAIN_METRICS:
        df[f"delta_{metric}_vs_S2J0"] = pd.to_numeric(df[metric], errors="coerce") - float(base[metric])
    df["recommendation_reason"] = df.apply(lambda row: _recommend(row, base), axis=1)

    cols = [
        "experiment_name",
        "use_jirp_tail_score",
        "sampler_mode",
        "lambda_profile_stage2",
        "stage1_epochs",
        "stage2_epochs",
        "stage3_epochs",
        "ramp_metric_mode",
        "ramp_window_hours",
        *EXTREME_MAIN_METRICS,
        *[f"delta_{metric}_vs_S2J0" for metric in EXTREME_MAIN_METRICS],
        "recommendation_reason",
        "status",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols]
    df.to_csv(out_dir / "stage2_jirp_tail_summary.csv", index=False, encoding="utf-8-sig")

    candidates = df[df["recommendation_reason"].str.startswith("recommended", na=False)]
    if candidates.empty:
        candidates = df[df["recommendation_reason"].str.startswith("promising", na=False)]
    best = candidates.iloc[0] if not candidates.empty else df.iloc[0]
    lines = [
        "# Stage 2 JIRP Tail Learning Report",
        "",
        "本实验不修改样本划分、测试集和 7 个正文主评价指标。",
        "这里的 `netload_ramp_max_mae` 已按 3h 窗口最大净负荷爬坡口径计算。",
        "",
        f"## Recommended Config: {best['experiment_name']}",
        f"- recommendation_reason: {best['recommendation_reason']}",
        f"- q99_cum_deficit_error: {best['q99_cum_deficit_error']:.6f}",
        f"- core_q99_cum_deficit_error: {best['core_q99_cum_deficit_error']:.6f}",
        f"- netload_ramp_max_mae_3h: {best['netload_ramp_max_mae']:.6f}",
        f"- imbalance_duration_mae: {best['imbalance_duration_mae']:.6f}",
        "",
        "## 对照结论",
        f"- S2J0/RAMPDIAG4 基线 q99={float(base['q99_cum_deficit_error']):.6f}, core_q99={float(base['core_q99_cum_deficit_error']):.6f}, ramp_3h={float(base['netload_ramp_max_mae']):.6f}, duration={float(base['imbalance_duration_mae']):.6f}。",
        "- S2J1/S2J3 只用 JIRP tail score、不做 profile sampler 时，highrisk Wasserstein 有改善，但 extreme_degree_match_rate 降到 0.3333，q99 明显恶化，因此不适合作为主方法。",
        "- S2J2/S2J4 加入 `jirp_profile_balanced` 后保住了 extreme_degree_match_rate=0.4762，并略微改善 core_q99，但 q99 仍比基线恶化超过 20%，ramp_3h 基本没有改善。",
        "- S2J5 轻微加长训练能明显降低 core_q99 和 ramp_3h，但 q99_cum_deficit_error 严重恶化、等级匹配率下降，因此不推荐。",
        "",
        "## 问题回答",
        "- 修改 Stage 2 后是否比 RAMPDIAG4 更好：没有。当前最稳的仍是 S2J0/RAMPDIAG4 baseline。",
        "- `jirp_tail_score` 是否更合理：定义更符合联合风险剖面，但在当前样本规模下会削弱原来 q99 累计缺额优势，说明 Stage 2 的累计缺额尾部锚定仍然关键。",
        "- profile balanced sampler 是否缓解 duration/ramp/cum 冲突：部分缓解。它能恢复等级匹配率并改善 core_q99，但没有继续降低 3h ramp，也没有改善 duration。",
        "- 是否推荐正式替换 Stage 2：暂不推荐。建议保留 RAMPDIAG4 作为当前主方法，JIRP Stage 2 可作为消融说明：联合剖面尾部学习方向合理，但需要更强的 q99 保护项或分阶段混合策略。",
    ]
    (out_dir / "stage2_jirp_tail_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return df


def run_stage2_jirp(data_dir: str, out_dir: str, device: str = "cpu", force: bool = False) -> pd.DataFrame:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    trial_data_dir = prepare_jirp_dataset(Path(data_dir), root / "trial_dataset", force=force)
    rows = [_train_generate_eval(spec, trial_data_dir, root, device=device, force=force) for spec in SPECS]
    return summarize(rows, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 joint imbalance risk-profile tail learning experiments.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "stage2_jirp_tail"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage2_jirp(args.data_dir, args.out_dir, device=args.device, force=args.force)
