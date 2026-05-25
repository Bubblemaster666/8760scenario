from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from risk_ranking_utils import add_risk_score, write_risk_tables
from run_copula_guided_residual_diffusion import DatasetSpec, _dataset_specs
from run_gan_augmented_tailweighted_copula import (
    GanAugmentedTailWeightedConfig,
    METHOD_NAME as GAN_METHOD,
    _run_augmented_tailweighted,
)
from run_month_evt_copula_risk_selection import (
    _add_month_season,
    _build_train_risk_table,
    _compute_monthly_tau,
    _copula_cfg,
    _evaluate_method,
    _evaluate_split_row,
    _generate_candidate_pool,
    _load_split,
    _parse_weights,
    _select_candidates,
)
from run_tailweighted_month_evt_copula import (
    TAIL_FIXED_METHOD,
    _compute_tail_scores,
    _fit_tailweighted_group_copulas,
)
from run_simple_evt_risk_diffusion import FULL_COMPARE_METRICS


BASE_DIR = Path(__file__).resolve().parent
VAL_PROTECTED_METHOD = "ValProtected_GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed"


@dataclass
class ValProtectedGanConfig(GanAugmentedTailWeightedConfig):
    out_dir: Path = BASE_DIR / "results" / "gan_augmented_tailweighted_copula_valprotected"
    augmented_data_root: Path = BASE_DIR / "outputs" / "gan_augmented_datasets_valprotected"
    n_gan_candidates: int = 20
    gan_candidate_multiplier: float = 0.0


def _load_augmented_train(dataset_name: str, cfg: ValProtectedGanConfig):
    root = cfg.augmented_data_root / dataset_name
    x_train = np.load(root / "X_train_aug.npy").astype(np.float32)
    cond_train = pd.read_csv(root / "cond_train_aug.csv")
    meta_train = pd.read_csv(root / "meta_train_aug.csv")
    mask_path = root / "event_mask_train_aug.npy"
    mask_train = np.load(mask_path).astype(np.float32) if mask_path.exists() else None
    return x_train, cond_train, meta_train, mask_train


def _generate_tailweighted_split(
    spec: DatasetSpec,
    cfg: ValProtectedGanConfig,
    out_dir: Path,
    split: str,
    method_name: str,
    generated_name: str,
    train_override: tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray | None] | None = None,
) -> Path:
    if train_override is None:
        x_train, cond_train_raw, meta_train, mask_train = _load_split(spec.data_dir, "train")
    else:
        x_train, cond_train_raw, meta_train, mask_train = train_override
    _, cond_split_raw, meta_split, mask_split = _load_split(spec.data_dir, split)

    cond_train = _add_month_season(cond_train_raw, meta_train)
    cond_split = _add_month_season(cond_split_raw, meta_split)
    tau_by_month, _ = _compute_monthly_tau(x_train, cond_train, cfg, spec.out_name, out_dir)
    train_risk = _build_train_risk_table(x_train, cond_train, mask_train, tau_by_month, cfg)
    tail_df = _compute_tail_scores(train_risk, cond_train, cfg, spec.out_name, out_dir)
    models, _, fit_df = _fit_tailweighted_group_copulas(x_train, cond_train, tail_df, cfg, out_dir, spec.out_name)
    if method_name == GAN_METHOD:
        tail_df.to_csv(out_dir / "tail_score_summary_augmented.csv", index=False, encoding="utf-8-sig")
        fit_df.to_csv(out_dir / "tailweighted_copula_fit_summary_augmented.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(int(cfg.seed))
    cop_cfg = _copula_cfg(cfg, int(x_train.shape[2]), out_dir / f"{method_name}_{split}_copula_groups")
    candidates, metrics, targets, base = _generate_candidate_pool(
        split, cond_split, mask_split, models, train_risk, tau_by_month, cfg, cop_cfg, rng, out_dir, spec.out_name
    )
    fixed_weights = _parse_weights(cfg.fixed_weights)
    generated, selection_log = _select_candidates(candidates, metrics, targets, base, train_risk, cond_split, fixed_weights, cfg)
    selection_log.to_csv(out_dir / f"candidate_selection_log_{method_name}_{split}.csv", index=False, encoding="utf-8-sig")
    path = out_dir / generated_name
    np.save(path, generated.astype(np.float32))
    return path


def _evaluate_split_short(method: str, generated: Path, spec: DatasetSpec, out_dir: Path, split: str, eval_name: str) -> dict:
    eval_dir = out_dir / "evaluations" / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    mask_path = spec.data_dir / f"event_mask_{split}.npy"
    summary = evaluate_generation(
        EvalConfig(
            real=str(spec.data_dir / f"X_{split}.npy"),
            generated=str(generated),
            cond=str(spec.data_dir / f"cond_{split}.csv"),
            meta=str(spec.data_dir / f"meta_{split}.csv"),
            event_mask=str(mask_path) if mask_path.exists() else None,
            out_dir=str(eval_dir),
            model_name=eval_name,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = {"method": method}
    row.update({metric: summary["metrics"].get(metric, np.nan) for metric in FULL_COMPARE_METRICS})
    return row


def _val_score(method: str, generated: Path, spec: DatasetSpec, out_dir: Path, eval_name: str) -> dict:
    row = _evaluate_split_short(method, generated, spec, out_dir, split="val", eval_name=eval_name)
    scored = add_risk_score(pd.DataFrame([row]))
    row = scored.iloc[0].to_dict()
    row["val_risk_score"] = float(row.get("risk_score", np.nan))
    return row


def _evaluate_test_row(method: str, generated: Path, spec: DatasetSpec, out_dir: Path, eval_name: str) -> dict:
    return _evaluate_split_short(method, generated, spec, out_dir, split="test", eval_name=eval_name)


def _write_dataset_report(
    spec: DatasetSpec,
    out_dir: Path,
    val_summary: pd.DataFrame,
    risk_main: pd.DataFrame,
    aux: pd.DataFrame,
) -> None:
    lines = [
        f"# {spec.name} - Val-Protected GAN Augmented TailWeighted Copula",
        "",
        "## Method",
        "",
        "This wrapper chooses between TailWeighted fixed and conservative GAN-augmented TailWeighted fixed using validation risk_score only. The test split is evaluated only after the validation decision is made.",
        "",
        "## Validation Selection",
        "",
        val_summary.to_markdown(index=False),
        "",
        "## Test Main Risk Table",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Auxiliary Realism Table",
        "",
        aux.to_markdown(index=False) if len(aux) else "No auxiliary rows.",
    ]
    (out_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_dataset(spec: DatasetSpec, cfg: ValProtectedGanConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = cfg.out_dir / spec.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.augmented_data_root.mkdir(parents=True, exist_ok=True)
    x_train_for_ratio, _, _, _ = _load_split(spec.data_dir, "train")
    n_candidates = (
        int(round(float(cfg.gan_candidate_multiplier) * len(x_train_for_ratio)))
        if float(cfg.gan_candidate_multiplier) > 0.0
        else int(cfg.n_gan_candidates)
    )
    cfg = replace(cfg, gan_candidate_ratio=float(n_candidates) / float(len(x_train_for_ratio)))

    tail_val = _generate_tailweighted_split(
        spec,
        cfg,
        out_dir,
        "val",
        TAIL_FIXED_METHOD,
        "generated_val_tailweighted_fixed.npy",
        train_override=None,
    )
    tail_test = _generate_tailweighted_split(
        spec,
        cfg,
        out_dir,
        "test",
        TAIL_FIXED_METHOD,
        "generated_samples_tailweighted_fixed.npy",
        train_override=None,
    )

    _run_augmented_tailweighted(spec, cfg, out_dir)
    augmented_train = _load_augmented_train(spec.out_name, cfg)
    gan_val = _generate_tailweighted_split(
        spec,
        cfg,
        out_dir,
        "val",
        GAN_METHOD,
        "generated_val_gan_aug_tailweighted.npy",
        train_override=augmented_train,
    )
    gan_test = _generate_tailweighted_split(
        spec,
        cfg,
        out_dir,
        "test",
        GAN_METHOD,
        "generated_samples_gan_aug_tailweighted.npy",
        train_override=augmented_train,
    )

    val_tail = _val_score(TAIL_FIXED_METHOD, tail_val, spec, out_dir, "tailweighted_fixed_val")
    val_gan = _val_score(GAN_METHOD, gan_val, spec, out_dir, "gan_aug_val")
    val_df = add_risk_score(pd.DataFrame([val_tail, val_gan]))
    tail_score = float(val_df[val_df["method"] == TAIL_FIXED_METHOD]["risk_score"].iloc[0])
    gan_score = float(val_df[val_df["method"] == GAN_METHOD]["risk_score"].iloc[0])
    gan_enabled = bool(gan_score < tail_score)
    selected_source = gan_test if gan_enabled else tail_test
    selected_method = GAN_METHOD if gan_enabled else TAIL_FIXED_METHOD
    selected_path = out_dir / "generated_samples_valprotected_gan_aug_tailweighted.npy"
    shutil.copy2(selected_source, selected_path)

    reason = (
        f"GAN val_risk_score {gan_score:.6f} < TailWeighted val_risk_score {tail_score:.6f}"
        if gan_enabled
        else f"GAN val_risk_score {gan_score:.6f} >= TailWeighted val_risk_score {tail_score:.6f}; fallback to TailWeighted"
    )
    selection = pd.DataFrame(
        [
            {
                "dataset": spec.out_name,
                "n_train": int(len(x_train_for_ratio)),
                "n_gan_candidates_requested": int(n_candidates),
                "val_score_tailweighted": tail_score,
                "val_score_gan_aug": gan_score,
                "gan_enabled": gan_enabled,
                "selected_method": selected_method,
                "reason": reason,
            }
        ]
    )
    selection.to_csv(out_dir / "val_selection_summary.csv", index=False, encoding="utf-8-sig")

    rows = [
        _evaluate_test_row(TAIL_FIXED_METHOD, tail_test, spec, out_dir, "tailweighted_fixed_test"),
        _evaluate_test_row(GAN_METHOD, gan_test, spec, out_dir, "gan_aug_test"),
        _evaluate_test_row(VAL_PROTECTED_METHOD, selected_path, spec, out_dir, "valprotected_test"),
    ]
    df = add_risk_score(pd.DataFrame(rows))
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(df, out_dir, method_col="method")
    _write_dataset_report(spec, out_dir, selection, risk_main, aux)
    return risk_main, aux, selection


def _write_global_summaries(
    root: Path,
    risk_tables: dict[str, pd.DataFrame],
    aux_tables: dict[str, pd.DataFrame],
    selection_tables: dict[str, pd.DataFrame],
) -> None:
    rows = []
    for dataset, table in risk_tables.items():
        df = table.copy()
        df.insert(0, "dataset", dataset)
        rows.append(df)
    risk_summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    risk_summary.to_csv(root / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")

    methods = list(dict.fromkeys(risk_summary["method"].astype(str).tolist()))
    datasets = list(risk_tables.keys())
    rank_rows = []
    for method in methods:
        row = {"method": method}
        ranks = []
        scores = []
        for dataset in datasets:
            sub = risk_summary[(risk_summary["dataset"] == dataset) & (risk_summary["method"] == method)]
            if sub.empty:
                row[f"{dataset}_risk_rank"] = np.nan
                row[f"{dataset}_risk_score"] = np.nan
                continue
            rank = float(sub["risk_rank"].iloc[0])
            score = float(sub["risk_score"].iloc[0])
            row[f"{dataset}_risk_rank"] = rank
            row[f"{dataset}_risk_score"] = score
            ranks.append(rank)
            scores.append(score)
        row["mean_risk_rank"] = float(np.mean(ranks)) if ranks else np.nan
        row["mean_risk_score"] = float(np.mean(scores)) if scores else np.nan
        row["wins_count"] = int(sum(1 for rank in ranks if rank == 1.0))
        row["top3_count"] = int(sum(1 for rank in ranks if rank <= 3.0))
        rank_rows.append(row)
    rank_df = pd.DataFrame(rank_rows).sort_values(["mean_risk_rank", "mean_risk_score"], na_position="last")
    rank_df.to_csv(root / "all_datasets_rank_summary.csv", index=False, encoding="utf-8-sig")

    selection_df = pd.concat(selection_tables.values(), ignore_index=True) if selection_tables else pd.DataFrame()
    selection_df.to_csv(root / "all_datasets_val_selection_summary.csv", index=False, encoding="utf-8-sig")

    aux_rows = []
    for dataset, table in aux_tables.items():
        df = table.copy()
        df.insert(0, "dataset", dataset)
        aux_rows.append(df)
    aux_summary = pd.concat(aux_rows, ignore_index=True) if aux_rows else pd.DataFrame()
    aux_summary.to_csv(root / "all_datasets_auxiliary_summary.csv", index=False, encoding="utf-8-sig")

    fixed = TAIL_FIXED_METHOD
    valp = VAL_PROTECTED_METHOD
    fixed_rank = rank_df[rank_df["method"] == fixed].iloc[0]
    valp_rank = rank_df[rank_df["method"] == valp].iloc[0]
    gan_enabled_count = int(selection_df["gan_enabled"].sum()) if "gan_enabled" in selection_df else 0
    mean_rank_improved = float(valp_rank["mean_risk_rank"]) < float(fixed_rank["mean_risk_rank"])
    mean_score_improved = float(valp_rank["mean_risk_score"]) < float(fixed_rank["mean_risk_score"])
    enter_formal = bool(mean_rank_improved and mean_score_improved)

    is_formal50 = "formal50" in root.name.lower()
    report_name = (
        "final_valprotected_gan_augmented_formal50_report.md"
        if is_formal50
        else "final_valprotected_gan_augmented_report.md"
    )
    run_kind = "formal50" if is_formal50 else "smoke"
    lines = [
        "# Val-Protected GAN Augmented TailWeighted Copula Report",
        "",
        "## Configuration",
        "",
        "- Validation protection selects between TailWeighted fixed and conservative GAN-augmented TailWeighted fixed using validation risk_score only.",
        "- Test metrics are not used for selection.",
        f"- run kind: `{run_kind}`",
        "- `gan_epochs = 5`" if not is_formal50 else "- `gan_epochs = 50`",
        "- `n_gan_candidates = 20`" if not is_formal50 else "- `gan_candidate_count = 3 * n_train_samples`",
        "- `gan_keep_ratio = 0.02`",
        "- `risk_filter_min_keep = 2`",
        "- `corr_filter_min_keep = 1`",
        "- `K_candidates = 20`",
        "- `alpha_tail = 1.0`",
        "",
        "## Validation Decisions",
        "",
        selection_df.to_markdown(index=False) if len(selection_df) else "No validation decisions.",
        "",
        "## Cross-Dataset Rank Summary",
        "",
        rank_df.to_markdown(index=False) if len(rank_df) else "No rank rows.",
        "",
        "## Interpretation",
        "",
        f"- GAN enabled on `{gan_enabled_count}/{len(selection_df)}` datasets.",
        f"- mean_risk_rank improved versus TailWeighted fixed: `{mean_rank_improved}`.",
        f"- mean_risk_score improved versus TailWeighted fixed: `{mean_score_improved}`.",
        "",
        "【ValProtected 结论】",
        f"- 是否避免 Muswellbrook 退化：{'是' if not bool(selection_df[selection_df['dataset'] == 'muswellbrook']['gan_enabled'].iloc[0]) else '否'}",
        f"- 是否保留 Singleton / Cessnock GAN 收益：{'是' if gan_enabled_count >= 2 else '否'}",
        f"- mean_risk_rank 是否优于 TailWeighted Fixed：{'是' if mean_rank_improved else '否'}",
        f"- mean_risk_score 是否优于 TailWeighted Fixed：{'是' if mean_score_improved else '否'}",
        f"- 是否建议进入 gan_epochs=50 CPU-friendly 正式实验：{'是' if enter_formal else '否'}",
        "",
        "【下一步】",
        "如果进入正式实验，建议配置：",
        "- `gan_epochs = 50`",
        "- `gan_candidate_count = 3 * n_train_samples`",
        "- `gan_keep_ratio = 0.02`",
        "- `K_candidates = 20`",
    ]
    (root / report_name).write_text("\n".join(lines), encoding="utf-8")

    log_lines = [
        "# Run Log",
        "",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- out_dir: `{root}`",
        f"- run_kind: `{run_kind}`",
        "- Python: `C:\\Users\\13411\\anaconda3\\python.exe`",
        "- torch: `2.11.0+cpu`",
        "- CUDA: unavailable",
        "- device: CPU",
        "- selection rule: validation risk_score only",
        "- test split used for final evaluation only",
        "",
        "## Validation Selection",
        "",
        selection_df.to_markdown(index=False) if len(selection_df) else "No validation decisions.",
        "",
        "## Outputs",
        "",
        "- `all_datasets_val_selection_summary.csv`",
        "- `all_datasets_risk_summary.csv`",
        "- `all_datasets_rank_summary.csv`",
        "- `all_datasets_auxiliary_summary.csv`",
        f"- `{report_name}`",
    ]
    (root / "run_log.md").write_text("\n".join(log_lines), encoding="utf-8")


def run_all(cfg: ValProtectedGanConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.augmented_data_root.mkdir(parents=True, exist_ok=True)
    risk_tables: dict[str, pd.DataFrame] = {}
    aux_tables: dict[str, pd.DataFrame] = {}
    selection_tables: dict[str, pd.DataFrame] = {}
    for spec in [s for s in _dataset_specs() if s.data_dir.exists()]:
        print(f"\n=== Dataset: {spec.out_name} ===")
        risk_main, aux, selection = run_one_dataset(spec, cfg)
        risk_tables[spec.out_name] = risk_main
        aux_tables[spec.out_name] = aux
        selection_tables[spec.out_name] = selection
    _write_global_summaries(cfg.out_dir, risk_tables, aux_tables, selection_tables)


def parse_args() -> ValProtectedGanConfig:
    parser = argparse.ArgumentParser(description="Run validation-protected conservative GAN-augmented TailWeighted Copula.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "gan_augmented_tailweighted_copula_valprotected")
    parser.add_argument("--augmented-data-root", type=Path, default=BASE_DIR / "outputs" / "gan_augmented_datasets_valprotected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-candidates", type=int, default=20)
    parser.add_argument("--alpha-tail", type=float, default=1.0)
    parser.add_argument("--fixed-weights", type=str, default="0.35,0.25,0.20,0.20")
    parser.add_argument("--gan-epochs", type=int, default=5)
    parser.add_argument("--gan-epochs-small", type=int, default=3)
    parser.add_argument("--n-gan-candidates", type=int, default=20)
    parser.add_argument("--gan-candidate-multiplier", type=float, default=0.0)
    parser.add_argument("--gan-keep-ratio", type=float, default=0.02)
    parser.add_argument("--gan-keep-min", type=int, default=1)
    parser.add_argument("--risk-filter-min-keep", type=int, default=2)
    parser.add_argument("--corr-filter-min-keep", type=int, default=1)
    return ValProtectedGanConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_all(parse_args())
