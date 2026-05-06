from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAT_METRICS = [
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
]

RISK_LOWER_BETTER = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

RISK_HIGHER_BETTER = [
    "extreme_degree_match_rate",
]

RISK_WEIGHTS = {
    "highrisk_wasserstein": 1.0,
    "highrisk_acf_mae": 1.0,
    "q99_cum_deficit_error": 2.0,
    "core_q99_cum_deficit_error": 2.0,
    "netload_ramp_max_mae": 1.0,
    "imbalance_duration_mae": 1.0,
    "extreme_degree_match_rate": 1.5,
}

PAPER_MAIN_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "extreme_degree_match_rate",
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

MAIN_COMPARE_ORDER = [
    "traditional_gaussian_copula",
    "plain_diffusion_baseline",
    "st_cdiff",
    "improved_diffusion",
    "enhanced_gan",
    "proposed",
]

ABLATION_ORDER = [
    "proposed",
    "no_evt_strict",
    "no_evt_continuous",
    "no_evt",
    "no_risk_loss",
    "no_month",
    "flat_condition",
]

BASELINE_NAMES = {
    "traditional_gaussian_copula",
    "traditional_baseline",
    "plain_diffusion_baseline",
    "conditional_ddpm",
    "st_cdiff",
    "improved_diffusion",
    "enhanced_gan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize model comparison with risk-oriented paper logic.")
    parser.add_argument("--compare", type=Path, default=Path("outputs/real_singleton_2018_2022_compare/evaluations/all_model_metrics.csv"))
    parser.add_argument("--guidance", type=Path, default=Path("outputs/guidance_sweep_proposed/guidance_summary.csv"))
    parser.add_argument("--tuning", type=Path, default=Path("outputs/proposed_tuning_variants/tuning_summary.csv"))
    parser.add_argument("--tuning-guidance", type=Path, default=Path("outputs/proposed_tuning_variants_guidance/summary.csv"))
    parser.add_argument("--dataset-summary", type=Path, default=Path("outputs/real_singleton_2018_2022_compare/dataset/dataset_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/result_summary"))
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_row(row: dict[str, str], group: str, source_file: Path, note: str = "") -> dict[str, object]:
    name = row.get("model_name") or row.get("variant") or row.get("experiment_name") or ""
    out: dict[str, object] = {
        "group": group,
        "experiment_name": name.strip(),
        "model_name": name.strip(),
        "status": row.get("status", "ok"),
        "note": note,
        "source_file": str(source_file),
    }
    for metric in STAT_METRICS + RISK_LOWER_BETTER + RISK_HIGHER_BETTER:
        out[metric] = safe_float(row.get(metric))
    for extra in ["severity_classification_method", "checkpoint_type_used_for_generation"]:
        if extra in row:
            out[extra] = row[extra]
    out["statistical_metrics"] = json.dumps({m: out.get(m) for m in STAT_METRICS}, ensure_ascii=False)
    out["extreme_metrics"] = json.dumps({m: out.get(m) for m in RISK_LOWER_BETTER + RISK_HIGHER_BETTER}, ensure_ascii=False)
    return out


def build_full_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    inputs = [
        (args.compare, "main_compare", ""),
        (args.guidance, "proposed_guidance_sweep", "same checkpoint, different guidance_scale"),
        (args.tuning, "proposed_retrain", "retrained proposed variant"),
        (args.tuning_guidance, "proposed_retrain_guidance", "retrained proposed variant with altered guidance_scale"),
    ]
    rows: list[dict[str, object]] = []
    for path, group, note in inputs:
        for row in read_csv_rows(path):
            normalized = normalize_row(row, group=group, source_file=path, note=note)
            if normalized["experiment_name"]:
                rows.append(normalized)
    return rows


def _rank_values(rows: list[dict[str, object]], metric: str, ascending: bool) -> dict[int, int]:
    values = []
    for idx, row in enumerate(rows):
        if row.get("status") == "failed":
            continue
        value = row.get(metric)
        if isinstance(value, float):
            values.append((idx, value))
    values.sort(key=lambda item: item[1], reverse=not ascending)
    ranks: dict[int, int] = {}
    last_value: float | None = None
    last_rank = 0
    for pos, (idx, value) in enumerate(values, start=1):
        if last_value is None or value != last_value:
            last_rank = pos
            last_value = value
        ranks[idx] = last_rank
    return ranks


def _baseline_best_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    baseline_rows = [
        row for row in rows
        if row.get("group") == "main_compare"
        and row.get("experiment_name") in BASELINE_NAMES
        and row.get("status") != "failed"
    ]
    if not baseline_rows:
        baseline_rows = [row for row in rows if row.get("status") != "failed" and row.get("experiment_name") != "proposed"]
    best: dict[str, float] = {}
    for metric in STAT_METRICS:
        values = [row[metric] for row in baseline_rows if isinstance(row.get(metric), float)]
        if values:
            best[metric] = min(values)
    return best


def attach_risk_oriented_scores(rows: list[dict[str, object]]) -> None:
    for metric in RISK_LOWER_BETTER:
        ranks = _rank_values(rows, metric, ascending=True)
        for idx, rank in ranks.items():
            rows[idx][f"{metric}_extreme_rank"] = rank
            rows[idx][f"{metric}_weighted_rank"] = rank * RISK_WEIGHTS[metric]
    for metric in RISK_HIGHER_BETTER:
        ranks = _rank_values(rows, metric, ascending=False)
        for idx, rank in ranks.items():
            rows[idx][f"{metric}_extreme_rank"] = rank
            rows[idx][f"{metric}_weighted_rank"] = rank * RISK_WEIGHTS[metric]

    for row in rows:
        weighted_sum = 0.0
        used_weight = 0.0
        for metric, weight in RISK_WEIGHTS.items():
            rank = row.get(f"{metric}_extreme_rank")
            if isinstance(rank, int):
                weighted_sum += rank * weight
                used_weight += weight
        row["final_extreme_score"] = round(weighted_sum / used_weight, 4) if used_weight else None
        row["recommendation_reason"] = _recommendation_reason(row)


def _statistical_status(row: dict[str, object]) -> str:
    penalty = float(row.get("statistical_penalty") or 0.0)
    max_excess = float(row.get("statistical_degradation_score") or 0.0)
    risk_score = row.get("risk_rank_score")
    if penalty <= 1e-12:
        return "acceptable"
    if max_excess <= 0.35:
        return "slightly_worse"
    if isinstance(risk_score, float) and risk_score <= 3.0:
        return "slightly_worse"
    return "unacceptable"


def _recommendation_reason(row: dict[str, object]) -> str:
    q99_rank = row.get("q99_cum_deficit_error_extreme_rank")
    core_q99_rank = row.get("core_q99_cum_deficit_error_extreme_rank")
    match_rank = row.get("extreme_degree_match_rate_extreme_rank")
    highrisk_w_rank = row.get("highrisk_wasserstein_extreme_rank")
    highrisk_acf_rank = row.get("highrisk_acf_mae_extreme_rank")
    ramp_rank = row.get("netload_ramp_max_mae_extreme_rank")
    tail_good = all(isinstance(rank, int) and rank <= 3 for rank in [q99_rank, core_q99_rank, match_rank])
    dist_good = all(isinstance(rank, int) and rank <= 3 for rank in [highrisk_w_rank, highrisk_acf_rank])
    risk_good = all(isinstance(rank, int) and rank <= 3 for rank in [q99_rank, core_q99_rank])
    ramp_weak = isinstance(ramp_rank, int) and ramp_rank >= 5
    if tail_good:
        return "recommended due to strong tail-risk and extreme-degree performance"
    if dist_good and not risk_good:
        return "good high-risk conditional distribution but weak risk matching"
    if risk_good and not dist_good:
        return "risk improved but high-risk conditional distribution needs review"
    if ramp_weak:
        return "tail risk is acceptable but ramp process is weak"
    return "not recommended for extreme scenario generation"


def sort_by_order(rows: list[dict[str, object]], order: list[str]) -> list[dict[str, object]]:
    order_map = {name: idx for idx, name in enumerate(order)}
    return sorted(rows, key=lambda row: order_map.get(str(row.get("experiment_name")), 999))


def load_dataset_context(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(["name" if col == "experiment_name" else col for col in columns]) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def build_markdown(dataset_context: dict[str, object], main_rows: list[dict[str, object]], ablation_rows: list[dict[str, object]]) -> str:
    lines = ["# Result Summary", ""]
    if dataset_context:
        lines.extend(
            [
                "## Dataset Context",
                "",
                f"- n_samples: {dataset_context.get('n_samples', '-')}",
                f"- split_counts: {dataset_context.get('split_counts', {})}",
                f"- event_type_counts: {json.dumps(dataset_context.get('event_type_counts', {}), ensure_ascii=False)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Evaluation Logic",
            "",
            "本文研究对象为重大天气事件下的风光荷联合极端场景生成，而非无条件常规场景生成。因此，正文主评价指标聚焦极端条件下的分布真实性、极端等级控制能力以及联合失衡风险刻画能力。全样本 Wasserstein、JS、ACF 和相关矩阵误差仅作为辅助诊断指标，用于判断生成结果是否发生明显整体失真，不作为方法优劣判断的主要依据。",
            "",
            "## Main Comparison",
            "",
        ]
    )
    lines.append(markdown_table(main_rows, ["experiment_name", *PAPER_MAIN_METRICS, "final_extreme_score", "recommendation_reason"]))
    lines.extend(["", "## Ablation", ""])
    lines.append(markdown_table(ablation_rows, ["experiment_name", *PAPER_MAIN_METRICS, "final_extreme_score", "recommendation_reason"]))
    lines.extend(["", "## Auxiliary Global Statistical Diagnostics", ""])
    lines.append(markdown_table(main_rows, ["experiment_name", *STAT_METRICS]))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_full_rows(args)
    attach_risk_oriented_scores(rows)

    main_rows = sort_by_order(
        [row for row in rows if row.get("group") == "main_compare" and row.get("experiment_name") in MAIN_COMPARE_ORDER],
        MAIN_COMPARE_ORDER,
    )
    ablation_rows = sort_by_order(
        [row for row in rows if row.get("group") == "main_compare" and row.get("experiment_name") in ABLATION_ORDER],
        ABLATION_ORDER,
    )
    key_rows = sorted(rows, key=lambda row: float(row.get("final_extreme_score") or 1e9))[:12]

    common_fields = [
        "group",
        "experiment_name",
        "status",
        "statistical_metrics",
        "extreme_metrics",
        *STAT_METRICS,
        *RISK_LOWER_BETTER,
        *RISK_HIGHER_BETTER,
        "final_extreme_score",
        "recommendation_reason",
        "severity_classification_method",
        "checkpoint_type_used_for_generation",
        "note",
        "source_file",
    ]

    write_csv(args.out_dir / "complete_result_summary.csv", rows, common_fields)
    write_csv(args.out_dir / "key_result_summary.csv", key_rows, common_fields)
    write_csv(args.out_dir / "main_compare_summary.csv", main_rows, common_fields)
    write_csv(
        args.out_dir / "main_compare_paper_table.csv",
        main_rows,
        ["experiment_name", *PAPER_MAIN_METRICS, "final_extreme_score", "recommendation_reason"],
    )
    write_csv(
        args.out_dir / "auxiliary_global_stat_table.csv",
        main_rows,
        ["experiment_name", *STAT_METRICS],
    )
    write_csv(
        args.out_dir / "ablation_paper_table.csv",
        ablation_rows,
        ["experiment_name", *PAPER_MAIN_METRICS, "final_extreme_score", "recommendation_reason"],
    )

    markdown = build_markdown(load_dataset_context(args.dataset_summary), main_rows, ablation_rows)
    (args.out_dir / "result_summary.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
