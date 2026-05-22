from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from risk_ranking_utils import (
    AUXILIARY_REALISM_METRICS,
    RISK_EVALUATION_EXPLANATION,
    RISK_MAIN_METRICS,
    RISK_RANKING_EXPLANATION,
    RISK_SCORE_WEIGHTS,
    add_risk_score,
)


STAT_METRICS = [
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
]

RISK_LOWER_BETTER = RISK_MAIN_METRICS

RISK_HIGHER_BETTER = [
    "extreme_degree_match_rate",
]

RISK_WEIGHTS = RISK_SCORE_WEIGHTS

PAPER_MAIN_METRICS = RISK_MAIN_METRICS

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
    name = row.get("model_name") or row.get("method") or row.get("variant") or row.get("experiment_name") or ""
    out: dict[str, object] = {
        "group": group,
        "experiment_name": name.strip(),
        "model_name": name.strip(),
        "status": row.get("status", "ok"),
        "note": note,
        "source_file": str(source_file),
    }
    metric_names = list(dict.fromkeys([*STAT_METRICS, *AUXILIARY_REALISM_METRICS, *RISK_LOWER_BETTER, *RISK_HIGHER_BETTER]))
    for metric in metric_names:
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
    base = pd.DataFrame(rows)
    base["_row_id"] = range(len(base))
    scored = add_risk_score(base)
    score_by_id = {int(row["_row_id"]): row for row in scored.to_dict(orient="records") if "_row_id" in row}
    metric_ranks = {metric: _rank_values(rows, metric, ascending=True) for metric in RISK_LOWER_BETTER}
    for idx, row in enumerate(rows):
        scored_row = score_by_id.get(idx)
        if scored_row:
            row["risk_score"] = scored_row.get("risk_score")
            row["risk_rank"] = scored_row.get("risk_rank")
            row["final_extreme_score"] = scored_row.get("risk_score")
            for metric in RISK_MAIN_METRICS:
                norm_col = f"norm_{metric}"
                if norm_col in scored_row:
                    row[norm_col] = scored_row.get(norm_col)
        for metric in RISK_LOWER_BETTER:
            if idx in metric_ranks[metric]:
                row[f"{metric}_risk_rank"] = metric_ranks[metric][idx]
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
    rank = row.get("risk_rank")
    q99_rank = row.get("q99_cum_deficit_error_risk_rank")
    core_rank = row.get("core_q99_cum_deficit_error_risk_rank")
    ramp_rank = row.get("netload_ramp_max_mae_risk_rank")
    duration_rank = row.get("imbalance_duration_mae_risk_rank")
    try:
        rank_int = int(rank) if rank is not None and str(rank) != "<NA>" else None
    except (TypeError, ValueError):
        rank_int = None
    if rank_int == 1:
        return "recommended by joint imbalance risk score"
    if all(isinstance(item, int) and item <= 3 for item in [q99_rank, core_rank]):
        return "strong cumulative tail risk, check ramp/duration as secondary risks"
    if isinstance(ramp_rank, int) and ramp_rank <= 3 and isinstance(duration_rank, int) and duration_rank <= 3:
        return "good ramp-duration risk but cumulative tail risk needs review"
    return "not preferred under joint imbalance risk ranking"


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
    aux_cols = [m for m in AUXILIARY_REALISM_METRICS if any(m in row for row in main_rows)]
    lines.extend(
        [
            "## Evaluation Logic",
            "",
            RISK_EVALUATION_EXPLANATION,
            RISK_RANKING_EXPLANATION,
            "",
            "## Main Risk Comparison",
            "",
            markdown_table(main_rows, ["experiment_name", *PAPER_MAIN_METRICS, "risk_score", "risk_rank", "recommendation_reason"]),
            "",
            "## Ablation",
            "",
            markdown_table(ablation_rows, ["experiment_name", *PAPER_MAIN_METRICS, "risk_score", "risk_rank", "recommendation_reason"]),
            "",
            "## Auxiliary Realism Diagnostics",
            "",
            markdown_table(main_rows, ["experiment_name", *aux_cols]),
        ]
    )
    return "`n".join(lines)


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
        *[m for m in AUXILIARY_REALISM_METRICS if any(m in row for row in rows)],
        *RISK_LOWER_BETTER,
        *RISK_HIGHER_BETTER,
        "risk_score",
        "risk_rank",
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
        ["experiment_name", *PAPER_MAIN_METRICS, "risk_score", "risk_rank", "extreme_degree_match_rate", "recommendation_reason"],
    )
    write_csv(
        args.out_dir / "risk_main_compare.csv",
        sorted(main_rows, key=lambda row: float(row.get("risk_score") or 1e9)),
        ["experiment_name", *PAPER_MAIN_METRICS, "risk_score", "risk_rank", "extreme_degree_match_rate", "recommendation_reason"],
    )
    write_csv(
        args.out_dir / "auxiliary_global_stat_table.csv",
        main_rows,
        ["experiment_name", *STAT_METRICS],
    )
    write_csv(
        args.out_dir / "auxiliary_realism_metrics.csv",
        main_rows,
        ["experiment_name", *[m for m in AUXILIARY_REALISM_METRICS if any(m in row for row in main_rows)], "realism_check_pass"],
    )
    write_csv(
        args.out_dir / "ablation_paper_table.csv",
        ablation_rows,
        ["experiment_name", *PAPER_MAIN_METRICS, "risk_score", "risk_rank", "extreme_degree_match_rate", "recommendation_reason"],
    )

    markdown = build_markdown(load_dataset_context(args.dataset_summary), main_rows, ablation_rows)
    (args.out_dir / "result_summary.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
