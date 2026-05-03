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
    "cum_deficit_mae",
    "q95_cum_deficit_error",
    "q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

RISK_HIGHER_BETTER = [
    "extreme_degree_match_rate",
    "extreme_degree_adjacent_match_rate",
]

RISK_WEIGHTS = {
    "cum_deficit_mae": 2.0,
    "q95_cum_deficit_error": 1.5,
    "q99_cum_deficit_error": 2.0,
    "netload_ramp_max_mae": 1.0,
    "imbalance_duration_mae": 1.0,
    "extreme_degree_match_rate": 1.0,
    "extreme_degree_adjacent_match_rate": 0.5,
}

STAT_TOLERANCE = {
    "mean_wasserstein": 0.25,
    "mean_js": 0.25,
    "acf_mae": 0.30,
    "corr_matrix_error": 0.30,
}

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
    out["risk_metrics"] = json.dumps({m: out.get(m) for m in RISK_LOWER_BETTER + RISK_HIGHER_BETTER}, ensure_ascii=False)
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
            rows[idx][f"{metric}_risk_rank"] = rank
            rows[idx][f"{metric}_weighted_rank"] = rank * RISK_WEIGHTS[metric]
    for metric in RISK_HIGHER_BETTER:
        ranks = _rank_values(rows, metric, ascending=False)
        for idx, rank in ranks.items():
            rows[idx][f"{metric}_risk_rank"] = rank
            rows[idx][f"{metric}_weighted_rank"] = rank * RISK_WEIGHTS[metric]

    total_weight = sum(RISK_WEIGHTS.values())
    for row in rows:
        weighted_sum = 0.0
        used_weight = 0.0
        for metric, weight in RISK_WEIGHTS.items():
            rank = row.get(f"{metric}_risk_rank")
            if isinstance(rank, int):
                weighted_sum += rank * weight
                used_weight += weight
        row["risk_rank_score"] = round(weighted_sum / used_weight, 4) if used_weight else None

    best_stats = _baseline_best_stats(rows)
    for row in rows:
        penalty = 0.0
        max_excess = 0.0
        degradation_details = {}
        for metric, tolerance in STAT_TOLERANCE.items():
            value = row.get(metric)
            best = best_stats.get(metric)
            if not isinstance(value, float) or best is None:
                continue
            degradation = (value - best) / (abs(best) + 1e-8)
            excess = max(0.0, degradation - tolerance)
            row[f"{metric}_degradation"] = degradation
            degradation_details[metric] = round(degradation, 4)
            penalty += excess
            max_excess = max(max_excess, excess)
        row["statistical_degradation_score"] = round(max_excess, 4)
        row["statistical_penalty"] = round(penalty, 4)
        risk_score = row.get("risk_rank_score")
        row["final_risk_oriented_score"] = round(float(risk_score) + penalty, 4) if isinstance(risk_score, float) else None
        row["statistical_degradation_details"] = json.dumps(degradation_details, ensure_ascii=False)
        row["statistical_status"] = _statistical_status(row)
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
    status = row.get("statistical_status")
    risk_score = row.get("risk_rank_score")
    acf_excess = float(row.get("acf_mae_degradation") or 0.0) - STAT_TOLERANCE["acf_mae"]
    if status == "unacceptable":
        return "not recommended due to statistical degradation"
    if isinstance(risk_score, float) and risk_score <= 3.0 and status == "acceptable":
        return "risk metrics improved while statistical metrics remain acceptable"
    if isinstance(risk_score, float) and risk_score <= 3.0 and status == "slightly_worse":
        return "risk-preferred but temporal-continuity slightly worse" if acf_excess > 0 else "risk improved with slight statistical degradation"
    if status == "acceptable":
        return "good distribution but weak risk improvement"
    return "risk improved but ACF degradation is too large" if acf_excess > 0 else "mixed trade-off"


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
            "The proposed method is selected by a risk-first, statistics-constrained rule. Wasserstein, JS, ACF, and correlation matrix error are used to check that statistical realism is not obviously degraded; cumulative deficit, tail cumulative deficit, ramp, duration, and severity matching drive the risk-oriented score.",
            "",
            "## Main Comparison",
            "",
        ]
    )
    lines.append(markdown_table(main_rows, ["experiment_name", *STAT_METRICS, "cum_deficit_mae", "q99_cum_deficit_error", "netload_ramp_max_mae", "imbalance_duration_mae", "extreme_degree_match_rate", "final_risk_oriented_score", "statistical_status", "recommendation_reason"]))
    lines.extend(["", "## Ablation", ""])
    lines.append(markdown_table(ablation_rows, ["experiment_name", "cum_deficit_mae", "q95_cum_deficit_error", "q99_cum_deficit_error", "netload_ramp_max_mae", "imbalance_duration_mae", "corr_matrix_error", "extreme_degree_match_rate", "final_risk_oriented_score", "recommendation_reason"]))
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
    key_rows = sorted(rows, key=lambda row: float(row.get("final_risk_oriented_score") or 1e9))[:12]

    common_fields = [
        "group",
        "experiment_name",
        "status",
        "statistical_metrics",
        "risk_metrics",
        *STAT_METRICS,
        *RISK_LOWER_BETTER,
        *RISK_HIGHER_BETTER,
        "risk_rank_score",
        "statistical_degradation_score",
        "statistical_penalty",
        "final_risk_oriented_score",
        "statistical_status",
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
        ["experiment_name", *STAT_METRICS, "cum_deficit_mae", "q99_cum_deficit_error", "netload_ramp_max_mae", "imbalance_duration_mae", "extreme_degree_match_rate", "final_risk_oriented_score", "statistical_status", "recommendation_reason"],
    )
    write_csv(
        args.out_dir / "ablation_paper_table.csv",
        ablation_rows,
        ["experiment_name", "cum_deficit_mae", "q95_cum_deficit_error", "q99_cum_deficit_error", "netload_ramp_max_mae", "imbalance_duration_mae", "corr_matrix_error", "extreme_degree_match_rate", "final_risk_oriented_score", "statistical_status", "recommendation_reason"],
    )

    markdown = build_markdown(load_dataset_context(args.dataset_summary), main_rows, ablation_rows)
    (args.out_dir / "result_summary.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
