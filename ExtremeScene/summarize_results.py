from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


CORE_METRICS = [
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
    "cum_deficit_mae",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

COMPARE_ORDER = [
    "traditional_gaussian_copula",
    "plain_diffusion_baseline",
    "enhanced_gan",
    "proposed",
    "no_evt",
    "no_risk_loss",
    "no_month",
    "flat_condition",
]

KEY_COMPARE_ROWS = [
    "traditional_gaussian_copula",
    "plain_diffusion_baseline",
    "enhanced_gan",
    "proposed",
]

KEY_PROPOSED_ROWS = [
    "proposed_g1p0",
    "tail_only_s12_s210_s30_g1p0",
    "risk_light_s12_s28_s34_g1p0",
    "low_lr_risk_light_g1p0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize model comparison and tuning results.")
    parser.add_argument(
        "--compare",
        type=Path,
        default=Path("outputs/real_singleton_2018_2022_compare/evaluations/all_model_metrics.csv"),
        help="Main model comparison CSV.",
    )
    parser.add_argument(
        "--guidance",
        type=Path,
        default=Path("outputs/guidance_sweep_proposed/guidance_summary.csv"),
        help="Guidance sweep CSV for the proposed checkpoint.",
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        default=Path("outputs/proposed_tuning_variants/tuning_summary.csv"),
        help="Retraining variant summary CSV.",
    )
    parser.add_argument(
        "--tuning-guidance",
        type=Path,
        default=Path("outputs/proposed_tuning_variants_guidance/summary.csv"),
        help="Guidance sweep CSV for retrained variants.",
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=Path("outputs/real_singleton_2018_2022_compare/dataset/dataset_summary.json"),
        help="Optional dataset summary JSON for context.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/result_summary"),
        help="Output directory for summary tables.",
    )
    return parser.parse_args()


def safe_float(value: str | None) -> float | None:
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_row(
    row: dict[str, str],
    group: str,
    experiment_name: str,
    source_file: Path,
    note: str = "",
) -> dict[str, object]:
    normalized: dict[str, object] = {
        "group": group,
        "experiment_name": experiment_name,
        "source_file": str(source_file),
        "note": note,
    }
    for metric in CORE_METRICS + [
        "q95_cum_deficit_error",
        "q99_cum_deficit_error",
        "extreme_degree_match_rate",
        "extreme_degree_adjacent_match_rate",
    ]:
        normalized[metric] = safe_float(row.get(metric))
    return normalized


def build_full_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for row in read_csv_rows(args.compare):
        experiment_name = row.get("model_name", "").strip()
        if not experiment_name:
            continue
        rows.append(
            normalize_row(
                row=row,
                group="main_compare",
                experiment_name=experiment_name,
                source_file=args.compare,
            )
        )

    for row in read_csv_rows(args.guidance):
        experiment_name = row.get("model_name", "").strip()
        if not experiment_name:
            continue
        rows.append(
            normalize_row(
                row=row,
                group="proposed_guidance_sweep",
                experiment_name=experiment_name,
                source_file=args.guidance,
                note="same checkpoint, different guidance_scale",
            )
        )

    for row in read_csv_rows(args.tuning):
        experiment_name = row.get("model_name", "").strip()
        if not experiment_name:
            continue
        rows.append(
            normalize_row(
                row=row,
                group="proposed_retrain",
                experiment_name=experiment_name,
                source_file=args.tuning,
                note="retrained proposed variant",
            )
        )

    for row in read_csv_rows(args.tuning_guidance):
        experiment_name = row.get("model_name", "").strip()
        if not experiment_name:
            continue
        rows.append(
            normalize_row(
                row=row,
                group="proposed_retrain_guidance",
                experiment_name=experiment_name,
                source_file=args.tuning_guidance,
                note="retrained proposed variant with altered guidance_scale",
            )
        )

    return rows


def attach_metric_ranks(rows: list[dict[str, object]], metrics: Iterable[str]) -> None:
    for metric in metrics:
        ranked = sorted(
            [row for row in rows if isinstance(row.get(metric), float)],
            key=lambda item: item[metric],  # type: ignore[index]
        )
        for idx, row in enumerate(ranked, start=1):
            row[f"{metric}_rank"] = idx

    for row in rows:
        ranks = [row.get(f"{metric}_rank") for metric in metrics if row.get(f"{metric}_rank") is not None]
        row["avg_rank_core_metrics"] = round(sum(ranks) / len(ranks), 3) if ranks else None


def select_key_rows(full_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    compare_lookup = {row["experiment_name"]: row for row in full_rows if row["group"] == "main_compare"}
    proposed_lookup = {
        row["experiment_name"]: row
        for row in full_rows
        if row["group"] in {"proposed_guidance_sweep", "proposed_retrain"}
    }

    selected: list[dict[str, object]] = []
    for name in KEY_COMPARE_ROWS:
        row = compare_lookup.get(name)
        if row:
            selected.append(row)

    for name in KEY_PROPOSED_ROWS:
        row = proposed_lookup.get(name)
        if row and row not in selected:
            selected.append(row)

    return selected


def load_dataset_context(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_number(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    headers = ["name" if col == "experiment_name" else col for col in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [fmt_number(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(
    dataset_context: dict[str, object],
    compare_rows: list[dict[str, object]],
    key_rows: list[dict[str, object]],
) -> str:
    lines: list[str] = ["# Result Summary", ""]

    if dataset_context:
        event_counts = dataset_context.get("event_type_counts", {})
        split_counts = dataset_context.get("split_counts", {})
        lines.extend(
            [
                "## Dataset Context",
                "",
                f"- n_samples: {dataset_context.get('n_samples', '-')}",
                f"- split_counts: {split_counts}",
                f"- event_type_counts: {json.dumps(event_counts, ensure_ascii=False)}",
                "",
            ]
        )

    lines.extend(["## Main Comparison", ""])
    lines.append(
        markdown_table(
            compare_rows,
            ["experiment_name"] + CORE_METRICS + ["avg_rank_core_metrics"],
        )
    )
    lines.append("")

    lines.extend(["## Proposed Tuning Snapshot", ""])
    lines.append(
        markdown_table(
            key_rows,
            ["experiment_name"] + CORE_METRICS + ["avg_rank_core_metrics"],
        )
    )
    lines.append("")
    lines.extend(
        [
            "## Recommendation",
            "",
            "- Recommended default: `proposed_g1p0` for the best overall distribution and cum_deficit trade-off without retraining.",
            "- If you care more about ACF, ramp, and duration balance: `tail_only_s12_s210_s30_g1p0`.",
            "- If you want stronger risk-side metrics and can accept weaker distribution fidelity: `risk_light_s12_s28_s34_g1p0`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rows = build_full_rows(args)
    attach_metric_ranks(full_rows, CORE_METRICS)

    compare_rows = [row for row in full_rows if row["group"] == "main_compare"]
    compare_rows.sort(key=lambda row: COMPARE_ORDER.index(row["experiment_name"]) if row["experiment_name"] in COMPARE_ORDER else 999)

    key_rows = select_key_rows(full_rows)

    csv_fields = [
        "group",
        "experiment_name",
        *CORE_METRICS,
        "q95_cum_deficit_error",
        "q99_cum_deficit_error",
        "extreme_degree_match_rate",
        "extreme_degree_adjacent_match_rate",
        *[f"{metric}_rank" for metric in CORE_METRICS],
        "avg_rank_core_metrics",
        "note",
        "source_file",
    ]

    write_csv(out_dir / "complete_result_summary.csv", full_rows, csv_fields)
    write_csv(out_dir / "main_compare_summary.csv", compare_rows, csv_fields)
    write_csv(out_dir / "key_result_summary.csv", key_rows, csv_fields)

    dataset_context = load_dataset_context(args.dataset_summary)
    markdown = build_markdown(dataset_context, compare_rows, key_rows)
    (out_dir / "result_summary.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
