from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STAT_THRESHOLDS = {
    "mean_wasserstein": 1.25,
    "mean_js": 1.25,
    "acf_mae": 1.30,
    "corr_matrix_error": 1.30,
}

SUMMARY_COLS = [
    "model_name",
    "checkpoint_type_used_for_generation",
    "checkpoint_stage_used_for_generation",
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
    "cum_deficit_mae",
    "q95_cum_deficit_error",
    "q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
    "extreme_degree_match_rate",
    "extreme_degree_adjacent_match_rate",
]

CORE_COLS = [
    "core_cum_deficit_mae",
    "core_q95_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "core_netload_ramp_max_mae",
    "core_imbalance_duration_mae",
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: pd.Series, name: str) -> float:
    if name not in row or pd.isna(row[name]):
        return float("nan")
    return float(row[name])


def _lt(a: float, b: float) -> bool:
    return np.isfinite(a) and np.isfinite(b) and a < b


def _ratio_ok(proposed: float, baseline: float, ratio: float) -> bool:
    if not np.isfinite(proposed) or not np.isfinite(baseline):
        return False
    if abs(baseline) <= 1e-12:
        return proposed <= ratio * 1e-12
    return proposed <= ratio * baseline


def _generated_allclose(out_dir: Path) -> tuple[bool, str]:
    p_path = out_dir / "models" / "proposed" / "generated_samples.npy"
    n_path = out_dir / "models" / "no_risk_loss" / "generated_samples.npy"
    if not p_path.exists() or not n_path.exists():
        return False, "generated sample file missing"
    proposed = np.load(p_path)
    no_risk = np.load(n_path)
    if proposed.shape != no_risk.shape:
        return False, f"different generated sample shapes: {proposed.shape} vs {no_risk.shape}"
    return bool(np.allclose(proposed, no_risk)), f"shape={proposed.shape}"


def _write_tuning_suggestions(out_dir: Path) -> None:
    text = """# Tuning Suggestions

Priority order:

1. Use final_model.pt
   checkpoint_type=final

2. Strengthen Stage 3
   stage1=12
   stage2=10
   stage3=4
   lambda_risk=0.04

3. Slightly increase risk weight
   stage1=12
   stage2=10
   stage3=2
   lambda_risk=0.06

4. Make cumulative deficit more dominant
   stage1=12
   stage2=10
   stage3=2
   lambda_risk=0.04
   lambda_cum=1.5
   lambda_ramp=0.15
   lambda_dur=0.25
"""
    (out_dir / "tuning_suggestions.md").write_text(text, encoding="utf-8")


def build_risk_check(data_dir: str | Path, out_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    metrics_path = out_dir / "evaluations" / "all_model_metrics.csv"
    metrics = pd.read_csv(metrics_path)

    rows = []
    warnings: list[str] = []
    event_mask_exists = (data_dir / "event_mask_test.npy").exists()
    if not event_mask_exists:
        warnings.append("event_mask_test.npy is missing; core_* metrics were skipped.")

    for model_name in ["proposed", "no_risk_loss"]:
        model_metrics = metrics.loc[metrics["model_name"] == model_name]
        if model_metrics.empty:
            warnings.append(f"{model_name} is missing from all_model_metrics.csv.")
            continue
        metric_row = model_metrics.iloc[0]
        gen_summary = _read_json(out_dir / "models" / model_name / "generation_summary.json")
        train_summary = _read_json(out_dir / "models" / model_name / "summary.json")
        row = {
            "model_name": model_name,
            "checkpoint_type_used_for_generation": gen_summary.get("resolved_checkpoint_type"),
            "checkpoint_stage_used_for_generation": gen_summary.get("checkpoint_stage"),
            "checkpoint_path_used_for_generation": gen_summary.get("checkpoint_path"),
            "lambda_risk_used": train_summary.get("lambda_risk_used"),
        }
        for col in SUMMARY_COLS[3:]:
            row[col] = _metric(metric_row, col)
        if event_mask_exists:
            for col in CORE_COLS:
                row[col] = _metric(metric_row, col)
        rows.append(row)

        if model_name == "proposed" and train_summary.get("best_risk_model_epoch") is None:
            warnings.append("proposed best_risk_model_epoch is empty; Stage 3 risk checkpoint may not have been saved.")
        if model_name == "proposed" and gen_summary.get("checkpoint_stage") != "stage3_risk":
            warnings.append("proposed generation did not use a stage3_risk checkpoint.")
        if model_name == "proposed" and float(train_summary.get("lambda_risk_used", 0.0) or 0.0) <= 0.0:
            warnings.append("proposed lambda_risk_used is not positive.")
        if model_name == "no_risk_loss" and float(train_summary.get("lambda_risk_used", -1.0) or 0.0) != 0.0:
            warnings.append("no_risk_loss lambda_risk_used is not 0.0.")
        warnings.extend([f"{model_name}: {w}" for w in gen_summary.get("warnings", [])])

    summary_df = pd.DataFrame(rows)
    ordered_cols = SUMMARY_COLS + (CORE_COLS if event_mask_exists else [])
    existing_cols = [c for c in ordered_cols if c in summary_df.columns]
    extra_cols = [c for c in summary_df.columns if c not in existing_cols]
    summary_df[existing_cols + extra_cols].to_csv(out_dir / "risk_check_summary.csv", index=False, encoding="utf-8-sig")

    if set(summary_df["model_name"]) >= {"proposed", "no_risk_loss"}:
        p = summary_df.set_index("model_name").loc["proposed"]
        n = summary_df.set_index("model_name").loc["no_risk_loss"]
        cum_better = _lt(float(p["cum_deficit_mae"]), float(n["cum_deficit_mae"]))
        q99_better = _lt(float(p["q99_cum_deficit_error"]), float(n["q99_cum_deficit_error"]))
        dur_better = _lt(float(p["imbalance_duration_mae"]), float(n["imbalance_duration_mae"]))
        ramp_better = _lt(float(p["netload_ramp_max_mae"]), float(n["netload_ramp_max_mae"]))
        stat_ok = all(
            _ratio_ok(float(p[col]), float(n[col]), threshold)
            for col, threshold in STAT_THRESHOLDS.items()
            if col in p and col in n
        )
        identical, identical_detail = _generated_allclose(out_dir)
        same_checkpoint = str(p.get("checkpoint_path_used_for_generation")) == str(n.get("checkpoint_path_used_for_generation"))
        if identical:
            verdict = "possible checkpoint or output reuse issue"
            warnings.append("generated samples are identical; possible checkpoint or seed/output reuse issue.")
        elif (cum_better and q99_better) or (cum_better and dur_better) or (q99_better and dur_better):
            verdict = "risk improved with acceptable statistical degradation" if stat_ok else "risk improved but statistical degradation needs review"
        else:
            verdict = "risk loss not effective, need to inspect Stage 3 and lambda_risk"

        if same_checkpoint:
            warnings.append("proposed and no_risk_loss used the same checkpoint_path.")

        report_lines = [
            "# Proposed Risk Check Report",
            "",
            f"Verdict: {verdict}",
            "",
            "## Risk Indicators",
            "",
            f"- cum_deficit_mae improved: {cum_better}",
            f"- q99_cum_deficit_error improved: {q99_better}",
            f"- imbalance_duration_mae improved: {dur_better}",
            f"- netload_ramp_max_mae improved: {ramp_better}",
            "",
            "## Statistical Degradation Check",
            "",
            f"- acceptable statistical degradation: {stat_ok}",
            f"- mean_wasserstein ratio threshold: {STAT_THRESHOLDS['mean_wasserstein']}",
            f"- mean_js ratio threshold: {STAT_THRESHOLDS['mean_js']}",
            f"- acf_mae ratio threshold: {STAT_THRESHOLDS['acf_mae']}",
            f"- corr_matrix_error ratio threshold: {STAT_THRESHOLDS['corr_matrix_error']}",
            "",
            "## Reuse Checks",
            "",
            f"- generated samples identical: {identical} ({identical_detail})",
            f"- checkpoint paths identical: {same_checkpoint}",
            "",
            "## Warnings",
            "",
        ]
        report_lines.extend([f"- {w}" for w in warnings] or ["- none"])
        report_lines.append("")
        (out_dir / "risk_check_report.md").write_text("\n".join(report_lines), encoding="utf-8")

        if verdict != "risk improved with acceptable statistical degradation":
            _write_tuning_suggestions(out_dir)
        else:
            stale_tuning_path = out_dir / "tuning_suggestions.md"
            if stale_tuning_path.exists():
                stale_tuning_path.unlink()

        return {
            "verdict": verdict,
            "cum_deficit_mae_improved": cum_better,
            "q99_cum_deficit_error_improved": q99_better,
            "imbalance_duration_mae_improved": dur_better,
            "netload_ramp_max_mae_improved": ramp_better,
            "statistical_degradation_acceptable": stat_ok,
            "generated_samples_identical": identical,
            "checkpoint_paths_identical": same_checkpoint,
            "warnings": warnings,
        }

    (out_dir / "risk_check_report.md").write_text("# Proposed Risk Check Report\n\nVerdict: incomplete metrics.\n", encoding="utf-8")
    return {"verdict": "incomplete metrics", "warnings": warnings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build proposed vs no_risk_loss risk check outputs.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build_risk_check(args.data_dir, args.out_dir), ensure_ascii=False, indent=2, default=str))
