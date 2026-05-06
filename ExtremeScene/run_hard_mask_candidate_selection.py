from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_generation import EvalConfig, evaluate_generation
from evt_fit import EVTConfig, fit_evt_and_label
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from risk_metrics import batch_hard_risk_metrics


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"
DEFAULT_OUT_DIR = BASE_DIR / "outputs" / "hard_mask_candidate_selection"

PAPER_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "extreme_degree_match_rate",
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]


@dataclass
class HardMaskSelectionConfig:
    data_dir: str = str(DEFAULT_DATA_DIR)
    out_dir: str = str(DEFAULT_OUT_DIR)
    num_candidates: int = 20
    seed: int = 42
    tau_cum_strict: float = 0.5
    tau_ramp_strict: float = 0.8
    tau_cum_relaxed: float = 0.8
    tau_ramp_relaxed: float = 1.2
    n_segments_target_proxy: int = 1


BASE_MODELS: dict[str, Path] = {
    "C4_30_20_6": BASE_DIR / "outputs" / "evt_transfer_longstage" / "C4_evttransfer_more_30_20_6" / "models" / "proposed" / "best_risk_model.pt",
    "O5_over500": BASE_DIR / "outputs" / "duration_over_tuning" / "O5_over500" / "models" / "proposed" / "best_risk_model.pt",
}

EXPERIMENTS: list[dict[str, Any]] = [
    {"name": "H1_C4_K20_hard_duration", "base_model": "C4_30_20_6", "threshold_mode": "strict"},
    {"name": "H2_C4_K20_hard_duration_relaxed", "base_model": "C4_30_20_6", "threshold_mode": "relaxed"},
    {"name": "H3_O5_K20_hard_duration", "base_model": "O5_over500", "threshold_mode": "strict"},
    {"name": "H4_O5_K20_hard_duration_relaxed", "base_model": "O5_over500", "threshold_mode": "relaxed"},
]


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _safe_normalized_abs_error(value: np.ndarray, target: np.ndarray, global_scale: float) -> np.ndarray:
    scale = np.maximum(np.abs(target), float(global_scale) + 1e-6)
    return np.abs(value - target) / scale


def _segment_stats(mask: np.ndarray, delta_t_hours: float) -> tuple[np.ndarray, np.ndarray]:
    n, k, t = mask.shape
    n_segments = np.zeros((n, k), dtype=float)
    longest = np.zeros((n, k), dtype=float)
    for i in range(n):
        for j in range(k):
            cur = 0
            segs = 0
            best = 0
            for v in mask[i, j].astype(bool):
                if v:
                    if cur == 0:
                        segs += 1
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            n_segments[i, j] = float(segs)
            longest[i, j] = float(best) * float(delta_t_hours)
    return n_segments, longest


def _severity_from_generated_cum(cum_flat: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"cum_deficit": np.asarray(cum_flat, dtype=float)})
    labeled, _ = fit_evt_and_label(df, EVTConfig(metric_col="cum_deficit"))
    return labeled["severity_level"].fillna(0).astype(int).to_numpy()


def _write_generated_long(samples: np.ndarray, cond: pd.DataFrame, out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for i, row in cond.reset_index(drop=True).iterrows():
        for t in range(samples.shape[2]):
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "generated_id": f"G{i:05d}",
                    "t": int(t),
                    "load": float(samples[i, 0, t]),
                    "wind_power": float(samples[i, 1, t]),
                    "solar_power": float(samples[i, 2, t]),
                    "event_type": row["event_type"],
                    "month": int(row["month"]),
                    "duration_hours": float(row.get("duration_hours", np.nan)),
                    "severity_level": int(row["severity_level"]),
                    "extreme_prob": float(row["extreme_prob"]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "generated_samples_long.csv", index=False, encoding="utf-8-sig")


def _ensure_candidates(base_name: str, ckpt: Path, cfg: HardMaskSelectionConfig) -> Path:
    cache_dir = Path(cfg.out_dir) / "_candidate_cache" / f"{base_name}_K{cfg.num_candidates}"
    candidate_path = cache_dir / "generated_candidates.npy"
    if candidate_path.exists():
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)
    generate_from_checkpoint(
        GenerationConfig(
            checkpoint=str(ckpt),
            checkpoint_type="best-risk",
            data_dir=cfg.data_dir,
            out_dir=str(cache_dir),
            split="test",
            guidance_scale=1.0,
            num_candidates_per_condition=int(cfg.num_candidates),
            candidate_selection_mode="risk_target",
            candidate_risk_weights="cum:1.0,ramp:0.3,duration:0.3",
            save_generated_candidates=True,
        )
    )
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate generation failed: {candidate_path}")
    return cache_dir


def _select_candidates(
    candidates: np.ndarray,
    cond: pd.DataFrame,
    cfg: HardMaskSelectionConfig,
    threshold_mode: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    n, k, c, t = candidates.shape
    if c != 3:
        raise ValueError(f"Expected candidates [N,K,3,T], got {candidates.shape}")

    tau = pd.to_numeric(cond["imbalance_tau"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta_t = float(pd.to_numeric(cond.get("delta_t_hours", pd.Series([1.0])), errors="coerce").fillna(1.0).iloc[0])
    tau_flat = np.repeat(tau, k)
    flat = candidates.reshape(n * k, c, t)
    metrics = batch_hard_risk_metrics(flat, tau=tau_flat, delta_t_hours=delta_t)
    gen_cum = metrics["cum_deficit"].reshape(n, k)
    gen_ramp = metrics["netload_ramp_max"].reshape(n, k)
    gen_dur = metrics["imbalance_duration"].reshape(n, k)
    gen_sev = _severity_from_generated_cum(metrics["cum_deficit"]).reshape(n, k)

    net = candidates[:, :, 0, :] - candidates[:, :, 1, :] - candidates[:, :, 2, :]
    exceed_mask = net > tau[:, None, None]
    n_segments, longest_segment = _segment_stats(exceed_mask, delta_t_hours=delta_t)

    target_cum = pd.to_numeric(cond["cum_deficit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None]
    target_ramp = pd.to_numeric(cond["netload_ramp_max"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None]
    target_dur = pd.to_numeric(cond["imbalance_duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None]
    target_sev = pd.to_numeric(cond["severity_level"], errors="coerce").fillna(0).astype(int).to_numpy()[:, None]

    cum_scale = float(np.nanstd(target_cum) + 1e-6)
    ramp_scale = float(np.nanstd(target_ramp) + 1e-6)
    dur_scale = float(np.nanstd(target_dur) + 1e-6)
    cum_err = _safe_normalized_abs_error(gen_cum, target_cum, cum_scale)
    ramp_err = _safe_normalized_abs_error(gen_ramp, target_ramp, ramp_scale)
    dur_err = _safe_normalized_abs_error(gen_dur, target_dur, dur_scale)
    sev_abs = np.abs(gen_sev - target_sev)
    segment_penalty = (
        np.abs(n_segments - float(cfg.n_segments_target_proxy))
        + np.abs(longest_segment - target_dur) / (target_dur + 1.0)
    )
    score = (
        1.0 * dur_err
        + 0.3 * cum_err
        + 0.2 * ramp_err
        + 0.3 * sev_abs
        + 0.3 * segment_penalty
    )
    fallback_score = 1.0 * cum_err + 0.3 * ramp_err + 0.3 * dur_err + 0.3 * sev_abs

    strict_valid = (sev_abs <= 1) & (cum_err <= float(cfg.tau_cum_strict)) & (ramp_err <= float(cfg.tau_ramp_strict))
    relaxed_valid = (sev_abs <= 1) & (cum_err <= float(cfg.tau_cum_relaxed)) & (ramp_err <= float(cfg.tau_ramp_relaxed))

    selected = np.zeros((n,), dtype=int)
    rows: list[dict[str, Any]] = []
    fallback_count = 0
    chosen_valid_counts: list[int] = []
    strict_counts: list[int] = []
    relaxed_counts: list[int] = []

    for i in range(n):
        strict_idx = np.where(strict_valid[i])[0]
        relaxed_idx = np.where(relaxed_valid[i])[0]
        strict_counts.append(int(len(strict_idx)))
        relaxed_counts.append(int(len(relaxed_idx)))

        stage = threshold_mode
        fallback = 0
        if threshold_mode == "relaxed":
            valid_idx = relaxed_idx
            if len(valid_idx) == 0:
                valid_idx = np.arange(k)
                stage = "fallback"
                fallback = 1
        else:
            valid_idx = strict_idx
            if len(valid_idx) == 0:
                valid_idx = relaxed_idx
                stage = "relaxed_after_strict_empty"
            if len(valid_idx) == 0:
                valid_idx = np.arange(k)
                stage = "fallback"
                fallback = 1

        if fallback:
            local_scores = fallback_score[i, valid_idx]
            fallback_count += 1
        else:
            local_scores = score[i, valid_idx]
        local_choice = int(valid_idx[int(np.argmin(local_scores))])
        selected[i] = local_choice
        chosen_valid_counts.append(int(len(valid_idx)) if not fallback else 0)

        rows.append(
            {
                "sample_id": cond.loc[i, "sample_id"],
                "selected_candidate_idx": local_choice,
                "selected_stage": stage,
                "fallback": int(fallback),
                "strict_valid_candidates": int(len(strict_idx)),
                "relaxed_valid_candidates": int(len(relaxed_idx)),
                "chosen_valid_candidates": int(len(valid_idx)) if not fallback else 0,
                "selected_score": float(score[i, local_choice]),
                "score_duration_component": float(dur_err[i, local_choice]),
                "score_cum_component": float(0.3 * cum_err[i, local_choice]),
                "score_ramp_component": float(0.2 * ramp_err[i, local_choice]),
                "score_severity_component": float(0.3 * sev_abs[i, local_choice]),
                "score_segment_component": float(0.3 * segment_penalty[i, local_choice]),
                "target_cum_deficit": float(target_cum[i, 0]),
                "generated_cum_deficit": float(gen_cum[i, local_choice]),
                "cum_normalized_abs_error": float(cum_err[i, local_choice]),
                "target_netload_ramp_max": float(target_ramp[i, 0]),
                "generated_netload_ramp_max": float(gen_ramp[i, local_choice]),
                "ramp_normalized_abs_error": float(ramp_err[i, local_choice]),
                "target_imbalance_duration": float(target_dur[i, 0]),
                "generated_imbalance_duration": float(gen_dur[i, local_choice]),
                "duration_normalized_abs_error": float(dur_err[i, local_choice]),
                "target_severity_level": int(target_sev[i, 0]),
                "generated_severity_level": int(gen_sev[i, local_choice]),
                "generated_n_segments": float(n_segments[i, local_choice]),
                "target_n_segments_proxy": int(cfg.n_segments_target_proxy),
                "generated_longest_segment": float(longest_segment[i, local_choice]),
                "segment_penalty": float(segment_penalty[i, local_choice]),
                "candidate_score_min": float(score[i].min()),
                "candidate_score_mean": float(score[i].mean()),
            }
        )

    selected_samples = candidates[np.arange(n), selected].astype(np.float32)
    diagnostics = {
        "fallback_rate": float(fallback_count / max(n, 1)),
        "average_valid_candidates": float(np.mean(chosen_valid_counts)),
        "average_strict_valid_candidates": float(np.mean(strict_counts)),
        "average_relaxed_valid_candidates": float(np.mean(relaxed_counts)),
    }
    return selected_samples, pd.DataFrame(rows), diagnostics


def run_hard_mask_candidate_selection(cfg: HardMaskSelectionConfig) -> pd.DataFrame:
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond = pd.read_csv(data_dir / "cond_test.csv").reset_index(drop=True)
    meta = pd.read_csv(data_dir / "meta_test.csv").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for base_name, ckpt in BASE_MODELS.items():
        cache = _ensure_candidates(base_name, ckpt, cfg)
        candidates = np.load(cache / "generated_candidates.npy").astype(np.float32)
        for exp in [item for item in EXPERIMENTS if item["base_model"] == base_name]:
            exp_dir = out_dir / exp["name"]
            model_dir = exp_dir / "models" / "proposed"
            eval_dir = exp_dir / "evaluations" / "proposed"
            model_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            selected, selection_df, diag = _select_candidates(candidates, cond, cfg, str(exp["threshold_mode"]))
            np.save(model_dir / "generated_samples.npy", selected)
            np.save(model_dir / "generated_candidates.npy", candidates)
            selection_df.to_csv(model_dir / "candidate_selection_summary.csv", index=False, encoding="utf-8-sig")
            cond.to_csv(model_dir / "selected_conditions.csv", index=False, encoding="utf-8-sig")
            meta.to_csv(model_dir / "selected_meta.csv", index=False, encoding="utf-8-sig")
            _write_generated_long(selected, cond, model_dir)

            generation_summary = {
                "num_generated": int(len(selected)),
                "split": "test",
                "base_model": base_name,
                "checkpoint_path": str(ckpt),
                "num_candidates_per_condition": int(cfg.num_candidates),
                "candidate_selection_mode": "hard_mask_duration",
                "threshold_mode": exp["threshold_mode"],
                "tau_cum_strict": cfg.tau_cum_strict,
                "tau_ramp_strict": cfg.tau_ramp_strict,
                "tau_cum_relaxed": cfg.tau_cum_relaxed,
                "tau_ramp_relaxed": cfg.tau_ramp_relaxed,
                **diag,
            }
            (model_dir / "generation_summary.json").write_text(json.dumps(generation_summary, ensure_ascii=False, indent=2), encoding="utf-8")

            event_mask_path = data_dir / "event_mask_test.npy"
            eval_summary = evaluate_generation(
                EvalConfig(
                    real=str(data_dir / "X_test.npy"),
                    generated=str(model_dir / "generated_samples.npy"),
                    cond=str(data_dir / "cond_test.csv"),
                    meta=str(data_dir / "meta_test.csv"),
                    out_dir=str(eval_dir),
                    model_name=exp["name"],
                    event_mask=str(event_mask_path) if event_mask_path.exists() else None,
                )
            )
            row = {
                "experiment_name": exp["name"],
                "base_model": base_name,
                "threshold_mode": exp["threshold_mode"],
                **generation_summary,
            }
            row.update(eval_summary["metrics"])
            (exp_dir / "metrics_row.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            pd.DataFrame([row]).to_csv(exp_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
            rows.append(row)

    summary = pd.DataFrame(rows)
    refs: list[dict[str, Any]] = []
    ref_paths = {
        "REF_E0_proposed": out_dir.parent / "main_compare_e0_fixed" / "paper_summary" / "main_compare_paper_table.csv",
        "REF_C4_30_20_6": out_dir.parent / "evt_transfer_longstage" / "C4_evttransfer_more_30_20_6" / "metrics_row.json",
        "REF_O5_over500": out_dir.parent / "duration_over_tuning" / "O5_over500" / "metrics_row.json",
    }
    main_path = ref_paths["REF_E0_proposed"]
    if main_path.exists():
        main = pd.read_csv(main_path)
        proposed = main.loc[main["experiment_name"].eq("proposed")]
        if not proposed.empty:
            refs.append({"experiment_name": "REF_E0_proposed", "base_model": "reference", **{m: proposed.iloc[0].get(m) for m in PAPER_METRICS}})
    for label in ["REF_C4_30_20_6", "REF_O5_over500"]:
        path = ref_paths[label]
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            refs.append({"experiment_name": label, "base_model": "reference", **{m: data.get(m) for m in PAPER_METRICS}})
    full = pd.concat([pd.DataFrame(refs), summary], ignore_index=True, sort=False)
    for base_name, base_exp in [("E0", "REF_E0_proposed"), ("C4", "REF_C4_30_20_6"), ("O5", "REF_O5_over500")]:
        base = full.loc[full["experiment_name"].eq(base_exp)]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for metric in PAPER_METRICS:
            full[f"delta_{metric}_vs_{base_name}"] = pd.to_numeric(full[metric], errors="coerce") - float(base_row[metric])
            full[f"pct_{metric}_vs_{base_name}"] = (pd.to_numeric(full[metric], errors="coerce") / max(abs(float(base_row[metric])), 1e-12) - 1.0) * 100.0

    full.to_csv(out_dir / "hard_mask_candidate_selection_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(full, out_dir)
    return full


def _write_report(df: pd.DataFrame, out_dir: Path) -> None:
    show_cols = [
        "experiment_name",
        "base_model",
        "threshold_mode",
        "fallback_rate",
        "average_valid_candidates",
        *PAPER_METRICS,
        "pct_q99_cum_deficit_error_vs_C4",
        "pct_core_q99_cum_deficit_error_vs_C4",
        "pct_imbalance_duration_mae_vs_C4",
        "pct_q99_cum_deficit_error_vs_O5",
        "pct_core_q99_cum_deficit_error_vs_O5",
        "pct_imbalance_duration_mae_vs_O5",
    ]
    candidates = df[df["experiment_name"].astype(str).str.startswith("H")].copy()
    recommendation = "none"
    if not candidates.empty:
        flags = []
        for _, row in candidates.iterrows():
            base = str(row.get("base_model", ""))
            duration_ok = float(row.get(f"pct_imbalance_duration_mae_vs_{base.split('_')[0] if base else 'C4'}", np.nan)) <= -20.0
            if base.startswith("C4"):
                q99_ok = float(row.get("pct_q99_cum_deficit_error_vs_C4", np.inf)) <= 30.0
                core_ok = float(row.get("core_q99_cum_deficit_error", np.inf)) <= 3.0
                acf_ok = float(row.get("highrisk_acf_mae", np.inf)) <= float(df.loc[df["experiment_name"].eq("REF_C4_30_20_6"), "highrisk_acf_mae"].iloc[0]) * 1.10
                degree_ok = float(row.get("extreme_degree_match_rate", -np.inf)) >= float(df.loc[df["experiment_name"].eq("REF_C4_30_20_6"), "extreme_degree_match_rate"].iloc[0])
            elif base.startswith("O5"):
                q99_ok = float(row.get("pct_q99_cum_deficit_error_vs_O5", np.inf)) <= 30.0
                core_ok = float(row.get("core_q99_cum_deficit_error", np.inf)) <= 3.0
                acf_ok = float(row.get("highrisk_acf_mae", np.inf)) <= float(df.loc[df["experiment_name"].eq("REF_O5_over500"), "highrisk_acf_mae"].iloc[0]) * 1.10
                degree_ok = float(row.get("extreme_degree_match_rate", -np.inf)) >= float(df.loc[df["experiment_name"].eq("REF_O5_over500"), "extreme_degree_match_rate"].iloc[0])
            else:
                q99_ok = core_ok = acf_ok = degree_ok = False
            ramp_ok = float(row.get("netload_ramp_max_mae", np.inf)) <= 7.0
            flags.append(duration_ok and q99_ok and core_ok and ramp_ok and acf_ok and degree_ok)
        candidates["meets_recommendation_rule"] = flags
        eligible = candidates[candidates["meets_recommendation_rule"]]
        pick_from = eligible if not eligible.empty else candidates
        recommendation = str(pick_from.sort_values(["imbalance_duration_mae", "q99_cum_deficit_error"]).iloc[0]["experiment_name"])

    lines = [
        "# Hard Mask Candidate Selection Report",
        "",
        "Selection uses only generated candidates and cond_test.csv targets. X_test is used only after selection for evaluation.",
        "",
        "## Summary",
        df[[c for c in show_cols if c in df.columns]].to_markdown(index=False),
        "",
        f"recommended_by_rule_or_lowest_duration: `{recommendation}`",
        "",
        "Recommendation rules: duration improves by at least 20%, q99 degradation <=30% vs the same base model, core_q99<=3.0, ramp<=7.0, highrisk_acf not obviously worse, and extreme_degree_match_rate does not decrease.",
    ]
    (out_dir / "hard_mask_candidate_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run_hard_mask_candidate_selection(HardMaskSelectionConfig())
    cols = ["experiment_name", "base_model", "fallback_rate", "average_valid_candidates", *PAPER_METRICS]
    print(result[[c for c in cols if c in result.columns]].to_string(index=False))
