from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from hierarchical_diffusion import (
    ConditionNormalizers,
    DiffusionScheduler,
    HierarchicalConditionalUNet1D,
    apply_physical_projection,
    build_condition_bundle,
    denormalize_x_np,
    sample_sequences,
)
from risk_metrics import batch_hard_risk_metrics
from risk_classifier import RiskClassifier1D


@dataclass
class GenerationConfig:
    checkpoint: Optional[str]
    data_dir: str
    out_dir: str
    split: str = "test"
    num_samples: Optional[int] = None
    event_type: Optional[str] = None
    month: Optional[int] = None
    severity_level: Optional[int] = None
    duration_hours: Optional[float] = None
    guidance_scale: Optional[float] = None
    checkpoint_type: str = "best"
    num_candidates_per_condition: int = 1
    candidate_selection_mode: str = "none"
    candidate_risk_weights: str = "cum:1.0,ramp:0.3,duration:0.3"
    save_generated_candidates: bool = True
    risk_guidance_mode: str = "none"
    risk_classifier_path: Optional[str] = None
    risk_guidance_scale: float = 0.0
    risk_guidance_start_step_ratio: float = 0.5
    risk_guidance_interval: int = 5


def _checkpoint_name(checkpoint_type: str) -> str:
    mapping = {"best": "best_model.pt", "best-risk": "best_risk_model.pt", "final": "final_model.pt"}
    if checkpoint_type not in mapping:
        raise ValueError(f"Unsupported checkpoint_type: {checkpoint_type}")
    return mapping[checkpoint_type]


def _resolve_checkpoint(cfg: GenerationConfig, out_dir: Path) -> tuple[Path, str, list[str]]:
    if cfg.checkpoint:
        return Path(cfg.checkpoint), cfg.checkpoint_type, []
    ckpt_path = out_dir / _checkpoint_name(cfg.checkpoint_type)
    if ckpt_path.exists():
        return ckpt_path, cfg.checkpoint_type, []
    warnings: list[str] = []
    if not ckpt_path.exists() and cfg.checkpoint_type == "best-risk":
        best_fallback = out_dir / "best_model.pt"
        if best_fallback.exists():
            warnings.append("best_risk_model.pt was not found; fell back to best_model.pt.")
            return best_fallback, "best-risk-fallback-best", warnings
        final_fallback = out_dir / "final_model.pt"
        if final_fallback.exists():
            warnings.append("best_risk_model.pt and best_model.pt were not found; fell back to final_model.pt.")
            return final_fallback, "best-risk-fallback-final", warnings
    return ckpt_path, cfg.checkpoint_type, warnings


def _load_condition_frame(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cond_df = pd.read_csv(data_dir / f"cond_{split}.csv")
    meta_df = pd.read_csv(data_dir / f"meta_{split}.csv")
    return cond_df, meta_df


def _parse_candidate_risk_weights(text: str) -> dict[str, float]:
    weights = {"cum": 1.0, "ramp": 0.3, "duration": 0.3}
    if not text:
        return weights
    for item in str(text).split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        weights[key.strip()] = float(value)
    return weights


def _condition_int_tensor(frame: pd.DataFrame, column: str, device: torch.device) -> torch.Tensor:
    if column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0).clip(0, 3).to_numpy()
    else:
        values = np.zeros((len(frame),), dtype=np.int64)
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _select_risk_guided_candidates(
    candidates: np.ndarray,
    selected_cond: pd.DataFrame,
    out_dir: Path,
    weights: dict[str, float],
) -> tuple[np.ndarray, pd.DataFrame]:
    n, k = candidates.shape[:2]
    tau = selected_cond["imbalance_tau"].astype(float).to_numpy() if "imbalance_tau" in selected_cond else np.zeros((n,), dtype=float)
    delta_t = float(selected_cond["delta_t_hours"].astype(float).iloc[0]) if "delta_t_hours" in selected_cond else 1.0
    flat = candidates.reshape(n * k, candidates.shape[2], candidates.shape[3])
    tau_rep = np.repeat(tau, k)
    metrics = batch_hard_risk_metrics(flat, tau=tau_rep, delta_t_hours=delta_t)
    gen_cum = metrics["cum_deficit"].reshape(n, k)
    gen_ramp = metrics["netload_ramp_max"].reshape(n, k)
    gen_dur = metrics["imbalance_duration"].reshape(n, k)

    target_cum = selected_cond["cum_deficit"].astype(float).to_numpy()[:, None]
    target_ramp = selected_cond["netload_ramp_max"].astype(float).to_numpy()[:, None]
    target_dur = selected_cond["imbalance_duration"].astype(float).to_numpy()[:, None]
    cum_scale = np.maximum(np.abs(target_cum), np.nanstd(target_cum) + 1e-6)
    ramp_scale = np.maximum(np.abs(target_ramp), np.nanstd(target_ramp) + 1e-6)
    dur_scale = np.maximum(np.abs(target_dur), np.nanstd(target_dur) + 1e-6)
    score = (
        float(weights.get("cum", 1.0)) * np.abs(gen_cum - target_cum) / cum_scale
        + float(weights.get("ramp", 0.3)) * np.abs(gen_ramp - target_ramp) / ramp_scale
        + float(weights.get("duration", 0.3)) * np.abs(gen_dur - target_dur) / dur_scale
    )
    best_idx = score.argmin(axis=1)
    selected = candidates[np.arange(n), best_idx]
    rows = []
    for i in range(n):
        j = int(best_idx[i])
        rows.append(
            {
                "sample_id": selected_cond.loc[i, "sample_id"],
                "selected_candidate_idx": j,
                "selected_score": float(score[i, j]),
                "target_cum_deficit": float(target_cum[i, 0]),
                "generated_cum_deficit": float(gen_cum[i, j]),
                "target_netload_ramp_max": float(target_ramp[i, 0]),
                "generated_netload_ramp_max": float(gen_ramp[i, j]),
                "target_imbalance_duration": float(target_dur[i, 0]),
                "generated_imbalance_duration": float(gen_dur[i, j]),
                "candidate_score_min": float(score[i].min()),
                "candidate_score_mean": float(score[i].mean()),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "candidate_selection_summary.csv", index=False, encoding="utf-8-sig")
    return selected.astype(np.float32), summary_df


def _filter_conditions(cond_df: pd.DataFrame, cfg: GenerationConfig) -> tuple[pd.DataFrame, str]:
    out = cond_df.copy()
    if cfg.event_type is not None:
        out = out[out["event_type"] == cfg.event_type]
    if cfg.month is not None:
        out = out[out["month"].astype(int) == int(cfg.month)]
    if cfg.severity_level is not None:
        out = out[out["severity_level"].astype(int) == int(cfg.severity_level)]

    duration_mode = "not_requested"
    if cfg.duration_hours is not None and not out.empty:
        duration = pd.to_numeric(out["duration_hours"], errors="coerce")
        nearest_idx = (duration - float(cfg.duration_hours)).abs().sort_values().index
        out = out.loc[nearest_idx].copy()
        out.iloc[0, out.columns.get_loc("duration_hours")] = float(cfg.duration_hours)
        out = out.iloc[:1].copy()
        duration_mode = "matched_nearest_and_overwritten"

    if cfg.num_samples is not None and len(out) > cfg.num_samples:
        out = out.iloc[: cfg.num_samples].copy()
    return out, duration_mode


def generate_from_checkpoint(cfg: GenerationConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, resolved_checkpoint_type, warnings = _resolve_checkpoint(cfg, out_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_dir = Path(cfg.data_dir)

    cond_df_full, meta_df_full = _load_condition_frame(data_dir, cfg.split)
    selected_cond, duration_mode = _filter_conditions(cond_df_full, cfg)
    if selected_cond.empty:
        raise ValueError("No conditions matched the requested filters.")
    selected_meta = meta_df_full.loc[selected_cond.index].reset_index(drop=True)
    selected_cond = selected_cond.reset_index(drop=True)

    cond_normalizers = ConditionNormalizers(**ckpt["condition_normalizers"])
    arrays, condition_meta = build_condition_bundle(
        selected_cond,
        selected_meta,
        seq_len=int(ckpt["seq_len"]),
        ablation=str(ckpt["train_config"]["ablation"]),
        normalizers=cond_normalizers,
        expected_event_types=len(ckpt["condition_meta"]["background"]["event_onehot"]),
        use_risk_profile_condition=bool(ckpt["train_config"].get("use_risk_profile_condition", False)),
        use_mask_condition=bool(ckpt["train_config"].get("use_mask_condition", False)),
        use_ramp_event_condition=bool(ckpt["train_config"].get("use_ramp_event_condition", False)),
        ramp_event_condition_scale=float(ckpt["train_config"].get("ramp_event_condition_scale", 1.0)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_dims = ckpt["cond_dims"]
    model = HierarchicalConditionalUNet1D(
        in_channels=int(ckpt["train_config"]["in_channels"]),
        base_channels=int(ckpt["train_config"]["base_channels"]),
        time_dim=int(ckpt["train_config"]["time_emb_dim"]),
        cond_dim=int(ckpt["train_config"]["cond_emb_dim"]),
        bg_dim=int(cond_dims["background"]),
        proc_dim=int(cond_dims["process"]),
        risk_dim=int(cond_dims["risk"]),
        flat_condition=bool(ckpt["flat_condition"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scheduler = DiffusionScheduler(
        steps=int(ckpt["train_config"]["diffusion_steps"]),
        beta_start=float(ckpt["train_config"]["beta_start"]),
        beta_end=float(ckpt["train_config"]["beta_end"]),
        device=device,
    ).to(device)

    bg = torch.from_numpy(arrays["background"]).to(device)
    proc = torch.from_numpy(arrays["process"]).to(device)
    risk = torch.from_numpy(arrays["risk"]).to(device)
    day_mask = torch.from_numpy(arrays["day_mask"]).to(device)
    guidance = float(cfg.guidance_scale if cfg.guidance_scale is not None else ckpt["train_config"].get("guidance_scale", 1.0))
    if (
        ckpt["train_config"].get("ablation") == "full"
        and cfg.checkpoint_type == "best-risk"
        and ckpt.get("checkpoint_stage") != "stage3_risk"
    ):
        warnings.append("proposed best-risk generation did not use a stage3_risk checkpoint.")

    num_candidates = max(1, int(cfg.num_candidates_per_condition))
    selection_mode = cfg.candidate_selection_mode.strip().lower()
    if num_candidates > 1 and selection_mode == "risk_target":
        bg_sample = bg.repeat_interleave(num_candidates, dim=0)
        proc_sample = proc.repeat_interleave(num_candidates, dim=0)
        risk_sample = risk.repeat_interleave(num_candidates, dim=0)
        day_mask_sample = day_mask.repeat_interleave(num_candidates, dim=0)
    else:
        bg_sample, proc_sample, risk_sample, day_mask_sample = bg, proc, risk, day_mask

    risk_guidance_classifier = None
    risk_guidance_targets = None
    classifier_x_mean = None
    classifier_x_std = None
    if cfg.risk_guidance_mode.strip().lower() == "classifier":
        if not cfg.risk_classifier_path:
            warnings.append("risk_guidance_mode=classifier but no risk_classifier_path was provided; guidance disabled.")
        else:
            clf_path = Path(cfg.risk_classifier_path)
            if not clf_path.exists():
                warnings.append(f"risk classifier checkpoint was not found: {clf_path}; guidance disabled.")
            else:
                clf_ckpt = torch.load(clf_path, map_location=device, weights_only=False)
                risk_guidance_classifier = RiskClassifier1D(in_channels=int(ckpt["train_config"]["in_channels"])).to(device)
                risk_guidance_classifier.load_state_dict(clf_ckpt["model_state"])
                risk_guidance_classifier.eval()
                classifier_x_mean = torch.as_tensor(clf_ckpt["x_mean"], dtype=torch.float32, device=device)
                classifier_x_std = torch.as_tensor(clf_ckpt["x_std"], dtype=torch.float32, device=device)
                risk_guidance_targets = {
                    "cum_level": _condition_int_tensor(selected_cond, "cum_level", device),
                    "ramp_level": _condition_int_tensor(selected_cond, "ramp_level", device),
                    "duration_level": _condition_int_tensor(selected_cond, "duration_level", device),
                    "severity_level": _condition_int_tensor(selected_cond, "severity_level", device),
                }
                if num_candidates > 1 and selection_mode == "risk_target":
                    risk_guidance_targets = {key: value.repeat_interleave(num_candidates) for key, value in risk_guidance_targets.items()}

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    gen_norm = sample_sequences(
        model=model,
        scheduler=scheduler,
        bg_cond=bg_sample,
        proc_cond=proc_sample,
        risk_cond=risk_sample,
        shape=(bg_sample.shape[0], int(ckpt["train_config"]["in_channels"]), int(ckpt["seq_len"])),
        guidance_scale=guidance,
        device=device,
        day_mask=day_mask_sample,
        x_mean=x_mean_t,
        x_std=x_std_t,
        risk_guidance_classifier=risk_guidance_classifier,
        risk_guidance_targets=risk_guidance_targets,
        risk_guidance_scale=float(cfg.risk_guidance_scale),
        risk_guidance_start_step_ratio=float(cfg.risk_guidance_start_step_ratio),
        risk_guidance_interval=int(cfg.risk_guidance_interval),
        classifier_x_mean=classifier_x_mean,
        classifier_x_std=classifier_x_std,
    ).detach()
    gen_denorm = denormalize_x_np(gen_norm.cpu().numpy(), x_mean, x_std).astype(np.float32)
    gen_proj_all = apply_physical_projection(torch.from_numpy(gen_denorm).to(device), day_mask_sample).cpu().numpy().astype(np.float32)
    candidate_summary_rows = []
    candidate_weights = _parse_candidate_risk_weights(cfg.candidate_risk_weights)
    if num_candidates > 1 and selection_mode == "risk_target":
        candidates = gen_proj_all.reshape(len(selected_cond), num_candidates, int(ckpt["train_config"]["in_channels"]), int(ckpt["seq_len"]))
        if cfg.save_generated_candidates:
            np.save(out_dir / "generated_candidates.npy", candidates.astype(np.float32))
        gen_proj, candidate_summary = _select_risk_guided_candidates(candidates, selected_cond, out_dir, candidate_weights)
        candidate_summary_rows = candidate_summary.to_dict(orient="records")
    else:
        gen_proj = gen_proj_all

    np.save(out_dir / "generated_samples.npy", gen_proj)
    long_rows = []
    for i, row in selected_cond.iterrows():
        for t in range(gen_proj.shape[2]):
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "generated_id": f"G{i:05d}",
                    "t": int(t),
                    "load": float(gen_proj[i, 0, t]),
                    "wind_power": float(gen_proj[i, 1, t]),
                    "solar_power": float(gen_proj[i, 2, t]),
                    "event_type": row["event_type"],
                    "month": int(row["month"]),
                    "duration_hours": float(row.get("duration_hours", np.nan)),
                    "severity_level": int(row["severity_level"]),
                    "extreme_prob": float(row["extreme_prob"]),
                }
            )
    pd.DataFrame(long_rows).to_csv(out_dir / "generated_samples_long.csv", index=False, encoding="utf-8-sig")
    selected_cond.to_csv(out_dir / "selected_conditions.csv", index=False, encoding="utf-8-sig")
    selected_meta.to_csv(out_dir / "selected_meta.csv", index=False, encoding="utf-8-sig")

    summary = {
        "num_generated": int(len(selected_cond)),
        "split": cfg.split,
        "guidance_scale": guidance,
        "ablation_name": ckpt["train_config"]["ablation"],
        "checkpoint_type": cfg.checkpoint_type,
        "resolved_checkpoint_type": resolved_checkpoint_type,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("checkpoint_epoch"),
        "checkpoint_stage": ckpt.get("checkpoint_stage"),
        "warnings": warnings,
        "num_candidates_per_condition": num_candidates,
        "candidate_selection_mode": selection_mode,
        "candidate_risk_weights": candidate_weights,
        "generated_candidates_saved": bool(num_candidates > 1 and selection_mode == "risk_target" and cfg.save_generated_candidates),
        "candidate_selection_mean_score": float(np.mean([row["selected_score"] for row in candidate_summary_rows])) if candidate_summary_rows else None,
        "risk_guidance_mode": cfg.risk_guidance_mode,
        "risk_classifier_path": cfg.risk_classifier_path,
        "risk_guidance_scale": float(cfg.risk_guidance_scale),
        "risk_guidance_start_step_ratio": float(cfg.risk_guidance_start_step_ratio),
        "risk_guidance_interval": int(cfg.risk_guidance_interval),
        "requested_duration_hours": cfg.duration_hours,
        "actual_condition_duration_hours": selected_cond["duration_hours"].astype(float).tolist() if "duration_hours" in selected_cond else [],
        "duration_generation_mode": duration_mode,
        "condition_meta": condition_meta,
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    model_summary_path = out_dir / "summary.json"
    if model_summary_path.exists():
        model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
        model_summary["checkpoint_type_used_for_generation"] = resolved_checkpoint_type
        model_summary["checkpoint_path_used_for_generation"] = str(checkpoint_path)
        model_summary["checkpoint_epoch_used_for_generation"] = ckpt.get("checkpoint_epoch")
        model_summary["checkpoint_stage_used_for_generation"] = ckpt.get("checkpoint_stage")
        model_summary["generation_warnings"] = warnings
        model_summary_path.write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate scenarios from a trained hierarchical diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-type", type=str, default="best", choices=["best", "best-risk", "final"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--event-type", type=str, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--severity-level", type=int, default=None)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--num-candidates-per-condition", type=int, default=1)
    parser.add_argument("--candidate-selection-mode", type=str, default="none", choices=["none", "risk_target"])
    parser.add_argument("--candidate-risk-weights", type=str, default="cum:1.0,ramp:0.3,duration:0.3")
    parser.add_argument("--save-generated-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--risk-guidance-mode", type=str, default="none", choices=["none", "classifier"])
    parser.add_argument("--risk-classifier-path", type=str, default=None)
    parser.add_argument("--risk-guidance-scale", type=float, default=0.0)
    parser.add_argument("--risk-guidance-start-step-ratio", type=float, default=0.5)
    parser.add_argument("--risk-guidance-interval", type=int, default=5)
    args = parser.parse_args()
    return GenerationConfig(
        checkpoint=args.checkpoint,
        checkpoint_type=args.checkpoint_type,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        split=args.split,
        num_samples=args.num_samples,
        event_type=args.event_type,
        month=args.month,
        severity_level=args.severity_level,
        duration_hours=args.duration_hours,
        guidance_scale=args.guidance_scale,
        num_candidates_per_condition=args.num_candidates_per_condition,
        candidate_selection_mode=args.candidate_selection_mode,
        candidate_risk_weights=args.candidate_risk_weights,
        save_generated_candidates=args.save_generated_candidates,
        risk_guidance_mode=args.risk_guidance_mode,
        risk_classifier_path=args.risk_classifier_path,
        risk_guidance_scale=args.risk_guidance_scale,
        risk_guidance_start_step_ratio=args.risk_guidance_start_step_ratio,
        risk_guidance_interval=args.risk_guidance_interval,
    )


if __name__ == "__main__":
    generate_from_checkpoint(parse_args())
