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
    HierarchicalConditionalUNet1D,
    DiffusionScheduler,
    apply_physical_projection,
    build_condition_bundle,
    denormalize_x_np,
    sample_sequences,
)


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
    guidance_scale: Optional[float] = None
    checkpoint_type: str = "best"


def _checkpoint_name(checkpoint_type: str) -> str:
    mapping = {
        "best": "best_model.pt",
        "best-risk": "best_risk_model.pt",
        "final": "final_model.pt",
    }
    if checkpoint_type not in mapping:
        raise ValueError(f"Unsupported checkpoint_type: {checkpoint_type}")
    return mapping[checkpoint_type]


def _resolve_checkpoint(cfg: GenerationConfig, out_dir: Path) -> tuple[Path, str]:
    if cfg.checkpoint:
        return Path(cfg.checkpoint), cfg.checkpoint_type
    ckpt_path = out_dir / _checkpoint_name(cfg.checkpoint_type)
    if not ckpt_path.exists() and cfg.checkpoint_type == "best-risk":
        fallback = out_dir / "best_model.pt"
        if fallback.exists():
            return fallback, "best-risk-fallback-best"
    return ckpt_path, cfg.checkpoint_type


def _load_condition_frame(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cond_df = pd.read_csv(data_dir / f"cond_{split}.csv")
    meta_df = pd.read_csv(data_dir / f"meta_{split}.csv")
    return cond_df, meta_df


def _filter_conditions(cond_df: pd.DataFrame, cfg: GenerationConfig) -> pd.DataFrame:
    out = cond_df.copy()
    if cfg.event_type is not None:
        out = out[out["event_type"] == cfg.event_type]
    if cfg.month is not None:
        out = out[out["month"].astype(int) == int(cfg.month)]
    if cfg.severity_level is not None:
        out = out[out["severity_level"].astype(int) == int(cfg.severity_level)]
    if cfg.num_samples is not None and len(out) > cfg.num_samples:
        out = out.iloc[: cfg.num_samples].copy()
    return out


def generate_from_checkpoint(cfg: GenerationConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, resolved_checkpoint_type = _resolve_checkpoint(cfg, out_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_dir = Path(cfg.data_dir)

    cond_df_full, meta_df_full = _load_condition_frame(data_dir, cfg.split)
    selected_cond = _filter_conditions(cond_df_full, cfg)
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
    guidance = float(cfg.guidance_scale if cfg.guidance_scale is not None else ckpt["train_config"]["guidance_scale"])

    gen_norm = sample_sequences(
        model=model,
        scheduler=scheduler,
        bg_cond=bg,
        proc_cond=proc,
        risk_cond=risk,
        shape=(len(selected_cond), int(ckpt["train_config"]["in_channels"]), int(ckpt["seq_len"])),
        guidance_scale=guidance,
        device=device,
    ).detach()
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    gen_denorm = denormalize_x_np(gen_norm.cpu().numpy(), x_mean, x_std).astype(np.float32)
    gen_proj = apply_physical_projection(torch.from_numpy(gen_denorm).to(device), day_mask).cpu().numpy().astype(np.float32)

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
    parser.add_argument("--guidance-scale", type=float, default=None)
    args = parser.parse_args()
    return GenerationConfig(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        split=args.split,
        num_samples=args.num_samples,
        event_type=args.event_type,
        month=args.month,
        severity_level=args.severity_level,
        guidance_scale=args.guidance_scale,
        checkpoint_type=args.checkpoint_type,
    )


if __name__ == "__main__":
    generate_from_checkpoint(parse_args())
