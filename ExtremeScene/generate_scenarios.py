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
    checkpoint: str
    data_dir: str
    out_dir: str
    split: str = "test"
    num_samples: Optional[int] = None
    event_type: Optional[str] = None
    month: Optional[int] = None
    severity_level: Optional[int] = None
    guidance_scale: Optional[float] = None


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
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "condition_meta": condition_meta,
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate scenarios from a trained hierarchical diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
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
    )


if __name__ == "__main__":
    generate_from_checkpoint(parse_args())
