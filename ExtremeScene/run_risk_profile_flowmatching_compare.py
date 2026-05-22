from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_generation import EvalConfig, evaluate_generation
from generate_scenarios import GenerationConfig, generate_from_checkpoint
from hierarchical_diffusion import (
    ConditionedWindowDataset,
    EMA,
    HierarchicalConditionalUNet1D,
    apply_physical_projection,
    compute_condition_normalizers,
    condition_dropout,
    denormalize_x_np,
    load_split_arrays,
)
from run_joint_profile_diffusion_experiment import prepare_joint_profile_dataset
from train_hierarchical_evt_diffusion import TrainConfig, resolve_device, set_seed, train_model
from risk_ranking_utils import (
    AUXILIARY_REALISM_METRICS,
    RISK_EVALUATION_EXPLANATION,
    RISK_MAIN_METRICS,
    RISK_RANKING_EXPLANATION,
    add_risk_score,
    write_risk_tables,
)


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class MethodSpec:
    method: str
    model_type: str
    use_risk_profile: bool
    source_generated: Path | None = None


SPECS = [
    MethodSpec(
        "A0_baseline_JRPD_best_3h",
        "diffusion",
        False,
        BASE_DIR / "outputs" / "ramp_window_retrain" / "models" / "RAMPDIAG4_JRPD_ramp3h" / "generated_samples.npy",
    ),
    MethodSpec("A1_flow_matching_original_condition", "flow_matching", False),
    MethodSpec(
        "A2_diffusion_with_risk_profile",
        "diffusion",
        True,
        BASE_DIR / "outputs" / "joint_profile_diffusion_on_ramp3h" / "models" / "JP1_profile_condition" / "generated_samples.npy",
    ),
    MethodSpec("A3_flow_matching_with_risk_profile", "flow_matching", True),
]

FULL_COMPARE_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "extreme_degree_match_rate",
    *RISK_MAIN_METRICS,
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
]


def _max_event_type_code(*frames: pd.DataFrame) -> int:
    max_code = 0
    for frame in frames:
        if "event_type_code" in frame.columns and not frame.empty:
            max_code = max(max_code, int(pd.to_numeric(frame["event_type_code"], errors="coerce").fillna(0).max()))
    return max_code


def _make_dataset(
    data_dir: Path,
    split: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    cond_normalizers,
    expected_event_types: int,
    use_risk_profile: bool,
) -> ConditionedWindowDataset:
    loaded = load_split_arrays(data_dir, split, include_event_mask=True, include_risk_profile=use_risk_profile)
    if use_risk_profile:
        x, cond, meta, event_mask, risk_profile = loaded
    else:
        x, cond, meta, event_mask = loaded
        risk_profile = None
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    return ConditionedWindowDataset(
        x_norm,
        cond,
        meta,
        seq_len=x.shape[2],
        ablation="full",
        normalizers=cond_normalizers,
        expected_event_types=expected_event_types,
        event_mask=event_mask,
        risk_profile=risk_profile,
        use_risk_profile_condition=True,
    )


def _flow_train(spec: MethodSpec, data_dir: Path, model_dir: Path, device_name: str, epochs: int, batch_size: int, steps: int, base_channels: int) -> dict:
    """Train Conditional Flow Matching.

    x: 风光荷三通道真实序列；x_t: Flow Matching 线性路径中间状态；
    t: 连续时间；condition: 原有分层条件；risk_profile: 36x4 联合风险过程矩阵；
    u_target: 目标速度场；u_pred: 模型预测速度场。
    """

    set_seed(42)
    torch.set_num_threads(1)
    device = resolve_device(device_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    x_train, cond_train, meta_train, _ = load_split_arrays(data_dir, "train", include_event_mask=True)
    x_val, cond_val, meta_val, _ = load_split_arrays(data_dir, "val", include_event_mask=True)
    x_mean = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    cond_normalizers = compute_condition_normalizers(cond_train)
    expected_event_types = _max_event_type_code(cond_train, cond_val) + 1

    train_ds = _make_dataset(data_dir, "train", x_mean, x_std, cond_normalizers, expected_event_types, spec.use_risk_profile)
    val_ds = _make_dataset(data_dir, "val", x_mean, x_std, cond_normalizers, expected_event_types, spec.use_risk_profile)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    bg_dim = train_ds.background.shape[1]
    proc_dim = train_ds.process.shape[1]
    risk_dim = train_ds.risk.shape[1]
    profile_channels = 4 if spec.use_risk_profile else 0
    model = HierarchicalConditionalUNet1D(
        in_channels=3,
        base_channels=base_channels,
        time_dim=128,
        cond_dim=128,
        bg_dim=bg_dim,
        proc_dim=proc_dim,
        risk_dim=risk_dim,
        flat_condition=False,
        profile_channels=profile_channels,
    ).to(device)
    ema = EMA(model, 0.995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    time_scale = 1000.0

    def run_epoch(loader: DataLoader, train: bool) -> float:
        model.train(train)
        total = 0.0
        count = 0
        for batch in loader:
            x, bg, proc, risk, _risk_targets, _day_mask, _event_mask, profile = batch
            x = x.to(device)
            bg = bg.to(device)
            proc = proc.to(device)
            risk = risk.to(device)
            profile = profile.to(device) if spec.use_risk_profile else None
            if train:
                bg_in, proc_in, risk_in = condition_dropout(bg, proc, risk, 0.10)
                profile_in = profile
                if profile_in is not None:
                    keep = (torch.rand(profile_in.size(0), device=device) > 0.10).float().view(-1, 1, 1)
                    profile_in = profile_in * keep
            else:
                bg_in, proc_in, risk_in, profile_in = bg, proc, risk, profile

            # Flow Matching: x0 是真实样本，x1 是高斯噪声，x_t 是二者线性插值。
            x0 = x
            x1 = torch.randn_like(x0)
            t = torch.rand(x0.size(0), device=device)
            x_t = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
            u_target = x1 - x0
            with torch.set_grad_enabled(train):
                u_pred = model(x_t, t * time_scale, bg_in, proc_in, risk_in, profile_in)
                loss = F.mse_loss(u_pred, u_target)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    ema.update(model)
            total += float(loss.item()) * x0.size(0)
            count += x0.size(0)
        return total / max(count, 1)

    history = []
    best_val = float("inf")
    best_epoch = None
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        ema_model = HierarchicalConditionalUNet1D(
            in_channels=3,
            base_channels=base_channels,
            time_dim=128,
            cond_dim=128,
            bg_dim=bg_dim,
            proc_dim=proc_dim,
            risk_dim=risk_dim,
            flat_condition=False,
            profile_channels=profile_channels,
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema.copy_to(ema_model)
        current_model = model
        model = ema_model
        val_loss = run_epoch(val_loader, train=False)
        model = current_model
        history.append({"epoch": epoch, "train_fm_loss": train_loss, "val_fm_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{spec.method} epoch {epoch:03d}/{epochs} | train_fm={train_loss:.4f} | val_fm={val_loss:.4f}")

    pd.DataFrame(history).to_csv(model_dir / "flow_matching_history.csv", index=False, encoding="utf-8-sig")
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt = {
        "model_state": best_state,
        "x_mean": x_mean,
        "x_std": x_std,
        "condition_normalizers": cond_normalizers.__dict__,
        "condition_meta": train_ds.condition_meta,
        "cond_dims": {"background": bg_dim, "process": proc_dim, "risk": risk_dim, "profile": profile_channels},
        "seq_len": 36,
        "model_type": "flow_matching",
        "best_epoch": best_epoch,
        "best_val_fm_loss": best_val,
        "time_scale": time_scale,
        "steps": steps,
        "base_channels": base_channels,
        "use_risk_profile": bool(spec.use_risk_profile),
    }
    torch.save(ckpt, model_dir / "flow_matching_model.pt")
    (model_dir / "flow_matching_summary.json").write_text(json.dumps({k: v for k, v in ckpt.items() if k != "model_state"}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return ckpt


@torch.no_grad()
def _flow_generate(spec: MethodSpec, data_dir: Path, model_dir: Path, out_path: Path, device_name: str, sample_steps: int) -> None:
    device = resolve_device(device_name)
    ckpt = torch.load(model_dir / "flow_matching_model.pt", map_location=device, weights_only=False)
    loaded = load_split_arrays(data_dir, "test", include_event_mask=True, include_risk_profile=spec.use_risk_profile)
    if spec.use_risk_profile:
        x_test, cond_test, meta_test, event_mask, risk_profile = loaded
    else:
        x_test, cond_test, meta_test, event_mask = loaded
        risk_profile = None
    normalizers = compute_condition_normalizers(pd.read_csv(data_dir / "cond_train.csv"))
    ds = ConditionedWindowDataset(
        ((x_test - ckpt["x_mean"]) / ckpt["x_std"]).astype(np.float32),
        cond_test,
        meta_test,
        seq_len=36,
        ablation="full",
        normalizers=normalizers,
        expected_event_types=len(ckpt["condition_meta"]["background"]["event_onehot"]),
        event_mask=event_mask,
        risk_profile=risk_profile if spec.use_risk_profile else None,
        use_risk_profile_condition=True,
    )
    bg = torch.from_numpy(ds.background).to(device)
    proc = torch.from_numpy(ds.process).to(device)
    risk = torch.from_numpy(ds.risk).to(device)
    profile = torch.from_numpy(ds.risk_profile).to(device) if spec.use_risk_profile else None
    day_mask = torch.from_numpy(ds.day_mask).to(device)
    profile_channels = int(ckpt["cond_dims"].get("profile", 0))
    model = HierarchicalConditionalUNet1D(
        in_channels=3,
        base_channels=int(ckpt["base_channels"]),
        time_dim=128,
        cond_dim=128,
        bg_dim=int(ckpt["cond_dims"]["background"]),
        proc_dim=int(ckpt["cond_dims"]["process"]),
        risk_dim=int(ckpt["cond_dims"]["risk"]),
        flat_condition=False,
        profile_channels=profile_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.randn((len(cond_test), 3, 36), device=device)
    dt = 1.0 / max(int(sample_steps), 1)
    for i in range(int(sample_steps)):
        t_value = 1.0 - i * dt
        t = torch.full((x.size(0),), t_value * float(ckpt["time_scale"]), device=device)
        u_pred = model(x, t, bg, proc, risk, profile)
        x = x - dt * u_pred
    gen = denormalize_x_np(x.cpu().numpy(), np.asarray(ckpt["x_mean"], dtype=np.float32), np.asarray(ckpt["x_std"], dtype=np.float32))
    gen = apply_physical_projection(torch.from_numpy(gen).to(device), day_mask).cpu().numpy().astype(np.float32)
    np.save(out_path, gen)


def _copy_or_train_diffusion(spec: MethodSpec, data_dir: Path, method_dir: Path, generated_out: Path, device: str, force_train: bool) -> None:
    if spec.source_generated is not None and spec.source_generated.exists() and not force_train:
        shutil.copy2(spec.source_generated, generated_out)
        return
    train_model(
        TrainConfig(
            data_dir=str(data_dir),
            out_dir=str(method_dir),
            ablation="full",
            seq_len=36,
            batch_size=32,
            lr=1e-4,
            weight_decay=1e-5,
            diffusion_steps=100,
            base_channels=64,
            guidance_scale=1.0,
            cond_dropout=0.10,
            ema_decay=0.995,
            stage1_epochs=12,
            stage2_epochs=10,
            stage3_epochs=2,
            lambda_tail=0.25,
            lambda_risk=0.04,
            lambda_cum=1.0,
            lambda_ramp=0.25,
            lambda_dur=0.35,
            lambda_recon=0.05,
            lambda_physics=0.02,
            lambda_resource=0.02,
            sampler_mode="risk_profile_balanced",
            use_risk_profile_condition=True,
            use_profile_loss=True,
            lambda_profile=0.05,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
            use_joint_profile_condition=spec.use_risk_profile,
            device=device,
            seed=42,
        )
    )
    generate_from_checkpoint(GenerationConfig(checkpoint=None, data_dir=str(data_dir), out_dir=str(method_dir), split="test", guidance_scale=1.0, checkpoint_type="best-risk"))
    shutil.copy2(method_dir / "generated_samples.npy", generated_out)


def _evaluate_method(method: str, generated: Path, data_dir: Path, out_dir: Path) -> dict:
    eval_dir = out_dir / "evaluations" / method
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(generated),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=method,
            event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    row = {"method": method}
    row.update({metric: summary["metrics"].get(metric, np.nan) for metric in FULL_COMPARE_METRICS})
    pd.DataFrame([row]).to_csv(out_dir / f"{method}_metrics.csv", index=False, encoding="utf-8-sig")
    return row


def _add_rank_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = add_risk_score(df)
    out["rank_by_q99"] = pd.to_numeric(out["q99_cum_deficit_error"], errors="coerce").rank(method="average", ascending=True)
    out["rank_by_ramp"] = pd.to_numeric(out["netload_ramp_max_mae"], errors="coerce").rank(method="average", ascending=True)
    out["rank_by_highrisk_acf"] = pd.to_numeric(out["highrisk_acf_mae"], errors="coerce").rank(method="average", ascending=True)
    return out.sort_values("risk_score").reset_index(drop=True)


def run(args: argparse.Namespace) -> pd.DataFrame:
    base_data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = prepare_joint_profile_dataset(base_data_dir, out_dir, force=bool(args.force_dataset))

    rows = []
    for spec in SPECS:
        method_dir = out_dir / "models" / spec.method
        method_dir.mkdir(parents=True, exist_ok=True)
        generated_out = out_dir / f"generated_samples_{spec.method[:2]}.npy"
        if spec.model_type == "flow_matching":
            if not (method_dir / "flow_matching_model.pt").exists() or bool(args.force_train):
                _flow_train(spec, data_dir, method_dir, args.device, epochs=int(args.epochs), batch_size=32, steps=int(args.sample_steps), base_channels=int(args.base_channels))
            if not generated_out.exists() or bool(args.force_train):
                _flow_generate(spec, data_dir, method_dir, generated_out, args.device, sample_steps=int(args.sample_steps))
        else:
            _copy_or_train_diffusion(spec, data_dir, method_dir, generated_out, args.device, force_train=bool(args.force_train))
        rows.append(_evaluate_method(spec.method, generated_out, data_dir, out_dir))

    df = _add_rank_summary(pd.DataFrame(rows))
    df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk_main, aux_realism = write_risk_tables(df, out_dir, method_col="method")
    root_results = BASE_DIR / "results"
    write_risk_tables(df, root_results, method_col="method")
    report = [
        "# Risk Profile + Flow Matching Compare",
        "",
        RISK_EVALUATION_EXPLANATION,
        RISK_RANKING_EXPLANATION,
        "",
        "- A0: JRPD_best_3h / RAMPDIAG4 baseline",
        "- A1: Conditional Flow Matching with original conditions",
        "- A2: DDPM/JRPD with 36x4 risk_profile condition",
        "- A3: Conditional Flow Matching with 36x4 risk_profile condition",
        "- netload_ramp_max_mae is evaluated with 3h window ramp.",
        "",
        "## Main Risk Ranking",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Complete Metrics",
        "",
        df[["method", *FULL_COMPARE_METRICS, "risk_score", "risk_rank", "rank_by_q99", "rank_by_ramp", "rank_by_highrisk_acf"]].to_markdown(index=False),
        "",
        "## Auxiliary Realism Metrics",
        "",
        aux_realism[[col for col in ["method", *AUXILIARY_REALISM_METRICS, "realism_check_pass"] if col in aux_realism.columns]].to_markdown(index=False),
    ]
    (out_dir / "compare_report.md").write_text("\n".join(report), encoding="utf-8")
    print(RISK_RANKING_EXPLANATION)
    print(RISK_EVALUATION_EXPLANATION)
    print(f"risk_main_compare: {out_dir / 'risk_main_compare.csv'}")
    print(f"root risk_main_compare: {root_results / 'risk_main_compare.csv'}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare JRPD, risk-profile diffusion, and Conditional Flow Matching.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "ramp_window_retrain" / "datasets" / "dataset_window_3h"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "results" / "risk_profile_flowmatching_compare"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--force-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-train", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
