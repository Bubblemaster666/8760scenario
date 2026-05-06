from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from risk_metrics import compute_windowed_netload_ramp


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class ExistingGeneration:
    experiment_name: str
    generated_path: str


DEFAULT_GENERATIONS = [
    ExistingGeneration("E0_proposed", "outputs/main_compare_e0_fixed/models/proposed/generated_samples.npy"),
    ExistingGeneration("T4_JRPD_sampler_loss", "outputs/three_method_trials/models/T4_JRPD_sampler_loss/generated_samples.npy"),
    ExistingGeneration("C4_evt_aug_more_30_20_6", "outputs/evt_transfer_longstage/C4_evttransfer_more_30_20_6/models/proposed/generated_samples.npy"),
    ExistingGeneration("traditional_gaussian_copula", "outputs/main_compare_e0_fixed/models/traditional_gaussian_copula/generation/generated_samples.npy"),
    ExistingGeneration("enhanced_gan", "outputs/main_compare_e0_fixed/models/enhanced_gan/generation/generated_samples.npy"),
    ExistingGeneration("plain_diffusion_baseline", "outputs/main_compare_e0_fixed/models/plain_diffusion_baseline/generation/generated_samples.npy"),
]


WINDOWS = {
    "1step": 1.0,
    "1h": 1.0,
    "2h": 2.0,
    "3h": 3.0,
}


def _safe_load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def _mae(real: np.ndarray, gen: np.ndarray) -> float:
    return float(np.nanmean(np.abs(gen - real)))


def _q_error(real: np.ndarray, gen: np.ndarray, q: float) -> float:
    return float(abs(np.nanquantile(gen, q) - np.nanquantile(real, q)))


def _window_ramps(x: np.ndarray, delta_t_hours: float) -> dict[str, np.ndarray]:
    out = {}
    for label, hours in WINDOWS.items():
        out[label] = np.asarray(compute_windowed_netload_ramp(x, delta_t_hours, hours, positive_only=True), dtype=float)
    stack = np.vstack([out["1h"], out["2h"], out["3h"]])
    out["multiscale"] = np.nanmax(stack, axis=0)
    return out


def _recommend(row: dict, copula_row: dict | None) -> str:
    if copula_row is None:
        return "diagnostic only"
    gap_1 = float(row["netload_ramp_1step_mae"]) - float(copula_row["netload_ramp_1step_mae"])
    gap_2 = float(row["netload_ramp_2h_mae"]) - float(copula_row["netload_ramp_2h_mae"])
    gap_3 = float(row["netload_ramp_3h_mae"]) - float(copula_row["netload_ramp_3h_mae"])
    if gap_2 < gap_1 * 0.75 or gap_3 < gap_1 * 0.75:
        return "2h/3h narrows gap to Copula; windowed ramp is worth retraining"
    if float(row["netload_ramp_2h_mae"]) < float(row["netload_ramp_1step_mae"]) * 0.75:
        return "2h ramp is smoother than one-step but Copula gap remains"
    return "one-step issue not resolved by windowed ramp"


def run_diagnostic(data_dir: str, out_dir: str, delta_t_hours: float | None = None) -> pd.DataFrame:
    data_root = Path(data_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    real = np.load(data_root / "X_test.npy").astype(np.float32)
    cond = pd.read_csv(data_root / "cond_test.csv")
    dt = float(delta_t_hours or (cond["delta_t_hours"].iloc[0] if "delta_t_hours" in cond.columns else 1.0))
    real_ramps = _window_ramps(real, dt)
    warnings: list[str] = []
    rows = []
    for item in DEFAULT_GENERATIONS:
        gen_path = BASE_DIR / item.generated_path
        gen = _safe_load(gen_path)
        if gen is None:
            warnings.append(f"missing generated result: {item.experiment_name} -> {gen_path}")
            continue
        if gen.shape != real.shape:
            warnings.append(f"shape mismatch skipped: {item.experiment_name}, generated={gen.shape}, real={real.shape}")
            continue
        gen_ramps = _window_ramps(gen, dt)
        row = {"experiment_name": item.experiment_name, "generated_path": str(gen_path)}
        for label in ["1step", "1h", "2h", "3h", "multiscale"]:
            col_label = "1step" if label == "1step" else label
            row[f"netload_ramp_{col_label}_mae"] = _mae(real_ramps[label], gen_ramps[label])
            row[f"q95_netload_ramp_{col_label}_error"] = _q_error(real_ramps[label], gen_ramps[label], 0.95)
            row[f"q99_netload_ramp_{col_label}_error"] = _q_error(real_ramps[label], gen_ramps[label], 0.99)
        rows.append(row)
    copula_row = next((r for r in rows if r["experiment_name"] == "traditional_gaussian_copula"), None)
    for row in rows:
        row["recommendation_reason"] = _recommend(row, copula_row)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "ramp_window_diagnostic_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Ramp Window Diagnostic Report",
        "",
        "本阶段只重算评价，不重训模型，不覆盖原始 cond 文件。",
        f"- delta_t_hours: {dt}",
        "",
        "## Findings",
    ]
    if not df.empty:
        for name in ["E0_proposed", "T4_JRPD_sampler_loss", "traditional_gaussian_copula"]:
            sub = df[df["experiment_name"] == name]
            if sub.empty:
                continue
            row = sub.iloc[0]
            lines.append(
                f"- {name}: 1step={row['netload_ramp_1step_mae']:.4g}, "
                f"2h={row['netload_ramp_2h_mae']:.4g}, 3h={row['netload_ramp_3h_mae']:.4g}, "
                f"multiscale={row['netload_ramp_multiscale_mae']:.4g}"
            )
        if copula_row is not None:
            lines.append("")
            lines.append("## Gap To Copula")
            for name in ["E0_proposed", "T4_JRPD_sampler_loss", "C4_evt_aug_more_30_20_6"]:
                sub = df[df["experiment_name"] == name]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                lines.append(
                    f"- {name}: gap_1step={row['netload_ramp_1step_mae'] - copula_row['netload_ramp_1step_mae']:.4g}, "
                    f"gap_2h={row['netload_ramp_2h_mae'] - copula_row['netload_ramp_2h_mae']:.4g}, "
                    f"gap_3h={row['netload_ramp_3h_mae'] - copula_row['netload_ramp_3h_mae']:.4g}"
                )
    lines += [
        "",
        "## Recommendation",
        "如果 2h/3h 明显缩小 proposed/JRPD 与 Copula 的 ramp 差距，则进入重训；否则说明问题不只是 one-step 颗粒度。",
        "",
        "## Warnings",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    (out_root / "ramp_window_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    (out_root / "ramp_window_diagnostic_warnings.json").write_text(json.dumps(warnings, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute net-load ramp diagnostics over 1h/2h/3h windows.")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "outputs" / "heavy_rain_tighten_test_v2" / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "outputs" / "ramp_window_diagnostic"))
    parser.add_argument("--delta-t-hours", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args.data_dir, args.out_dir, delta_t_hours=args.delta_t_hours)
