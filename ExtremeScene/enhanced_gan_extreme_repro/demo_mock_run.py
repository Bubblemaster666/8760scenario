from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from enhanced_gan_extreme import EnhancedGANConfig, train_enhanced_gan, generate_from_conditions, evaluate_generation


def build_mock_extreme_dataset(out_dir: str, n: int = 120, seq_len: int = 24, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(seq_len)
    X = np.zeros((n, 3, seq_len), dtype=np.float32)
    rows = []
    event_names = ["寒潮", "暴雪/风吹雪", "大风/沙尘暴", "高温"]
    for i in range(n):
        e = int(rng.integers(0, 4))
        month = int(rng.choice([1, 2, 3, 4, 7, 8, 11, 12]))
        severity = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
        low_wind = int(e in [1, 3] or (rng.random() < 0.25))
        low_irr = int(e in [0, 1, 2] or (rng.random() < 0.25))
        sev_scale = 1.0 + 0.12 * severity
        load = 580 + 60 * np.sin(2 * np.pi * (t - 7) / 24) + rng.normal(0, 10, seq_len)
        wind = 150 + 35 * np.sin(2 * np.pi * (t + rng.uniform(0, 6)) / 24) + rng.normal(0, 8, seq_len)
        solar = 210 * np.maximum(0, np.sin(np.pi * (t - 6) / 13)) + rng.normal(0, 6, seq_len)
        if e == 0:  # cold wave: load up, pv down
            load += 80 * sev_scale
            solar *= 0.45
            wind *= 0.85
        elif e == 1:  # blizzard: load up, pv/wind down
            load += 65 * sev_scale
            solar *= 0.25
            wind *= 0.50
        elif e == 2:  # dust storm: pv down, wind volatile
            solar *= 0.30
            wind *= np.clip(1.2 + 0.35 * np.sin(2 * np.pi * t / 8), 0.3, 1.6)
        else:  # high temperature: load up, wind down
            load += 95 * sev_scale
            wind *= 0.55
            solar *= 0.85
        if low_wind:
            wind *= 0.55
        if low_irr:
            solar *= 0.60
        solar[(t < 6) | (t >= 20)] = 0.0
        X[i, 0] = np.clip(load, 0, None)
        X[i, 1] = np.clip(wind, 0, None)
        X[i, 2] = np.clip(solar, 0, None)
        net = X[i, 0] - X[i, 1] - X[i, 2]
        tau = 0.0
        cum = float(np.maximum(0, net - tau).sum())
        ramp = float(np.diff(net).max())
        dur = float((net > tau).sum())
        # smaller means more extreme, mock only
        extreme_prob = float(np.clip(np.exp(-cum / 9000), 0.001, 0.2))
        rows.append(
            {
                "sample_id": f"E{i:04d}",
                "event_type": event_names[e],
                "event_type_code": e,
                "month": month,
                "season": "",
                "low_wind_flag": low_wind,
                "low_irradiance_flag": low_irr,
                "duration_hours": seq_len,
                "severity_level": severity,
                "extreme_prob": extreme_prob,
                "cum_deficit": cum,
                "netload_ramp_max": ramp,
                "imbalance_duration": dur,
            }
        )
    np.save(out / "X.npy", X)
    pd.DataFrame(rows).to_csv(out / "cond.csv", index=False, encoding="utf-8-sig")
    meta = pd.DataFrame({
        "sample_id": [r["sample_id"] for r in rows],
        "core_start_time": ["2025-01-01 00:00:00"] * n,
        "core_end_time": ["2025-01-01 23:00:00"] * n,
        "window_start_time": ["2025-01-01 00:00:00"] * n,
        "window_end_time": ["2025-01-01 23:00:00"] * n,
    })
    meta.to_csv(out / "meta.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    root = Path("demo_enhanced_gan_outputs")
    data_dir = root / "dataset"
    model_dir = root / "model"
    build_mock_extreme_dataset(str(data_dir), n=96, seq_len=24, seed=1)
    cfg = EnhancedGANConfig(
        data_dir=str(data_dir),
        output_dir=str(model_dir),
        epochs=2,
        batch_size=32,
        n_critic=1,
        hidden_dim=64,
        z_dim=32,
        device="cpu",
        acf_max_lag=8,
        generate_per_test_condition=1,
    )
    train_summary = train_enhanced_gan(cfg)
    gen_dir = model_dir / "generation"
    gen_summary = generate_from_conditions(str(model_dir), str(model_dir / "cond_test.csv"), str(gen_dir), n_per_condition=1, device="cpu")
    eval_summary = evaluate_generation(str(model_dir / "X_test.npy"), str(gen_dir / "generated_samples.npy"), str(gen_dir / "generated_conditions.csv"), str(model_dir / "evaluation"), "enhanced_gan_demo", cfg)
    print(json.dumps({"train": train_summary, "generate": gen_summary, "evaluate": eval_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
