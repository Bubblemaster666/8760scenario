from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_mock_dataset(out_dir: Path, n: int = 96, t: int = 24, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    hours = np.arange(t)
    X = np.zeros((n, 3, t), dtype=np.float32)
    rows = []
    meta = []
    event_names = ["寒潮", "暴雪/风吹雪", "大风/沙尘暴", "高温"]
    for i in range(n):
        event_code = i % 4
        sev = (i // 4) % 4
        month = int(rng.integers(1, 13))
        low_w = int(event_code in (0, 3) or sev >= 2)
        low_s = int(event_code in (1, 2) or sev >= 2)
        base_load = 80 + 8 * np.sin(2 * np.pi * (hours - 7) / 24) + 4 * rng.normal(size=t)
        event_load = 6 * sev + (event_code in (0, 3)) * 8
        load = base_load + event_load
        wind = 28 + 8 * np.sin(2 * np.pi * (hours + month) / 24) + 5 * rng.normal(size=t)
        if low_w:
            wind *= 0.45 + 0.08 * rng.random()
        if event_code == 2:
            wind += 10 * np.sin(2 * np.pi * hours / 6)
        solar_shape = np.maximum(0, np.sin(np.pi * (hours - 6) / 13))
        solar = 38 * solar_shape + 2 * rng.normal(size=t)
        solar = np.clip(solar, 0, None)
        if low_s:
            solar *= 0.35 + 0.1 * rng.random()
        solar[(hours < 6) | (hours >= 20)] = 0
        X[i, 0] = np.clip(load, 0, None)
        X[i, 1] = np.clip(wind, 0, None)
        X[i, 2] = np.clip(solar, 0, None)
        net = X[i, 0] - X[i, 1] - X[i, 2]
        rows.append({
            "sample_id": f"s{i:04d}",
            "event_type": event_names[event_code],
            "event_type_code": event_code,
            "month": month,
            "season": (month % 12) // 3 + 1,
            "low_wind_flag": low_w,
            "low_irradiance_flag": low_s,
            "duration_hours": t,
            "extreme_prob": max(0.002, 0.2 / (sev + 1)),
            "severity_level": sev,
            "cum_deficit": float(np.maximum(net, 0).sum()),
            "netload_ramp_max": float(np.diff(net).max()),
            "imbalance_duration": float((net > 0).sum()),
        })
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        meta.append({
            "sample_id": f"s{i:04d}",
            "window_start_time": start,
            "window_end_time": start + pd.Timedelta(hours=t-1),
            "core_start_time": start + pd.Timedelta(hours=3),
            "core_end_time": start + pd.Timedelta(hours=t-4),
        })
    np.save(out_dir / "X.npy", X)
    pd.DataFrame(rows).to_csv(out_dir / "cond.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(meta).to_csv(out_dir / "meta.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    data_dir = root / "demo_data"
    out_dir = root / "demo_outputs"
    build_mock_dataset(data_dir)
    cmd = [
        sys.executable,
        str(root / "traditional_copula_baseline.py"),
        "pipeline",
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--min-group-size", "6",
        "--n-per-condition", "1",
        "--acf-max-lag", "12",
        "--temporal-smooth-strength", "0.12",
    ]
    subprocess.run(cmd, check=True)
    print(f"Demo finished. Outputs: {out_dir}")
