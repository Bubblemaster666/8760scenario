"""Create a tiny mock extreme dataset and run the plain DDPM pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mock_dataset"
OUT = ROOT / "outputs" / "plain_ddpm_demo"
DATA.mkdir(exist_ok=True, parents=True)

rng = np.random.default_rng(123)
N, T = 120, 24
hours = np.arange(T)
X = []
cond_rows = []
meta_rows = []
for i in range(N):
    event = rng.choice(["cold_wave", "sandstorm", "heat_wave"])
    sev = int(rng.integers(0, 4))
    load = 80 + 10*np.sin((hours-7)/24*2*np.pi) + 6*sev + rng.normal(0, 2, T)
    wind = 25 + 8*np.sin((hours+i%5)/24*2*np.pi) + rng.normal(0, 4, T)
    solar = np.maximum(0, 35*np.sin((hours-6)/12*np.pi)) + rng.normal(0, 2, T)
    if event == "sandstorm":
        solar *= 0.45
        wind += 8
    if event == "cold_wave":
        load += 10
        solar *= 0.65
    if event == "heat_wave":
        load += 8
        wind *= 0.75
    solar[(hours < 6) | (hours >= 20)] = 0
    wind = np.maximum(wind, 0)
    solar = np.maximum(solar, 0)
    load = np.maximum(load, 0)
    X.append(np.stack([load, wind, solar], axis=0))
    net = load - wind - solar
    cond_rows.append({
        "sample_id": i,
        "event_type": event,
        "event_type_code": {"cold_wave":0, "sandstorm":1, "heat_wave":2}[event],
        "month": int(rng.integers(1, 13)),
        "low_wind_flag": int(wind.mean() < 22),
        "low_irradiance_flag": int(solar.mean() < 8),
        "duration_hours": T,
        "severity_level": sev,
        "cum_deficit": float(np.maximum(net, 0).sum()),
        "netload_ramp_max": float(np.diff(net).max()),
        "imbalance_duration": float((net > 0).sum()),
    })
    start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
    meta_rows.append({
        "sample_id": i,
        "window_start_time": start.isoformat(),
        "window_end_time": (start + pd.Timedelta(hours=T-1)).isoformat(),
        "core_start_time": (start + pd.Timedelta(hours=4)).isoformat(),
        "core_end_time": (start + pd.Timedelta(hours=20)).isoformat(),
    })
X = np.asarray(X, dtype=np.float32)
np.save(DATA / "X.npy", X)
pd.DataFrame(cond_rows).to_csv(DATA / "cond.csv", index=False)
pd.DataFrame(meta_rows).to_csv(DATA / "meta.csv", index=False)

cmd = [
    sys.executable, str(ROOT / "plain_ddpm_baseline.py"), "pipeline",
    "--data-dir", str(DATA),
    "--output-dir", str(OUT),
    "--epochs", "3",
    "--batch-size", "32",
    "--diffusion-steps", "50",
    "--base-channels", "32",
    "--n-blocks", "3",
    "--device", "cpu",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Done. Metrics:", OUT / "evaluation" / "metrics_summary.csv")
