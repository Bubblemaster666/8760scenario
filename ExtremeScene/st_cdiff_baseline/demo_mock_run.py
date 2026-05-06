from pathlib import Path
import subprocess
import sys

OUT = Path("outputs/st_cdiff_demo")
OUT.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable,
    "st_cdiff_baseline.py",
    "pipeline",
    "--use-mock",
    "--output-dir", str(OUT),
    "--mock-n", "60",
    "--seq-len", "24",
    "--epochs", "2",
    "--batch-size", "32",
    "--diffusion-steps", "20",
    "--base-channels", "24",
    "--acf-max-lag", "8",
    "--device", "cpu",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
print(f"Done. See {OUT.resolve()}")
