# Traditional Statistical Extreme Baseline

This package implements a traditional statistical baseline for wind-solar-load extreme scenario generation.

Chinese method name:

**传统统计基线：极端样本分组-边缘分布拟合-Gaussian Copula联合建模-抽样生成联合场景-时序修正-反归一化与物理约束**

It is designed as a fair baseline for the thesis experiments. It does **not** use EVT-guided diffusion training, tail-sensitive loss, hierarchical condition injection, or joint imbalance consistency loss.

## Input format

Put these files in `data_dir`:

```text
X.npy       # [N, 3, T] or [N, T, 3]
cond.csv    # condition and label table
meta.csv    # optional time metadata
```

Channel order is fixed:

```text
0 = load
1 = wind_power
2 = solar_power
```

Recommended `cond.csv` fields:

```text
sample_id,event_type,event_type_code,month,season,
low_wind_flag,low_irradiance_flag,duration_hours,
extreme_prob,severity_level,cum_deficit,netload_ramp_max,imbalance_duration
```

## Method

1. Extreme sample grouping by event/severity/resource state.
2. Empirical marginal distribution fitting for each scalar dimension.
3. Gaussian Copula joint modeling across all channels and time steps.
4. Monte Carlo sampling from the Gaussian copula.
5. Temporal correction: mild smoothing and ramp clipping.
6. Physical projection: non-negative power and night-time PV zeroing.
7. Unified evaluation.

## Run a quick demo

```bash
pip install -r requirements.txt
python demo_mock_run.py
```

## Run full pipeline on your data

```bash
python traditional_copula_baseline.py pipeline \
  --data-dir outputs/dataset \
  --output-dir outputs/models/traditional_copula \
  --group-cols event_type_code,severity_level,low_wind_flag,low_irradiance_flag \
  --fallback-group-cols event_type_code,severity_level;event_type_code;global \
  --min-group-size 8 \
  --n-per-condition 1 \
  --imbalance-tau 0 \
  --acf-max-lag 12
```

## Separate commands

Fit:

```bash
python traditional_copula_baseline.py fit --data-dir outputs/dataset --output-dir outputs/models/traditional_copula
```

Generate:

```bash
python traditional_copula_baseline.py generate \
  --model-dir outputs/models/traditional_copula \
  --cond-csv outputs/models/traditional_copula/dataset_split/cond_test.csv \
  --output-dir outputs/models/traditional_copula/generation \
  --n-per-condition 1
```

Evaluate:

```bash
python traditional_copula_baseline.py evaluate \
  --real outputs/models/traditional_copula/dataset_split/X_test.npy \
  --generated outputs/models/traditional_copula/generation/generated_samples.npy \
  --cond outputs/models/traditional_copula/generation/generated_conditions.csv \
  --output-dir outputs/models/traditional_copula/evaluation \
  --model-name traditional_gaussian_copula
```

## Outputs

```text
traditional_copula_model.pkl
group_fit_summary.csv
pipeline_summary.json

dataset_split/
  X_train.npy, X_test.npy, cond_train.csv, cond_test.csv

generation/
  generated_samples.npy          # [N, 3, T]
  generated_conditions.csv
  generated_samples_long.csv
  generation_summary.json

evaluation/
  metrics_summary.csv
  risk_metrics_real_vs_generated.csv
  metrics_by_event_type.csv
  metrics_by_severity_level.csv
  evaluation_summary.json
  figures/
```

## Evaluation metrics

Statistical fidelity:

- Wasserstein distance
- JS divergence
- ACF error
- correlation matrix error

Joint imbalance risk:

- cumulative deficit error
- max net-load ramp error
- imbalance duration error
- q95/q99 errors
- extreme degree match rate

Net load is defined as:

```text
N(t) = load(t) - wind_power(t) - solar_power(t)
```

Risk metrics use:

```text
cum_deficit = sum(max(0, N(t) - tau)) * delta_t
netload_ramp_max = max(N(t) - N(t-1))
imbalance_duration = sum(1(N(t) > tau)) * delta_t
```
