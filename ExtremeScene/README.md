# ExtremeScene

This folder contains the paper-oriented workflow for:

`重大天气事件下风光荷联合极端场景生成及年度嵌入`

The code keeps the original mock-data and legacy diffusion scripts, and adds a
modular pipeline for:

- unified dataset export for mock and real data
- unified joint-imbalance risk metrics
- EVT probability and severity labeling
- hierarchical EVT-risk diffusion training with ablations
- unified generation and evaluation
- lightweight annual embedding

## Research Goal

The main target is to generate joint load-wind-solar extreme segments that keep:

- statistical realism
- temporal continuity
- cross-variable correlation
- stronger joint imbalance risk semantics under major weather events
- compatibility with lightweight annual embedding into background scenarios

## Data Format

The unified dataset interface exports:

- `X.npy`
  - shape `[N, 3, T]`
  - channel order is fixed:
    - `0 = load`
    - `1 = wind_power`
    - `2 = solar_power`
- `cond.csv`
  - at least includes:
    - `sample_id`
    - `event_type`
    - `event_type_code`
    - `month`
    - `season`
    - `low_wind_flag`
    - `low_irradiance_flag`
    - `duration_hours`
    - `extreme_prob`
    - `tail_score`
    - `severity_level`
    - `cum_deficit`
    - `netload_ramp_max`
    - `imbalance_duration`
    - `imbalance_tau`
- `meta.csv`
  - at least includes:
    - `sample_id`
    - `core_start_time`
    - `core_end_time`
    - `window_start_time`
    - `window_end_time`
    - `original_start_time`
    - `original_end_time`
- `event_type_mapping.json`
- split files:
  - `X_train.npy`, `X_val.npy`, `X_test.npy`
  - `cond_train.csv`, `cond_val.csv`, `cond_test.csv`
  - `meta_train.csv`, `meta_val.csv`, `meta_test.csv`

The canonical event types are:

- `寒潮`
- `暴雪/风吹雪`
- `大风/沙尘暴`
- `高温`

Low-resource states are not treated as main event classes. They are represented
through `low_wind_flag` and `low_irradiance_flag`.

## Core Modules

- `data_interface.py`
  - unified event detection -> risk metric -> EVT -> tensor export
- `risk_metrics.py`
  - shared net-load, cumulative deficit, ramp, and imbalance-duration logic
- `train_hierarchical_evt_diffusion.py`
  - proposed method with hierarchical background/process/risk conditions
- `generate_scenarios.py`
  - unified conditional generation output
- `evaluate_generation.py`
  - unified paper-oriented evaluation
- `annual_embedding.py`
  - lightweight annual embedding loop
- `run_experiments.py`
  - unified model/ablation runner
- `run_paper_pipeline.py`
  - end-to-end paper demo

## Quick Start With Mock Data

Run a small CPU-friendly demo:

```powershell
python run_paper_pipeline.py --use-mock --seq-len 24 --out-dir outputs/demo
```

This will:

- create a mock dataset
- split train/val/test
- train `traditional_baseline`, `conditional_ddpm`, `proposed`
- train the required ablations:
  - `no_evt`
  - `no_risk_loss`
  - `no_month`
  - `flat_condition`
- generate test scenarios
- evaluate them
- export `all_model_metrics.csv`

## Replace With Real Data

Prepare a CSV with at least:

- `time`
- `load`
- `wind_power`
- `solar_power`
- optional weather columns:
  - `temp`
  - `wind_speed`
  - `irradiance`
  - `snowfall`
  - `visibility`

Then run:

```powershell
python run_paper_pipeline.py --real-data-csv your_real_data.csv --seq-len 24 --out-dir outputs/real_case
```

If your columns differ, update the column mapping in `data_interface.py` or wire a
custom `ColumnMapping` into your own launcher.

## Train The Proposed Method

```powershell
python train_hierarchical_evt_diffusion.py --data-dir outputs/demo/dataset --out-dir outputs/demo/models/proposed --ablation full --device cpu
```

The trainer supports three stages:

- Stage 1: distribution learning
- Stage 2: tail reinforcement
- Stage 3: risk consistency learning

Recommended paper-level `proposed_final` configuration:

- `stage1_epochs = 12`
- `stage2_epochs = 10`
- `stage3_epochs = 2`
- `diffusion_steps = 100`
- `base_channels = 64`
- `batch_size = 32`
- `guidance_scale = 1.0`
- `lambda_tail = 0.25`
- `lambda_risk = 0.04`
- `lambda_cum = 1.0`
- `lambda_ramp = 0.25`
- `lambda_dur = 0.35`
- `lambda_recon = 0.05`
- `lambda_physics = 0.02`
- `lambda_resource = 0.02`
- `cond_dropout = 0.10`
- `ema_decay = 0.995`
- `learning_rate = 1e-4`
- `weight_decay = 1e-5`

The stage policy is intentionally conservative:

- Stage 1 uses diffusion loss plus light reconstruction and physical losses.
- Stage 2 adds tail-sensitive loss but keeps risk consistency disabled.
- Stage 3 adds risk consistency with a light global weight, where cumulative deficit is the dominant risk term and ramp/duration are auxiliary terms.

The main outputs are:

- `best_model.pt`
- `training_history.csv`
- `loss_curve.png`
- `condition_meta.json`
- `normalization_stats.npz`
- `summary.json`

## Run Ablation Experiments

```powershell
python run_experiments.py --data-dir outputs/demo/dataset --out-dir outputs/final --methods proposed no_evt no_risk_loss no_month flat_condition --stage1-epochs 12 --stage2-epochs 10 --stage3-epochs 2 --diffusion-steps 100 --base-channels 64 --batch-size 32
```

Supported ablations:

- `full`
- `no_evt`
- `no_risk_loss`
- `no_month`
- `flat_condition`

Each run records:

- `ablation_name`
- `innovation_flags`
- `ablation_note`
- full `train_config`

inside `summary.json`.

Recommended ablation settings keep the same formal training budget as the full model:

- `proposed/full`: EVT continuous risk, tail-sensitive loss, light risk consistency, month features, hierarchical condition.
- `no_evt`: removes continuous `extreme_prob` and `tail_score`; tail weighting falls back to `severity_level`.
- `no_risk_loss`: keeps EVT conditions and tail loss, but forces `lambda_risk = 0`.
- `no_month`: zeros the month sine/cosine background features.
- `flat_condition`: replaces hierarchical background/process/risk encoders with one flat condition encoder.

## Tuning Experiments

Run the six proposed-method tuning presets:

```powershell
python run_tuning_experiments.py --data-dir outputs/demo/dataset --out-dir outputs/tuning --device cpu
```

The presets are:

- `P1_tail_stable`: `12/10/0`, `lambda_tail=0.25`, `lambda_risk=0`, `guidance_scale=1.0`
- `P2_risk_light`: `12/10/2`, `lambda_tail=0.25`, `lambda_risk=0.04`, `guidance_scale=1.0`
- `P3_risk_mid`: `16/10/4`, `lambda_tail=0.25`, `lambda_risk=0.05`, `guidance_scale=1.0`
- `P4_tail_strong`: `12/10/2`, `lambda_tail=0.35`, `lambda_risk=0.04`, `guidance_scale=1.0`
- `P5_low_guidance`: `12/10/2`, `lambda_tail=0.25`, `lambda_risk=0.04`, `guidance_scale=0.8`
- `P6_high_guidance`: `12/10/2`, `lambda_tail=0.25`, `lambda_risk=0.04`, `guidance_scale=1.2`

All presets share `diffusion_steps=100`, `base_channels=64`, `batch_size=32`, `lambda_cum=1.0`, `lambda_ramp=0.25`, `lambda_dur=0.35`, `lambda_recon=0.05`, `lambda_physics=0.02`, `lambda_resource=0.02`, `cond_dropout=0.10`, `ema_decay=0.995`, `learning_rate=1e-4`, and `weight_decay=1e-5`.

Outputs:

- `outputs/tuning/<preset>/best_model.pt`
- `outputs/tuning/<preset>/generated_samples.npy`
- `outputs/tuning/<preset>/evaluation/metrics_summary.csv`
- `outputs/tuning/tuning_metrics_summary.csv`

`tuning_metrics_summary.csv` includes a weighted `rank_score` across cumulative deficit, q99 cumulative deficit, distribution, ACF, correlation, ramp, and duration metrics. Lower is better.

## Unified Scenario Generation

```powershell
python generate_scenarios.py --checkpoint outputs/demo/models/proposed/best_model.pt --data-dir outputs/demo/dataset --out-dir outputs/demo/models/proposed --split test
```

Outputs:

- `generated_samples.npy`
- `generated_samples_long.csv`
- `generation_summary.json`

The generation post-process applies:

- `load >= 0`
- `wind_power >= 0`
- `solar_power >= 0`
- `night solar_power = 0`

## Unified Evaluation

```powershell
python evaluate_generation.py --real outputs/demo/dataset/X_test.npy --generated outputs/demo/models/proposed/generated_samples.npy --cond outputs/demo/dataset/cond_test.csv --meta outputs/demo/dataset/meta_test.csv --model-name proposed --out-dir outputs/demo/evaluations/proposed
```

Main outputs:

- `metrics_summary.csv`
- `metrics_by_event_type.csv`
- `metrics_by_severity.csv`
- `risk_metrics_real_vs_generated.csv`
- `figures/`

Key figures include:

- `typical_generated_curve.png`
- `netload_curve.png`
- `acf_comparison.png`
- `corr_matrix_real.png`
- `corr_matrix_generated.png`
- `risk_metric_boxplot.png`

## Unified Experiment Runner

```powershell
python run_experiments.py --data-dir outputs/demo/dataset --out-dir outputs/demo
```

Default methods:

- `traditional_gaussian_copula`
- `plain_diffusion_baseline`
- `enhanced_gan`
- `proposed`
- `no_evt`
- `no_risk_loss`
- `no_month`
- `flat_condition`

It exports:

- `outputs/demo/evaluations/all_model_metrics.csv`

The proposed and ablation methods now use the paper-level default training
configuration unless command-line arguments override it.

## Annual Embedding

```powershell
python annual_embedding.py --background outputs/demo/annual/regular_background.csv --segments outputs/demo/models/proposed/generated_samples.npy --cond outputs/demo/dataset/cond_test.csv --out-dir outputs/demo/annual_embedding
```

Or run a mock background:

```powershell
python annual_embedding.py --use-mock-background --segments outputs/demo/models/proposed/generated_samples.npy --cond outputs/demo/dataset/cond_test.csv --out-dir outputs/demo/annual_embedding
```

Outputs:

- `annual_scenario_baseline.csv`
- `annual_scenario_proposed.csv`
- `annual_embedding_summary.csv`
- `boundary_smoothing_metrics.csv`
- `annual_embedding_boundary.png`

## End-to-End Demo Command

```powershell
python run_paper_pipeline.py --use-mock --seq-len 24 --out-dir outputs/demo
```

## Notes

- The proposed model keeps hierarchical conditions as a real code structure.
- `flat_condition` is a real flat-condition ablation, not just a renamed tensor.
- Evaluation uses the same risk metric definitions as dataset labeling.
- The workflow supports CPU-only runs. Small mock runs are designed to finish on CPU.
