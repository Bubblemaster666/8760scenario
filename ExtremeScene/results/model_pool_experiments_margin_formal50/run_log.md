# Model Pool Margin Formal50 Run Log

## Environment

- Python: `C:\Users\13411\anaconda3\python.exe`
- Device: CPU
- CUDA: unavailable
- Working directory: `C:\Users\13411\Desktop\work\ExtremeScene\8760scenario\ExtremeScene`

## Code Changes

- Added `ValSelected_Model_Ensemble_Margin` support in `run_model_pool_experiments.py`.
- Added margin sweep for validation-set model selection:
  - `margin=0.00`
  - `margin=0.05`
  - `margin=0.10`
- Implemented selection rule:
  - Choose a non-TailWeighted candidate only when `best_non_copula_val_score <= tailweighted_val_score - margin`.
  - Otherwise fallback to `TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed`.
- Added formal50 GAN validation/test samples into the model-pool candidate set when existing results are available.
- Added per-dataset `margin_selection_summary.csv`.
- Added global `all_margins_selection_summary.csv`.
- Updated final report generation to distinguish:
  - overall best method across the full model pool;
  - best margin ensemble variant.

## Run Command

```powershell
$env:MPLCONFIGDIR='C:\Users\13411\Desktop\work\ExtremeScene\8760scenario\ExtremeScene\results\tmp_mplconfig'
& 'C:\Users\13411\anaconda3\python.exe' run_model_pool_experiments.py `
  --out-dir results\model_pool_experiments_margin_formal50 `
  --augmented-data-root outputs\model_pool_augmented_datasets_margin_formal50 `
  --epochs 50 `
  --k-candidates 20 `
  --candidate-count 20 `
  --selection-margins 0.00,0.05,0.10 `
  --alpha-tail 1.0 `
  --fixed-weights 0.35,0.25,0.20,0.20
```

## Datasets

- singleton: `outputs/ramp_window_retrain/datasets/dataset_window_3h`
- muswellbrook: `outputs/muswellbrook_ramp_window_retrain/datasets/dataset_window_3h`
- cessnock_or_newarea: `outputs/cessnock_south_ramp_window_retrain/datasets/dataset_window_3h`

## Key Results

Overall best method:

- `TransformerVAE_Augmented_TailWeighted_Copula`
- mean_risk_rank: `1.6667`
- mean_risk_score: `0.3547`
- wins_count: `2`
- top3_count: `3`

Margin ensemble results:

- `margin=0.00`: mean_risk_rank `3.0000`, mean_risk_score `0.5144`
- `margin=0.05`: mean_risk_rank `3.0000`, mean_risk_score `0.5144`
- `margin=0.10`: mean_risk_rank `4.3333`, mean_risk_score `0.5765`

TailWeighted Fixed baseline:

- mean_risk_rank: `5.6667`
- mean_risk_score: `0.5771`

## Conclusion

- The best overall formal50 model-pool method is `TransformerVAE_Augmented_TailWeighted_Copula`.
- The best margin ensemble variant is `margin=0.00`, with `margin=0.05` tied and preferred if a more conservative selector is needed.
- `margin=0.10` is too conservative in this run because it falls back to TailWeighted on cessnock_or_newarea and loses performance.
- The margin selector still shows validation overfitting risk because GAN can look very strong on validation but does not dominate on test.

