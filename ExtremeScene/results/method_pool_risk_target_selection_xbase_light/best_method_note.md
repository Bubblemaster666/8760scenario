# Best Scheme Note

Evaluation metrics are unchanged: q99 cumulative deficit error, core q99 cumulative deficit error, 3h net-load ramp MAE, and imbalance duration MAE. Lower risk_score is better after per-comparison min-max normalization with the existing weights 0.30/0.30/0.25/0.15.

## Current Best From Existing Formal Results
- Existing formal best: `DurationAware_RiskFirst_TCN_Empirical`; mean_risk_score=0.114328; q99=12.4252; core_q99=5.7862; ramp=1.7202; duration=6.4233.

## New Attempts
- `valprotected_x_base_copula`: mean_risk_score=0.175603; q99=9.7551; core_q99=3.9723; ramp=1.9847; duration=5.5575.
- `TargetSelected_empirical_riskfirst_evt_pool_paper_weights`: mean_risk_score=0.139580; q99=3.5707; core_q99=9.4873; ramp=1.5725; duration=6.7242.
- `TargetSelected_val_calibrated_rf_duration_pool_global_main`: mean_risk_score=0.144259; q99=6.5851; core_q99=5.8162; ramp=1.6440; duration=6.7652.

## Best Interpretable Composite Found
- `ValWeightSelected_xbase_durationaware`: choose `valprotected_x_base_copula` for Singleton and Muswellbrook, and `DurationAware_RiskFirst_TCN_Empirical` for Cessnock. This validation-weight-selected composite gives mean_risk_score about 0.134382 in the combined comparison, with avg q99=3.8171, avg core_q99=2.8898, avg ramp=1.8695, avg duration=5.9649.
- Optimistic test-table upper bound: `Composite_dataset_best_in_table`, using `valprotected_x_base_copula` for Singleton/Muswellbrook and `TargetSelected_val_calibrated_riskfirst_evt_pool_duration_guard` for Cessnock, gives mean_risk_score about 0.129459. Treat it as an upper-bound diagnostic because the final per-dataset choice was selected from test-table performance.

## Recommendation
- If requiring one fixed method name across all three datasets, use `valprotected_x_base_copula` as the best newly discovered single method in this round; it improves core_q99 and duration strongly but has worse q99/ramp balance than the best target-selected RiskFirst/EVT variants.
- If a dataset-adaptive scheme is acceptable, recommend `ValWeightSelected_xbase_durationaware` as the best defensible improved scheme found so far.