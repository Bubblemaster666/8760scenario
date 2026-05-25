# Model Pool Margin Formal50 Experiment Report

## Rank Summary

| method                                                           |   singleton_risk_rank |   singleton_risk_score |   muswellbrook_risk_rank |   muswellbrook_risk_score |   cessnock_or_newarea_risk_rank |   cessnock_or_newarea_risk_score |   mean_risk_rank |   mean_risk_score |   wins_count |   top3_count |
|:-----------------------------------------------------------------|----------------------:|-----------------------:|-------------------------:|--------------------------:|--------------------------------:|---------------------------------:|-----------------:|------------------:|-------------:|-------------:|
| TransformerVAE_Augmented_TailWeighted_Copula                     |                     1 |               0.277843 |                        1 |                  0.236186 |                               3 |                         0.550024 |          1.66667 |          0.354684 |            2 |            3 |
| ValSelected_Model_Ensemble                                       |                     3 |               0.488845 |                        2 |                  0.309064 |                               4 |                         0.745179 |          3       |          0.514362 |            0 |            2 |
| ValSelected_Model_Ensemble_Margin_0.00                           |                     3 |               0.488845 |                        2 |                  0.309064 |                               4 |                         0.745179 |          3       |          0.514362 |            0 |            2 |
| ValSelected_Model_Ensemble_Margin_0.05                           |                     3 |               0.488845 |                        2 |                  0.309064 |                               4 |                         0.745179 |          3       |          0.514362 |            0 |            2 |
| Conditional_Transformer_Risk_Generator                           |                     9 |               0.656189 |                        2 |                  0.309064 |                               2 |                         0.379755 |          4.33333 |          0.448336 |            0 |            2 |
| ValSelected_Model_Ensemble_Margin_0.10                           |                     3 |               0.488845 |                        2 |                  0.309064 |                               8 |                         0.931691 |          4.33333 |          0.576533 |            0 |            2 |
| GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed |                     3 |               0.488845 |                        8 |                  0.645361 |                               4 |                         0.745179 |          5       |          0.626461 |            0 |            1 |
| TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed               |                     2 |               0.345736 |                        7 |                  0.453921 |                               8 |                         0.931691 |          5.66667 |          0.577116 |            0 |            1 |
| Conditional_NormalizingFlow_Risk_Generator                       |                     8 |               0.6      |                        9 |                  0.858692 |                               1 |                         0.11137  |          6       |          0.523354 |            1 |            1 |

## Margin Selection Summary

|   margin | method_name                            | selected_methods                                                                                                                                                                                                      |   mean_rank |   mean_score |
|---------:|:---------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|-------------:|
|     0    | ValSelected_Model_Ensemble_Margin_0.00 | singleton=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed, muswellbrook=Conditional_Transformer_Risk_Generator, cessnock_or_newarea=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed |     3       |     0.514362 |
|     0.05 | ValSelected_Model_Ensemble_Margin_0.05 | singleton=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed, muswellbrook=Conditional_Transformer_Risk_Generator, cessnock_or_newarea=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed |     3       |     0.514362 |
|     0.1  | ValSelected_Model_Ensemble_Margin_0.10 | singleton=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed, muswellbrook=Conditional_Transformer_Risk_Generator, cessnock_or_newarea=TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed               |     4.33333 |     0.576533 |

## Validation Model Selection

| dataset             | method                                                           |   highrisk_wasserstein |   highrisk_acf_mae |   extreme_degree_match_rate |   val_q99 |   val_core_q99 |   val_ramp |   val_duration |   mean_wasserstein |   mean_js |   acf_mae |   corr_matrix_error |   norm_q99_cum_deficit_error |   norm_core_q99_cum_deficit_error |   norm_netload_ramp_max_mae |   norm_imbalance_duration_mae |   val_risk_score |   risk_rank |   priority | selected_margin_0.00   | selected_margin_0.05   | selected_margin_0.10   |
|:--------------------|:-----------------------------------------------------------------|-----------------------:|-------------------:|----------------------------:|----------:|---------------:|-----------:|---------------:|-------------------:|----------:|----------:|--------------------:|-----------------------------:|----------------------------------:|----------------------------:|------------------------------:|-----------------:|------------:|-----------:|:-----------------------|:-----------------------|:-----------------------|
| singleton           | GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed |               1.16485  |          0.0882506 |                    0.44     |  54.1453  |      28.7111   |    1.40545 |        8.32    |           0.306206 | 0.0609278 | 0.0538899 |           0.0625164 |                    0         |                          0        |                  0          |                     0.133333  |         0.02     |           1 |          2 | True                   | True                   | True                   |
| singleton           | TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed               |               1.18855  |          0.103698  |                    0.48     |  54.5373  |      32.7539   |    1.54125 |        8.28    |           0.270007 | 0.0617437 | 0.0522831 |           0.0587675 |                    0.0114866 |                          0.399336 |                  0.640247   |                     0.0666667 |         0.293309 |           2 |          0 | False                  | False                  | False                  |
| singleton           | TransformerVAE_Augmented_TailWeighted_Copula                     |               1.15227  |          0.085932  |                    0.44     |  61.7995  |      31.2534   |    1.61756 |        8.24    |           0.246601 | 0.0546905 | 0.0508096 |           0.04861   |                    0.224303  |                          0.251118 |                  1          |                     0         |         0.392626 |           3 |          1 | False                  | False                  | False                  |
| singleton           | Conditional_Transformer_Risk_Generator                           |               0.957314 |          0.0394877 |                    0.44     |  73.923   |      36.3045   |    1.53392 |        8.84    |           0.266614 | 0.0990684 | 0.0303697 |           0.0727697 |                    0.579578  |                          0.750059 |                  0.605672   |                     1         |         0.700309 |           4 |          3 | False                  | False                  | False                  |
| singleton           | Conditional_NormalizingFlow_Risk_Generator                       |               1.0182   |          0.0341831 |                    0.32     |  88.2695  |      38.8349   |    1.40588 |        8.64    |           0.371756 | 0.0895068 | 0.0450963 |           0.0606911 |                    1         |                          1        |                  0.00205359 |                     0.666667  |         0.700513 |           5 |          4 | False                  | False                  | False                  |
| muswellbrook        | Conditional_Transformer_Risk_Generator                           |               0.570606 |          0.0561831 |                    0.416667 |   2.44948 |       0.349007 |    1.65774 |        8.54167 |           0.232001 | 0.0756532 | 0.0480902 |           0.0545251 |                    0         |                          0        |                  0.490754   |                     0.529412  |         0.2021   |           1 |          3 | True                   | True                   | True                   |
| muswellbrook        | Conditional_NormalizingFlow_Risk_Generator                       |               0.703517 |          0.0515922 |                    0.375    |   7.55197 |       2.8922   |    1.55848 |        8.16667 |           0.331647 | 0.102742  | 0.0686659 |           0.0898659 |                    0.356056  |                          0.434728 |                  0.0985196  |                     0         |         0.261865 |           2 |          4 | False                  | False                  | False                  |
| muswellbrook        | TransformerVAE_Augmented_TailWeighted_Copula                     |               0.469477 |          0.0783737 |                    0.458333 |  16.6977  |       1.48745  |    1.65183 |        8.875   |           0.296562 | 0.0649029 | 0.0718155 |           0.0516559 |                    0.994253  |                          0.194603 |                  0.467418   |                     1         |         0.623511 |           3 |          1 | False                  | False                  | False                  |
| muswellbrook        | GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed |               0.478135 |          0.0628239 |                    0.375    |  16.7801  |       4.69994  |    1.53355 |        8.75    |           0.227829 | 0.0549296 | 0.0581239 |           0.0567208 |                    1         |                          0.74374  |                  0          |                     0.823529  |         0.646652 |           4 |          2 | False                  | False                  | False                  |
| muswellbrook        | TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed               |               0.622163 |          0.0701285 |                    0.375    |   7.95347 |       6.19908  |    1.7866  |        8.70833 |           0.213343 | 0.0545945 | 0.0691758 |           0.062315  |                    0.384074  |                          1        |                  1          |                     0.764706  |         0.779928 |           5 |          0 | False                  | False                  | False                  |
| cessnock_or_newarea | GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed |               0.86663  |          0.0955831 |                    0.307692 |  34.2382  |      17.1909   |    2.0053  |        4.23077 |           0.358166 | 0.0482403 | 0.0772103 |           0.0655813 |                    0.374275  |                          0.704616 |                  0          |                     0         |         0.323667 |           1 |          2 | True                   | True                   | False                  |
| cessnock_or_newarea | Conditional_Transformer_Risk_Generator                           |               0.559464 |          0.0697044 |                    0.269231 |  21.1815  |       3.88149  |    2.52245 |        6.46154 |           0.267285 | 0.0615692 | 0.035034  |           0.0316897 |                    0         |                          0        |                  1          |                     1         |         0.4      |           2 |          3 | False                  | False                  | False                  |
| cessnock_or_newarea | TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed               |               0.945348 |          0.101974  |                    0.384615 |  36.2614  |      16.6199   |    2.1198  |        4.73077 |           0.335754 | 0.0427612 | 0.0775141 |           0.045986  |                    0.43227   |                          0.674388 |                  0.221407   |                     0.224138  |         0.42097  |           3 |          0 | False                  | False                  | True                   |
| cessnock_or_newarea | TransformerVAE_Augmented_TailWeighted_Copula                     |               0.636302 |          0.1047    |                    0.307692 |  38.4263  |      22.7704   |    2.17883 |        4.80769 |           0.311456 | 0.0429543 | 0.0731875 |           0.0543903 |                    0.494328  |                          1        |                  0.33555    |                     0.258621  |         0.570979 |           4 |          1 | False                  | False                  | False                  |
| cessnock_or_newarea | Conditional_NormalizingFlow_Risk_Generator                       |               1.22934  |          0.0896345 |                    0.461538 |  56.0669  |      17.6072   |    2.30148 |        5.26923 |           0.554397 | 0.0841785 | 0.060588  |           0.0778474 |                    1         |                          0.726655 |                  0.572716   |                     0.465517  |         0.731003 |           5 |          4 | False                  | False                  | False                  |

【实验结论】
- 最优方法：TransformerVAE_Augmented_TailWeighted_Copula
- 最优 margin：0.00
- 是否超过 TailWeighted Fixed：是
- 是否推荐作为最终主方法：是

【各 margin 对比】
margin=0.00:
- selected methods: singleton=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed, muswellbrook=Conditional_Transformer_Risk_Generator, cessnock_or_newarea=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed
- mean_rank: 3.0000
- mean_score: 0.5144

margin=0.05:
- selected methods: singleton=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed, muswellbrook=Conditional_Transformer_Risk_Generator, cessnock_or_newarea=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed
- mean_rank: 3.0000
- mean_score: 0.5144

margin=0.10:
- selected methods: singleton=GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed, muswellbrook=Conditional_Transformer_Risk_Generator, cessnock_or_newarea=TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed
- mean_rank: 4.3333
- mean_score: 0.5765

【最终推荐】
- 推荐使用哪个 margin：0.05
- 原因：在 margin ensemble 内，该 margin 的 mean_rank=3.0000、mean_score=0.5144；TailWeighted Fixed 的 mean_rank=5.6667、mean_score=0.5771。
- 额外结论：本轮全局最优是 TransformerVAE_Augmented_TailWeighted_Copula，mean_rank=1.6667、mean_score=0.3547；margin ensemble 没有超过它。

【各数据集选择】
singleton:
- selected method: GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed
- reason: GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed val_score 0.020000 <= TailWeighted 0.293309 - margin 0.05

muswellbrook:
- selected method: Conditional_Transformer_Risk_Generator
- reason: Conditional_Transformer_Risk_Generator val_score 0.202100 <= TailWeighted 0.779928 - margin 0.05

cessnock_or_newarea:
- selected method: GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed
- reason: GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed val_score 0.323667 <= TailWeighted 0.420970 - margin 0.05

【风险】
- 是否可能验证集过拟合：是，尤其是 GAN 在验证集上分数很低，但测试集没有对应领先，说明单纯 margin 不能完全避免过拟合选择。
- 是否需要多随机种子：是，建议至少 3 个 seed 验证 selection margin 的稳定性。
- 下一步最小改动：保持模型池不变，增加多 seed 复验；若继续使用 ensemble，建议增加非 Copula 候选的测试前验证稳健性约束，而不是继续新增模型。