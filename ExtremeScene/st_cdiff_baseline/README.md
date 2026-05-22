# ST-CDiff：联合时空相关增强型条件扩散对比组

该代码包实现一个**改进扩散模型对比组**，用于风光荷极端场景生成实验中的 baseline。

定位：

- 强于普通条件扩散：加入时间相关增强、风光荷通道耦合增强、条件自适应融合。
- 弱于本文 proposed 方法：**不使用** EVT 连续极端概率、tail_score、尾部敏感损失、联合失衡一致性损失、背景-过程-风险分层条件结构和三阶段训练。

因此它适合用于证明：仅增强时空相关性能够改善 ACF 和相关系数矩阵，但不足以充分刻画累计缺额、最大净负荷爬坡、持续失衡时长等极端风险指标。

## 输入格式

`data_dir` 目录下需要：

```text
X.npy       # shape = [N, 3, T]，也兼容 [N, T, 3]
cond.csv
meta.csv    # 可选
```

通道顺序固定：

```text
0 = load
1 = wind_power
2 = solar_power
```

`cond.csv` 建议包含：

```text
sample_id
event_type
event_type_code
month
low_wind_flag
low_irradiance_flag
duration_hours
severity_level
```

注意：本对照组不会使用 `extreme_prob`、`tail_score`、`cum_deficit`、`netload_ramp_max`、`imbalance_duration` 作为训练输入或损失。

## 模型结构

ST-CDiff = 普通 DDPM + 三个轻量增强模块：

1. `Temporal Enhancement Module`：残差 Conv1D + dilation + 轻量时间注意力，用于增强时间连续性。
2. `Channel Relation Module`：3×3 可学习通道混合矩阵，用于增强 load / wind / solar 耦合关系。
3. `Condition Gating Module`：FiLM 条件调制，用于自适应融合事件类型、月份、资源状态、持续时间、等级等条件。

训练目标仅为标准 DDPM 噪声预测损失：

```text
loss = MSE(eps_pred, eps)
```

## 快速 demo

```bash
pip install -r requirements.txt
python demo_mock_run.py
```

## 训练

```bash
python st_cdiff_baseline.py train \
  --data-dir outputs/dataset \
  --output-dir outputs/models/st_cdiff \
  --epochs 100 \
  --batch-size 64 \
  --diffusion-steps 100 \
  --base-channels 64 \
  --device cpu
```

## 生成

```bash
python st_cdiff_baseline.py generate \
  --model-dir outputs/models/st_cdiff \
  --cond-csv outputs/models/st_cdiff/dataset_split/cond_test.csv \
  --output-dir outputs/models/st_cdiff/generation \
  --n-per-condition 1 \
  --device cpu
```

## 评价

```bash
python st_cdiff_baseline.py evaluate \
  --real outputs/models/st_cdiff/dataset_split/X_test.npy \
  --generated outputs/models/st_cdiff/generation/generated_samples.npy \
  --cond outputs/models/st_cdiff/generation/generated_conditions.csv \
  --output-dir outputs/models/st_cdiff/evaluation \
  --model-name ST-CDiff \
  --acf-max-lag 12 \
  --imbalance-tau 0
```

## 一键流程

```bash
python st_cdiff_baseline.py pipeline \
  --data-dir outputs/dataset \
  --output-dir outputs/models/st_cdiff \
  --epochs 100 \
  --batch-size 64 \
  --diffusion-steps 100 \
  --device cpu
```

## 输出

```text
model/
  st_cdiff_model.pt
  config.json
  scaler.json
  condition_meta.json
  training_history.csv
  loss_curve.png
  training_summary.json
  dataset_split/
    X_train.npy / X_val.npy / X_test.npy
    cond_train.csv / cond_val.csv / cond_test.csv
  generation/
    generated_samples.npy
    generated_conditions.csv
    generated_samples_long.csv
  evaluation/
    metrics_summary.csv
    risk_metrics_real_vs_generated.csv
    metrics_by_event_type.csv
    metrics_by_severity_level.csv
    figures/
```

## 论文建议命名

中文：联合时空相关增强型条件扩散模型

英文：ST-CDiff, Spatio-Temporal Enhanced Conditional Diffusion
