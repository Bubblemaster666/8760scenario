# Plain DDPM Baseline

这是“只用普通扩散”的对比组，作为最干净的扩散模型基线。

## 定位

该模型只学习风光荷极端片段的经验分布，不使用任何增强机制：

- 不使用天气/事件标签条件；
- 不使用 EVT `extreme_prob` 或 `tail_score`；
- 不使用尾部敏感损失；
- 不使用联合失衡风险一致性损失；
- 不使用背景—过程—风险分层条件结构；
- 不使用联合时空相关增强模块；
- 不使用 GAN 或 Copula。

它用于回答：**仅使用标准 DDPM 学习极端样本分布，能够达到什么水平？**

## 输入格式

`X.npy`：

```text
shape = [N, 3, T]
channel order = [load, wind_power, solar_power]
```

也兼容 `[N, T, 3]`，脚本会自动转换。

可选：

```text
cond.csv
meta.csv
```

它们不参与训练，只用于测试划分、分组评价和生成结果记录。

## 运行 demo

```bash
pip install -r requirements.txt
python demo_mock_run.py
```

## 训练

```bash
python plain_ddpm_baseline.py train \
  --data-dir outputs/dataset \
  --output-dir outputs/models/plain_ddpm \
  --epochs 100 \
  --batch-size 64 \
  --diffusion-steps 200 \
  --device cpu
```

## 生成

```bash
python plain_ddpm_baseline.py generate \
  --model-dir outputs/models/plain_ddpm \
  --output-dir outputs/models/plain_ddpm/generation \
  --use-test-count \
  --device cpu
```

## 评价

```bash
python plain_ddpm_baseline.py evaluate \
  --real outputs/models/plain_ddpm/X_test.npy \
  --generated outputs/models/plain_ddpm/generation/generated_samples.npy \
  --cond outputs/models/plain_ddpm/cond_test.csv \
  --output-dir outputs/models/plain_ddpm/evaluation \
  --model-name plain_ddpm
```

## 一键 pipeline

```bash
python plain_ddpm_baseline.py pipeline \
  --data-dir outputs/dataset \
  --output-dir outputs/models/plain_ddpm \
  --epochs 100 \
  --batch-size 64 \
  --diffusion-steps 200 \
  --device cpu
```

## 输出

```text
best_model.pt
normalization_stats.npz
training_history.csv
summary.json

generation/generated_samples.npy
generation/generated_samples_long.csv
generation/generated_conditions.csv

evaluation/metrics_summary.csv
evaluation/risk_metrics_real_vs_generated.csv
evaluation/metrics_by_event_type.csv
evaluation/metrics_by_severity_level.csv
```

## 论文中建议名称

中文：普通 DDPM 基线 / 无条件扩散基线

英文：Plain DDPM / Unconditional DDPM Baseline

## 与其他对比组的区别

| 方法 | 使用内容 |
|---|---|
| Plain DDPM | 仅标准 DDPM 噪声预测损失 |
| 普通条件扩散 | DDPM + 普通条件向量 |
| 联合时空增强扩散 | DDPM + 时间/变量相关增强 + 条件调制 |
| 本文方法 | 分层条件 + EVT 连续概率 + 尾部敏感训练 + 联合失衡风险一致性约束 |
