# Enhanced GAN Extreme Wind-Solar-Load Baseline

这个代码包用于复现并整理 **Enhanced GAN-Based Joint Wind-Solar-Load Scenario Generation With Extreme Weather Labelling** 的核心思想，作为论文中的 **GAN 极端场景生成对比基线**。

## 1. 方法定位

本代码实现的是对比基线，不是本文提出方法。它保留 Enhanced GAN 的主要思想：

- 极端天气/场景标签条件输入；
- 条件 GAN 框架；
- Wasserstein 距离 + Gradient Penalty，即 WGAN-GP；
- 生成器和判别器使用不同初始学习率，并随 epoch 衰减；
- 生成结果进行功率范围、爬坡约束、夜间光伏置零等物理修正；
- 输出统一格式，方便和扩散模型对比。

它**不包含**：

- EVT 尾部敏感损失；
- 联合失衡一致性损失；
- 分层条件扩散结构；
- 年度嵌入模块。

这样做是为了保证它作为 GAN 基线时不偷用本文方法的创新点。

## 2. 输入数据格式

推荐数据目录包含：

```text
X.npy
cond.csv
meta.csv  可选
```

其中：

```text
X.npy shape = [N, 3, T]
通道顺序 = [load, wind_power, solar_power]
```

也兼容 `[N, T, 3]`，代码会自动转为 `[N, 3, T]`。

`cond.csv` 至少建议包含：

```text
sample_id
event_type
event_type_code
month
low_wind_flag
low_irradiance_flag
duration_hours
severity_level
extreme_prob
cum_deficit
netload_ramp_max
imbalance_duration
```

若已有划分，也可以提供：

```text
X_train.npy
X_test.npy
cond_train.csv
cond_test.csv
```

## 3. 一键 demo

```bash
python demo_mock_run.py
```

会生成 mock 极端样本，训练 2 个 epoch，生成样本并评价。

## 4. 训练

```bash
python enhanced_gan_extreme.py train ^
  --data-dir outputs/dataset ^
  --output-dir outputs/models/enhanced_gan ^
  --epochs 100 ^
  --batch-size 64 ^
  --n-critic 4 ^
  --gp-lambda 10 ^
  --device cpu
```

默认不使用 `extreme_prob` 作为条件，避免 GAN 基线过度接近本文 EVT 条件模型。若需要更强基线，可加：

```bash
--use-extreme-prob-condition
```

## 5. 生成

```bash
python enhanced_gan_extreme.py generate ^
  --model-dir outputs/models/enhanced_gan ^
  --cond-csv outputs/models/enhanced_gan/cond_test.csv ^
  --output-dir outputs/models/enhanced_gan/generation ^
  --n-per-condition 1 ^
  --device cpu
```

输出：

```text
generated_samples.npy          # [N, 3, T]
generated_samples_long.csv
generated_conditions.csv
generation_summary.json
```

## 6. 评价

```bash
python enhanced_gan_extreme.py evaluate ^
  --real outputs/models/enhanced_gan/X_test.npy ^
  --generated outputs/models/enhanced_gan/generation/generated_samples.npy ^
  --cond outputs/models/enhanced_gan/generation/generated_conditions.csv ^
  --output-dir outputs/models/enhanced_gan/evaluation ^
  --model-name enhanced_gan
```

评价指标包括：

### 统计真实性与相关性

- Wasserstein 距离：load / wind / solar 及均值；
- JS 散度：load / wind / solar 及均值；
- 自相关函数误差 ACF-MAE：load / wind / solar 及均值；
- 相关系数矩阵误差：Frobenius norm 和 MAE。

### 联合失衡风险指标

统一净负荷：

```text
N(t) = L(t) - W(t) - S(t)
```

指标：

- 累计缺额误差；
- 最大净负荷爬坡误差；
- 持续失衡时长误差；
- 95% / 99% 分位误差；
- 极端等级匹配率。

## 7. 一键 pipeline

```bash
python enhanced_gan_extreme.py pipeline ^
  --data-dir outputs/dataset ^
  --output-dir outputs/models/enhanced_gan ^
  --epochs 100 ^
  --batch-size 64 ^
  --n-per-condition 1 ^
  --device cpu
```

## 8. 主要输出文件

```text
enhanced_gan_model.pt
scaler.json
condition_meta.json
physical_limits.npz
training_history.csv
training_summary.json
X_train.npy / X_val.npy / X_test.npy
cond_train.csv / cond_val.csv / cond_test.csv
generation/generated_samples.npy
evaluation/metrics_summary.csv
evaluation/risk_metrics_real_vs_generated.csv
evaluation/figures/*.png
```

## 9. 论文中建议命名

中文：

```text
计及极端天气标签的增强 WGAN-GP 方法
```

英文：

```text
EW-Labeled Enhanced WGAN-GP
```

论文中建议说明：该方法作为 GAN 类深度生成基线，用于比较本文条件扩散模型在统计分布、时间连续性、风光荷相关性和联合失衡风险刻画方面的改进。
