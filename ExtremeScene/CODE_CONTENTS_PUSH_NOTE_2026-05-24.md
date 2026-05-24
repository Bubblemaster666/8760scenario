# 本次推送代码内容说明（2026-05-24）

本文档用于说明本次提交中新增或修改的核心源码文件、主要功能、输入输出以及推荐运行方式。

说明范围仅包含源码与说明文档，不包含大体积中间结果、训练输出目录、原始数据文件和生成样本文件。

---

## 1. 本次代码改动目标

本次推送主要围绕两个方向展开：

1. **OpenEnergyHub / CAISO 外部验证数据接入**
   - 下载、清洗、标准化 5 分钟与 1 小时风光荷气象数据；
   - 接入现有“重大天气事件下风光荷联合极端场景生成”管线；
   - 支持极端样本截取阈值扫描，以扩充外部验证样本库。

2. **GAN 增强的 TailWeighted Month EVT-Copula 方法试验**
   - GAN 仅用于训练集尾部样本增强；
   - Copula 仍作为最终场景生成统计先验；
   - 对 GAN 样本增加物理筛选、风险筛选和相关性筛选；
   - 在四组数据集上统一比较。

---

## 2. 新增/修改文件说明

### 2.1 `prepare_openenergyhub_caiso.py`

**作用：**
- 下载 OpenEnergyHub/CAISO 数据集；
- 解析原始字段；
- 生成标准化 5 分钟数据表和 1 小时重采样数据表；
- 输出数据质量检查报告。

**主要输出：**
- `data/openenergyhub_caiso_raw/Database.csv`
- `data/processed/openenergyhub_caiso_5min_wind_solar_load_weather.csv`
- `data/processed/openenergyhub_caiso_hourly_wind_solar_load_weather.csv`
- `data/processed/openenergyhub_caiso_data_quality_report.md`

**适用场景：**
- 外部验证数据准备；
- 新区域风光荷气象时序标准化。

---

### 2.2 `run_paper_pipeline.py`

**本次修改点：**
- 增加了可调的极端事件截取阈值参数；
- 增加了风险筛选样本数保护项；
- 便于对外部数据进行“宽松/平衡/更严格”不同口径的样本库扫描。

**本次新增的重要参数：**
- `min_event_hours`
- `use_adaptive_thresholds`
- `cold_temp_quantile`
- `cold_drop_24h_quantile`
- `adaptive_cold_drop_min`
- `heat_temp_quantile`
- `high_wind_speed_quantile`
- `snowfall_quantile`
- `min_samples_after_screen`

**作用：**
- 不改主模型结构，仅通过样本库构造阈值调节外部验证样本规模与风险纯度。

---

### 2.3 `run_openenergyhub_threshold_sweep.py`

**作用：**
- 对 CAISO 外部验证集运行多组极端事件截取阈值扫描；
- 比较不同阈值组合下的样本数、高风险样本覆盖率、事件类型平衡度和零累计缺额比例；
- 帮助选择更适合外部验证的极端样本库口径。

**当前使用过的典型配置：**
- `baseline_current`
- `balanced_relaxed`
- `relaxed_event_q95`
- `relaxed_loose_screen`
- `more_relaxed`

**主要输出：**
- `results/openenergyhub_caiso_threshold_sweep/openenergyhub_threshold_sweep_summary.csv`
- `results/openenergyhub_caiso_threshold_sweep/openenergyhub_threshold_sweep_report.md`

**当前建议：**
- `balanced_relaxed` 作为外部验证最均衡的样本库口径。

---

### 2.4 `run_gan_augmented_tailweighted_copula.py`

**作用：**
- 实现新方法：
  `GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed`

**核心流程：**
1. 读取训练集真实极端样本；
2. 计算联合失衡风险指标；
3. 基于 tail score 选择尾部真实样本；
4. 训练条件 WGAN-GP 尾部增强器；
5. 生成 GAN 候选尾部样本；
6. 对候选样本做：
   - 物理筛选
   - 风险筛选
   - 相关性筛选
7. 将保留下来的 GAN 样本加入训练集；
8. 在增强训练集上重新拟合 TailWeighted Month EVT-Copula；
9. 用固定风险筛选权重生成测试场景；
10. 输出四组数据集的统一评价结果。

**当前四组数据集：**
- `singleton`
- `muswellbrook`
- `cessnock_or_newarea`
- `openenergyhub_caiso_balanced_relaxed`

**主要输出目录：**
- `results/gan_augmented_tailweighted_copula/`

**主要输出文件：**
- `all_datasets_rank_summary.csv`
- `all_datasets_risk_summary.csv`
- `all_datasets_auxiliary_summary.csv`
- `final_gan_augmented_tailweighted_copula_report.md`

**当前实验结论：**
- GAN 训练能跑通；
- 但四组数据中，GAN 候选样本均在风险筛选阶段被全部剔除；
- 因此该方法当前实际退化为原始 `TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed`；
- 结论是“代码已实现，实验已完成，但当前版本不建议替代原主方法”。

---

## 3. 本次推送涉及的主要方法逻辑

### 3.1 TailWeighted Month EVT-Copula

已有逻辑保留：
- 月份 / 季节 / 全局 Copula fallback；
- 基于联合失衡风险指标构造 `tail_score`；
- 用 `sample_weight = 1 + alpha_tail * tail_score` 对训练样本加权；
- 根据 `extreme_prob` 与月份条件确定目标风险；
- 在候选 Copula 场景中按固定风险加权误差选择最终样本。

### 3.2 GAN 增强版本

新增逻辑：
- GAN 不直接生成最终测试场景；
- 只用于给训练集“补充高风险尾部样本”；
- 最终测试集场景仍由 TailWeighted Copula 生成。

这保持了 Copula 的统计稳定性，同时尝试提升尾部覆盖能力。

---

## 4. 推荐运行方式

### 4.1 数据下载与清洗

```powershell
python prepare_openenergyhub_caiso.py
```

### 4.2 CAISO 阈值扫描

```powershell
python run_openenergyhub_threshold_sweep.py
```

### 4.3 GAN 增强 TailWeighted Copula

```powershell
python run_gan_augmented_tailweighted_copula.py ^
  --out-dir results/gan_augmented_tailweighted_copula ^
  --augmented-data-root outputs/gan_augmented_datasets ^
  --k-candidates 20 ^
  --alpha-tail 1.0 ^
  --fixed-weights 0.35,0.25,0.20,0.20
```

---

## 5. 本次未纳入推送的内容

为了避免仓库过大或结果文件污染版本管理，本次默认**不建议**把以下内容一并推送：

- `outputs/`
- `results/`
- `data/openenergyhub_caiso_raw/`
- `data/processed/`
- 大体积 `.npy`
- 生成场景文件
- 训练中间检查点

这些内容适合本地保留或单独归档，不建议和源码混推。

---

## 6. 当前代码状态结论

本次推送后的代码状态可以概括为：

1. 已支持 OpenEnergyHub/CAISO 外部验证数据下载与标准化；
2. 已支持 CAISO 外部样本库阈值扫描；
3. 已支持四数据集统一运行 GAN 增强 TailWeighted Copula；
4. 当前 GAN 增强版本实现完整，但实验结果暂不建议作为最终主方法；
5. 当前更推荐继续保留：
   - `TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed`
   作为稳健主方法候选。

---

## 7. 后续最小改动建议

如果继续沿 GAN 增强方向迭代，最小建议不是继续加大 GAN 模型，而是先改样本保留机制：

1. 将风险筛选从硬阈值改成更温和的局部尾部筛选；
2. 按 `month + event_type` 局部训练和局部过滤 GAN 样本；
3. 先只增强单一事件类型（如高温或大风），验证是否能稳定保留 GAN 样本。

