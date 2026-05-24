# 本次上传的数据包说明（2026-05-24）

本次随代码一并上传的数据包，主要用于 **OpenEnergyHub / CAISO 外部验证**。

## 1. 上传内容

### 1.1 原始数据

目录：
- `ExtremeScene/data/openenergyhub_caiso_raw/`

主要文件：
- `Database.csv`
- `download_metadata.json`

说明：
- `Database.csv` 是从 OpenEnergyHub / Mendeley 公共数据入口下载得到的原始表；
- 保留原始文件是为了后续复现字段映射、清洗和重采样过程。

---

### 1.2 清洗和标准化后的数据

目录：
- `ExtremeScene/data/processed/`

主要文件：
- `openenergyhub_caiso_5min_wind_solar_load_weather.csv`
- `openenergyhub_caiso_hourly_wind_solar_load_weather.csv`
- `openenergyhub_caiso_hourly_pipeline_input.csv`
- `openenergyhub_caiso_data_quality_report.md`

说明：
- `5min` 文件为标准字段名下的 5 分钟原始对齐版本；
- `hourly` 文件为 1 小时重采样版本；
- `hourly_pipeline_input` 是为了接入当前极端场景生成主流程所做的兼容输入表；
- `data_quality_report` 记录时间范围、缺失情况、负值检查、夜间光伏检查和外部验证可用性判断。

---

### 1.3 外部验证样本库（balanced_relaxed）

目录：
- `ExtremeScene/outputs/openenergyhub_caiso_threshold_sweep/balanced_relaxed/dataset/`

说明：
- 这是在 CAISO 数据上经过阈值扫描后选出的推荐样本库版本；
- 该版本在样本数、高风险覆盖和事件类型平衡之间相对更均衡；
- 主要用于后续外部验证方法对比。

典型文件包括：
- `X_train.npy`
- `X_val.npy`
- `X_test.npy`
- `cond_train.csv`
- `cond_val.csv`
- `cond_test.csv`
- `meta_train.csv`
- `meta_val.csv`
- `meta_test.csv`
- `event_mask_train.npy`
- `event_mask_val.npy`
- `event_mask_test.npy`
- `timeseries_input.csv`
- `dataset_summary.json`

---

## 2. 这些数据的用途

本次上传的数据主要用于以下任务：

1. OpenEnergyHub/CAISO 外部验证数据复现；
2. 外部极端样本库构建；
3. `balanced_relaxed` 外部验证口径下的方法比较；
4. TailWeighted / Month EVT-Copula / Simple Diffusion / GAN 增强方法的统一外部测试。

---

## 3. 未包含的内容

为了避免仓库体积过大，本次**没有**上传以下大规模中间结果：

- 各方法完整 `results/` 目录下的生成样本与评价输出；
- 多轮实验中间 checkpoint；
- 其他区域大体积原始 8760 文件；
- 大量重复性可再生输出。

这些内容仍建议本地保存或单独归档。

---

## 4. 推荐配合源码使用

与本次数据包配套的主要脚本包括：

- `prepare_openenergyhub_caiso.py`
- `run_openenergyhub_threshold_sweep.py`
- `run_paper_pipeline.py`
- `run_gan_augmented_tailweighted_copula.py`

如果后续要重建同样的数据流程，优先使用上述脚本，而不是手工改动数据文件。

