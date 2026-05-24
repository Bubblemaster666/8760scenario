# Ausgrid 三组区域数据说明（2026-05-24）

本次上传的是当前项目中实际用于区域实验对比的三组 **最终成品数据表**，即已经整理好的：

- 风电
- 光伏
- 负荷
- 天气

统一融合后的年度小时级 8760 CSV。

---

## 1. 三组区域

本次上传的三组区域为：

1. **Singleton / Singleton North**
2. **Muswellbrook**
3. **Cessnock South / new area**

---

## 2. 上传文件范围

### 2.1 Singleton

目录：
- `ExtremeScene/`

文件：
- `singleton_wind_solar_load_weather_hourly_8760_FY2019.csv`
- `singleton_wind_solar_load_weather_hourly_8760_FY2020.csv`
- `singleton_wind_solar_load_weather_hourly_8760_FY2021.csv`
- `singleton_wind_solar_load_weather_hourly_8760_FY2022.csv`

### 2.2 Muswellbrook

目录：
- `ExtremeScene/`

文件：
- `muswellbrook_wind_solar_load_weather_hourly_8760_FY2019.csv`
- `muswellbrook_wind_solar_load_weather_hourly_8760_FY2020.csv`
- `muswellbrook_wind_solar_load_weather_hourly_8760_FY2021.csv`
- `muswellbrook_wind_solar_load_weather_hourly_8760_FY2022.csv`

### 2.3 Cessnock South / new area

目录：
- `ausgrid_output/`

文件：
- `cessnock_south_wind_solar_load_weather_hourly_8760_FY2019.csv`
- `cessnock_south_wind_solar_load_weather_hourly_8760_FY2020.csv`
- `cessnock_south_wind_solar_load_weather_hourly_8760_FY2021.csv`
- `cessnock_south_wind_solar_load_weather_hourly_8760_FY2022.csv`

说明：
- 当前 `Cessnock South` 的成品表还保存在 `ausgrid_output/` 下；
- 这是历史整理路径遗留，不影响使用；
- 后续如果你希望目录统一，我可以再帮你整理到 `ExtremeScene/` 或单独 `datasets/` 目录。

---

## 3. 数据口径

这些 CSV 都是**最终成品输入表**，用于后续：

- 极端事件识别
- 样本库构造
- 风光荷联合极端场景生成训练
- 各方法区域对比实验

相对于原始负荷、天气、风光中间文件，这些表已经是更适合直接接入主流程的版本。

---

## 4. 本次没有一并上传的内容

本次只上传三组区域的**成品 8760 小时级融合表**，没有一起上传：

- 原始负荷分路文件
- 原始天气中间表
- 原始风光拼接过程文件
- 其他实验输出和中间结果

这样做的目的是让仓库优先保留“可直接复现实验的数据成品”，避免数据层次过于混杂。

---

## 5. 后续建议

如果后续希望结构更统一，建议把三组区域的成品数据进一步整理成：

- `datasets/singleton/`
- `datasets/muswellbrook/`
- `datasets/cessnock_or_newarea/`

并统一附带：

- README / 数据字段说明
- 时间范围说明
- 缺失值与处理说明

这样更适合后续论文复现和外部协作。

