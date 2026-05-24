# OpenEnergyHub CAISO Data Quality Report

## 1. Download Status
- Download success: `True`
- OpenEnergyHub dataset page: https://openenergyhub.ornl.gov/explore/dataset/renewable-energy-and-electricity-demand-time-series-dataset-with-exogenous-varia/
- Linked Mendeley page: https://data.mendeley.com/datasets/fdfftr3tc2/1
- Raw file source used: `existing-local-file`
- Raw file size (bytes): `29752780`
- Note: Raw CSV already existed locally and was reused.

## 2. Raw-to-Standard Field Mapping
| Original field | Standard field |
|---|---|
| `Time` | `time` |
| `Electric_demand` | `load` |
| `Wind_production` | `wind_power` |
| `PV_production` | `solar_power` |
| `Temperature` | `temp` |
| `Humidity` | `humidity` |
| `Wind_speed` | `wind_speed` |
| `DHI` | `dhi` |
| `DNI` | `dni` |
| `GHI` | `ghi` |
| `Season` | `season_code_raw` |
| `Day_of_the_week` | `day_of_week_raw` |

## 3. Time Coverage
- Start time: `2019-01-01T00:00:00`
- End time: `2021-12-31T23:55:00`
- Raw record count: `315648`
- Distinct timestamps after de-duplication: `315648`
- 5 min aligned record count: `315648`
- Hourly resampled record count: `26304`
- Strict 5-minute interval before reindexing: `True`
- Duplicate timestamps: `0`
- Missing timestamps in full 5-minute grid: `0`
- Invalid timestamps: `0`
- Timezone info present: `False`
- Note on timezone: The source timestamps are naive strings without explicit timezone metadata.

## 4. Missing Rate by Field
| Field | Missing rate |
|---|---:|
| `time` | 0.000000 |
| `load` | 0.000000 |
| `wind_power` | 0.000000 |
| `solar_power` | 0.000000 |
| `temp` | 0.000000 |
| `humidity` | 0.000000 |
| `wind_speed` | 0.000000 |
| `dhi` | 0.000000 |
| `dni` | 0.000000 |
| `ghi` | 0.000000 |
| `season` | 0.000000 |
| `day_of_week` | 0.000000 |
| `source_row` | 0.000000 |
| `season_code_raw` | 0.000000 |
| `day_of_week_raw` | 0.000000 |

## 5. Wind/Solar/Load Summary
| Variable | min | max | mean | q95 | q99 |
|---|---:|---:|---:|---:|---:|
| load | 14662.000000 | 47067.000000 | 24833.695287 | 35349.000000 | 40531.530000 |
| wind_power | 0.000000 | 5743.000000 | 2019.745625 | 4328.000000 | 4911.000000 |
| solar_power | 0.000000 | 13191.000000 | 3567.493895 | 11210.000000 | 12333.000000 |

## 6. Physical Checks
- Raw negative values before cleaning:
  - `load`: 0
  - `wind_power`: 216
  - `solar_power`: 112829
- Cleaning applied in processed outputs:
  - `load`, `wind_power`, `solar_power`, `wind_speed`, `dhi`, `dni`, `ghi` clipped at lower bound `0`
  - `humidity` clipped to `[0, 100]`
- Negative value count:
  - `load`: 0
  - `wind_power`: 0
  - `solar_power`: 0
- Night solar proxy used: `ghi <= 1 W/m^2`
- Night solar mean: `23.668374`
- Night solar max: `4539.000000`
- Spike check:
  - `load` threshold `56869.530000`, count `0`
  - `wind_power` threshold `11472.000000`, count `0`
  - `solar_power` threshold `36189.000000`, count `0`

## 7. Weather Variable Ranges
| Variable | min | max | mean | q95 | q99 |
|---|---:|---:|---:|---:|---:|
| temp | -0.540000 | 39.020000 | 17.472179 | 31.720000 | 34.460000 |
| humidity | 11.572000 | 88.688000 | 51.261462 | 77.916000 | 82.720000 |
| wind_speed | 0.660000 | 8.540000 | 2.545759 | 4.520000 | 5.660000 |
| dhi | 0.000000 | 431.000000 | 53.779907 | 189.600000 | 262.600000 |
| dni | 0.000000 | 999.800000 | 288.058615 | 897.400000 | 948.000000 |
| ghi | 0.000000 | 1058.200000 | 221.787985 | 843.000000 | 977.400000 |

Interpretation:
- `DHI/DNI/GHI` are irradiance-style variables in `W/m^2`, so hourly resampling used arithmetic mean rather than energy integration.
- `season` in the standardized outputs is recomputed from timestamp month using Northern Hemisphere California seasons because this dataset comes from CAISO.

## 8. Suitability for Extreme Scenario Generation
- The time axis is complete and is suitable for event window extraction.
- After cleaning, the main power variables do not contain negative values.
- Key fields are highly complete and the dataset is usable for external validation.
- However, night-time low-GHI periods still show non-trivial solar output, which suggests possible timezone or irradiance-power alignment issues.

## 9. Preliminary Event Identification Suggestions
- Heat-wave events: start with `temp` monthly or seasonal `q90 / q95`.
- Strong-wind events: start with `wind_speed` `q90 / q95`.
- Low-irradiance events: start with `ghi` `q10`.
- Sharp irradiance drop: use 3-hour `ghi` decline magnitude.
- Net-load extremes: define `net_load = load - wind_power - solar_power`.
- Joint imbalance risk: combine `cum_deficit`, 3-hour net-load ramp, and imbalance duration after choosing a month-specific net-load threshold.
