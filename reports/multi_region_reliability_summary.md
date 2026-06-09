# Multi-region reliability summary

This external check applies the same baseline-first decision rule to Poltava, Vinnytsia, Cherkasy, and Ukraine.
Baseline methods are control models; ML is recommended only when it clears the practical MAE margin.

## Cherkasy

- Maize: XGBoost (within expected error), gain 0.33 t/ha, coverage 100.0%.
- Sunflower: ARIMA (baseline safer), gain 0.04 t/ha, coverage 100.0%.
- Wheat: LightGBM (within expected error), gain 0.08 t/ha, coverage 66.7%.

## Poltava

- Maize: LightGBM (outside validation error scale), gain 0.17 t/ha, coverage 33.3%.
- Sunflower: LightGBM (within expected error), gain 0.13 t/ha, coverage 100.0%.
- Wheat: FORECAST.LINEAR (baseline safer), gain -0.05 t/ha, coverage 66.7%.

## Ukraine

- Maize: ElasticNet (within expected error), gain 0.11 t/ha, coverage 66.7%.
- Sunflower: ARIMA (baseline safer), gain -0.15 t/ha, coverage 100.0%.
- Wheat: FORECAST.LINEAR (baseline safer), gain -0.28 t/ha, coverage 66.7%.

## Vinnytsia

- Maize: XGBoost (within expected error), gain 0.47 t/ha, coverage 100.0%.
- Sunflower: LightGBM (within expected error), gain 0.06 t/ha, coverage 66.7%.
- Wheat: FORECAST.LINEAR (baseline safer), gain -0.17 t/ha, coverage 33.3%.
