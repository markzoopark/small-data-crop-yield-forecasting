# Reliability summary

This is a small-data benchmark, not a production forecasting system.
The workflow recommends ML only when it beats the best simple baseline by the configured practical MAE margin.

## Maize

- Recommended method: LightGBM (machine_learning)
- Recommended MAE: 0.69 t/ha
- Best ML MAE: 0.69 t/ha
- Best baseline MAE: 0.86 t/ha
- ML gain vs baseline: 0.17 t/ha
- Test coverage inside validation-residual band: 33.3%
- Warning label: outside validation error scale
- Top useful feature group: mineral-treated share

For maize, ML improves MAE, but the test errors fall outside the small validation-residual scale too often. Use the forecast with caution; the strongest diagnostic group is mineral-treated share.

## Sunflower

- Recommended method: LightGBM (machine_learning)
- Recommended MAE: 0.04 t/ha
- Best ML MAE: 0.04 t/ha
- Best baseline MAE: 0.17 t/ha
- ML gain vs baseline: 0.13 t/ha
- Test coverage inside validation-residual band: 100.0%
- Warning label: within expected error
- Top useful feature group: phosphorus fertiliser

For sunflower, ML clears the baseline-first rule and the test errors are mostly within the empirical validation-residual band. The strongest diagnostic group is phosphorus fertiliser.

## Wheat

- Recommended method: FORECAST.LINEAR (baseline)
- Recommended MAE: 0.49 t/ha
- Best ML MAE: 0.54 t/ha
- Best baseline MAE: 0.49 t/ha
- ML gain vs baseline: -0.05 t/ha
- Test coverage inside validation-residual band: 66.7%
- Warning label: baseline safer
- Top useful feature group: mineral-treated share

For wheat, the transparent baseline is recommended because the selected ML model does not clear the practical improvement margin. Treat this crop as the negative/control case.
