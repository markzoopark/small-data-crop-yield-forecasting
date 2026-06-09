# Small data crop yield forecasting

This is a small research project for crop yield forecasting from official agricultural statistics.

The main idea is simple: do not say that machine learning is better just because a model was trained. The code first compares ML models with simple baselines. ML is recommended only when it beats the best baseline by a practical margin. The forecast also gets a small reliability check based on validation errors.

The current project uses official AgroStats data for 2010-2024.

Territories:

- Poltava
- Vinnytsia
- Cherkasy
- Ukraine

Poltava is the main case study. Vinnytsia, Cherkasy, and Ukraine are added as an external check, so the workflow is not tested only on one oblast.

Crops:

- wheat
- maize
- sunflower

This is a benchmark for small annual datasets. It is not a production forecasting system.

## What is inside

- `src/agrostats/` - the Python code
- `data/raw/agrostats/poltava/` - raw AgroStats CSV files used here
- `data/raw/agrostats/vinnytsia/` - external region check data
- `data/raw/agrostats/cherkasy/` - external region check data
- `data/raw/agrostats/ukraine/` - national-level scale check data
- `reports/` - generated metrics and reliability summaries
- `tests/` - small tests for unit conversion, diagnostics, and reliability logic

## What the code does

1. Loads official AgroStats CSV files.
2. Normalises units.
3. Builds lagged and moving-average features.
4. Trains ElasticNet, XGBoost, and LightGBM.
5. Compares them with simple baselines:
   - naive lag-1
   - linear trend
   - LINEST with lags
   - ARIMA
6. Adds reliability checks:
   - empirical validation-residual bands
   - test coverage inside those bands
   - feature-group ablation
   - baseline-first recommended method
   - forecast cards for each crop

## How to run

Create an environment and install packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_all.py --region poltava --languages uk,en
```

Import the extra downloaded regions again if needed:

```bash
python scripts/import_multi_region_data.py
```

Run the external multi-region check:

```bash
python scripts/run_multi_region.py
```

Run tests:

```bash
python -m pytest -q
```

## Main outputs

Important files in `reports/`:

- `metrics_leaderboard.csv` - best ML model per crop
- `metrics_baselines_summary.csv` - best simple baselines
- `recommended_methods.csv` - baseline-first final recommendation
- `forecast_cards.csv` - one short decision-support row per crop
- `prediction_bands.csv` - empirical validation-residual bands on test forecasts
- `feature_group_ablation.csv` - what happens when feature groups are removed
- `reliability_summary.md` - simple human-readable summary
- `data_inventory.csv` - checks that each territory has the expected 91 files
- `multi_region_recommended_methods.csv` - final decision for 4 territories x 3 crops
- `multi_region_forecast_cards.csv` - forecast cards for all region-crop cases
- `region_comparison_summary.csv` - short comparison by territory
- `decision_threshold_sensitivity.csv` - checks margins 0.00, 0.03, 0.05, 0.10 t/ha
- `novelty_evidence_table.csv` - compact evidence table for reporting the workflow results

## Important warning

The dataset is very small: one annual observation per crop for each territory and year. The results should be read as a reproducible benchmark and decision-support example, not as a final operational forecast.

The important point is not that Excel-style functions are "better". They are control baselines. ML has to earn recommendation by beating them by a practical margin.

The multi-region check gives a stronger robustness check:

- maize is usually the clearest ML-positive case
- wheat often stays with a transparent baseline
- sunflower is mixed by territory
- some ML wins still receive a reliability warning
