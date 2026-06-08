# Small data crop yield forecasting

This is a small research project for crop yield forecasting from official agricultural statistics.

The main idea is simple: do not say that machine learning is better just because a model was trained. The code first compares ML models with simple baselines. ML is recommended only when it beats the best baseline by a practical margin. The forecast also gets a small reliability check based on validation errors.

The current example uses Poltava region, Ukraine, for 2010-2024. Crops:

- wheat
- maize
- sunflower

This is a benchmark for small annual datasets. It is not a production forecasting system.

## What is inside

- `src/agrostats/` - the Python code
- `data/raw/agrostats/poltava/` - raw AgroStats CSV files used here
- `reports/` - generated metrics and reliability summaries
- `paper/` - the draft article and figures
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

Run tests:

```bash
python -m pytest -q
```

Build the draft article:

```bash
python scripts/build_reliability_article.py
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

Article:

- `paper/kopishynska_small_data_crop_forecasting_reliability.docx`

## Important warning

The dataset is very small: one annual regional observation per crop for each year. The results should be read as a reproducible benchmark and decision-support example, not as a final operational forecast.

The wheat result is especially useful because it shows why the baseline-first rule matters: for wheat, the simple baseline is safer than the selected ML model. For maize and sunflower, ML is more useful.
