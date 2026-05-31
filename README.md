# Bankruptcy Prediction Experiment

This project trains multiple classifiers on annual financial statement data and evaluates them across prediction horizons.

## Requirements

Install the Python packages listed in `requirements.txt`.

## Run

```bash
python something_v2.py --input-file "data v2/financial_data.csv" --results-dir "results v2" --visualizations-dir "visualizations v2"
```

## CLI options

- `--input-file`: path to the source CSV
- `--results-dir`: folder for `results_summary.csv` and `results_per_fold.csv`
- `--visualizations-dir`: folder for generated plots
- `--min-year`: minimum year to keep from the dataset
- `--max-horizon`: maximum prediction horizon to generate

## Outputs

- `results_summary.csv`
- `results_per_fold.csv`
- `model_performance_comparison.png`
- `false_negative_rate_by_model_and_horizon.png`
- `predicted_2025_risk.csv`
- `predicted_2025_risk_top20.png`
