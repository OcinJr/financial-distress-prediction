# Financial Distress & Bankruptcy Prediction Experiment

This project trains multiple classifiers on annual financial statement data and evaluates them across prediction horizons.

## Requirements

Install the Python packages listed in `requirements.txt`.

## Run

To run the pipeline and generate predictions:
```bash
python financial_distress_prediction.py --input-file "data/financial_data.csv" --results-dir "results" --visualizations-dir "visualizations"
```
Or run the Jupyter Notebook: [financial_distress_prediction.ipynb](financial_distress_prediction.ipynb)

## CLI options

- `--input-file`: path to the source CSV (default: `data/financial_data.csv`)
- `--results-dir`: folder for results output (default: `results`)
- `--visualizations-dir`: folder for generated plots (default: `visualizations`)
- `--min-year`: minimum year to keep from the dataset (default: `2021`)
- `--max-horizon`: maximum prediction horizon to generate (default: `3`)

## Outputs

All outputs are saved to their respective directories:

### Results (`results/`)
- `results_summary.csv`: aggregated evaluation metrics across folds.
- `results_per_fold.csv`: detailed metrics for each of the 5 folds.
- `results_thresholds.csv`: tuned classification thresholds and their corresponding validation MCC.

### Visualizations (`visualizations/`)
- `recall_sensitivity.png`: Model sensitivity comparison.
- `pr_auc.png`: Precision-Recall Area Under Curve comparison.
- `fnr_mcc_comparison.png`: Combined stacked plot displaying FNR and MCC side-by-side.
- `false_negative_rate_by_model_and_horizon.png`: FNR comparison plot.

