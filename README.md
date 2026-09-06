# Financial Distress & Bankruptcy Prediction Experiment

This project trains multiple classifiers on annual financial statement data from Borsa
İstanbul companies and evaluates them across 1-, 2- and 3-year prediction horizons. The
prediction target is derived from the audit opinion: an unqualified opinion is treated as
healthy, and any modified opinion (qualified, adverse, or disclaimer) as distressed.

## Requirements

Install the Python packages listed in `requirements.txt`.

## Run

To run the pipeline and generate predictions:
```bash
python financial_distress_prediction.py --input-file "data/financial_data.csv" --results-dir "results" --visualizations-dir "visualizations"
```
Or run the Jupyter Notebook: [financial_distress_prediction.ipynb](financial_distress_prediction.ipynb)

Note that the notebook bypasses argparse and uses the defaults below. To vary `min_year`
from a notebook, call `run_experiment_per_year(...)` directly instead of `main()`.

## CLI options

- `--input-file`: path to the source CSV (default: `data/financial_data.csv`)
- `--results-dir`: folder for results output (default: `results`)
- `--visualizations-dir`: folder for generated plots (default: `visualizations`)
- `--min-year`: minimum year to keep from the dataset (default: `2015`)
- `--max-horizon`: maximum prediction horizon to generate (default: `3`, capped at `3`)

## Method notes

These affect how the results should be read:

- **Company-aware cross-validation.** Folds are split with `StratifiedGroupKFold` grouped by
  company, so every row of a firm stays on the same side of a split. A firm contributes
  several near-identical yearly rows with correlated labels, so a row-wise split would leak
  one year of a company into training while testing on another.
- **Thresholds are tuned at the real class prevalence.** SMOTE runs inside an
  `imblearn` pipeline, so it is applied only to the training portion of each inner fold. The
  out-of-fold probabilities used to pick each threshold therefore reflect the imbalanced
  distribution the model faces at test time, not a balanced one.
- **Rows with a missing audit opinion are dropped**, not treated as distressed. They carry no
  ground truth, and defaulting them into the positive class would contaminate it.
- **Horizons with too few positives are skipped.** A horizon needs at least 25 positive rows
  to support the nested cross-validation; below that it is reported as excluded for
  insufficient positive-class support rather than as a result.

## Outputs

All outputs are saved to their respective directories:

### Results (`results/`)
- `results_summary.csv`: metrics averaged across folds, with fold-to-fold standard deviations
  and a `Folds` column giving the number of folds each row averages over.
- `results_per_fold.csv`: metrics for each individual fold.
- `results_thresholds.csv`: tuned classification thresholds with their fold-to-fold
  dispersion (`Best Threshold (SD)`) and the corresponding validation MCC.

### Visualizations (`visualizations/`)
- `fig_threshold_dispersion.png`: every per-fold threshold selection, by model and horizon.
- `fig_performance.png`: MCC, PR-AUC and FNR by horizon, with one-standard-deviation error bars.

Figures are grayscale with hatch patterns and sized to a 4.6 in text width at 1200 dpi, so
they can be placed in a manuscript without rescaling.
