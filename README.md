# Decision Threshold Stability in Multi-Horizon Financial Distress Prediction

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20477366.svg)](https://doi.org/10.5281/zenodo.20477366)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20477953.svg)](https://doi.org/10.5281/zenodo.20477953)

Analysis code for the paper *Decision Threshold Stability in Multi-Horizon Financial
Distress Prediction on Imbalanced Turkish Firms*.

The pipeline trains four classifiers on annual financial statement data from Borsa İstanbul
companies. The classifiers are Random Forest, XGBoost, LightGBM and a stacking ensemble.
Each one is evaluated at 1-, 2- and 3-year prediction horizons. For each model and horizon,
the pipeline tunes a decision threshold by nested cross-validation. It then measures how
stable that threshold is across folds and random seeds.

The prediction target comes from the audit opinion. An unqualified opinion is labelled
healthy. Any modified opinion (qualified, adverse or disclaimer) is labelled distressed.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows; on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python financial_distress_prediction.py
```

This runs the full experiment with the default settings: five seeds, horizons 1 to 3 and
data from 2015 onward. Tables are written to `results/` and figures to `visualizations/`.

## Requirements

- Python 3.11 (tested with 3.11.9)
- The exact package versions are pinned in [requirements.txt](requirements.txt): pandas,
  numpy, matplotlib, scikit-learn, imbalanced-learn, xgboost and lightgbm.

## Data

The pipeline expects one CSV file at `data/financial_data.csv`. If the file is missing,
download `financial_data.csv` from the Zenodo dataset record and place it in `data/`:

- **Zenodo (exact file used in the paper):**
  [10.5281/zenodo.20477953](https://doi.org/10.5281/zenodo.20477953)
- **Original source (Kaggle):**
  [10.34740/kaggle/ds/7115872](https://doi.org/10.34740/kaggle/ds/7115872)

Use the Zenodo copy to reproduce the paper. The Kaggle dataset can change over time, so a
newer download may give different results. The file's MD5 checksum is
`359a881da46e9f4099a9a902883859a0`.

## Running

### Full experiment

```bash
python financial_distress_prediction.py
```

One seed takes about 20–25 minutes on a desktop CPU, so the default five-seed run takes
about 2 hours.

### Reference run only (faster)

```bash
python financial_distress_prediction.py --seeds 42
```

Seed 42 is the reference run behind Tables 1–2 and Fig. 2. A single-seed run skips the
cross-seed analyses (Tables S1–S6 and the seed-dispersion figure).

### Rebuild tables and figures without retraining

```bash
python financial_distress_prediction.py --from-results
```

This reads the probability files that an earlier run saved in `results/`. It then
regenerates every table and figure in a few seconds. Use it after changing only the
plotting or table code.

### Jupyter notebook

[financial_distress_prediction.ipynb](financial_distress_prediction.ipynb) runs the same
pipeline. The notebook ignores command-line arguments and always uses the defaults. To change
a setting such as `min_year`, call `run_experiment_per_year(...)` directly instead of
`main()`.

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--input-file` | `data/financial_data.csv` | Path to the source CSV. |
| `--results-dir` | `results` | Folder for tables and saved probabilities. |
| `--visualizations-dir` | `visualizations` | Folder for figures. |
| `--min-year` | `2015` | Earliest fiscal year to keep. |
| `--max-horizon` | `3` | Longest prediction horizon (maximum 3). |
| `--seeds` | `42 7 123 2024 31415` | Seeds for the repeated cross-validation. Each seed is one full repetition. |
| `--fixed-model-seed` | off | Keep the SMOTE and learner seeds at 42 so that only the company partition changes between seeds. |
| `--from-results` | off | Skip training and rebuild outputs from a previous run's files. |

Other settings, such as the threshold grid, the number of trees, the learning rate and the
winsorisation limit, are constants in the `CONFIGURATION` block at the top of
[financial_distress_prediction.py](financial_distress_prediction.py).

## Method notes

These choices affect how the results should be read:

- **Company-aware cross-validation.** Folds are split with `StratifiedGroupKFold` grouped by
  company, so all rows of one firm stay on the same side of every split. A firm contributes
  several near-identical yearly rows with correlated labels. A row-wise split would therefore
  train on one year of a company and test on another.
- **Thresholds are tuned at the real class prevalence.** SMOTE runs inside an `imblearn`
  pipeline, so it only resamples the training part of each inner fold. The out-of-fold
  probabilities used to pick each threshold therefore have the same imbalance as the test
  data, not a balanced one.
- **Group-aware stacking.** The meta-classifier is trained on company-grouped out-of-fold
  probabilities from the three base models, computed on real rows only. No synthetic SMOTE
  rows are used.
- **Rows with a missing audit opinion are dropped.** They have no ground truth. Labelling
  them distressed would contaminate the positive class.
- **Horizons with too few positives are skipped.** A horizon needs at least 25 positive rows
  for nested cross-validation. Below that, it is reported as excluded instead of producing a
  result.
- **Repeated seeds.** Each non-reference seed changes the outer split, the inner split, SMOTE
  and the learners at the same time. Model differences across seeds are tested with the
  corrected resampled t-test, using Holm correction for multiple comparisons.

## Outputs

### Tables (`results/`)

| File | Contents |
|---|---|
| `table1_thresholds.csv` | Paper Table 1: tuned thresholds and their fold-to-fold SD (reference seed). |
| `table2_performance.csv` | Paper Table 2: MCC, PR-AUC and FNR (reference seed). |
| `horizon_means.csv` | Mean MCC and PR-AUC per horizon. |
| `tableS1_threshold_sd_across_seeds.csv` | Threshold SD summarised across seeds. |
| `tableS2_performance_across_seeds.csv` | Performance summarised across seeds. |
| `tableS3_dispersion_tests.csv` | Levene (median-centred) tests of threshold dispersion, Random Forest vs. each other model. |
| `tableS4_pairwise_tests.csv` | Pairwise model comparisons (corrected t-test, Holm). |
| `tableS5_means_95ci.csv` | Means with 95% confidence intervals. |
| `tableS6_horizon_tests.csv` | Kruskal–Wallis test of the horizon effect for each model. |
| `tableS7_calibration.csv` | Brier score, ECE and MCC-surface shape per model and horizon. |
| `results_summary.csv` | Metrics per seed, model and horizon, averaged over folds. |
| `results_per_fold.csv` | Metrics for every fold. |
| `results_thresholds.csv` | Tuned thresholds with SD and validation MCC. |
| `threshold_sd_by_seed.csv`, `horizon_means_across_seeds.csv`, `calibration_per_fold.csv` | Supporting data for the tables above. |
| `inner_mcc_curves.csv.gz`, `test_probabilities.csv.gz`, `oof_probabilities.csv.gz` | Saved probabilities that `--from-results` uses. |
| `run_info.txt` | Package versions, seeds and runtime. |

Tables S1–S6 are only produced when more than one seed is run.

### Figures (`visualizations/`)

| File | Contents |
|---|---|
| `fig2_threshold_dispersion` | Paper Fig. 2: every per-fold threshold, by model and horizon. |
| `fig3_calibration` | Paper Fig. 3: calibration and the shape of the MCC curve. |
| `figS_performance` | MCC, PR-AUC and FNR by horizon with ±1 SD error bars. |
| `figS_threshold_sd_by_seed` | Threshold SD for each seed (multi-seed runs only). |

Each figure is saved as `.pdf`, `.svg` and `.png` (1200 dpi). The figures are grayscale with
hatch patterns and sized to a 4.6 in text width, so they fit a manuscript without rescaling.

## Citation

If you use this code, please cite it using [CITATION.cff](CITATION.cff) or the Zenodo record:

- **Code (all versions):** [10.5281/zenodo.20477366](https://doi.org/10.5281/zenodo.20477366)
- **Data (all versions):** [10.5281/zenodo.20477953](https://doi.org/10.5281/zenodo.20477953)

To cite the exact code version behind a set of results, use that version's DOI from the
Zenodo record. For example, v2.1.0 is
[10.5281/zenodo.24008980](https://doi.org/10.5281/zenodo.24008980).

## License

The code is released under the [MIT License](LICENSE). The dataset is distributed separately
under the license stated on its Zenodo record.
