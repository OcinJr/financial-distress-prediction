"""
Decision Threshold Stability in Multi-Horizon Financial Distress Prediction
on Imbalanced Turkish Firms -- experiment code.

Pipeline follows SEMMA (paper Sect. 3):
  Sample  -> process_turkish_panel_data()   (Sect. 3.1, Eq. 1)
  Explore -> explore_report()               (Sect. 3.2)
  Modify  -> cleaning in process_turkish_panel_data() + make_base_pipeline() (Sect. 3.3)
  Model   -> build_models()                 (Sect. 3.4, Eq. 2)
  Assess  -> find_best_threshold(), calculate_metrics_at_threshold(),
             export_paper_tables(), save_threshold_dispersion_figure() (Sect. 3.5, Eq. 3-4)
  Robustness -> export_seed_robustness()    (repeated CV seeds, review revision)

Seeds: the reference run (seed 42; outer 42, inner 0) reproduces Tables 1-2 and Fig. 2.
Further seeds repeat the whole outer CV on new company partitions (default 4 extra).

Outputs (results dir):  results_per_fold.csv, results_summary.csv, results_thresholds.csv
                        (all with a Seed column),
                        table1_thresholds.csv, table2_performance.csv, horizon_means.csv
                        (reference seed = the paper's tables),
                        tableS1_threshold_sd_across_seeds.csv, tableS2_performance_across_
                        seeds.csv, tableS3_dispersion_tests.csv, threshold_sd_by_seed.csv,
                        horizon_means_across_seeds.csv, run_info.txt
Outputs (figures dir):  fig2_threshold_dispersion (paper Fig. 2, reference seed),
                        fig3_calibration (paper Fig. 3, all seeds),
                        figS_threshold_sd_by_seed, figS_performance (supplementary);
                        each as .svg (vector, embedded in the paper), .pdf (vector) and
                        .png (1200 dpi fallback)

Command line:   python financial_distress_prediction.py --input-file data/financial_data.csv
                python financial_distress_prediction.py --seeds 42      (reference run only)
                python financial_distress_prediction.py --from-results  (redraw tables and
                    figures from a finished run in --results-dir, no retraining)
In a notebook:  main(["--input-file", "data/financial_data.csv"])
"""

import argparse
import os
import platform
import sys
import time
import warnings
from inspect import signature

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.base import clone, BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import (
    recall_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

# XGBoost 3.3.0 and later give different XGBoost fits on this data than the version the
# submitted tables were produced with; 3.0.5 and 3.2.0 reproduce them exactly.
# Pin with:  pip install "xgboost==3.2.0"
_xgb_version = tuple(int(p) for p in xgb.__version__.split(".")[:2] if p.isdigit())
if _xgb_version >= (3, 3):
    warnings.warn(f"xgboost {xgb.__version__} detected; the paper's XGBoost and stacking "
                  f"results were produced with xgboost <= 3.2.0 and will not reproduce "
                  f"exactly. Install xgboost==3.2.0 to match them.")

# Cosmetic only: LightGBM records feature names at fit time and receives a bare array at
# predict time inside the pipeline. Column ORDER is preserved, so predictions are unaffected.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


# ========== CONFIGURATION (paper Sect. 3.4 - 3.5) ==========
SEED = 42                 # reference seed: outer split, SMOTE, every learner, stacking split
INNER_CV_SEED = 0         # inner split of the REFERENCE run (the out-of-fold probabilities
                          # used to select t*). 0 is the value behind the submitted tables.

# Repeated cross-validation seeds (Reviewer 5: re-evaluate "across 5-10 different random
# seeds (for both fold partitioning and SMOTE resampling)"; Reviewer 6: "repeated CV").
# Each seed is one complete repetition of the 5-fold outer CV. The first entry, SEED, is the
# reference run and reproduces the submitted tables exactly (outer 42, inner 0, model 42).
# Every other seed s sets outer = inner = SMOTE = learner seed = s, so the company
# partition, the resampling and the learners all change between repetitions.
# Use --fixed-model-seed to vary only the partition.
SEEDS = [SEED, 7, 123, 2024, 31415]
VARY_MODEL_SEED = True

N_SPLITS = 5              # outer and inner group k-fold
N_ESTIMATORS = 300
LEARNING_RATE = 0.05      # XGBoost and LightGBM
XGB_MAX_DEPTH = 4
WINSOR_LIMIT = 100_000    # features clipped to [-WINSOR_LIMIT, WINSOR_LIMIT]
# T = {0.05, 0.06, ..., 0.95}. Deliberately NOT rounded: Random Forest probabilities are
# multiples of 1/300 and can sit exactly on a grid value, so rounding the grid flips the
# ">=" comparison at those ties and changes the RF results behind the submitted tables.
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
MIN_POSITIVES_FOR_NESTED_CV = 25

# True  -> the meta-classifier is trained on GROUP-aware out-of-fold probabilities of the
#          three base pipelines, computed on real (non-synthetic) rows. This is what the
#          paper describes: every split partitions companies and SMOTE stays inside
#          training folds.
# False -> legacy behaviour: sklearn StackingClassifier(cv=5) placed after SMOTE. Its
#          internal split is row-wise and runs on SMOTE-augmented data, so the same firm and
#          synthetic rows can sit on both sides. Kept only to reproduce the earlier tables.
GROUP_AWARE_STACKING = True

# Inner cross-validation parallelism. The base learners already use n_jobs=-1, so raising
# this nests parallelism and can oversubscribe the CPU. Increase only if you have cores spare.
CV_N_JOBS = 1


# ========== SAFE PIPELINE COMPONENTS ==========
class SafeIterativeImputer(BaseEstimator, TransformerMixin):
    """IterativeImputer initialised from column medians; falls back to median imputation."""

    def __init__(self, max_iter=10, random_state=SEED,
                 initial_strategy="median", skip_complete=True):
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_strategy = initial_strategy
        self.skip_complete = skip_complete

    def fit(self, X, y=None):
        try:
            self.imputer_ = IterativeImputer(
                max_iter=self.max_iter, random_state=self.random_state,
                initial_strategy=self.initial_strategy, skip_complete=self.skip_complete
            ).fit(X)
        except Exception as e:
            print(f"[!] IterativeImputer failed ({e}). Falling back to median imputation.")
            self.imputer_ = SimpleImputer(strategy="median").fit(X)
        return self

    def transform(self, X):
        return self.imputer_.transform(X)


class SafeSMOTE(SMOTE):
    """SMOTE whose neighbourhood size adapts to the minority count of the fold (Sect. 3.3)."""

    def fit_resample(self, X, y):
        y_arr = np.asarray(y).astype(int)
        n_minority = int(min(np.bincount(y_arr, minlength=2)))
        if n_minority < 2:
            print(f"[!] Only {n_minority} minority sample(s) in this fold - skipping SMOTE.")
            return X, y
        self.k_neighbors = min(5, n_minority - 1)
        return super().fit_resample(X, y)


def make_base_pipeline(classifier, seed=SEED):
    """Modify phase (Sect. 3.3): imputation -> robust scaling -> SMOTE -> classifier.

    Every step is refitted on whatever training data the pipeline receives, so nothing is
    learned from a validation or test fold and no synthetic row reaches one.
    """
    return ImbPipeline([
        ("imputer", SafeIterativeImputer(max_iter=10, random_state=seed,
                                         initial_strategy="median", skip_complete=True)),
        ("scaler", RobustScaler()),
        ("smote", SafeSMOTE(random_state=seed)),
        ("classifier", classifier),
    ])


def seed_plan(seeds=None, vary_model_seed=None):
    """One dict per repetition: {'seed', 'outer', 'inner', 'model'}; the reference run first."""
    seeds = list(SEEDS if seeds is None else seeds)
    vary = VARY_MODEL_SEED if vary_model_seed is None else vary_model_seed
    if SEED in seeds:                       # reference run always first
        seeds = [SEED] + [s for s in seeds if s != SEED]
    plan = []
    for s in dict.fromkeys(seeds):          # de-duplicate, keep order
        if s == SEED:
            plan.append({"seed": s, "outer": SEED, "inner": INNER_CV_SEED, "model": SEED})
        else:
            plan.append({"seed": s, "outer": s, "inner": s,
                         "model": s if vary else SEED})
    return plan


class GroupStackingClassifier(BaseEstimator, ClassifierMixin):
    """Stacking ensemble of Eq. (2) with a company-grouped internal split.

    Level 0: each base learner is a full pipeline (imputer, scaler, SMOTE, model).
    Level 1: logistic regression g trained on out-of-fold probabilities f_1..f_3(x_i),
    produced by StratifiedGroupKFold over companies on real rows only.
    After the meta-classifier is fitted, each base pipeline is refitted on all the data,
    exactly as sklearn's StackingClassifier does.
    """

    def __init__(self, estimators, final_estimator, n_splits=N_SPLITS, random_state=SEED):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        if groups is None:
            print("[!] GroupStackingClassifier fitted without groups - row-wise split used.")
            cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(X, y))
        else:
            cv = StratifiedGroupKFold(self.n_splits, shuffle=True,
                                      random_state=self.random_state)
            splits = list(cv.split(X, y, np.asarray(groups)))

        meta_features = np.column_stack([
            cross_val_predict(clone(est), X, y, cv=splits, method="predict_proba",
                              n_jobs=1)[:, 1]
            for _, est in self.estimators
        ])
        self.final_estimator_ = clone(self.final_estimator).fit(meta_features, y)
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        return self

    def _meta_features(self, X):
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(self._meta_features(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ========== COLUMN TRANSLATION MAP ==========
COLUMN_TRANSLATIONS = {
    "Şirket Adı":                             "Company Name",
    "Şirketin Kodu":                          "Company Code",
    "Periyot":                                "Period",
    "Yıl":                                    "Year",
    "Dönen Varlıklar":                        "Current Assets",
    "Duran Varlıklar":                        "Non-Current Assets",
    "Toplam Varlıklar":                       "Total Assets",
    "Kısa Vadeli Yükümlülükler":              "Short-Term Liabilities",
    "Uzun Vadeli Yükümlülükler":              "Long-Term Liabilities",
    "Toplam Yükümlülükler":                   "Total Liabilities",
    "Toplam Özkaynaklar":                     "Total Equity",
    "Ana Ortaklığa Ait Özkaynaklar":          "Equity Attributable to Parent",
    "Kontrol Gücü Olmayan Kaynaklar":         "Non-Controlling Interests",
    "Toplam Kaynaklar":                       "Total Sources",
    "Cari Oran":                              "Current Ratio",
    "Dönen Varlıklar / Aktif (%)":            "Current Assets to Total Assets (%)",
    "Asit Test Oranı":                        "Quick Ratio",
    "Nakit Oranı":                            "Cash Ratio",
    "Aktif Karlılık (%)":                     "Return on Assets (%)",
    "Faaliyet Kar Marjı":                     "Operating Profit Margin",
    "Esas Faaliyet Kar Marjı":                "Core Operating Profit Margin",
    "Brüt Kar Marjı (%)":                     "Gross Profit Margin (%)",
    "FAVÖK Marjı (%)":                        "EBITDA Margin (%)",
    "Net Kar Marjı":                          "Net Profit Margin",
    "Özsermaye Karlılığı (%)":                "Return on Equity (%)",
    "VAFÖK Marjı":                            "EBITDA Margin (Alt)",
    "ROCE Oranı":                             "Return on Capital Employed",
    "Finansman Gider / Net Satış":            "Financing Cost to Net Sales",
    "Borç Kaynak Oranı":                      "Debt to Source Ratio",
    "Duran Varlıklar / Aktif ":               "Non-Current Assets to Total Assets",
    "Duran Varlıklar / Maddi Özkaynak":       "Non-Current Assets to Tangible Equity",
    "Esas Faaliyet Karı / Kısa Vadeli Borç":  "Core Operating Profit to Short-Term Debt",
    "FAVÖK / Kısa Vade Borç":                 "EBITDA to Short-Term Debt",
    "Net Borç / FAVÖK":                       "Net Debt to EBITDA",
    "Kısa Vade Borç / Aktif":                 "Short-Term Debt to Total Assets",
    "Kısa Vade Borç / Dönen Varlık":          "Short-Term Debt to Current Assets",
    "Kısa Vade Borç / Özsermaye":             "Short-Term Debt to Equity",
    "Kısa Vade Borç / Toplam Borç":           "Short-Term Debt to Total Debt",
    "Net Satışlar / Kısa Vade Borç":          "Net Sales to Short-Term Debt",
    "Özsermaye / Aktif":                      "Equity to Total Assets",
    "Özsermaye / Maddi Duran Varlıklar":      "Equity to Tangible Fixed Assets",
    "Toplam Borç / Özsermaye":                "Total Debt to Equity",
    "Aktif Devir Hızı":                       "Asset Turnover",
    "Alacak Devir Hızı":                      "Receivables Turnover",
    "Dönen Varlıklar Devir Hızı":             "Current Assets Turnover",
    "Ticari Borçlar Devir Hızı":              "Trade Payables Turnover",
    "Finansal Kaldıraç":                      "Financial Leverage",
    "Stok Devir Hızı":                        "Inventory Turnover",
    "Altman Z-Skoru":                         "Altman Z-Score",
    "Springate Skoru":                        "Springate Score",
    "Zmijewski Skoru":                        "Zmijewski Score",
    "L Model Skoru":                          "L Model Score",
    "Görüs Tipi":                             "Audit Opinion",
}

# The four categories of the source vocabulary (Sect. 3.1). Unqualified = healthy (0);
# any modified opinion = distressed (1), Eq. (1).
TARGET_LABEL_TRANSLATIONS = {
    "Olumlu":                                 "Unqualified",
    "Şartlı":                                 "Qualified",
    "Görüş bildirmekten kaçınma":             "Disclaimer of Opinion",
    "Olumsuz":                                "Adverse",
}

PLACEHOLDER_TOKENS = ["?", "NA", "NaN", "", " "]
ID_COLS = ["Company Name", "Company Code", "Period", "Audit Opinion"]


# ========== 1. Explore (Sect. 3.2) ==========
def explore_report(df):
    """Profile the four data-quality problems named in Sect. 3.2. Reads only, changes nothing.

    Expects translated column names, before placeholder replacement and numeric coercion.
    """
    feature_cols = [c for c in df.columns if c not in ID_COLS + ["Year"]]
    raw = df[feature_cols]

    n_empty = int(raw.isna().sum().sum())
    n_placeholder = int(raw.isin(PLACEHOLDER_TOKENS).sum().sum())
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    n_nonfinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    n_extreme = int((finite.abs() > WINSOR_LIMIT).sum().sum())
    cols_extreme = int(((finite.abs() > WINSOR_LIMIT).any()).sum())
    opinions = df["Audit Opinion"]
    n_unmapped = int((~opinions.isin(TARGET_LABEL_TRANSLATIONS.keys())).sum())

    print("\n[EXPLORE] Data-quality profile (annual reports, after year filter)")
    print(f"    rows x feature columns       : {raw.shape[0]} x {raw.shape[1]}")
    print(f"    empty cells                  : {n_empty}")
    print(f"    placeholder strings          : {n_placeholder}  {PLACEHOLDER_TOKENS}")
    print(f"    non-finite values (+/-inf)   : {n_nonfinite}")
    print(f"    |value| > {WINSOR_LIMIT:,}          : {n_extreme} cells in {cols_extreme} columns")
    print(f"    unmapped / absent opinions   : {n_unmapped}")
    return {"empty": n_empty, "placeholder": n_placeholder, "nonfinite": n_nonfinite,
            "extreme": n_extreme, "extreme_cols": cols_extreme, "unmapped": n_unmapped}


# ========== 2. Sample + Modify (Sect. 3.1, 3.3) ==========
def process_turkish_panel_data(file_path, max_horizon=3, min_year=2015):
    print(f"[*] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)

    COMPANY_COL = "Şirket Adı"
    YEAR_COL    = "Yıl"
    TARGET_COL  = "Görüs Tipi"
    PERIOD_COL  = "Periyot"

    if COMPANY_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(
            f"[!] Critical Error: Columns '{COMPANY_COL}' or '{TARGET_COL}' not found. "
            f"Please check your CSV headers."
        )

    # Sample: annual reports from min_year onward
    df = df[df[PERIOD_COL] == "Yıllık"].copy()
    print(f"[*] Rows after annual filter: {len(df)}")
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[YEAR_COL] >= min_year].copy()
    print(f"[*] Rows after {min_year}+ filter: {len(df)}")

    # Modify (1/5): harmonise column names
    df.rename(columns=COLUMN_TRANSLATIONS, inplace=True)

    explore_report(df)

    # Modify (2/5): placeholder tokens -> missing
    df.replace(PLACEHOLDER_TOKENS, np.nan, inplace=True)

    # Modify (3/5): cast feature columns to numeric. read_csv types any column holding "?"
    # as object; without this, the object-column drop below would delete real features.
    for col in df.columns:
        if col not in ID_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target, Eq. (1): unmapped or absent opinions are removed, not folded into a class
    df["Audit Opinion"] = df["Audit Opinion"].map(TARGET_LABEL_TRANSLATIONS)
    n_before = len(df)
    df = df.dropna(subset=["Audit Opinion"]).copy()
    print(f"[*] Dropped {n_before - len(df)} rows with missing/unmapped Audit Opinion.")
    df["class"] = np.where(df["Audit Opinion"] == "Unqualified", 0, 1)

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # The calendar-year join below needs exactly one row per company-year.
    n_dupes = int(df.duplicated(subset=["Company Name", "Year"]).sum())
    if n_dupes:
        print(f"[!] {n_dupes} duplicate company-year rows - keeping the first of each.")
        df = df.drop_duplicates(subset=["Company Name", "Year"], keep="first").copy()

    df = df.sort_values(by=["Company Name", "Year"]).reset_index(drop=True)

    if max_horizon > 3:
        print(f"[!] max_horizon={max_horizon} is too high for the 2015+ window. Capping at 3.")
        max_horizon = 3

    horizon_datasets = {}

    for horizon in range(1, max_horizon + 1):
        print(f"\n[*] Generating {horizon}-Year Horizon dataset...")

        # Eq. (1): features of year t carry the label of the SAME company in year t + h.
        # A calendar-year join, not a positional shift, so gaps in a firm's history never
        # pull in a label from the wrong year.
        target_col_name = f"Target_{horizon}Y"
        future = (
            df[["Company Name", "Year", "class"]]
            .rename(columns={"class": target_col_name})
            .assign(Year=lambda d: d["Year"] - horizon)
        )
        horizon_df = df.merge(future, on=["Company Name", "Year"], how="inner")

        # Modify (4/5): remove identifier columns (and any remaining non-numeric column)
        columns_to_drop = ["Company Name", "Company Code", "Period",
                           "Year", "Audit Opinion", "class", target_col_name]
        object_cols = horizon_df.select_dtypes(include=["object", "string"]).columns
        columns_to_drop.extend([c for c in object_cols if c not in columns_to_drop])

        X = horizon_df.drop(columns=columns_to_drop, errors="ignore")
        y = horizon_df[target_col_name].astype(int)

        # Company identity: never a feature, used only by the group-aware splitters.
        groups = horizon_df["Company Name"].to_numpy()

        X = X.apply(pd.to_numeric, errors="coerce")

        # Zero-variance columns (constant or entirely missing) carry no information. A
        # column that is constant on the whole horizon is constant in every fold, so this
        # learns nothing from any test fold. Checked on the raw values, BEFORE winsorizing,
        # in the order Sect. 3.3 states. (After clipping, balance-sheet levels such as
        # Total Assets become 100,000 in almost every row; they are kept, as in the
        # submitted run.)
        zero_var = X.columns[X.nunique(dropna=True) <= 1].tolist()
        if zero_var:
            print(f"[*] Removing {len(zero_var)} zero-variance column(s): {zero_var}")
            X = X.drop(columns=zero_var)
        else:
            print("[*] No zero-variance columns found.")

        # Modify (5/5): infinities -> missing, winsorize to fixed limits. Both use fixed
        # constants rather than sample statistics, so applying them before the split
        # introduces no leakage.
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.clip(lower=-WINSOR_LIMIT, upper=WINSOR_LIMIT)
        n_saturated = int(((X.abs() >= WINSOR_LIMIT).mean() > 0.9).sum())
        if horizon == 1 and n_saturated:
            print(f"[!] {n_saturated} column(s) sit at the +/-{WINSOR_LIMIT:,} limit in more "
                  f"than 90% of rows after winsorizing (balance-sheet levels in TRY).")

        if horizon == 1:
            print(f"[*] {X.shape[1]} features reaching the model: {X.columns.tolist()}")

        n_distressed_firms = len(np.unique(groups[y.to_numpy() == 1]))
        print(f"[+] {horizon}-Year Horizon - n = {len(y)} firm-years from "
              f"{len(np.unique(groups))} companies | distressed: {int(y.sum())} "
              f"observations from {n_distressed_firms} companies "
              f"({y.mean():.1%}) | features: {X.shape[1]}")

        horizon_datasets[f"{horizon}-Year Horizon"] = (X, y, groups)

    return horizon_datasets


# ========== 3. Model (Sect. 3.4) ==========
def _rf(seed=SEED):
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)


def _xgb(seed=SEED):
    return xgb.XGBClassifier(eval_metric="logloss", random_state=seed,
                             n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                             max_depth=XGB_MAX_DEPTH, tree_method="hist", n_jobs=-1)


def _lgbm(seed=SEED):
    return lgb.LGBMClassifier(random_state=seed, n_estimators=N_ESTIMATORS,
                              learning_rate=LEARNING_RATE, verbosity=-1, n_jobs=-1)


def _meta(seed=SEED):
    return LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed)


def build_models(seed=SEED):
    """Return {name: (estimator, needs_groups)}. Each estimator is a complete pipeline."""
    models = {
        "Random Forest": (make_base_pipeline(_rf(seed), seed), False),
        "XGBoost":       (make_base_pipeline(_xgb(seed), seed), False),
        "LightGBM":      (make_base_pipeline(_lgbm(seed), seed), False),
    }
    if GROUP_AWARE_STACKING:
        models["Stacking Ensemble"] = (GroupStackingClassifier(
            estimators=[("Random Forest", make_base_pipeline(_rf(seed), seed)),
                        ("XGBoost", make_base_pipeline(_xgb(seed), seed)),
                        ("LightGBM", make_base_pipeline(_lgbm(seed), seed))],
            final_estimator=_meta(seed), n_splits=N_SPLITS, random_state=seed,
        ), True)
    else:
        models["Stacking Ensemble"] = (make_base_pipeline(StackingClassifier(
            estimators=[("Random Forest", _rf(seed)), ("XGBoost", _xgb(seed)),
                        ("LightGBM", _lgbm(seed))],
            final_estimator=_meta(seed), cv=5, n_jobs=1,
        ), seed), False)
    return models


def _oof_probabilities(estimator, X, y, groups, cv, needs_groups):
    """Out-of-fold probabilities; passes groups to fit() when the estimator needs them."""
    kwargs = {}
    if needs_groups:
        key = "params" if "params" in signature(cross_val_predict).parameters else "fit_params"
        kwargs[key] = {"groups": groups}
    return cross_val_predict(estimator, X, y, groups=groups, cv=cv,
                             method="predict_proba", n_jobs=CV_N_JOBS, **kwargs)[:, 1]


# ========== 4. Assess (Sect. 3.5) ==========
def find_best_threshold(y_true, y_prob):
    """Eq. (3): t* = argmax over T of MCC(y, y_hat_t). Ties keep the lowest t.

    Returns the unrounded grid value; round only for display.
    """
    best_threshold, best_score = 0.5, -np.inf
    for threshold in THRESHOLD_GRID:
        score = matthews_corrcoef(y_true, (y_prob >= threshold).astype(int))
        if score > best_score:
            best_score, best_threshold = score, threshold
    return float(best_threshold), float(best_score)


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    """MCC (Eq. 4), threshold-free PR-AUC and FNR = FN / (FN + TP)."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "Recall (Sensitivity)":  recall_score(y_true, y_pred, zero_division=0),
        "FNR (False Neg Rate)":  fnr,
        "PR-AUC":                average_precision_score(y_true, y_prob),
        "MCC":                   matthews_corrcoef(y_true, y_pred),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


# ========== 5. Paper tables ==========
MODEL_ORDER = ["LightGBM", "Random Forest", "XGBoost", "Stacking Ensemble"]
MODEL_SHORT = {"LightGBM": "LGBM", "Random Forest": "RF",
               "XGBoost": "XGB", "Stacking Ensemble": "Stacking"}


def _horizon_order(df):
    """Sort '1-Year Horizon', '2-Year Horizon', ... numerically rather than as strings."""
    return sorted(df["Horizon"].dropna().unique(), key=lambda h: int(str(h).split("-")[0]))


def _mean_sd(values, scale=1.0, decimals=3):
    v = np.asarray(values, dtype=float) * scale
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{v.mean():.{decimals}f} ({sd:.{decimals}f})"


def export_paper_tables(df_fold_results, results_dir="results"):
    """Write Table 1, Table 2 and the horizon means in the paper's 'mean (SD)' format.

    SD is the sample standard deviation over the five outer folds (ddof = 1).
    """
    os.makedirs(results_dir, exist_ok=True)
    horizons = _horizon_order(df_fold_results)
    models = [m for m in MODEL_ORDER if m in set(df_fold_results["Model"])]
    label = {h: h.replace(" Horizon", "").lower() for h in horizons}

    def cell(model, horizon, col):
        return df_fold_results.loc[(df_fold_results["Model"] == model) &
                                   (df_fold_results["Horizon"] == horizon), col]

    # Table 1: selected thresholds
    t1 = pd.DataFrame([{"Model": m, **{label[h]: _mean_sd(cell(m, h, "Best Threshold"))
                                       for h in horizons}} for m in models])
    t1["Model"] = t1["Model"].replace({"Stacking Ensemble": "Stacking ensemble"})

    # Table 2: predictive performance
    rows = []
    for h in horizons:
        for m in models:
            rows.append({"Horizon": label[h], "Model": MODEL_SHORT[m],
                         "MCC": _mean_sd(cell(m, h, "MCC")),
                         "PR-AUC": _mean_sd(cell(m, h, "PR-AUC")),
                         "FNR (%)": _mean_sd(cell(m, h, "FNR (False Neg Rate)"), 100, 1)})
    t2 = pd.DataFrame(rows)

    # Horizon means across models (abstract / Sect. 4.2), computed from unrounded values
    model_means = (df_fold_results.groupby(["Horizon", "Model"])[["MCC", "PR-AUC"]]
                   .mean().groupby("Horizon").mean().reindex(horizons))
    thr_sd = (df_fold_results.groupby(["Horizon", "Model"])["Best Threshold"].std()
              .unstack("Horizon").reindex(index=models, columns=horizons))
    hm = model_means.round(3).rename(index=label)

    t1.to_csv(os.path.join(results_dir, "table1_thresholds.csv"), index=False)
    t2.to_csv(os.path.join(results_dir, "table2_performance.csv"), index=False)
    hm.to_csv(os.path.join(results_dir, "horizon_means.csv"))

    print("\n========== TABLE 1: thresholds, mean (SD) ==========")
    print(t1.to_string(index=False))
    print("\n========== TABLE 2: performance, mean (SD) ==========")
    print(t2.to_string(index=False))
    print("\n========== Mean across models per horizon ==========")
    print(hm.to_string())
    print("\n========== Threshold SD range per model (stability, Sect. 4.1) ==========")
    for m in models:
        print(f"    {m:<18} {thr_sd.loc[m].min():.3f} - {thr_sd.loc[m].max():.3f}")
    fnr_best = (df_fold_results.groupby(["Horizon", "Model"])["FNR (False Neg Rate)"]
                .mean().min())
    print(f"    Lowest mean FNR of any configuration: {fnr_best:.1%}")
    return t1, t2, hm


# ========== 6. Publication figures ==========
# Grayscale with distinct markers, because Springer prints in black and white; sized to the
# 4.6 in LNCS/CCIS text width at 1200 dpi so no rescaling is needed in Word.
FIG_WIDTH_IN = 4.6
FIG_DPI = 1200


def _apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "text.color": "black",
    })


def save_threshold_dispersion_figure(df_fold_results, output_dir="visualizations",
                                     filename="fig2_threshold_dispersion.png", dpi=FIG_DPI):
    """Paper Fig. 2: every per-fold t*, grouped by model and horizon; bars = fold mean."""
    _apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)

    horizons = _horizon_order(df_fold_results)
    models = [m for m in MODEL_ORDER if m in set(df_fold_results["Model"])]
    marks = ["o", "s", "^", "D", "v"]
    rng = np.random.default_rng(0)          # fixed jitter keeps the figure reproducible

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.9))
    for xi, model in enumerate(models):
        for hi, horizon in enumerate(horizons):
            t = df_fold_results.loc[(df_fold_results["Model"] == model) &
                                    (df_fold_results["Horizon"] == horizon),
                                    "Best Threshold"].values
            if len(t) == 0:
                continue
            offset = (hi - (len(horizons) - 1) / 2) * 0.22
            x = xi + offset + rng.uniform(-0.03, 0.03, len(t))
            ax.scatter(x, t, marker=marks[hi % len(marks)], s=17, facecolor="white",
                       edgecolor="black", linewidth=0.8, zorder=3,
                       label=horizon.replace(" Horizon", "").lower() if xi == 0 else "")
            ax.plot([xi + offset - 0.09, xi + offset + 0.09], [t.mean()] * 2,
                    color="black", lw=1.4, zorder=4)

    ax.axhline(0.5, color="0.6", ls=":", lw=0.8, zorder=1)
    ax.text(len(models) - 0.48, 0.515, "default 0.5", fontsize=5.6, color="0.35", ha="right")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=7.5)
    ax.set_ylabel("Selected threshold $t^*$", fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=6.2, ncol=len(horizons), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), handletextpad=0.3, columnspacing=1.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.75")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.87, bottom=0.11)

    path = os.path.join(output_dir, filename)
    _save_figure(fig, path, dpi)
    plt.close(fig)
    print(f"[+] Fig. 2 saved to: {path}")
    return path


def save_performance_figure(df_fold_results, output_dir="visualizations",
                            filename="figS_performance.png", dpi=FIG_DPI):
    """Supplementary (not in the paper): MCC, PR-AUC and FNR by horizon, +/- 1 SD."""
    _apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)

    horizons = _horizon_order(df_fold_results)
    models = [m for m in MODEL_ORDER if m in set(df_fold_results["Model"])]
    fills = ["white", "0.78", "0.55", "0.30"]
    hatches = ["", "///", "...", "xx"]
    panels = [("MCC", "MCC", 1.0),
              ("PR-AUC", "PR-AUC", 1.0),
              ("FNR (False Neg Rate)", "False negative rate (%)", 100.0)]

    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_WIDTH_IN, 5.0))
    x = np.arange(len(horizons))
    bw = 0.76 / max(len(models), 1)

    for ax, (col, label, scale) in zip(axes, panels):
        for i, model in enumerate(models):
            mu, sd = [], []
            for horizon in horizons:
                v = df_fold_results.loc[(df_fold_results["Model"] == model) &
                                        (df_fold_results["Horizon"] == horizon), col]
                mu.append(v.mean() * scale)
                sd.append(v.std(ddof=1) * scale if len(v) > 1 else 0.0)
            ax.bar(x + (i - (len(models) - 1) / 2) * bw, mu, bw, yerr=sd, capsize=1.8,
                   label=MODEL_SHORT.get(model, model), facecolor=fills[i % len(fills)],
                   edgecolor="black", linewidth=0.7, hatch=hatches[i % len(hatches)],
                   error_kw=dict(lw=0.7, ecolor="black"))
        ax.set_xticks(x)
        ax.set_xticklabels([h.replace(" Horizon", "").lower() for h in horizons], fontsize=7)
        ax.set_ylabel(label, fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", ls=":", lw=0.4, color="0.7")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].legend(fontsize=6.2, ncol=len(models), frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 1.30), handlelength=1.4, columnspacing=1.1)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.90, bottom=0.06, hspace=0.36)

    path = os.path.join(output_dir, filename)
    _save_figure(fig, path, dpi)
    plt.close(fig)
    print(f"[+] Supplementary performance figure saved to: {path}")
    return path


def save_seed_dispersion_figure(df_all, output_dir="visualizations",
                                filename="figS_threshold_sd_by_seed.png", dpi=FIG_DPI):
    """Supplementary: within-seed fold SD of t*, one marker per seed, by model and horizon."""
    if "Seed" not in df_all or df_all["Seed"].nunique() < 2:
        return None
    _apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    horizons = _horizon_order(df_all)
    models = [m for m in MODEL_ORDER if m in set(df_all["Model"])]
    sd = (df_all.groupby(["Seed", "Horizon", "Model"])["Best Threshold"].std()
          .rename("sd").reset_index())
    marks = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.9))
    for xi, model in enumerate(models):
        for hi, horizon in enumerate(horizons):
            v = sd.loc[(sd["Model"] == model) & (sd["Horizon"] == horizon), "sd"].values
            offset = (hi - (len(horizons) - 1) / 2) * 0.22
            ax.scatter(np.full(len(v), xi + offset), v, marker=marks[hi % len(marks)], s=17,
                       facecolor="white", edgecolor="black", linewidth=0.8, zorder=3,
                       label=horizon.replace(" Horizon", "").lower() if xi == 0 else "")
            if len(v):
                ax.plot([xi + offset - 0.09, xi + offset + 0.09], [v.mean()] * 2,
                        color="black", lw=1.4, zorder=4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=7.5)
    ax.set_ylabel("Fold-to-fold SD of $t^*$", fontsize=7.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=6.2, ncol=len(horizons), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), handletextpad=0.3, columnspacing=1.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.75")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.87, bottom=0.11)
    path = os.path.join(output_dir, filename)
    _save_figure(fig, path, dpi)
    plt.close(fig)
    print(f"[+] Supplementary seed-dispersion figure saved to: {path}")
    return path


def save_publication_figures(df_all, output_dir="visualizations", dpi=FIG_DPI):
    """Fig. 2 and the performance figure from the reference seed; seed figure from all."""
    ref = _reference_rows(df_all)
    return [save_threshold_dispersion_figure(ref, output_dir, dpi=dpi),
            save_performance_figure(ref, output_dir, dpi=dpi),
            save_seed_dispersion_figure(df_all, output_dir, dpi=dpi)]


# ========== 7. Repeated-seed robustness (review revision) ==========
def _reference_rows(df_all):
    """Rows of the reference run (SEED), i.e. the run behind Tables 1-2 and Fig. 2."""
    if "Seed" not in df_all:
        return df_all
    ref = df_all[df_all["Seed"] == SEED]
    return ref if len(ref) else df_all[df_all["Seed"] == df_all["Seed"].iloc[0]]


def export_seed_robustness(df_all, results_dir="results"):
    """Does the stability ranking survive new company partitions?

    For every seed, the fold-to-fold SD of t* is computed exactly as in Table 1; the tables
    below summarise that SD across seeds. Repetitions share the same data, so the
    Brown-Forsythe tests on pooled thresholds are descriptive, not confirmatory.
    """
    if "Seed" not in df_all or df_all["Seed"].nunique() < 2:
        print("[*] Only one seed run - repeated-seed analysis skipped.")
        return None
    from scipy.stats import levene

    os.makedirs(results_dir, exist_ok=True)
    seeds = list(dict.fromkeys(df_all["Seed"]))
    horizons = _horizon_order(df_all)
    models = [m for m in MODEL_ORDER if m in set(df_all["Model"])]
    label = {h: h.replace(" Horizon", "").lower() for h in horizons}

    # Within-seed fold SD of t*, the Table 1 quantity, for every seed
    sd = (df_all.groupby(["Seed", "Horizon", "Model"])["Best Threshold"].std()
          .rename("Threshold SD").reset_index())
    sd.to_csv(os.path.join(results_dir, "threshold_sd_by_seed.csv"), index=False)

    # Table S1: mean within-seed SD, with its min-max across seeds
    s1 = []
    for m in models:
        row = {"Model": m}
        for h in horizons:
            v = sd.loc[(sd["Model"] == m) & (sd["Horizon"] == h), "Threshold SD"]
            row[label[h]] = f"{v.mean():.3f} ({v.min():.3f}-{v.max():.3f})"
        s1.append(row)
    s1 = pd.DataFrame(s1)

    # How often each model has the lowest threshold SD (seed x horizon cells)
    lowest = sd.loc[sd.groupby(["Seed", "Horizon"])["Threshold SD"].idxmin(), "Model"]
    n_cells = sd.groupby(["Seed", "Horizon"]).ngroups
    wins = lowest.value_counts().reindex(models, fill_value=0)

    # Table S3: dispersion of RF thresholds vs each other model, pooled over seeds x folds
    tests = []
    if "Random Forest" in models:
        for h in horizons:
            rf = df_all.loc[(df_all["Model"] == "Random Forest") &
                            (df_all["Horizon"] == h), "Best Threshold"]
            for m in models:
                if m == "Random Forest":
                    continue
                other = df_all.loc[(df_all["Model"] == m) &
                                   (df_all["Horizon"] == h), "Best Threshold"]
                stat, p = levene(rf, other, center="median")
                tests.append({"Horizon": label[h], "Comparison": f"RF vs {MODEL_SHORT[m]}",
                              "n per model": len(rf), "pooled SD RF": round(rf.std(), 3),
                              f"pooled SD other": round(other.std(), 3),
                              "Brown-Forsythe W": round(stat, 3), "p": round(p, 4)})
    s3 = pd.DataFrame(tests)

    # Table S2: performance, mean of seed means (between-seed SD)
    perf = (df_all.groupby(["Seed", "Horizon", "Model"])
            [["MCC", "PR-AUC", "FNR (False Neg Rate)"]].mean().reset_index())
    s2 = []
    for h in horizons:
        for m in models:
            p = perf[(perf["Horizon"] == h) & (perf["Model"] == m)]
            s2.append({"Horizon": label[h], "Model": MODEL_SHORT[m],
                       "MCC": _mean_sd(p["MCC"]), "PR-AUC": _mean_sd(p["PR-AUC"]),
                       "FNR (%)": _mean_sd(p["FNR (False Neg Rate)"], 100, 1)})
    s2 = pd.DataFrame(s2)

    hm = (perf.groupby(["Seed", "Horizon"])[["MCC", "PR-AUC"]].mean().reset_index()
          .groupby("Horizon")[["MCC", "PR-AUC"]].agg(["mean", "min", "max"])
          .reindex(horizons).rename(index=label).round(3))

    s1.to_csv(os.path.join(results_dir, "tableS1_threshold_sd_across_seeds.csv"), index=False)
    s2.to_csv(os.path.join(results_dir, "tableS2_performance_across_seeds.csv"), index=False)
    s3.to_csv(os.path.join(results_dir, "tableS3_dispersion_tests.csv"), index=False)
    hm.to_csv(os.path.join(results_dir, "horizon_means_across_seeds.csv"))

    print(f"\n========== REPEATED SEEDS: {len(seeds)} repetitions {seeds} ==========")
    print("Table S1 - fold-to-fold SD of t*: mean over seeds (min-max)")
    print(s1.to_string(index=False))
    print(f"\nLowest threshold SD, out of {n_cells} seed x horizon cells:")
    for m in models:
        print(f"    {m:<18} {wins[m]}")
    print("\nTable S2 - performance: mean of seed means (between-seed SD)")
    print(s2.to_string(index=False))
    print("\nMean across models per horizon: mean (min, max over seeds)")
    print(hm.to_string())
    if len(s3):
        print("\nTable S3 - RF threshold dispersion vs others (pooled; descriptive)")
        print(s3.to_string(index=False))
    return {"S1": s1, "S2": s2, "S3": s3, "wins": wins, "horizon_means": hm}


# ========== 7b. Paired significance tests (Reviewer 6) ==========
def corrected_resampled_t(diffs, n_splits=N_SPLITS):
    """Corrected repeated k-fold t-test (Nadeau and Bengio 2003; Bouckaert and Frank 2004).

    diffs: per-fold differences between two models, paired by (seed, fold). The variance is
    inflated by n_test/n_train because the training sets of different folds overlap.
    Returns mean difference, its 95% CI, t and two-sided p.
    """
    from scipy import stats
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    ratio = 1.0 / (n_splits - 1)                    # n_test / n_train for k-fold
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / n + ratio) * var)
    if se == 0:
        return d.mean(), (d.mean(), d.mean()), 0.0, 1.0
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    half = stats.t.ppf(0.975, df=n - 1) * se
    return d.mean(), (d.mean() - half, d.mean() + half), t, p


def _holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def export_significance_tests(df_all, results_dir="results"):
    """Pairwise model comparisons per horizon (corrected t-test, Holm-adjusted), 95% CIs of
    each model's mean, and a horizon effect test per model (Kruskal-Wallis on fold values)."""
    from scipy.stats import kruskal
    os.makedirs(results_dir, exist_ok=True)
    horizons = _horizon_order(df_all)
    models = [m for m in MODEL_ORDER if m in set(df_all["Model"])]
    label = {h: h.replace(" Horizon", "").lower() for h in horizons}
    metrics = [("MCC", "MCC"), ("PR-AUC", "PR-AUC")]

    rows = []
    for h in horizons:
        sub = df_all[df_all["Horizon"] == h]
        for col, name in metrics:
            wide = sub.pivot_table(index=["Seed", "Fold"], columns="Model", values=col)
            block = []
            for i, a in enumerate(models):
                for b in models[i + 1:]:
                    pair = wide[[a, b]].dropna()
                    mean, ci, t, p = corrected_resampled_t(pair[a] - pair[b])
                    block.append({"Horizon": label[h], "Metric": name,
                                  "Comparison": f"{MODEL_SHORT[a]} - {MODEL_SHORT[b]}",
                                  "n folds": len(pair), "Mean diff": mean,
                                  "CI low": ci[0], "CI high": ci[1], "t": t, "p": p})
            adj = _holm([r["p"] for r in block])
            for r, pa in zip(block, adj):
                r["p (Holm)"] = pa
            rows.extend(block)
    pairs = pd.DataFrame(rows)

    ci_rows = []
    for h in horizons:
        for m in models:
            v = df_all.loc[(df_all["Horizon"] == h) & (df_all["Model"] == m)]
            row = {"Horizon": label[h], "Model": MODEL_SHORT[m]}
            for col, name in metrics + [("FNR (False Neg Rate)", "FNR")]:
                mean, ci, _, _ = corrected_resampled_t(v[col].to_numpy())
                scale = 100 if name == "FNR" else 1
                dec = 1 if name == "FNR" else 3
                row[name] = (f"{mean * scale:.{dec}f} "
                             f"[{ci[0] * scale:.{dec}f}, {ci[1] * scale:.{dec}f}]")
            ci_rows.append(row)
    cis = pd.DataFrame(ci_rows)

    kw_rows = []
    for m in (models if len(horizons) > 1 else []):
        for col, name in metrics:
            groups_ = [df_all.loc[(df_all["Model"] == m) & (df_all["Horizon"] == h), col]
                       for h in horizons]
            stat, p = kruskal(*groups_)
            kw_rows.append({"Model": MODEL_SHORT[m], "Metric": name,
                            "H": round(stat, 3), "p": round(p, 4)})
    kw = pd.DataFrame(kw_rows)

    pairs.round(4).to_csv(os.path.join(results_dir, "tableS4_pairwise_tests.csv"), index=False)
    cis.to_csv(os.path.join(results_dir, "tableS5_means_95ci.csv"), index=False)
    kw.to_csv(os.path.join(results_dir, "tableS6_horizon_tests.csv"), index=False)

    print("\n========== PAIRED TESTS (corrected repeated k-fold t, Holm per horizon x metric) ==")
    print(pairs.round(4).to_string(index=False))
    print(f"\n    Holm-adjusted p < 0.05: {int((pairs['p (Holm)'] < 0.05).sum())} of {len(pairs)}")
    print("\n========== Model means with corrected 95% CI ==========")
    print(cis.to_string(index=False))
    print("\n========== Horizon effect per model (Kruskal-Wallis on fold values) ==========")
    print(kw.to_string(index=False))
    return {"pairs": pairs, "ci": cis, "horizon": kw}


# ========== 7c. Calibration analysis (Reviewer 6) ==========
def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return ece


def calibration_summary(test_df, curves_df, df_all, results_dir="results", tol=0.01):
    """Per model and horizon: Brier score, ECE and probability mass in the tails (held-out
    folds), and the shape of the inner MCC surface that t* is read from: the share of the grid
    within `tol` of the maximum MCC and the span between the farthest near-optimal thresholds.
    The span is then related to the observed fold-to-fold threshold SD."""
    from scipy.stats import spearmanr
    os.makedirs(results_dir, exist_ok=True)
    grid_cols = [c for c in curves_df.columns if c[:1].isdigit()]
    grid = np.array([float(c) for c in grid_cols])

    cal = []
    for (s, h, f, m), g in test_df.groupby(["Seed", "Horizon", "Fold", "Model"]):
        y, p = g["y"].to_numpy(float), g["p"].to_numpy(float)
        cal.append({"Seed": s, "Horizon": h, "Fold": f, "Model": m,
                    "Brier": np.mean((p - y) ** 2), "ECE": _ece(y, p),
                    "Tails": np.mean((p < 0.1) | (p > 0.9)),
                    "Middle": np.mean((p >= 0.3) & (p <= 0.7))})
    cal = pd.DataFrame(cal)

    surf = []
    for _, r in curves_df.iterrows():
        c = r[grid_cols].to_numpy(float)
        near = grid[c >= c.max() - tol]
        surf.append({"Seed": r["Seed"], "Horizon": r["Horizon"], "Fold": r["Fold"],
                     "Model": r["Model"], "Plateau share": len(near) / len(grid),
                     "Near-optimal span": near.max() - near.min()})
    surf = pd.DataFrame(surf)

    keys = ["Seed", "Horizon", "Fold", "Model"]
    per_fold = cal.merge(surf, on=keys)
    summary = (per_fold.groupby(["Model", "Horizon"])
               [["Brier", "ECE", "Tails", "Middle", "Plateau share", "Near-optimal span"]]
               .mean().round(3).reset_index())

    # Does a wide near-optimal span go with an unstable threshold? One point per
    # seed x horizon x model: mean span over folds vs SD of t* over folds.
    cell = (per_fold.groupby(["Seed", "Horizon", "Model"])["Near-optimal span"].mean()
            .rename("span").reset_index()
            .merge(df_all.groupby(["Seed", "Horizon", "Model"])["Best Threshold"].std()
                   .rename("thr_sd").reset_index(), on=["Seed", "Horizon", "Model"]))
    rho, p_rho = spearmanr(cell["span"], cell["thr_sd"])

    summary.to_csv(os.path.join(results_dir, "tableS7_calibration.csv"), index=False)
    per_fold.to_csv(os.path.join(results_dir, "calibration_per_fold.csv"), index=False)
    print("\n========== CALIBRATION AND MCC-SURFACE SHAPE (mean over seeds x folds) ==========")
    print(summary.to_string(index=False))
    print(f"\n    Spearman(near-optimal span, threshold SD) over {len(cell)} seed-horizon-model "
          f"cells: rho = {rho:.3f}, p = {p_rho:.2g}")
    return {"summary": summary, "per_fold": per_fold, "rho": rho, "p_rho": p_rho, "cells": cell}


def save_calibration_figure(test_df, curves_df, oof_df=None, output_dir="visualizations",
                            filename="fig3_calibration.png", dpi=FIG_DPI, bins=10):
    """Fig. 3: (a) reliability diagram of held-out probabilities, (b) density of inner
    out-of-fold probabilities, (c) inner MCC minus its fold maximum across the threshold grid,
    averaged over folds (a flatter curve = more thresholds close to the optimum).
    All seeds, folds and horizons pooled."""
    _apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    models = [m for m in MODEL_ORDER if m in set(test_df["Model"])]
    styles = {"LightGBM": dict(color="black", ls="-", marker="o"),
              "Random Forest": dict(color="0.45", ls="--", marker="s"),
              "XGBoost": dict(color="black", ls=":", marker="^"),
              "Stacking Ensemble": dict(color="0.6", ls="-.", marker="D")}
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 1.95))
    edges = np.linspace(0, 1, bins + 1)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="0.75", lw=0.7, zorder=1)
    for m in models:
        g = test_df[test_df["Model"] == m]
        idx = np.clip(np.digitize(g["p"], edges[1:-1]), 0, bins - 1)
        xs, ys = [], []
        for b in range(bins):
            sel = idx == b
            if sel.sum() >= 20:
                xs.append(g["p"].to_numpy()[sel].mean()); ys.append(g["y"].to_numpy()[sel].mean())
        st = styles[m]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=0.9, marker=st["marker"],
                ms=2.4, mfc="white", mew=0.6, label=MODEL_SHORT[m])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", fontsize=6.5)
    ax.set_ylabel("Observed distress rate", fontsize=6.5)
    ax.set_title("(a) Reliability", fontsize=7)

    ax = axes[1]
    oof_hist_edges = np.linspace(0, 1, 21)
    dens_src = oof_df if oof_df is not None else test_df
    for m in models:
        # inner out-of-fold probabilities: the scores t* is actually selected on
        g = dens_src[dens_src["Model"] == m]["p"].to_numpy()
        h, _ = np.histogram(g, bins=oof_hist_edges, density=True)
        st = styles[m]
        ax.plot((oof_hist_edges[:-1] + oof_hist_edges[1:]) / 2, h, color=st["color"],
                ls=st["ls"], lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability", fontsize=6.5)
    ax.set_ylabel("Density", fontsize=6.5)
    ax.set_title("(b) Score distribution", fontsize=7)

    ax = axes[2]
    grid_cols = [c for c in curves_df.columns if c[:1].isdigit()]
    grid = np.array([float(c) for c in grid_cols])
    for m in models:
        c = curves_df[curves_df["Model"] == m][grid_cols].to_numpy(float)
        rel = c - c.max(axis=1, keepdims=True)
        st = styles[m]
        ax.plot(grid, rel.mean(axis=0), color=st["color"], ls=st["ls"], lw=0.9)
    ax.set_xlim(0.05, 0.95)
    ax.set_xlabel("Threshold $t$", fontsize=6.5)
    ax.set_ylabel("MCC $-$ max MCC", fontsize=6.5)
    ax.set_title("(c) Inner MCC", fontsize=7)

    for ax in axes:
        ax.tick_params(labelsize=5.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=5.6, frameon=False, ncol=4, loc="upper center",
                   bbox_to_anchor=(1.75, 1.42), handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.78, bottom=0.2, wspace=0.48)
    path = os.path.join(output_dir, filename)
    _save_figure(fig, path, dpi)
    plt.close(fig)
    print(f"[+] Fig. 3 saved to: {path} (+ .pdf, .svg)")
    return path


def _save_figure(fig, path, dpi=FIG_DPI):
    """Raster at `dpi` plus vector PDF and SVG copies (Reviewer 5, Fig. 2 resolution).

    The vector files are written deterministically (no timestamp, fixed SVG id salt), so the
    same results always give byte-identical PDF/SVG files. The SVG is the version embedded in
    the paper; the PNG is only Word's fallback for viewers without SVG support."""
    fig.savefig(path, dpi=dpi, facecolor="white", metadata={"Software": None})
    base = os.path.splitext(path)[0]
    fig.savefig(base + ".pdf", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                          "Creator": None})
    with plt.rc_context({"svg.hashsalt": "fdp-paper103", "svg.fonttype": "path"}):
        fig.savefig(base + ".svg", facecolor="white", metadata={"Date": None, "Creator": None})


# ========== 8. Environment (Sect. 3.6) ==========
def environment_info(plan=None):
    info = {
        "Python": platform.python_version(),
        "OS": f"{platform.system()} {platform.release()}",
        "Processor": platform.processor() or platform.machine(),
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "xgboost": xgb.__version__,
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "GROUP_AWARE_STACKING": GROUP_AWARE_STACKING,
    }
    if plan is not None:
        info["Seed plan (seed: outer/inner/model)"] = "; ".join(
            f"{r['seed']}: {r['outer']}/{r['inner']}/{r['model']}" for r in plan)
    return info


# ========== 9. Main experiment loop ==========
def run_experiment_per_year(file_path, max_horizon=3, min_year=2015,
                            seeds=None, vary_model_seed=None, collect=None):
    """Run the nested grouped CV for every horizon and every seed in the plan.

    seeds=[42] reproduces the submitted single-seed results. The default (SEEDS) adds four
    repetitions in which the partition, SMOTE and the learners all change.

    collect: optional dict; when given, it is filled with
      "curves": one row per seed/horizon/fold/model with the inner out-of-fold MCC at every
                grid threshold (the surface t* is chosen from, Eq. 3),
      "test":   held-out probabilities (y, p) for the calibration analysis,
      "oof":    inner out-of-fold probabilities on the training partition.
    """
    if collect is not None:
        for k in ("curves", "test", "oof"):
            collect.setdefault(k, [])
    datasets_dict = process_turkish_panel_data(file_path, max_horizon=max_horizon,
                                               min_year=min_year)
    plan = seed_plan(seeds, vary_model_seed)
    print("\n[*] Seed plan (seed: outer / inner / model):")
    for r in plan:
        print(f"    {r['seed']}: {r['outer']} / {r['inner']} / {r['model']}"
              + ("   <- reference run (Tables 1-2, Fig. 2)" if r["seed"] == SEED else ""))
    all_results = []

    for horizon_name, (X, y, groups) in datasets_dict.items():
        print(f"\n{'=' * 70}\nSTARTING EXPERIMENT FOR HORIZON: {horizon_name}\n{'=' * 70}")

        # Three levels of grouped CV are nested; with too few positives an inner training
        # fold can lose the positive class entirely. Skip rather than report noise.
        n_positives = int(y.sum())
        if n_positives < MIN_POSITIVES_FOR_NESTED_CV:
            print(f"[!] Skipping {horizon_name}: only {n_positives} positive(s), below the "
                  f"{MIN_POSITIVES_FOR_NESTED_CV} needed for the nested {N_SPLITS}-fold CV.")
            continue

        for rep in plan:
            models = build_models(rep["model"])

            # Outer split: companies, not rows (Sect. 3.5)
            outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                            random_state=rep["outer"])

            for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups),
                                                             start=1):
                print(f"\n[*] Seed {rep['seed']} | Fold {fold_num}/{N_SPLITS} | {horizon_name}")

                overlap = set(groups[train_idx]) & set(groups[test_idx])
                assert not overlap, (f"[!] Company overlap in fold {fold_num}: {len(overlap)} "
                                     f"firms in both train and test - grouping is not working.")

                print(f"    train: {len(train_idx)} rows / {len(set(groups[train_idx]))} firms"
                      f" | test: {len(test_idx)} rows / {len(set(groups[test_idx]))} firms"
                      f" | test positives: {int(y.iloc[test_idx].sum())}")

                X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
                y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
                groups_train = groups[train_idx].copy()

                for model_name, (model, needs_groups) in models.items():
                    print(f"   -> {model_name}...")
                    try:
                        estimator = clone(model)

                        # Inner grouped split on the training partition only: out-of-fold
                        # probabilities at the observed prevalence, never on SMOTE output.
                        inner_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                                        random_state=rep["inner"])
                        y_oof_prob = _oof_probabilities(estimator, X_train, y_train,
                                                        groups_train, inner_cv, needs_groups)
                        best_threshold, validation_mcc = find_best_threshold(y_train,
                                                                             y_oof_prob)

                        # Refit on the full training partition (imputation, scaling, SMOTE
                        # included), then apply t* unchanged to the held-out companies.
                        fit_kwargs = {"groups": groups_train} if needs_groups else {}
                        estimator.fit(X_train, y_train, **fit_kwargs)
                        y_prob = estimator.predict_proba(X_test)[:, 1]
                        fold_metrics = calculate_metrics_at_threshold(y_test, y_prob,
                                                                      best_threshold)
                    except (ValueError, IndexError) as e:
                        print(f"      [!] {model_name} failed on seed {rep['seed']} fold "
                              f"{fold_num} of {horizon_name} ({e}) - skipping.")
                        continue

                    if collect is not None:
                        key = {"Seed": rep["seed"], "Horizon": horizon_name,
                               "Fold": fold_num, "Model": model_name}
                        yt = np.asarray(y_train)
                        curve = {f"{t:.2f}": matthews_corrcoef(yt, (y_oof_prob >= t).astype(int))
                                 for t in THRESHOLD_GRID}
                        collect["curves"].append({**key, **curve})
                        collect["test"].append(pd.DataFrame({
                            **key, "row": X_test.index.to_numpy(),
                            "y": np.asarray(y_test), "p": y_prob}))
                        collect["oof"].append(pd.DataFrame({
                            **key, "y": yt, "p": y_oof_prob}))

                    all_results.append({
                        "Dataset":        f"Turkish_Audit_{min_year}_plus",
                        "Seed":           rep["seed"],
                        "Horizon":        horizon_name,
                        "Fold":           fold_num,
                        "Model":          model_name,
                        "Best Threshold": best_threshold,
                        "Validation MCC": validation_mcc,
                        **fold_metrics,
                    })
                    print(f"      Threshold: {best_threshold:.2f} | "
                          f"PR-AUC: {fold_metrics['PR-AUC']:.4f} | "
                          f"FNR: {fold_metrics['FNR (False Neg Rate)']:.4f} | "
                          f"MCC: {fold_metrics['MCC']:.4f}")

    df_fold_results = pd.DataFrame(all_results)
    if df_fold_results.empty:
        print("[!] No horizon produced any results. Check the positive counts above.")
        return pd.DataFrame(), df_fold_results

    keys = ["Dataset", "Seed", "Horizon", "Model"]
    metric_columns = ["Recall (Sensitivity)", "FNR (False Neg Rate)", "PR-AUC", "MCC"]
    aggregation = {m: "mean" for m in metric_columns}
    aggregation.update({"Best Threshold": "mean", "Validation MCC": "mean",
                        "TN": "sum", "FP": "sum", "FN": "sum", "TP": "sum"})
    df_results = df_fold_results.groupby(keys, as_index=False).agg(aggregation)

    sd_cols = metric_columns + ["Best Threshold"]
    df_std = (df_fold_results.groupby(keys, as_index=False)[sd_cols].std()
              .rename(columns={m: f"{m} (SD)" for m in sd_cols}))
    df_results = df_results.merge(df_std, on=keys, how="left")
    df_counts = (df_fold_results.groupby(keys, as_index=False).size()
                 .rename(columns={"size": "Folds"}))
    df_results = df_results.merge(df_counts, on=keys, how="left")

    incomplete = df_results[df_results["Folds"] < N_SPLITS]
    if len(incomplete):
        print(f"[!] Averaged over fewer than {N_SPLITS} folds - footnote these in the paper:")
        print(incomplete[["Seed", "Horizon", "Model", "Folds"]].to_string(index=False))
    else:
        print(f"[+] Every seed-model-horizon combination completed all {N_SPLITS} folds.")

    round_columns = (metric_columns + [f"{m} (SD)" for m in sd_cols]
                     + ["Best Threshold", "Validation MCC"])
    df_results[round_columns] = df_results[round_columns].round(4)
    df_results = df_results.sort_values(["Seed", "Horizon", "Model"]).reset_index(drop=True)
    return df_results, df_fold_results


def write_outputs(df_fold_results, curves, test, oof, results_dir, visualizations_dir):
    """Every paper table (Tables 1-4, S1-S7) and figure (Figs. 2-3, supplementary) from the
    fold-level results. Used after training and by --from-results."""
    print(f"\n[*] Paper tables below use the reference run (seed {SEED}).")
    export_paper_tables(_reference_rows(df_fold_results), results_dir)
    export_seed_robustness(df_fold_results, results_dir)
    save_publication_figures(df_fold_results, visualizations_dir)
    if df_fold_results["Seed"].nunique() > 1:
        export_significance_tests(df_fold_results, results_dir)
    calibration_summary(test, curves, df_fold_results, results_dir)
    save_calibration_figure(test, curves, oof, visualizations_dir)


def rebuild_outputs(results_dir="results", visualizations_dir="visualizations"):
    """Regenerate tables and figures from a finished run without retraining any model."""
    def need(name):
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run the experiment first.")
        return pd.read_csv(path)
    folds = need("results_per_fold.csv")
    curves = need("inner_mcc_curves.csv.gz")
    test = need("test_probabilities.csv.gz")
    oof = need("oof_probabilities.csv.gz")
    print(f"[*] Rebuilding outputs from '{results_dir}': {len(folds)} fold rows, "
          f"seeds {sorted(folds['Seed'].unique())}")
    write_outputs(folds, curves, test, oof, results_dir, visualizations_dir)
    print(f"[+] Tables in '{results_dir}', figures in '{visualizations_dir}'.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Multi-horizon financial distress prediction with threshold-stability "
                    "analysis (Turkish audit-opinion panel)."
    )
    parser.add_argument("--input-file", default=os.path.join("data", "financial_data.csv"),
                        help="Path to the input CSV file.")
    parser.add_argument("--results-dir", default="results",
                        help="Directory where CSV outputs will be saved.")
    parser.add_argument("--visualizations-dir", default="visualizations",
                        help="Directory where figures will be saved.")
    parser.add_argument("--min-year", type=int, default=2015,
                        help="Minimum year to keep from the dataset.")
    parser.add_argument("--max-horizon", type=int, default=3,
                        help="Maximum prediction horizon to generate.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Cross-validation seeds (default {SEEDS}). '--seeds {SEED}' "
                             f"reproduces the single-seed reference run only.")
    parser.add_argument("--from-results", action="store_true",
                        help="Do not retrain: rebuild every table and figure from the files a "
                             "previous run saved in --results-dir (results_per_fold.csv, "
                             "inner_mcc_curves.csv.gz, test_probabilities.csv.gz, "
                             "oof_probabilities.csv.gz).")
    parser.add_argument("--fixed-model-seed", action="store_true",
                        help=f"Keep the learner and SMOTE seed at {SEED} in every repetition "
                             f"(vary only the company partition).")
    return parser


def main(argv=None):
    is_jupyter = "ipykernel" in sys.modules or "google.colab" in sys.modules
    if argv is None and is_jupyter:
        argv = []
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.visualizations_dir, exist_ok=True)
    if args.from_results:
        rebuild_outputs(args.results_dir, args.visualizations_dir)
        return

    vary = not args.fixed_model_seed
    plan = seed_plan(args.seeds, vary)
    env = environment_info(plan)
    print("[*] Environment:")
    for k, v in env.items():
        print(f"    {k:<38} {v}")

    start = time.perf_counter()
    collect = {}
    df_results, df_fold_results = run_experiment_per_year(
        args.input_file, max_horizon=args.max_horizon, min_year=args.min_year,
        seeds=args.seeds, vary_model_seed=vary, collect=collect)
    minutes = (time.perf_counter() - start) / 60
    if df_results.empty:
        return

    print("\n\n========== FINAL AGGREGATED RESULTS (per seed) ==========")
    print(df_results.to_string(index=False))

    df_results.to_csv(os.path.join(args.results_dir, "results_summary.csv"), index=False)
    df_fold_results.to_csv(os.path.join(args.results_dir, "results_per_fold.csv"), index=False)
    df_results[["Seed", "Horizon", "Model", "Best Threshold", "Best Threshold (SD)",
                "Validation MCC"]].to_csv(
        os.path.join(args.results_dir, "results_thresholds.csv"), index=False)

    curves = pd.DataFrame(collect["curves"])
    test = pd.concat(collect["test"], ignore_index=True)
    oof = pd.concat(collect["oof"], ignore_index=True)
    curves.to_csv(os.path.join(args.results_dir, "inner_mcc_curves.csv.gz"), index=False)
    test.to_csv(os.path.join(args.results_dir, "test_probabilities.csv.gz"), index=False)
    oof.to_csv(os.path.join(args.results_dir, "oof_probabilities.csv.gz"), index=False)

    # Tables 1-2 and Fig. 2 come from the reference seed, as in the paper
    write_outputs(df_fold_results, curves, test, oof, args.results_dir, args.visualizations_dir)

    env["Runtime (minutes)"] = f"{minutes:.1f}"
    with open(os.path.join(args.results_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        for k, v in env.items():
            f.write(f"{k}: {v}\n")
    print(f"\n[+] Done in {minutes:.1f} minutes. Results in '{args.results_dir}', "
          f"figures in '{args.visualizations_dir}'.")


if __name__ == "__main__":
    main()