import argparse
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

# Cosmetic only: LightGBM records feature names at fit time and receives a bare array at
# predict time inside the pipeline. Column ORDER is preserved, so predictions are unaffected.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Inner cross-validation parallelism. The base learners already use n_jobs=-1, so raising
# this nests parallelism and can oversubscribe the CPU. Increase only if you have cores spare.
CV_N_JOBS = 1


# ========== SAFE PIPELINE COMPONENTS ==========
class SafeIterativeImputer(BaseEstimator, TransformerMixin):
    """IterativeImputer that falls back to median imputation if MICE fails.

    Without the fallback, a single sparse fold aborts the whole run.
    """

    def __init__(self, max_iter=10, random_state=42,
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
    """SMOTE that adapts k_neighbors to the minority count of the fold it is fitted on.

    SMOTE needs at least k_neighbors + 1 minority samples (6 by default) and raises
    otherwise; grouped folds at the longer horizons can fall below that.
    """

    def fit_resample(self, X, y):
        y_arr = np.asarray(y).astype(int)
        n_minority = int(min(np.bincount(y_arr, minlength=2)))
        if n_minority < 2:
            print(f"[!] Only {n_minority} minority sample(s) in this fold - skipping SMOTE.")
            return X, y
        self.k_neighbors = min(5, n_minority - 1)
        return super().fit_resample(X, y)


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

TARGET_LABEL_TRANSLATIONS = {
    "Olumlu":                                 "Unqualified",
    "Şartlı":                                 "Qualified",
    "Görüş bildirmekten kaçınma":             "Disclaimer of Opinion",
    "Olumsuz":                                "Adverse",
}


# ========== 1. Process Panel Data from CSV ==========
def process_turkish_panel_data(file_path, max_horizon=2, min_year=2015):
    print(f"[*] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)

    df.replace(["?", "NA", "NaN", "", " "], np.nan, inplace=True)

    # --- COLUMN NAMES ---
    COMPANY_COL = "Şirket Adı"
    YEAR_COL    = "Yıl"
    TARGET_COL  = "Görüs Tipi"
    PERIOD_COL  = "Periyot"

    if COMPANY_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(
            f"[!] Critical Error: Columns '{COMPANY_COL}' or '{TARGET_COL}' not found. "
            f"Please check your CSV headers."
        )

    # Keep annual reports only
    df = df[df[PERIOD_COL] == "Yıllık"].copy()
    print(f"[*] Rows after annual filter: {len(df)}")

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[YEAR_COL] >= min_year].copy()
    print(f"[*] Rows after {min_year}+ filter: {len(df)}")

    df.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
    print(f"[*] Columns translated to English.")

    # read_csv types any column holding "?" or "NA" as object, and the replace() above swaps
    # the values without changing the dtype. Coerce here so that the object-column drop
    # further down removes only genuine identifier columns, never features.
    ID_COLS = ["Company Name", "Company Code", "Period", "Audit Opinion"]
    n_coerced = 0
    for col in df.columns:
        if col not in ID_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_coerced += 1
    print(f"[*] Coerced {n_coerced} non-identifier columns to numeric.")

    df["Audit Opinion"] = df["Audit Opinion"].map(TARGET_LABEL_TRANSLATIONS)

    n_before = len(df)
    df = df.dropna(subset=["Audit Opinion"]).copy()
    print(f"[*] Dropped {n_before - len(df)} rows with missing/unmapped Audit Opinion.")

    # Binarize Audit Opinion: Unqualified = 0 (healthy), everything else = 1 (distressed)
    healthy_labels = ["Unqualified"]
    df["class"] = np.where(df["Audit Opinion"].isin(healthy_labels), 0, 1)

    print(f"[*] Class distribution:\n{df['class'].value_counts().rename({0: 'Healthy (0)', 1: 'Distressed (1)'}).to_string()}")

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # The merge below needs exactly one row per company-year.
    n_dupes = int(df.duplicated(subset=["Company Name", "Year"]).sum())
    if n_dupes:
        print(f"[!] {n_dupes} duplicate company-year rows - keeping the first of each.")
        df = df.drop_duplicates(subset=["Company Name", "Year"], keep="first").copy()

    df = df.sort_values(by=["Company Name", "Year"]).reset_index(drop=True)

    # Allow up to 3-year horizon for the 2015+ annual window.
    if max_horizon > 3:
        print(f"[!] Warning: max_horizon={max_horizon} is too high for the current window. Capping at 3.")
        max_horizon = 3

    horizon_datasets = {}

    for horizon in range(1, max_horizon + 1):
        print(f"\n[*] Generating {horizon}-Year Horizon dataset...")
        horizon_df = df.copy()

        # Label each row with the outcome h years ahead, matched on the calendar year.
        # A positional shift(-horizon) would mislabel any company with a gap in its
        # reporting history — e.g. putting a 2-year outcome in the 1-year dataset.
        target_col_name = f"Target_{horizon}Y"
        future = (
            df[["Company Name", "Year", "class"]]
            .rename(columns={"class": target_col_name})
            .assign(Year=lambda d: d["Year"] - horizon)
        )
        horizon_df = horizon_df.merge(future, on=["Company Name", "Year"], how="inner")

        # Drop features that could cause target leakage
        columns_to_drop = [
            "Company Name", "Company Code", "Period",
            "Year", "Audit Opinion", "class", target_col_name
        ]

        object_cols = horizon_df.select_dtypes(include=["object", "string"]).columns
        columns_to_drop.extend([c for c in object_cols if c not in columns_to_drop])

        X = horizon_df.drop(columns=columns_to_drop, errors="ignore")
        y = horizon_df[target_col_name].astype(int)

        # Company identity, captured before it is dropped from the features. It never enters
        # the model — the cross-validation splitter uses it to keep all rows belonging to one
        # company on the same side of every split.
        groups = horizon_df["Company Name"].to_numpy()

        X = X.apply(pd.to_numeric, errors="coerce")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.clip(lower=-100000, upper=100000)

        if horizon == 1:
            print(f"[*] {X.shape[1]} features reaching the model: {X.columns.tolist()}")

        # Company-level counts as well as row-level ones. The paper needs both: the positive
        # ROWS overstate the independent evidence, because one distressed firm contributes a
        # positive row in every year it holds a modified opinion.
        n_distressed_firms = len(np.unique(groups[y.to_numpy() == 1]))
        print(f"[+] {horizon}-Year Horizon — X shape: {X.shape}, "
              f"Companies: {len(np.unique(groups))} "
              f"({n_distressed_firms} distressed), "
              f"Class balance: {y.value_counts().to_dict()}")

        # groups travels with X and y — row order is identical for all three.
        horizon_datasets[f"{horizon}-Year Horizon"] = (X, y, groups)

    return horizon_datasets


# ========== 2. Threshold tuning & Metrics ==========
def find_best_threshold(y_true, y_prob, metric="MCC"):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold, best_score = 0.5, -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == "MCC":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "F1-Score":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "Balanced Accuracy":
            score = balanced_accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score, best_threshold = score, threshold

    return float(best_threshold), float(best_score)


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
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


# ========== 3. Publication Figures ==========
# Two properties matter for submission: output is grayscale with hatch patterns, because
# Springer prints in black and white and colour-only distinctions disappear; and figures are
# sized to the 4.6 in text width at 1200 dpi, so they need no rescaling in Word (rescaling
# shrinks in-figure text below the 6 pt minimum Springer requires).

MODEL_ORDER = ["LightGBM", "Random Forest", "XGBoost", "Stacking Ensemble"]
MODEL_SHORT = {"LightGBM": "LGBM", "Random Forest": "RF",
               "XGBoost": "XGB", "Stacking Ensemble": "Stacking"}
FIG_WIDTH_IN = 4.6          # Springer LNCS/CCIS text width is 12.2 cm
FIG_DPI = 1200              # Springer asks >= 800 dpi for line art, 1200 preferred


def _horizon_order(df):
    """Sort '1-Year Horizon', '2-Year Horizon', ... numerically rather than as strings."""
    return sorted(df["Horizon"].dropna().unique(),
                  key=lambda h: int(str(h).split("-")[0]))


def _apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "text.color": "black",
    })


def save_threshold_dispersion_figure(df_fold_results, output_dir="visualizations",
                                     filename="fig_threshold_dispersion.png", dpi=FIG_DPI):
    """Every per-fold threshold selection, grouped by model and horizon.

    Plotting the individual folds rather than their mean is the entire point of this
    figure, so it takes df_fold_results and not the aggregated table.
    """
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
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[+] Threshold dispersion figure saved to: {path}")
    return path


def save_performance_figure(df_fold_results, output_dir="visualizations",
                            filename="fig_performance.png", dpi=FIG_DPI):
    """MCC, PR-AUC and FNR by horizon, with one-standard-deviation error bars.

    The error bars are not decoration: they are what shows the models overlapping.
    """
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
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[+] Performance figure saved to: {path}")
    return path


def save_publication_figures(df_fold_results, output_dir="visualizations", dpi=FIG_DPI):
    """Both paper figures in one call."""
    return [save_threshold_dispersion_figure(df_fold_results, output_dir, dpi=dpi),
            save_performance_figure(df_fold_results, output_dir, dpi=dpi)]


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ========== 4. Main Experiment Loop ==========
def run_experiment_per_year(file_path, max_horizon=3, min_year=2015):
    datasets_dict = process_turkish_panel_data(
        file_path,
        max_horizon=max_horizon,
        min_year=min_year,
    )
    all_results = []

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            eval_metric="logloss", random_state=42,
            n_estimators=300, learning_rate=0.05,
            max_depth=4, tree_method="hist", n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            random_state=42, n_estimators=300,
            learning_rate=0.05, verbosity=-1, n_jobs=-1
        ),
        "Stacking Ensemble": StackingClassifier(
            estimators=[
                ("Random Forest", RandomForestClassifier(
                    n_estimators=300, random_state=42, n_jobs=-1
                )),
                ("XGBoost", xgb.XGBClassifier(
                    eval_metric="logloss", random_state=42,
                    n_estimators=300, learning_rate=0.05,
                    max_depth=4, tree_method="hist", n_jobs=-1
                )),
                ("LightGBM", lgb.LGBMClassifier(
                    random_state=42, n_estimators=300,
                    learning_rate=0.05, verbosity=-1, n_jobs=-1
                )),
            ],
            final_estimator=LogisticRegression(
                solver="liblinear", max_iter=5000, random_state=42
            ),
            cv=5, n_jobs=1,
        ),
    }

    for horizon_name, (X, y, groups) in datasets_dict.items():
        print(f"\n{'=' * 70}")
        print(f"STARTING EXPERIMENT FOR HORIZON: {horizon_name}")
        print(f"{'=' * 70}")

        # Three levels of 5-fold CV sit on top of each other here: the outer split, the
        # inner split used for threshold tuning, and the Stacking Ensemble's own cv=5. With
        # too few positives an outer training fold is left with 1-2 of them, and splitting
        # that five ways leaves some inner folds with none — which StackingClassifier.fit()
        # cannot survive. Resampling does not create real positive examples, so skip any
        # horizon that cannot support the nested design.
        MIN_POSITIVES_FOR_NESTED_CV = 25   # 5 outer folds x ~5 positives per test fold
        n_positives = int(y.sum())
        if n_positives < MIN_POSITIVES_FOR_NESTED_CV:
            print(
                f"[!] Skipping {horizon_name}: only {n_positives} positive(s) in the full "
                f"horizon dataset, below the {MIN_POSITIVES_FOR_NESTED_CV} needed to support "
                f"the nested 5-fold CV. Report this horizon as excluded for insufficient "
                f"positive-class support rather than as a result."
            )
            continue

        # Group by company so every row of a firm lands in the same fold. A firm contributes
        # several near-identical yearly rows with correlated labels, so a row-wise split
        # would put its 2015 row in training and its 2016 row in test — group leakage that
        # inflates every reported metric.
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y, groups), start=1):
            print(f"\n[*] Fold {fold_num}/5 for {horizon_name}...")

            # Fail loudly if a company ever straddles the split.
            overlap = set(groups[train_idx]) & set(groups[test_idx])
            assert not overlap, (
                f"[!] Company overlap in fold {fold_num}: {len(overlap)} firms in both "
                f"train and test — grouping is not working."
            )

            # Group-aware folds are far less even than row-wise ones, and a test fold with
            # very few positives makes MCC noisy — so report the composition of each.
            print(
                f"    train: {len(train_idx)} rows / {len(set(groups[train_idx]))} firms | "
                f"test: {len(test_idx)} rows / {len(set(groups[test_idx]))} firms | "
                f"test positives: {int(y.iloc[test_idx].sum())}"
            )

            X_train = X.iloc[train_idx].copy()
            X_test  = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test  = y.iloc[test_idx].copy()

            groups_train = groups[train_idx].copy()

            for model_name, model in models.items():
                print(f"\n   -> Training {model_name}...")

                # The horizon-level guard above catches the common case, but a single
                # fold/model pair can still hit a degenerate CV split. A run of this length
                # should not lose the horizons it already computed over one bad fold, so log
                # the failure, skip that pair, and keep going.
                try:
                    pipeline = ImbPipeline([
                        ("imputer", SafeIterativeImputer(
                            max_iter=10, random_state=42, initial_strategy="median", skip_complete=True
                        )),
                        ("scaler", RobustScaler()),
                        ("smote", SafeSMOTE(random_state=42)),
                        ("classifier", clone(model))
                    ])

                    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

                    y_oof_prob = cross_val_predict(
                        pipeline, X_train, y_train,
                        groups=groups_train,
                        cv=inner_cv,
                        method="predict_proba",
                        n_jobs=CV_N_JOBS
                    )[:, 1]

                    best_threshold, validation_mcc = find_best_threshold(
                        y_train, y_oof_prob, metric="MCC"
                    )

                    pipeline.fit(X_train, y_train)
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                    fold_metrics = calculate_metrics_at_threshold(y_test, y_prob, best_threshold)
                except (ValueError, IndexError) as e:
                    print(
                        f"      [!] {model_name} failed on fold {fold_num} of {horizon_name} "
                        f"({e}) — skipping this fold/model combination."
                    )
                    continue

                all_results.append({
                    "Dataset":          f"Turkish_Audit_{min_year}_plus",
                    "Horizon":          horizon_name,
                    "Fold":             fold_num,
                    "Model":            model_name,
                    "Best Threshold":   best_threshold,
                    "Validation MCC":   validation_mcc,
                    **fold_metrics,
                })

                print(
                    f"      Threshold: {best_threshold:.2f} | "
                    f"Recall: {fold_metrics['Recall (Sensitivity)']:.4f} | "
                    f"PR-AUC: {fold_metrics['PR-AUC']:.4f} | "
                    f"FNR: {fold_metrics['FNR (False Neg Rate)']:.4f} | "
                    f"MCC: {fold_metrics['MCC']:.4f}"
                )

    df_fold_results = pd.DataFrame(all_results)

    if df_fold_results.empty:
        print("[!] No horizon produced any results. Check the positive counts above.")
        return pd.DataFrame(), df_fold_results

    metric_columns = [
        "Recall (Sensitivity)",
        "FNR (False Neg Rate)",
        "PR-AUC",
        "MCC",
    ]
    aggregation = {m: "mean" for m in metric_columns}
    aggregation.update({
        "Best Threshold": "mean",
        "Validation MCC": "mean",
        "TN": "sum", "FP": "sum", "FN": "sum", "TP": "sum",
    })

    df_results = df_fold_results.groupby(
        ["Dataset", "Horizon", "Model"], as_index=False
    ).agg(aggregation)

    # Fold-to-fold standard deviations. Group-aware folds vary more, so the paper reports
    # dispersion next to the means rather than means alone.
    df_std = df_fold_results.groupby(
        ["Dataset", "Horizon", "Model"], as_index=False
    )[metric_columns].std().rename(columns={m: f"{m} (SD)" for m in metric_columns})
    df_results = df_results.merge(df_std, on=["Dataset", "Horizon", "Model"], how="left")

    # Standard deviation of the SELECTED THRESHOLD across folds. This is the headline
    # quantity for the paper's stability finding, so it belongs in the summary table.
    df_thr_sd = (df_fold_results
                 .groupby(["Dataset", "Horizon", "Model"], as_index=False)["Best Threshold"]
                 .std().rename(columns={"Best Threshold": "Best Threshold (SD)"}))
    df_results = df_results.merge(df_thr_sd, on=["Dataset", "Horizon", "Model"], how="left")

    df_counts = (df_fold_results
                 .groupby(["Dataset", "Horizon", "Model"], as_index=False)
                 .size().rename(columns={"size": "Folds"}))
    df_results = df_results.merge(df_counts, on=["Dataset", "Horizon", "Model"], how="left")

    incomplete = df_results[df_results["Folds"] < 5]
    if len(incomplete):
        print("[!] Averaged over fewer than 5 folds — footnote these in the paper:")
        print(incomplete[["Horizon", "Model", "Folds"]].to_string(index=False))

    round_columns = (
        metric_columns
        + [f"{m} (SD)" for m in metric_columns]
        + ["Best Threshold", "Best Threshold (SD)", "Validation MCC"]
    )
    df_results[round_columns] = df_results[round_columns].round(4)
    df_results = df_results.sort_values(["Horizon", "Model"]).reset_index(drop=True)

    return df_results, df_fold_results


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train financial distress prediction models and save results/plots."
    )
    parser.add_argument(
        "--input-file",
        default=os.path.join("data", "financial_data.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where CSV outputs will be saved.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default="visualizations",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2015,
        help="Minimum year to keep from the dataset.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=3,
        help="Maximum prediction horizon to generate.",
    )
    return parser


def main():
    import sys
    is_jupyter = 'ipykernel' in sys.modules or 'google.colab' in sys.modules
    # NOTE: in a notebook, argparse is bypassed and the DEFAULTS above are used. To vary
    # min_year from a notebook, call run_experiment_per_year(...) directly instead of main().
    args = build_arg_parser().parse_args(args=[] if is_jupyter else None)

    ensure_output_dir(args.results_dir)
    ensure_output_dir(args.visualizations_dir)

    df_results, df_fold_results = run_experiment_per_year(
        args.input_file,
        max_horizon=args.max_horizon,
        min_year=args.min_year,
    )

    if df_results.empty:
        return

    print("\n\n========== FINAL AGGREGATED RESULTS ==========")
    print(df_results.to_string(index=False))

    results_summary_path = os.path.join(args.results_dir, "results_summary.csv")
    results_fold_path = os.path.join(args.results_dir, "results_per_fold.csv")
    results_thresholds_path = os.path.join(args.results_dir, "results_thresholds.csv")
    df_results.to_csv(results_summary_path, index=False)
    df_fold_results.to_csv(results_fold_path, index=False)

    df_thresholds = df_results[["Horizon", "Model", "Best Threshold",
                                "Best Threshold (SD)", "Validation MCC"]].copy()
    df_thresholds.to_csv(results_thresholds_path, index=False)

    print(f"\n[+] Results saved to:\n  - Summary: {results_summary_path}\n  - Folds: {results_fold_path}\n  - Thresholds: {results_thresholds_path}")

    # The paper figures take the per-fold table, not the aggregated one.
    save_publication_figures(df_fold_results, args.visualizations_dir)


# ========== 5. Run ==========
if __name__ == "__main__":
    main()