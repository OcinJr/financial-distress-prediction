import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.base import clone
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
import xgboost as xgb
import lightgbm as lgb


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
def process_turkish_panel_data(file_path, max_horizon=2, min_year=2021):
    print(f"[*] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Standardize missing values
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

    # Keep data from min_year onward
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[YEAR_COL] >= min_year].copy()
    print(f"[*] Rows after {min_year}+ filter: {len(df)}")

    # Translate column names to English
    df.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
    print(f"[*] Columns translated to English.")

    # Translate target label values to English
    df["Audit Opinion"] = df["Audit Opinion"].map(TARGET_LABEL_TRANSLATIONS)

    # Binarize Audit Opinion: Unqualified = 0 (healthy), everything else = 1 (distressed)
    healthy_labels = ["Unqualified"]
    df["class"] = np.where(df["Audit Opinion"].isin(healthy_labels), 0, 1)

    print(f"[*] Class distribution:\n{df['class'].value_counts().rename({0: 'Healthy (0)', 1: 'Distressed (1)'}).to_string()}")

    # Sort chronologically
    df = df.sort_values(by=["Company Name", "Year"]).reset_index(drop=True)

    # Allow up to 3-year horizon for the 2021+ annual window.
    if max_horizon > 3:
        print(f"[!] Warning: max_horizon={max_horizon} is too high for the current window. Capping at 3.")
        max_horizon = 3

    horizon_datasets = {}

    for horizon in range(1, max_horizon + 1):
        print(f"\n[*] Generating {horizon}-Year Horizon dataset...")
        horizon_df = df.copy()

        # Shift the target UP to predict future distress
        target_col_name = f"Target_{horizon}Y"
        horizon_df[target_col_name] = (
            horizon_df.groupby("Company Name")["class"].shift(-horizon)
        )
        horizon_df = horizon_df.dropna(subset=[target_col_name])

        # Drop features that could cause target leakage
        columns_to_drop = [
            "Company Name", "Company Code", "Period",
            "Year", "Audit Opinion", "class", target_col_name
        ]
        
        object_cols = horizon_df.select_dtypes(include=["object", "string"]).columns
        columns_to_drop.extend([c for c in object_cols if c not in columns_to_drop])

        X = horizon_df.drop(columns=columns_to_drop, errors="ignore")
        y = horizon_df[target_col_name].astype(int)

        X = X.apply(pd.to_numeric, errors="coerce")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.clip(lower=-100000, upper=100000)

        print(f"[+] {horizon}-Year Horizon — X shape: {X.shape}, "
              f"Class balance: {y.value_counts().to_dict()}")

        horizon_datasets[f"{horizon}-Year Horizon"] = (X, y)

    return horizon_datasets


# ========== 2. Remove highly correlated features ==========
def remove_highly_correlated_features(X_train, X_test, threshold=0.95):
    print("[*] Checking Multicollinearity (Threshold: > 0.95)...")
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced  = X_test.drop(columns=to_drop)
    print(f"[+] Dropped {len(to_drop)} redundant features.")
    return X_train_reduced, X_test_reduced


# ========== 3. Preprocessing per fold ==========
def preprocess_financial_split(X_train, X_test, y_train, use_correlation_filter=False):
    print("[*] Missing Values Imputation (MICE)...")
    imputer = IterativeImputer(
        max_iter=10, random_state=42,
        initial_strategy="median", skip_complete=True
    )
    try:
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train), columns=X_train.columns
        )
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test), columns=X_test.columns
        )
    except Exception as e:
        print(f"[!] IterativeImputer failed ({e}). Falling back to median imputation...")
        fallback = SimpleImputer(strategy="median")
        X_train_imputed = pd.DataFrame(
            fallback.fit_transform(X_train), columns=X_train.columns
        )
        X_test_imputed = pd.DataFrame(
            fallback.transform(X_test), columns=X_test.columns
        )

    print("[*] Normalizing with Robust Scaler...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed), columns=X_test.columns
    )

    if use_correlation_filter:
        print("[*] Correlation filtering active...")
        X_train_scaled, X_test_scaled = remove_highly_correlated_features(
            X_train_scaled, X_test_scaled, threshold=0.95
        )

    print("[*] Handling Class Imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("[+] Preprocessing complete.\n")
    return X_train_resampled, X_test_scaled, y_train_resampled


# ========== 4. Threshold tuning & Metrics ==========
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
    
    # Calculate False Negative Rate safely
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "Recall (Sensitivity)":  recall_score(y_true, y_pred, zero_division=0),
        "FNR (False Neg Rate)":  fnr,
        "PR-AUC":                average_precision_score(y_true, y_prob),
        "MCC":                   matthews_corrcoef(y_true, y_pred),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def make_safe_filename(text):
    safe_text = str(text).lower()
    for old, new in [(" ", "_"), ("/", "_"), ("(", ""), (")", ""), ("-", "_")]:
        safe_text = safe_text.replace(old, new)
    return safe_text


def save_performance_visualization(df_results, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = df_results.copy()
    horizon_order = sorted(
        plot_df["Horizon"].dropna().unique(),
        key=lambda x: int(str(x).split("-")[0])
    )
    plot_df["Horizon"] = pd.Categorical(
        plot_df["Horizon"],
        categories=horizon_order,
        ordered=True
    )

    saved_paths = []
    
    # 1. Save Recall and PR-AUC separately
    separate_metrics = ["Recall (Sensitivity)", "PR-AUC"]
    for metric in separate_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=plot_df, x="Model", y=metric, hue="Horizon", ax=ax
        )
        ax.set_title(f"Model Comparison - {metric}", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Horizon", loc="best")
        fig.tight_layout()

        safe_name = make_safe_filename(metric)
        output_path = os.path.join(output_dir, f"{safe_name}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"[+] Performance graph saved to: {output_path}")
        saved_paths.append(output_path)

    # 2. Save FNR and MCC combined (one on top of the other)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    
    # Top subplot: FNR
    sns.barplot(
        data=plot_df, x="Model", y="FNR (False Neg Rate)", hue="Horizon", ax=ax1
    )
    ax1.set_title("Model Comparison - FNR (False Neg Rate)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("FNR")
    ax1.legend(title="Horizon", loc="best")
    
    # Bottom subplot: MCC
    sns.barplot(
        data=plot_df, x="Model", y="MCC", hue="Horizon", ax=ax2
    )
    ax2.set_title("Model Comparison - MCC", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MCC")
    ax2.tick_params(axis="x", rotation=20)
    ax2.legend(title="Horizon", loc="best")
    
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, "fnr_mcc_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[+] Combined FNR and MCC graph saved to: {output_path}")
    saved_paths.append(output_path)
        
    return saved_paths


def save_fnr_visualization(df_results, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = df_results.copy()
    horizon_order = sorted(
        plot_df["Horizon"].dropna().unique(),
        key=lambda x: int(str(x).split("-")[0])
    )
    plot_df["Horizon"] = pd.Categorical(
        plot_df["Horizon"],
        categories=horizon_order,
        ordered=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="Model",
        y="FNR (False Neg Rate)",
        hue="Horizon",
        ax=ax
    )
    ax.set_title("False Negative Rate by Model and Horizon")
    ax.set_xlabel("")
    ax.set_ylabel("FNR")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Horizon", loc="best")

    fig.tight_layout()
    output_path = os.path.join(output_dir, "false_negative_rate_by_model_and_horizon.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] FNR graph saved to: {output_path}")
    return output_path


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ========== 5. Main Experiment Loop ==========
def run_experiment_per_year(file_path, max_horizon=3, min_year=2021):
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

    for horizon_name, (X, y) in datasets_dict.items():
        print(f"\n{'=' * 70}")
        print(f"STARTING EXPERIMENT FOR HORIZON: {horizon_name}")
        print(f"{'=' * 70}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n[*] Fold {fold_num}/5 for {horizon_name}...")

            X_train = X.iloc[train_idx].copy()
            X_test  = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test  = y.iloc[test_idx].copy()

            X_train_ready, X_test_ready, y_train_ready = preprocess_financial_split(
                X_train, X_test, y_train, use_correlation_filter=False
            )

            for model_name, model in models.items():
                print(f"\n   -> Training {model_name}...")
                oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                y_oof_prob = cross_val_predict(
                    clone(model), X_train_ready, y_train_ready,
                    cv=oof_cv, method="predict_proba", n_jobs=-1
                )[:, 1]

                best_threshold, validation_mcc = find_best_threshold(
                    y_train_ready, y_oof_prob, metric="MCC"
                )

                clf = clone(model)
                clf.fit(X_train_ready, y_train_ready)
                y_prob = clf.predict_proba(X_test_ready)[:, 1]
                fold_metrics = calculate_metrics_at_threshold(y_test, y_prob, best_threshold)

                all_results.append({
                    "Dataset":          "Turkish_Audit_2021_2024",
                    "Horizon":          horizon_name,
                    "Fold":             fold_num,
                    "Model":            model_name,
                    "Best Threshold":   best_threshold,
                    "Validation MCC":   validation_mcc,
                    **fold_metrics,
                })

                # Print fold results to console
                print(
                    f"      Threshold: {best_threshold:.2f} | "
                    f"Recall: {fold_metrics['Recall (Sensitivity)']:.4f} | "
                    f"PR-AUC: {fold_metrics['PR-AUC']:.4f} | "
                    f"FNR: {fold_metrics['FNR (False Neg Rate)']:.4f} | "
                    f"MCC: {fold_metrics['MCC']:.4f}"
                )

    df_fold_results = pd.DataFrame(all_results)

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

    df_results[metric_columns + ["Best Threshold", "Validation MCC"]] = (
        df_results[metric_columns + ["Best Threshold", "Validation MCC"]].round(4)
    )
    df_results = df_results.sort_values(["Horizon", "Model"]).reset_index(drop=True)

    return df_results, df_fold_results


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train bankruptcy prediction models and save results/plots."
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
        default=2021,
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
    args = build_arg_parser().parse_args(args=[] if is_jupyter else None)

    ensure_output_dir(args.results_dir)
    ensure_output_dir(args.visualizations_dir)

    df_results, df_fold_results = run_experiment_per_year(
        args.input_file,
        max_horizon=args.max_horizon,
        min_year=args.min_year,
    )

    print("\n\n========== FINAL AGGREGATED RESULTS ==========")
    print(df_results.to_string(index=False))

    results_summary_path = os.path.join(args.results_dir, "results_summary.csv")
    results_fold_path = os.path.join(args.results_dir, "results_per_fold.csv")
    results_thresholds_path = os.path.join(args.results_dir, "results_thresholds.csv")
    df_results.to_csv(results_summary_path, index=False)
    df_fold_results.to_csv(results_fold_path, index=False)
    
    # Save thresholds separately
    df_thresholds = df_results[["Horizon", "Model", "Best Threshold", "Validation MCC"]].copy()
    df_thresholds.to_csv(results_thresholds_path, index=False)
    
    print(f"\n[+] Results saved to:\n  - Summary: {results_summary_path}\n  - Folds: {results_fold_path}\n  - Thresholds: {results_thresholds_path}")

    save_performance_visualization(df_results, args.visualizations_dir)
    save_fnr_visualization(df_results, args.visualizations_dir)


# ========== 6. Run ==========
if __name__ == "__main__":
    main()
