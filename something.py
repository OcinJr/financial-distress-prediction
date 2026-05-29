import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import arff
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
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


# ========== Load and combine all ARFF files from the given folder ==========
def load_and_combine_arff(folder_path):
    print(f"[*] Searching for .arff files in folder: {folder_path}")
    arff_files = glob.glob(os.path.join(folder_path, '*.arff'))

    if not arff_files:
        raise FileNotFoundError("No .arff files found. Please check your folder path.")

    print(f"[*] Found {len(arff_files)} files. Starting merge...")

    all_data = []
    for file in sorted(arff_files):
        data, meta = arff.loadarff(file)
        df_temp = pd.DataFrame(data)

        for col in df_temp.select_dtypes([object]).columns:
            df_temp[col] = df_temp[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        all_data.append(df_temp)

    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"[+] Merge complete! Data size: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns.\n")
    return df_combined


# ========== Remove highly correlated features in training data ==========
def remove_highly_correlated_features(X_train, X_test, threshold=0.95):
    print("[*] Checking Multicollinearity (Threshold: > 0.95)...")
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced = X_test.drop(columns=to_drop)

    print(f"[+] Found {len(to_drop)} redundant features. These features have been removed.")
    return X_train_reduced, X_test_reduced


# ========== Perform imputation, scaling, optional correlation filtering, and SMOTE per fold ==========
def preprocess_financial_split(
    X_train,
    X_test,
    y_train,
    use_correlation_filter=False
):
    print("[*] Missing Values Imputation (MICE)...")
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy='median',
        skip_complete=True
    )
    try:
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns
        )
    except Exception as e:
        print(f"[!] IterativeImputer failed ({e}). Fallback to median imputation...")
        fallback_imputer = SimpleImputer(strategy='median')
        X_train_imputed = pd.DataFrame(
            fallback_imputer.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_imputed = pd.DataFrame(
            fallback_imputer.transform(X_test),
            columns=X_test.columns
        )

    print("[*] Normalizing with Robust Scaler...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

    if use_correlation_filter:
        print("[*] Correlation filtering active. Running feature selection...")
        X_train_ready, X_test_ready = remove_highly_correlated_features(
            X_train_scaled,
            X_test_scaled,
            threshold=0.95
        )
    else:
        print("[*] Correlation filtering inactive. All scaled features are used.")
        X_train_ready = X_train_scaled
        X_test_ready = X_test_scaled

    print("[*] Handling Imbalanced Data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_ready, y_train)

    print("[+] Preprocessing Complete!\n")
    return X_train_resampled, X_test_ready, y_train_resampled


# ========== Find best probability threshold based on validation metric ==========
def find_best_threshold(y_true, y_prob, metric="MCC"):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == "MCC":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "F1-Score":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "Balanced Accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unrecognized metric threshold tuning: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)


# ========== Calculate all evaluation metrics using a specific probability threshold ==========
def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc,
        "PR-AUC": pr_auc,
        "Balanced Accuracy": balanced_acc,
        "MCC": mcc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

# ========== Run experiments per dataset with Stratified 5-Fold and threshold tuning ==========
def run_experiment_per_year(folder_path):
    arff_files = sorted(glob.glob(os.path.join(folder_path, '*.arff')))

    if not arff_files:
        raise FileNotFoundError(".arff files not found!")

    all_results = []

    horizon_map = {
        "1year.arff": "Bankruptcy after 5 years",
        "2year.arff": "Bankruptcy after 4 years",
        "3year.arff": "Bankruptcy after 3 years",
        "4year.arff": "Bankruptcy after 2 years",
        "5year.arff": "Bankruptcy after 1 year"
    }

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cpu",
            n_jobs=-1
        ),

        "LightGBM": lgb.LGBMClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            verbosity=-1,
            n_jobs=-1
        ),

        "Stacking Ensemble": StackingClassifier(
            estimators=[
                ("Random Forest", RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1
                )),
                ("XGBoost", xgb.XGBClassifier(
                    eval_metric='logloss',
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    device="cpu",
                    n_jobs=-1
                )),
                ("LightGBM", lgb.LGBMClassifier(
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    verbosity=-1,
                    n_jobs=-1
                ))
            ],
            final_estimator=LogisticRegression(
                solver='liblinear',
                max_iter=5000,
                random_state=42
            ),
            cv=5,
            n_jobs=1
        )
    }

    for file in arff_files:
        file_name = os.path.basename(file)
        horizon = horizon_map.get(file_name, "Unknown horizon")

        print(f"\n{'=' * 70}")
        print(f"STARTING EXPERIMENT FOR DATASET: {file_name}")
        print(f"HORIZON: {horizon}")
        print(f"{'=' * 70}")

        data, meta = arff.loadarff(file)
        df = pd.DataFrame(data)

        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            )

        print("[*] Converting '?' to Null...")
        df.replace('?', np.nan, inplace=True)

        X = df.drop('class', axis=1).astype(float)
        y = df['class'].astype(int)

        print("[*] Cleaning Infinity values...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        X = X.clip(lower=-100000, upper=100000)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n[*] Fold {fold_num}/5 for {file_name}...")
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            X_train_ready, X_test_ready, y_train_ready = preprocess_financial_split(
                X_train,
                X_test,
                y_train,
                use_correlation_filter=False
            )

            print(f"[*] Number of training data after SMOTE: {X_train_ready.shape[0]}")
            print(f"[*] Number of fold testing data: {X_test_ready.shape[0]}")

            for model_name, model in models.items():
                print(f"\n   -> Training {model_name}...")

                # --- Threshold tuning via OOF cross_val_predict (no leakage, single fit) ---
                print(f"      [*] Getting OOF probabilities for threshold tuning...")
                oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                y_oof_prob = cross_val_predict(
                    clone(model),
                    X_train_ready,
                    y_train_ready,
                    cv=oof_cv,
                    method="predict_proba",
                    n_jobs=-1
                )[:, 1]
                best_threshold, validation_mcc = find_best_threshold(
                    y_train_ready,
                    y_oof_prob,
                    metric="MCC"
                )

                # --- Final model: fit once on all training data ---
                clf = clone(model)
                clf.fit(X_train_ready, y_train_ready)

                y_prob = clf.predict_proba(X_test_ready)[:, 1]
                fold_metrics = calculate_metrics_at_threshold(
                    y_test,
                    y_prob,
                    best_threshold
                )

                all_results.append({
                    "Dataset": file_name,
                    "Horizon": horizon,
                    "Fold": fold_num,
                    "Model": model_name,
                    "Best Threshold": best_threshold,
                    "Validation MCC": validation_mcc,
                    **fold_metrics
                })

                print(
                    f"      Threshold: {best_threshold:.2f} | "
                    f"OOF MCC: {validation_mcc:.4f} | "
                    f"AUC-ROC: {fold_metrics['AUC-ROC']:.4f} | "
                    f"PR-AUC: {fold_metrics['PR-AUC']:.4f} | "
                    f"Recall: {fold_metrics['Recall (Sensitivity)']:.4f} | "
                    f"F1: {fold_metrics['F1-Score']:.4f} | "
                    f"MCC: {fold_metrics['MCC']:.4f}"
                )

    df_fold_results = pd.DataFrame(all_results)

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall (Sensitivity)",
        "F1-Score",
        "AUC-ROC",
        "PR-AUC",
        "Balanced Accuracy",
        "MCC"
    ]

    aggregation = {metric: "mean" for metric in metric_columns}
    aggregation.update({
        "Best Threshold": "mean",
        "Validation MCC": "mean",
        "TN": "sum",
        "FP": "sum",
        "FN": "sum",
        "TP": "sum"
    })

    df_results = df_fold_results.groupby(
        ["Dataset", "Horizon", "Model"],
        as_index=False
    ).agg(aggregation)

    df_results[metric_columns + ["Best Threshold", "Validation MCC"]] = (
        df_results[metric_columns + ["Best Threshold", "Validation MCC"]].round(4)
    )

    dataset_order = ["1year.arff", "2year.arff", "3year.arff", "4year.arff", "5year.arff"]
    df_results["Dataset"] = pd.Categorical(
        df_results["Dataset"],
        categories=dataset_order,
        ordered=True
    )

    df_results = df_results.sort_values(["Dataset", "Model"]).reset_index(drop=True)

    return df_results, df_fold_results


# ========== Style numeric tables to highlight the best values ==========
def style_metric_table(metric_table):
    return metric_table.style\
        .highlight_max(axis=1, color='#c6efce')\
        .format("{:.4f}")\
        .set_properties(**{'text-align': 'center', 'border': '1px solid black'})\
        .set_table_styles([
            dict(
                selector='th',
                props=[('text-align', 'center'), ('background-color', '#f2f2f2')]
            )
        ])


# ========== Style text tables for a neater output appearance ==========
def style_text_table(table):
    return table.style\
        .set_properties(**{'text-align': 'center', 'border': '1px solid black'})\
        .set_table_styles([
            dict(
                selector='th',
                props=[('text-align', 'center'), ('background-color', '#f2f2f2')]
            )
        ])


# ========== Create mean and standard deviation summary from each fold's results ==========
def build_mean_std_summary(df_fold_results, metrics):
    summary = df_fold_results.groupby(
        ["Dataset", "Horizon", "Model"],
        as_index=False
    )[metrics].agg(["mean", "std"])

    summary.columns = [
        " ".join(column).strip() if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    formatted_summary = summary[["Dataset", "Horizon", "Model"]].copy()
    for metric in metrics:
        mean_col = f"{metric} mean"
        std_col = f"{metric} std"
        formatted_summary[metric] = summary.apply(
            lambda row: f"{row[mean_col]:.4f} +/- {row[std_col]:.4f}",
            axis=1
        )

    return summary.round(4), formatted_summary


# ========== Determine the best model per dataset for each evaluation metric ==========
def build_best_model_summary(df_results, metrics):
    summary_rows = []

    for dataset, group in df_results.groupby("Dataset", observed=False):
        row = {"Dataset": dataset}

        for metric in metrics:
            best_idx = group[metric].idxmax()
            best_row = group.loc[best_idx]
            row[metric] = f"{best_row['Model']} ({best_row[metric]:.4f})"

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ========== Calculate model ranking based on important metrics for imbalanced data ==========
def build_overall_ranking(df_results):
    ranking_metrics = [
        "PR-AUC",
        "MCC",
        "F1-Score",
        "Recall (Sensitivity)",
        "Balanced Accuracy"
    ]

    ranked = df_results.copy()

    for metric in ranking_metrics:
        ranked[f"{metric} Rank"] = ranked.groupby("Dataset")[metric] \
            .rank(ascending=False, method="min")

    rank_columns = [f"{metric} Rank" for metric in ranking_metrics]
    ranked["Average Rank"] = ranked[rank_columns].mean(axis=1)

    overall_ranking = ranked.groupby("Model", as_index=False)["Average Rank"] \
        .mean() \
        .sort_values("Average Rank") \
        .reset_index(drop=True)

    overall_ranking["Overall Position"] = overall_ranking["Average Rank"] \
        .rank(ascending=True, method="min") \
        .astype(int)

    return ranked, overall_ranking


# ========== Create confusion matrix summary and important error rates ==========
def build_confusion_summary(df_results):
    summary = df_results[[
        "Dataset",
        "Horizon",
        "Model",
        "TN",
        "FP",
        "FN",
        "TP"
    ]].copy()

    actual_negative = summary["TN"] + summary["FP"]
    actual_positive = summary["FN"] + summary["TP"]
    predicted_negative = summary["TN"] + summary["FN"]
    predicted_positive = summary["FP"] + summary["TP"]

    summary["Actual Non-Bankrupt"] = actual_negative
    summary["Actual Bankrupt"] = actual_positive
    summary["Predicted Non-Bankrupt"] = predicted_negative
    summary["Predicted Bankrupt"] = predicted_positive
    summary["False Negative Rate"] = np.where(
        actual_positive == 0,
        0,
        summary["FN"] / actual_positive
    )
    summary["False Positive Rate"] = np.where(
        actual_negative == 0,
        0,
        summary["FP"] / actual_negative
    )
    summary["Specificity"] = np.where(
        actual_negative == 0,
        0,
        summary["TN"] / actual_negative
    )

    rate_columns = [
        "False Negative Rate",
        "False Positive Rate",
        "Specificity"
    ]
    summary[rate_columns] = summary[rate_columns].round(4)

    return summary


# ========== Determine the model with the lowest False Negative Rate on each dataset ==========
def build_lowest_fnr_summary(confusion_summary):
    summary_rows = []

    for dataset, group in confusion_summary.groupby("Dataset", observed=False):
        best_idx = group["False Negative Rate"].idxmin()
        best_row = group.loc[best_idx]
        summary_rows.append({
            "Dataset": dataset,
            "Model": best_row["Model"],
            "False Negative Rate": best_row["False Negative Rate"],
            "FN": best_row["FN"],
            "Actual Bankrupt": best_row["Actual Bankrupt"]
        })

    return pd.DataFrame(summary_rows)


# ========== Convert text to safe filename ==========
def make_safe_filename(text):
    safe_text = text.lower()
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("(", ""),
        (")", ""),
        ("-", "_")
    ]:
        safe_text = safe_text.replace(old, new)

    return safe_text


# ========== Create and save experimental result visualization graphs ==========
def save_visualizations(
    df_results,
    confusion_summary,
    overall_ranking=None,
    output_dir="visualizations"
):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    saved_paths = []
    key_metrics = [
        "AUC-ROC",
        "PR-AUC",
        "F1-Score",
        "Recall (Sensitivity)",
        "MCC",
        "Balanced Accuracy"
    ]

    for metric in key_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=df_results,
            x="Dataset",
            y=metric,
            hue="Model",
            ax=ax
        )
        ax.set_title(f"{metric} by Dataset and Model")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        output_path = os.path.join(
            output_dir,
            f"{make_safe_filename(metric)}_by_dataset_model.png"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df_results,
        x="Dataset",
        y="Best Threshold",
        hue="Model",
        marker="o",
        ax=ax
    )
    ax.set_title("Best Threshold by Dataset and Model")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Best Threshold")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    threshold_path = os.path.join(output_dir, "best_threshold_by_dataset_model.png")
    fig.tight_layout()
    fig.savefig(threshold_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(threshold_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=confusion_summary,
        x="Dataset",
        y="False Negative Rate",
        hue="Model",
        ax=ax
    )
    ax.set_title("False Negative Rate by Dataset and Model")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("False Negative Rate")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fnr_path = os.path.join(output_dir, "false_negative_rate_by_dataset_model.png")
    fig.tight_layout()
    fig.savefig(fnr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(fnr_path)

    if overall_ranking is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=overall_ranking.sort_values("Average Rank"),
            x="Model",
            y="Average Rank",
            ax=ax
        )
        ax.set_title("Overall Model Ranking")
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Rank")
        ax.tick_params(axis="x", rotation=20)
        ranking_path = os.path.join(output_dir, "overall_model_ranking.png")
        fig.tight_layout()
        fig.savefig(ranking_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(ranking_path)

    return saved_paths


# ========== Export all experiment result tables to a multi-sheet Excel file ==========
def export_results_to_excel(output_path, sheets):
    try:
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, dataframe in sheets.items():
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[+] Excel file successfully saved: {output_path}")
    except ImportError as e:
        print(f"[!] Excel export skipped because dependency is not available: {e}")
    except ValueError as e:
        print(f"[!] Excel export skipped: {e}")

if __name__ == "__main__":
    folder_path = 'data'

    try:
        results_table, fold_results = run_experiment_per_year(folder_path)

        print("\n\n🏆 STRATIFIED 5-FOLD EXPERIMENT COMPARISON RESULTS TABLE 🏆")

        from IPython.display import display
        metrics_to_display = [
            "Accuracy",
            "Precision",
            "Recall (Sensitivity)",
            "F1-Score",
            "AUC-ROC",
            "PR-AUC",
            "Balanced Accuracy",
            "MCC"
        ]

        results_mean_std_numeric, results_mean_std_formatted = build_mean_std_summary(
            fold_results,
            metrics_to_display
        )

        print("\n\n=== MEAN +/- STD RESULTS TABLE PER METRIC ===")
        for metric in metrics_to_display:
            print(f"\n=== {metric.upper()} ===")
            metric_table = results_mean_std_formatted.pivot(
                index='Dataset',
                columns='Model',
                values=metric
            )
            display(style_text_table(metric_table))

        print("\n=== AVERAGE BEST THRESHOLD PER DATASET AND MODEL ===")
        threshold_table = results_table.pivot(
            index='Dataset',
            columns='Model',
            values='Best Threshold'
        )
        display(style_metric_table(threshold_table))

        print("\n=== BEST MODEL PER DATASET FOR EACH METRIC ===")
        best_model_summary = build_best_model_summary(results_table, metrics_to_display)
        display(best_model_summary)

        print("\n=== OVERALL MODEL RANKING BASED ON IMBALANCED METRICS ===")
        ranked_results, overall_ranking = build_overall_ranking(results_table)
        display(style_text_table(overall_ranking))

        print("\n=== CONFUSION MATRIX SUMMARY AND FALSE NEGATIVE RATE ===")
        confusion_summary = build_confusion_summary(results_table)
        display(style_text_table(confusion_summary))

        print("\n=== MODEL WITH LOWEST FALSE NEGATIVE RATE PER DATASET ===")
        lowest_fnr_summary = build_lowest_fnr_summary(confusion_summary)
        display(style_text_table(lowest_fnr_summary))

        visualization_paths = save_visualizations(
            results_table,
            confusion_summary,
            overall_ranking
        )
        visualization_index = pd.DataFrame({
            "Visualization File": visualization_paths
        })

        results_table.to_csv("bankruptcy_model_comparison_results.csv", index=False)
        fold_results.to_csv("bankruptcy_model_comparison_fold_results.csv", index=False)
        results_mean_std_numeric.to_csv("bankruptcy_model_comparison_mean_std_numeric.csv", index=False)
        results_mean_std_formatted.to_csv("bankruptcy_model_comparison_mean_std_formatted.csv", index=False)
        best_model_summary.to_csv("bankruptcy_model_best_summary.csv", index=False)
        ranked_results.to_csv("bankruptcy_model_ranked_results.csv", index=False)
        overall_ranking.to_csv("bankruptcy_model_overall_ranking.csv", index=False)
        confusion_summary.to_csv("bankruptcy_model_confusion_summary.csv", index=False)
        lowest_fnr_summary.to_csv("bankruptcy_model_lowest_fnr_summary.csv", index=False)
        visualization_index.to_csv("bankruptcy_model_visualization_files.csv", index=False)
        export_results_to_excel(
            "bankruptcy_model_comparison_results.xlsx",
            {
                "Mean CV Results": results_table,
                "Fold Results": fold_results,
                "Mean Std Numeric": results_mean_std_numeric,
                "Mean Std Formatted": results_mean_std_formatted,
                "Best Model Summary": best_model_summary,
                "Ranked Results": ranked_results,
                "Overall Ranking": overall_ranking,
                "Confusion Summary": confusion_summary,
                "Lowest FNR Summary": lowest_fnr_summary,
                "Visualization Files": visualization_index
            }
        )
        print(f"\n[+] {len(visualization_paths)} graphs have been saved in the visualizations folder.")
        print(
            "\n[+] Aggregate results, per-fold results, mean/std, confusion matrix, "
            "visualizations, and best model summary have been saved."
        )

    except Exception as e:
        print(f"\n[!] AN ERROR OCCURRED: {e}")
