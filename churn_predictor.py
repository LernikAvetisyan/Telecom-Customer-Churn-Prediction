# final churn_predictor.py
#
# Telco Customer Churn: (Based on our Assignments 1–5)
#
# This script consolidates:
#   Assignments 1–4: data cleaning, outlier handling (IQR), discretization, normalization (preview),
#           feature creation, feature selection + one-hot encoding (32 final features)
#   Assignment 5: modeling (LogReg + RF + HGB), stratified CV, OOF threshold selection (≥0.80 recall),
#        test evaluation, ranking, and deployment artifacts.
#
# Success criteria (from team reports): hit ≥ 80% recall on churners, and aim for accuracy ≥ 70%
# while keeping precision/business cost reasonable. (We pick the smallest threshold meeting
# recall target, then prefer higher F1/PR-AUC when ranking.) 
#
#
# We will run this by using the code below:
#   python churn_predictor.py

from __future__ import annotations
import os, json
from typing import Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score
)
import joblib

# Global config / constants

RANDOM_STATE = 42
MIN_RECALL_TARGET = 0.80

RAW_CSV = "data/Telco-Customer-Churn.csv"
A4_MODEL_TABLE = "data/telco_preprocessed_for_model.csv"
A4_Z_PREVIEW = "data/telco_preprocessed_with_normalized.csv"
FEATURE_NAMES_TXT = "data/feature_names_after_encoding.txt"
FEATURE_CARD_JSON = "data/feature_card.json"

ARTIFACT_DIR = "artifacts"

# A1–A4: Data Preparation (CRISP-DM: Data Preparation stage)

def load_and_basic_clean(csv_path: str) -> pd.DataFrame:
    """
    Assignments 1/3: Load raw CSV and fix known data issues

    Decisions and actions we made:
      1) Convert 'TotalCharges' to numeric, blank strings → NaN (coerce)
      2) Impute missing TotalCharges as MonthlyCharges * tenure (keeps rows)
      3) Map label 'Churn' to numeric (Yes→1, No→0)

    Returns a minimally cleaned DataFrame preserving original fields.
    """
    df = pd.read_csv(csv_path)
    # TotalCharges: blanks → NaN → numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask_nan_tc = df["TotalCharges"].isna()
    df.loc[mask_nan_tc, "TotalCharges"] = (
        df.loc[mask_nan_tc, "MonthlyCharges"] * df.loc[mask_nan_tc, "tenure"]
    )
    # Label map
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df


def fill_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assignment 3: Defensive null handling beyond TotalCharges imputation

    Decisions & actions:
      1) For object (categorical) columns: fill with mode
      2) For numeric columns: fill with median
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: Iterable[str], k: float = 1.5) -> pd.DataFrame:
    """
    Assignments 3/4: Outlier handling using the 1.5*IQR rule on selected numeric columns

    Decisions & actions:
      1) We clip extreme values of MonthlyCharges and TotalCharges to reduce undue influence
      2) We record how many values would have been clipped (often 0 in this dataset)

    Returns a DataFrame with capped columns, and prints counts for transparency
    """
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        before = ((df[col] < low) | (df[col] > high)).sum()
        df[col] = df[col].clip(lower=low, upper=high)
        print(f"[IQR cap] {col}: clipped {before} outliers (low={low:.3f}, high={high:.3f})")
    return df


def discretize_tenure_and_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assignment 4: Discretization for interpretability and linear-model stability

    Decisions & actions:
      1) tenure_group ∈ {up_to_1yr, 1_to_2yrs, 2_to_4yrs, 4_to_6yrs}
      2) monthly_tier ∈ {low, medium, high} using 33%/66% quantiles of MonthlyCharges
    """
    df = df.copy()
    # Tenure bins
    tenure_bins = [-0.1, 12, 24, 48, 72]
    tenure_labels = ["up_to_1yr", "1_to_2yrs", "2_to_4yrs", "4_to_6yrs"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)

    # Monthly charges tiers by quantiles
    q33, q66 = df["MonthlyCharges"].quantile([0.33, 0.66])
    df["monthly_tier"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "medium", "high"]
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assignment 4: Simple, business-meaningful features.

    New features:
      1) avg_spend_per_month = TotalCharges / max(tenure, 1)
      2) is_long_contract    = 1 if Contract ∈ {One year, Two year}
      3) is_electronic_check = 1 if PaymentMethod == 'Electronic check'
      4) num_services        = count of 'Yes' across internet add-ons
                              after mapping 'No internet service' → 'No'
    """
    df = df.copy()

    # Avoid divide-by-zero for new customers
    denom = df["tenure"].replace(0, 1)
    df["avg_spend_per_month"] = df["TotalCharges"] / denom

    df["is_long_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    svc_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for c in svc_cols:
        if c in df.columns:
            df[c] = df[c].replace({"No internet service": "No"})
    df["num_services"] = (df[svc_cols] == "Yes").sum(axis=1)
    return df


def normalization_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assignment 4: Normalization preview (z-scores) for documentation ONLY

    Decisions & actions:
      1) Add *_z columns for tenure, MonthlyCharges, avg_spend_per_month
      2) We DO NOT use these columns for training (to avoid leakage), modeling uses a
        scaler inside the CV pipeline for LR only
    """
    df = df.copy()
    for base in ["tenure", "MonthlyCharges", "avg_spend_per_month"]:
        mu = df[base].mean()
        sd = df[base].std(ddof=0) or 1.0
        df[f"{base}_z"] = (df[base] - mu) / sd
    return df


def select_and_encode(df: pd.DataFrame, drop_redundant: bool = True) -> pd.DataFrame:
    """
    Assignment 4: Feature selection + encoding → final modeling table

    Decisions & actions:
      1) Drop 'customerID' (identifier)
      2) Drop 'TotalCharges' (≈ tenure × MonthlyCharges) to reduce redundancy,
        keep engineered 'avg_spend_per_month' instead.
      3) One-hot encode categoricals with drop_first=True (avoid dummy trap)
    """
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if drop_redundant and "TotalCharges" in df.columns:
        df = df.drop(columns=["TotalCharges"])
    df = pd.get_dummies(df, drop_first=True)
    return df


def save_a4_artifacts(df_encoded: pd.DataFrame, df_with_z: pd.DataFrame) -> None:
    """
    Assignment 4: Persist modeling table, preview table, and feature metadata
    """
    os.makedirs(os.path.dirname(A4_MODEL_TABLE), exist_ok=True)
    # Modeling table (no *_z)
    df_encoded.to_csv(A4_MODEL_TABLE, index=False)
    # Preview (with *_z)
    df_with_z.to_csv(A4_Z_PREVIEW, index=False)
    # Feature names (X only)
    X_cols = [c for c in df_encoded.columns if c != "Churn"]
    with open(FEATURE_NAMES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(X_cols))
    # Feature card
    feature_card = {
        "rows": int(df_encoded.shape[0]),
        "cols_total": int(df_encoded.shape[1]),
        "cols_X": int(len(X_cols)),
        "cols_y": 1,
        "target": "Churn"
    }
    with open(FEATURE_CARD_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_card, f, indent=2)
    print(f"[A4] Saved: {A4_MODEL_TABLE}, {A4_Z_PREVIEW}, {FEATURE_NAMES_TXT}, {FEATURE_CARD_JSON}")



# Assignment 5: Modeling, Evaluation, Deployment
def build_models() -> Dict[str, object]:
    """
    Assignment 5: Instantiate three models:

      A) Logistic Regression with StandardScaler and class_weight='balanced'
      B) Random Forest with class_weight='balanced'
      C) HistGradientBoosting (boosted trees)
    """
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=0.5,
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,          # can try 12 if needed
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,             # alternatively use max_leaf_nodes=31
        max_iter=300,
        random_state=RANDOM_STATE
    )
    return {"LR": lr, "RF": rf, "HGB": hgb}


def evaluate_cv(model, X, y, cv) -> Dict[str, Tuple[float, float]]:
    """
    5-fold stratified CV with primary=recall. Secondaries=F1, ROC-AUC, PR-AUC
    Returns a dict: {metric: (mean, std)} using test-fold scores only
    """
    scoring = {
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cvres = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )
    return {m: (cvres[f"test_{m}"].mean(), cvres[f"test_{m}"].std()) for m in scoring}


"""
    Helper to see five raw fold scores per mod
    """

def evaluate_cv_raw(model, X, y, cv) -> pd.DataFrame:
    """Return per-fold CV scores (test-fold only) for Recall, F1, ROC-AUC, PR-AUC"""
    scoring = {"recall": "recall", "f1": "f1", "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    cvres = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    n_folds = cv.get_n_splits()
    return pd.DataFrame({
        "fold": np.arange(1, n_folds + 1, dtype=int),
        "recall": cvres["test_recall"],
        "f1": cvres["test_f1"],
        "roc_auc": cvres["test_roc_auc"],
        "pr_auc": cvres["test_pr_auc"],
    })



def oof_probs(model, X, y, cv) -> np.ndarray:
    """
    Assignment 5: Out-of-fold probabilities (positive class) for threshold selection
    without peeking at the test set.
    """
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def threshold_sweep(y_true, probas, grid=None, min_recall=MIN_RECALL_TARGET) -> Tuple[pd.DataFrame, float]:
    """
    Assignment 5: Build recall/precision/F1 across thresholds, choose best F1 among those
    that meet min_recall (or the highest-recall threshold if none meet)
    """
    if grid is None:
        grid = np.linspace(0.10, 0.90, 17)  # 0.10, 0.15, ..., 0.90
    rows = []
    for thr in grid:
        y_pred = (probas >= thr).astype(int)
        rows.append({
            "threshold": float(thr),
            "recall": float(recall_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred))
        })
    df_thr = pd.DataFrame(rows).sort_values("threshold")
    feasible = df_thr[df_thr["recall"] >= min_recall]
    if feasible.empty:
        best_thr = df_thr.sort_values(["recall", "f1"], ascending=[False, False]).iloc[0]["threshold"]
    else:
        best_thr = feasible.sort_values("f1", ascending=False).iloc[0]["threshold"]
    return df_thr, float(best_thr)


def fit_select_threshold_and_test(model, X_train, y_train, X_test, y_test, cv, name: str) -> Dict:
    """
    Assignment 5: CV summary → OOF threshold → fit on full train → test evaluation at chosen threshold.
    Returns a dict with all metrics, confusion matrix, chosen threshold, and threshold sweep.
    """
    cv_summary = evaluate_cv(model, X_train, y_train, cv=cv)
    oof = oof_probs(model, X_train, y_train, cv=cv)
    sweep, best_thr = threshold_sweep(y_train, oof, min_recall=MIN_RECALL_TARGET)

    # Fit and evaluate on test
    model.fit(X_train, y_train)
    probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_t = (probs_test >= best_thr).astype(int)

    res = {
        "model": name,
        "threshold": best_thr,
        "recall": float(recall_score(y_test, y_pred_t)),
        "precision": float(precision_score(y_test, y_pred_t, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_t)),
        "accuracy": float(accuracy_score(y_test, y_pred_t)),
        "roc_auc": float(roc_auc_score(y_test, probs_test)),
        "pr_auc": float(average_precision_score(y_test, probs_test)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_t).tolist(),  # [[TN, FP],[FN, TP]]
        "cv_summary": {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in cv_summary.items()},
        "threshold_sweep": sweep.to_dict(orient="records")
    }
    return res


def rank_results(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Assignment 5: Rank models by Recall (primary), then F1, then PR-AUC.
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "threshold": r["threshold"],
            "recall": r["recall"],
            "precision": r["precision"],
            "f1": r["f1"],
            "accuracy": r["accuracy"],
            "roc_auc": r["roc_auc"],
            "pr_auc": r["pr_auc"]
        })
    df = pd.DataFrame(rows).sort_values(by=["recall", "f1", "pr_auc"], ascending=[False, False, False]).reset_index(drop=True)
    return df

#   Deployment helpers (inference)
def save_model_artifacts(model_obj, model_name, threshold, feature_names, out_dir=ARTIFACT_DIR):
    """
    Persist the trained model and the exact inference contract

    Writes:
      - {out_dir}/model.pkl
      - {out_dir}/inference_config.json  (keys: model, threshold, feature_names)

    Parameters
    ----------
    model_obj: sklearn estimator (e.g., Pipeline for LR)
    model_name: str
        Short identifier for the model (e.g., "LR", "RF", "HGB").
    threshold: float
        Operating probability cutoff chosen from OOF CV (e.g., 0.45).
    feature_names: Iterable[str]
        Exact training-time feature order expected at inference.
    out_dir: str
        Directory to store artifacts (default: ARTIFACT_DIR).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save sklearn estimator
    joblib.dump(model_obj, os.path.join(out_dir, "model.pkl"))

    # Save inference contract
    cfg = {
        "model": str(model_name),
        "threshold": float(threshold),
        "feature_names": list(feature_names),
    }
    with open(os.path.join(out_dir, "inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_artifacts(art_dir=ARTIFACT_DIR):
    """
    Load the deployed model and contract.
    Returns: model, feature_names, threshold
    """
    model = joblib.load(os.path.join(art_dir, "model.pkl"))
    with open(os.path.join(art_dir, "inference_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_names = cfg["feature_names"]
    threshold = float(cfg["threshold"])
    return model, feature_names, threshold

def predict_from_features(df_features: pd.DataFrame,
                          model, feature_names, threshold: float) -> pd.DataFrame:
    """
    Score already-preprocessed rows (Assignment 4 rules).
    - Enforces exact training-time column order.
    - Fills missing one-hot columns with 0, drops extras.
    Returns: churn_prob (float), churn_pred (0/1)
    """
    X_inf = df_features.reindex(columns=feature_names, fill_value=0)
    probs = model.predict_proba(X_inf)[:, 1]
    yhat = (probs >= threshold).astype(int)
    return pd.DataFrame({"churn_prob": probs, "churn_pred": yhat})


def export_churn_predictions(
    model, 
    threshold: float, 
    df_encoded: pd.DataFrame, 
    out_dir: str = "Example", 
    n_rows: str | int = "10"
) -> None:
    """
    Export predicted churn customers using the given model and threshold.
    - Uses ALL rows from the encoded modeling table (X = 32 features).
    - Adds two columns: predicted_churn (0/1) and churn_probability (float).
    - Saves to Example/churn_predictions_{tag}.csv and .xlsx
      where {tag} is 'topK' for a number, or 'full' for FULL.

    Parameters
    ----------
    model: fitted sklearn estimator with predict_proba
    threshold: float
        Probability cutoff for classifying churn.
    df_encoded: DataFrame
        The full encoded modeling table with 'Churn' column inside.
    out_dir: str
        Output directory for files. Default: 'Example'
    n_rows: str or int
        'FULL' to save all predicted churn rows
        or an integer (e.g. 10, 25) to save the top-K by probability.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Build X for inference
    X_all = df_encoded.drop(columns=["Churn"])
    # Predict probabilities and labels at the chosen threshold
    probs = model.predict_proba(X_all)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Construct output table with all 32 features
    df_out = X_all.copy()
    df_out["predicted_churn"] = preds
    df_out["churn_probability"] = probs

    # Keep only predicted churners, sort by probability descending
    df_churn = df_out[df_out["predicted_churn"] == 1].sort_values(
        "churn_probability", ascending=False
    )

    # Resolve save mode: FULL or top-K
    tag = "top10"
    if isinstance(n_rows, str) and n_rows.upper() == "FULL":
        df_save = df_churn
        tag = "full"
    else:
        try:
            k = int(n_rows)
        except Exception:
            k = 10
        df_save = df_churn.head(k)
        tag = f"top{k}"

    # Save CSV
    csv_path = os.path.join(out_dir, f"churn_predictions_{tag}.csv")
    df_save.to_csv(csv_path, index=False)

    # Save Excel (fallback if engine not installed)
    xlsx_path = os.path.join(out_dir, f"churn_predictions_{tag}.xlsx")
    wrote_xlsx = False
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            df_save.to_excel(w, index=False, sheet_name="Predictions")
        wrote_xlsx = True
    except Exception:
        try:
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
                df_save.to_excel(w, index=False, sheet_name="Predictions")
            wrote_xlsx = True
        except Exception:
            pass

    # Console message for your demo
    if wrote_xlsx:
        print(f"[Export] Saved {len(df_save)} churn predictions to {csv_path} and {xlsx_path}")
    else:
        print(f"[Export] Saved {len(df_save)} churn predictions to {csv_path}")
        print("         Install 'openpyxl' or 'xlsxwriter' to also write Excel.")



# Main driver (runs A1–A4 then A5)
def main():
    #  Assignments 1–4: Build the modeling table
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    df = load_and_basic_clean(RAW_CSV)
    df = fill_remaining_nulls(df)
    df = cap_outliers_iqr(df, cols=["MonthlyCharges", "TotalCharges"], k=1.5)
    df = discretize_tenure_and_monthly(df)
    df = engineer_features(df)

    # Normalization preview (not used in modeling, for documentation only)
    df_preview = normalization_preview(df)

    # Final modeling table
    df_encoded = select_and_encode(df, drop_redundant=True)
    save_a4_artifacts(df_encoded, df_preview)

    # Assignment 5: Modeling
    # Build X/y
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"].astype(int)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    print(f"[Split] Train size={X_train.shape}, Test size={X_test.shape}, "
          f"Churn% Train={train_ratio:.3f}, Test={test_ratio:.3f}")

    # Models and CV
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    trained = {}
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    for name, model in models.items():
        print(f"\n===== {name} =====")
        # CV → OOF threshold → fit on full train → evaluate on test
        res = fit_select_threshold_and_test(model, X_train, y_train, X_test, y_test, cv, name)
        results[name] = res

        # Show 5-fold CV mean±std per metric
        cvsum = res["cv_summary"]
        print("CV (5-fold) mean±std → "
              f"Recall {cvsum['recall']['mean']:.3f}±{cvsum['recall']['std']:.3f}, "
              f"F1 {cvsum['f1']['mean']:.3f}±{cvsum['f1']['std']:.3f}, "
              f"ROC-AUC {cvsum['roc_auc']['mean']:.3f}±{cvsum['roc_auc']['std']:.3f}, "
              f"PR-AUC {cvsum['pr_auc']['mean']:.3f}±{cvsum['pr_auc']['std']:.3f}")

        # Per-fold raw CV scores (print + save)
        cv_folds_df = evaluate_cv_raw(model, X_train, y_train, cv)
        print("CV per-fold scores:")
        print(cv_folds_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        cv_folds_df.to_csv(os.path.join(ARTIFACT_DIR, f"cv_folds_{name}.csv"), index=False)

        # Refit model for saving, keep trained object
        model.fit(X_train, y_train)
        trained[name] = model

        # Save threshold sweep for the report
        pd.DataFrame(res["threshold_sweep"]).to_csv(
            os.path.join(ARTIFACT_DIR, f"threshold_sweeps_{name}.csv"), index=False
        )

        # Pretty print — test metrics at chosen threshold
        print(f"Chosen threshold: {res['threshold']:.2f}")
        print(f"Test Recall={res['recall']:.3f} | Precision={res['precision']:.3f} | "
              f"F1={res['f1']:.3f} | Acc={res['accuracy']:.3f} | "
              f"ROC-AUC={res['roc_auc']:.3f} | PR-AUC={res['pr_auc']:.3f}")
        print("Confusion Matrix [[TN FP]\n                 [FN TP]]:")
        # (matrix print intentionally suppressed)


    # Ranking and save
    rank_df = rank_results(results) # type: ignore
    print("\n===== Model Ranking (Recall → F1 → PR-AUC) =====")
    print(rank_df.to_string(index=False))
    rank_df.to_csv(os.path.join(ARTIFACT_DIR, "results_table.csv"), index=False)

    # Best model + save artifacts (model + inference contract)
    best = rank_df.iloc[0]
    best_name = str(best["model"])
    best_threshold = float(best["threshold"])
    save_model_artifacts(trained[best_name], best_name, best_threshold, X.columns, out_dir=ARTIFACT_DIR) # type: ignore

    # Explicit announcement of the best model (right under the ranking)
    print(f"\nBest model: {best_name} (threshold={best_threshold:.2f})")

    # ===== Automatic success-criteria checks (no hard-coded claims) =====
    print("\n===== Success Criteria (per model, test set) =====")
    for name, res in results.items(): # type: ignore
        rec_ok = res["recall"]   >= 0.80
        acc_ok = res["accuracy"] >= 0.70
        print(f"{name}: Recall≥80%={'YES' if rec_ok else 'NO '} "
              f"(rec={res['recall']:.2%}), "
              f"Acc≥70%={'YES' if acc_ok else 'NO '} "
              f"(acc={res['accuracy']:.2%})")

    best_metrics = results[best_name] # type: ignore
    rec_ok = best_metrics["recall"]   >= 0.80
    acc_ok = best_metrics["accuracy"] >= 0.70
    print("\n===== Success Criteria (best model only) =====")
    print(f"{best_name}: Recall≥80%={'YES' if rec_ok else 'NO '} "
          f"(rec={best_metrics['recall']:.2%}), "
          f"Acc≥70%={'YES' if acc_ok else 'NO '} "
          f"(acc={best_metrics['accuracy']:.2%})")

    # --- Optional: inference smoke test (set env var to enable) ---
    if os.environ.get("SMOKE_TEST_INFERENCE") == "1":
        model_deployed, feat_names, thr = load_artifacts()
        df_inf = pd.read_csv(A4_MODEL_TABLE).drop(columns=["Churn"])
        preds = predict_from_features(df_inf.head(5), model_deployed, feat_names, thr)
        print("\n[SmokeTest] First 5 predictions:")
        print(preds.to_string(index=False))

    #                Explanation what to run to see tables
    # Demo export: churn predictions from the chosen BEST model
    # Usage:
    #   python churn_predictor.py          -> saves top 10 predicted churn rows
    #   python churn_predictor.py 25       -> saves top 25 predicted churn rows
    #   python churn_predictor.py FULL     -> saves ALL predicted churn rows
    # ------------------------------------------------------------
    export_arg = "10"
    if len(sys.argv) > 1:
        export_arg = sys.argv[1]  # "FULL" or an integer like "25"

    # Use the trained best model object and the full encoded table
    best_model_obj = trained[best_name]  # already refit on full training set above
    export_churn_predictions(
        model=best_model_obj,
        threshold=best_threshold,
        df_encoded=df_encoded,   # includes X (32 features) plus 'Churn'
        out_dir="Example",
        n_rows=export_arg
    )






if __name__ == "__main__":
    main()