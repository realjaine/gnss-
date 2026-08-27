"""
train_and_compare.py
----------------------
Day: train + compare classifiers on gnss_dataset_combined.csv, pick a winner,
run SHAP explainability on it, and save it as gnss_model.pkl.

Steps (matches the plan):
  1. Load + stratified 80/20 split
  2. Baseline: XGBoost, full per-class report + confusion matrix
  3. Comparison: XGBoost vs Random Forest vs MLPClassifier (same split)
  4. Pick winner from the comparison table
  5. SHAP explainability on the winner
  6. Save winner as gnss_model.pkl
"""

import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

DATA_PATH = "gnss_dataset_combined.csv"

FEATURES = [
    "cn0_dbhz",
    "num_satellites",
    "pseudorange_m",
    "pseudorange_std_m",
    "doppler_hz",
    "signal_dropout",
]


# ---------------------------------------------------------------------------
# STEP 1: LOAD + SPLIT
# ---------------------------------------------------------------------------
def load_and_prepare(path: str) -> pd.DataFrame:
    """
    signal_dropout: 1/0 flag for epochs where the receiver lost lock
    entirely (pseudorange/doppler missing) — a real jamming symptom, so we
    turn it into a feature instead of deleting those rows or letting a raw
    NaN confuse the models.
    """
    df = pd.read_csv(path)
    df["signal_dropout"] = df["pseudorange_m"].isna().astype(int)
    df["pseudorange_m"] = df["pseudorange_m"].fillna(0)
    df["doppler_hz"] = df["doppler_hz"].fillna(0)
    return df


def get_split():
    df = load_and_prepare(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # genuine/jammed/spoofed -> 0/1/2
    joblib.dump(le, "label_encoder.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Loaded {len(df)} rows. Train: {len(X_train)}  Test: {len(X_test)} (stratified 80/20)\n")
    return X_train, X_test, y_train, y_test, le


# ---------------------------------------------------------------------------
# STEP 2: BASELINE — XGBoost
# ---------------------------------------------------------------------------
def train_xgboost_baseline(X_train, X_test, y_train, y_test, le):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_test)
    print("=" * 60)
    print("STEP 2 — XGBOOST BASELINE — per-class report")
    print("=" * 60)
    print(classification_report(y_test, preds, target_names=le.classes_))

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_), "\n")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — XGBoost Baseline")
    plt.tight_layout()
    plt.savefig("confusion_matrix_xgboost.png", dpi=150)
    plt.close()

    return model, train_time, preds


# ---------------------------------------------------------------------------
# STEP 3: COMPARISON — XGBoost vs Random Forest vs MLP
# ---------------------------------------------------------------------------
def train_comparison_models(X_train, X_test, y_train, y_test, xgb_model, xgb_time, le):
    results = []

    # XGBoost (reuse from step 2 so we don't retrain)
    preds = xgb_model.predict(X_test)
    results.append(_score_row("XGBoost", y_test, preds, xgb_time))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    start = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    rf_preds = rf.predict(X_test)
    results.append(_score_row("Random Forest", y_test, rf_preds, rf_time))

    # MLP — neural nets are sensitive to feature scale, so we standardize
    # inputs first (XGBoost/RF are tree-based and don't need this)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=500, random_state=42, early_stopping=True
    )
    start = time.time()
    mlp.fit(X_train_scaled, y_train)
    mlp_time = time.time() - start
    mlp_preds = mlp.predict(X_test_scaled)
    results.append(_score_row("MLP (Neural Net)", y_test, mlp_preds, mlp_time))

    table = pd.DataFrame(results)
    print("=" * 60)
    print("STEP 3 — MODEL COMPARISON TABLE")
    print("=" * 60)
    print(table.to_string(index=False))
    print()
    table.to_csv("model_comparison_table.csv", index=False)

    return {"XGBoost": xgb_model, "Random Forest": rf, "MLP (Neural Net)": mlp}, table, scaler


def _score_row(name, y_test, preds, train_time):
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "train_time_sec": round(train_time, 3),
    }


# ---------------------------------------------------------------------------
# STEP 4: PICK A WINNER
# ---------------------------------------------------------------------------
def pick_winner(table: pd.DataFrame) -> str:
    """
    Picks by F1 (macro) first since that balances precision/recall across
    all 3 classes equally — important here because a jammed/spoofed miss is
    more costly than a genuine miss, so accuracy alone can be misleading.
    Ties broken by training time (faster is more practical for a laptop-only
    setup with no GPU).
    """
    best = table.sort_values(["f1_macro", "train_time_sec"], ascending=[False, True]).iloc[0]
    return best["model"]


# ---------------------------------------------------------------------------
# STEP 5: SHAP EXPLAINABILITY (on the winner, if tree-based; MLP handled separately)
# ---------------------------------------------------------------------------
def run_shap(model, model_name, X_test, le):
    print("=" * 60)
    print(f"STEP 5 — SHAP explainability on {model_name}")
    print("=" * 60)

    if model_name == "MLP (Neural Net)":
        # SHAP's fast TreeExplainer doesn't apply to MLPs; use KernelExplainer
        # on a small background sample to keep runtime reasonable on a laptop
        background = shap.sample(X_test, 50, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        X_for_plot = X_test.sample(100, random_state=42)
        shap_values = explainer.shap_values(X_for_plot)
    else:
        explainer = shap.TreeExplainer(model)
        X_for_plot = X_test
        shap_values = explainer.shap_values(X_for_plot)

    # Newer SHAP versions return either a list of per-class arrays, or one
    # 3D array shaped (rows, features, classes). Normalize to a list of
    # per-class 2D arrays so the rest of this function doesn't care which
    # version produced it.
    if isinstance(shap_values, list):
        shap_values_per_class = shap_values
    elif shap_values.ndim == 3:
        shap_values_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_values_per_class = [shap_values]  # binary/single-output case

    # Plot 1 -- clean report-ready bar chart: mean |SHAP| per feature per class,
    # side by side. This is the single plot most people put in a report because
    # it answers "what matters overall" without visual clutter.
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values_per_class, X_for_plot, class_names=list(le.classes_),
        plot_type="bar", show=False
    )
    plt.title(f"SHAP Feature Importance by Class — {model_name}")
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary_bar.png (report-ready overview)")

    # Plot 2 -- one detailed beeswarm per class, saved separately so each is
    # readable on its own (this is the "why did it call THIS row spoofed"
    # level of detail, good for a methodology section or viva questions).
    for i, class_name in enumerate(le.classes_):
        plt.figure(figsize=(7, 4.5))
        shap.summary_plot(shap_values_per_class[i], X_for_plot, show=False)
        plt.title(f"SHAP Detail — class: {class_name}")
        plt.tight_layout()
        plt.savefig(f"shap_detail_{class_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved shap_detail_{class_name}.png")
    print()


# ---------------------------------------------------------------------------
# STEP 6: SAVE WINNER
# ---------------------------------------------------------------------------
def save_winner(model, model_name, le, scaler=None):
    bundle = {
        "model": model,
        "model_name": model_name,
        "label_encoder": le,
        "features": FEATURES,
        "scaler": scaler,  # only non-None if the winner is the MLP
    }
    joblib.dump(bundle, "gnss_model.pkl")
    print(f"Saved winning model ({model_name}) as gnss_model.pkl")
    print("Bundle contains: model, model_name, label_encoder, features list, scaler (if needed)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = get_split()

    xgb_model, xgb_time, _ = train_xgboost_baseline(X_train, X_test, y_train, y_test, le)

    models, table, scaler = train_comparison_models(
        X_train, X_test, y_train, y_test, xgb_model, xgb_time, le
    )

    winner_name = pick_winner(table)
    print("=" * 60)
    print(f"STEP 4 — WINNER: {winner_name}")
    print("=" * 60)
    winner_row = table[table["model"] == winner_name].iloc[0]
    print(winner_row.to_string(), "\n")

    winner_model = models[winner_name]
    winner_scaler = scaler if winner_name == "MLP (Neural Net)" else None

    run_shap(winner_model, winner_name, X_test, le)

    save_winner(winner_model, winner_name, le, winner_scaler)
