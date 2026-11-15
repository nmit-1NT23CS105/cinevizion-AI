import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss
)

# ============================================================
# CINEVISION AI - STEP 11
# JOINT MULTI-OBJECTIVE THRESHOLD OPTIMIZER
# (NON-DESTRUCTIVE EXTENSION)
# ============================================================

from pathlib import Path

BASE_DIR = str(
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "pipeline"
    else Path(__file__).resolve().parent
)
SPLIT_DIR = os.path.join(BASE_DIR, "dataset", "splits")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

VAL_PREDS_PATH = os.path.join(EVAL_DIR, "validation_predictions.csv")
TEST_PREDS_PATH = os.path.join(EVAL_DIR, "test_predictions.csv")
TTA_PREDS_PATH = os.path.join(EVAL_DIR, "ensemble_test_predictions.csv")

VAL_CSV = os.path.join(SPLIT_DIR, "validation.csv")
TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")

OPTIMAL_THRESHOLDS_V2 = os.path.join(EVAL_DIR, "optimal_thresholds_v2.json")
V2_TEST_METRICS = os.path.join(EVAL_DIR, "v2_test_metrics.csv")
V2_SUMMARY_REPORT = os.path.join(EVAL_DIR, "step11_threshold_optimization_report.txt")

TARGET_GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Crime",
    "Drama",
    "Horror",
    "Romance",
    "Thriller"
]

print("=" * 70)
print("CINEVISION AI - STEP 11: JOINT MULTI-OBJECTIVE THRESHOLD OPTIMIZER")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA & PREDICTIONS
# ------------------------------------------------------------
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

y_val = val_df[TARGET_GENRES].values.astype(int)
y_test = test_df[TARGET_GENRES].values.astype(int)

# Load existing probabilities
val_probs_df = pd.read_csv(VAL_PREDS_PATH)
test_probs_df = pd.read_csv(TEST_PREDS_PATH)

prob_cols = [f"{g}_probability" for g in TARGET_GENRES]
val_probs = val_probs_df[prob_cols].values
test_probs = test_probs_df[prob_cols].values

# Load original thresholds for comparison
with open(os.path.join(EVAL_DIR, "optimal_thresholds.json"), "r") as f:
    orig_thresholds = json.load(f)

print(f"Loaded validation set: {val_probs.shape}, test set: {test_probs.shape}")
print("Original thresholds:", orig_thresholds)

# ------------------------------------------------------------
# 2. OBJECTIVE FUNCTION FOR JOINT OPTIMIZATION
# ------------------------------------------------------------
# We want to maximize Macro F1, while ensuring high exact match and
# avoiding severe recall collapse on rare classes (like Horror/Crime).
#
# Optimization is performed STRICTLY on validation probabilities.
# Test data is NEVER used in the optimization loop.
# ------------------------------------------------------------

def objective(thresholds):
    # Clip thresholds to [0.15, 0.85]
    thresholds = np.clip(thresholds, 0.15, 0.85)
    
    bin_preds = (val_probs >= thresholds).astype(int)
    
    # 1. Macro F1
    macro_f1 = f1_score(y_val, bin_preds, average="macro", zero_division=0)
    
    # 2. Exact Match Ratio
    exact_match = np.mean(np.all(y_val == bin_preds, axis=1))
    
    # 3. Minimum Recall across all genres (penalize if any genre drops below 0.45)
    recalls = [recall_score(y_val[:, i], bin_preds[:, i], zero_division=0) for i in range(len(TARGET_GENRES))]
    min_recall = np.min(recalls)
    avg_recall = np.mean(recalls)
    
    # Multi-objective composite score:
    # 70% Macro F1 + 15% Exact Match + 15% Min Recall
    composite = 0.70 * macro_f1 + 0.15 * exact_match + 0.15 * min_recall
    
    # Minimize negative composite
    return -composite

# ------------------------------------------------------------
# 3. RUN BOUNDED OPTIMIZATION
# ------------------------------------------------------------
print("\nRunning Nelder-Mead joint multi-objective optimization on validation set...")

initial_guess = np.array([orig_thresholds[g] for g in TARGET_GENRES])

res = minimize(
    objective,
    initial_guess,
    method="Nelder-Mead",
    options={"maxiter": 1500, "disp": True}
)

best_thresholds_arr = np.clip(np.round(res.x, 2), 0.15, 0.85)
optimal_thresholds_v2 = {g: float(t) for g, t in zip(TARGET_GENRES, best_thresholds_arr)}

print("\n" + "=" * 70)
print("OPTIMIZATION RESULTS")
print("=" * 70)
print("Original Thresholds vs Optimized V2 Thresholds:")
for g in TARGET_GENRES:
    diff = optimal_thresholds_v2[g] - orig_thresholds[g]
    print(f"  {g:10s} : {orig_thresholds[g]:.2f}  -->  {optimal_thresholds_v2[g]:.2f} (delta: {diff:+.2f})")

# ------------------------------------------------------------
# 4. EVALUATE ON VALIDATION AND TEST SET
# ------------------------------------------------------------
def evaluate(probs, y_true, thresh_dict, name):
    thresh_arr = np.array([thresh_dict[g] for g in TARGET_GENRES])
    bin_preds = (probs >= thresh_arr).astype(int)
    
    macro_p = precision_score(y_true, bin_preds, average="macro", zero_division=0)
    macro_r = recall_score(y_true, bin_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, bin_preds, average="macro", zero_division=0)
    
    micro_f1 = f1_score(y_true, bin_preds, average="micro", zero_division=0)
    h_loss = hamming_loss(y_true, bin_preds)
    exact_match = np.mean(np.all(y_true == bin_preds, axis=1))
    partial_match = np.mean(np.any((y_true == 1) & (bin_preds == 1), axis=1))
    
    return {
        "Configuration": name,
        "Macro F1": macro_f1,
        "Micro F1": micro_f1,
        "Macro Precision": macro_p,
        "Macro Recall": macro_r,
        "Hamming Loss": h_loss,
        "Exact Match": exact_match,
        "Partial Match": partial_match
    }, bin_preds

# Test evaluation with Original Thresholds vs V2 Thresholds
val_v1, _ = evaluate(val_probs, y_val, orig_thresholds, "Validation (V1 Thresholds)")
val_v2, _ = evaluate(val_probs, y_val, optimal_thresholds_v2, "Validation (V2 Joint Thresholds)")

test_v1, _ = evaluate(test_probs, y_test, orig_thresholds, "Test (V1 Thresholds - Original)")
test_v2, bin_test_v2 = evaluate(test_probs, y_test, optimal_thresholds_v2, "Test (V2 Joint Thresholds)")

# Also evaluate with TTA probabilities if available
if os.path.exists(TTA_PREDS_PATH):
    tta_df = pd.read_csv(TTA_PREDS_PATH)
    tta_cols = [f"{g}_prob" for g in TARGET_GENRES]
    tta_probs = tta_df[tta_cols].values
    test_tta_v2, bin_tta_v2 = evaluate(tta_probs, y_test, optimal_thresholds_v2, "Test (TTA + V2 Joint Thresholds)")
    results_summary = pd.DataFrame([val_v1, val_v2, test_v1, test_v2, test_tta_v2])
else:
    results_summary = pd.DataFrame([val_v1, val_v2, test_v1, test_v2])

print("\n" + "=" * 70)
print("BENCHMARK COMPARISON TABLE")
print("=" * 70)
print(results_summary.to_string(index=False))

# Per-Genre Metrics on Test Set with V2 Thresholds
per_genre_v2 = []
for i, g in enumerate(TARGET_GENRES):
    p = precision_score(y_test[:, i], bin_test_v2[:, i], zero_division=0)
    r = recall_score(y_test[:, i], bin_test_v2[:, i], zero_division=0)
    f1 = f1_score(y_test[:, i], bin_test_v2[:, i], zero_division=0)
    roc = roc_auc_score(y_test[:, i], test_probs[:, i])
    pr = average_precision_score(y_test[:, i], test_probs[:, i])
    per_genre_v2.append({
        "Genre": g,
        "V1_Thresh": orig_thresholds[g],
        "V2_Thresh": optimal_thresholds_v2[g],
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "ROC_AUC": roc,
        "PR_AUC": pr
    })

per_genre_df = pd.DataFrame(per_genre_v2)
print("\n" + "=" * 70)
print("PER-GENRE TEST METRICS WITH V2 THRESHOLDS")
print("=" * 70)
print(per_genre_df.to_string(index=False))

# ------------------------------------------------------------
# 5. SAVE ARTIFACTS
# ------------------------------------------------------------
with open(OPTIMAL_THRESHOLDS_V2, "w", encoding="utf-8") as f:
    json.dump(optimal_thresholds_v2, f, indent=4)

per_genre_df.to_csv(V2_TEST_METRICS, index=False)

with open(V2_SUMMARY_REPORT, "w", encoding="utf-8") as f:
    f.write("CINEVISION AI - STEP 11 THRESHOLD OPTIMIZATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write("OPTIMAL V2 THRESHOLDS:\n")
    f.write(json.dumps(optimal_thresholds_v2, indent=4) + "\n\n")
    f.write("EVALUATION COMPARISON:\n")
    f.write(results_summary.to_string(index=False) + "\n\n")
    f.write("PER-GENRE V2 BREAKDOWN:\n")
    f.write(per_genre_df.to_string(index=False) + "\n")

print(f"\nSaved V2 thresholds to:\n  {OPTIMAL_THRESHOLDS_V2}")
print(f"Saved V2 metrics to:\n  {V2_TEST_METRICS}")
print(f"Saved summary report to:\n  {V2_SUMMARY_REPORT}")
print("STEP 11 COMPLETE.")

