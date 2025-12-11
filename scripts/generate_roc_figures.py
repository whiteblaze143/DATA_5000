#!/usr/bin/env python3
"""
Generate sensitivity-specificity and AUROC plots for key clinical features across model variants
and save them into docs/figures/ for inclusion in the LaTeX report.

- Uses per-variant `clinical_features_per_sample.csv` under results/eval/<variant>/
- Produces: 
  - docs/figures/sensitivity_specificity_qt_baseline.png
  - docs/figures/roc_qt_variants.png  (baseline/hybrid/physics)
  - docs/figures/roc_qrs_variants.png (baseline/hybrid/physics)

Usage: python3 scripts/generate_roc_figures.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def compute_roc_from_scores(y_true, y_score):
    # y_true: array-like of 0/1
    # y_score: continuous scores
    # returns fpr, tpr, thresholds, auc_val
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    # Remove NaNs
    mask = (~np.isnan(y_score)) & (~np.isnan(y_true))
    y_true = y_true[mask]
    y_score = y_score[mask]
    # If no positive or no negative classes, return trivial ROC
    P = y_true.sum()
    N = len(y_true) - P
    # Sort descending by score
    desc_idx = np.argsort(-y_score, kind='mergesort')
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]
    # Compute TPR and FPR at each unique score threshold
    cum_TP = np.cumsum(y_true_sorted)
    cum_FP = np.cumsum(1 - y_true_sorted)
    # For thresholds we take unique values of y_score_sorted
    uniq_scores, idx = np.unique(y_score_sorted, return_index=True)
    tpr = np.concatenate(([0.0], cum_TP[idx - 1] / P if P > 0 else np.zeros_like(idx), [1.0])) if P>0 else np.array([0.0,1.0])
    fpr = np.concatenate(([0.0], cum_FP[idx - 1] / N if N > 0 else np.zeros_like(idx), [1.0])) if N>0 else np.array([0.0,1.0])
    # AUC via trapezoid rule
    # ensure fpr,tpr sorted asc by fpr
    order = np.argsort(fpr)
    fpr_s = fpr[order]
    tpr_s = tpr[order]
    auc_val = np.trapz(tpr_s, fpr_s)
    return fpr_s, tpr_s, np.concatenate((uniq_scores, [uniq_scores[-1]])) if len(uniq_scores)>0 else np.array([]), auc_val

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "eval"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["baseline", "hybrid", "physics"]
CHEST_LEADS = ["V1", "V2", "V3", "V5", "V6"]

# Clinically meaningful thresholds
QT_THRESHOLD_MS = 450  # QT prolongation threshold (ms)
QRS_THRESHOLD_MS = 120  # QRS prolongation threshold (ms)

# Read per-variant sample data and build per-variant samples for chest leads
variant_dfs = {}
for v in VARIANTS:
    vdir = EVAL_DIR / v
    file = vdir / "clinical_features_per_sample.csv"
    if not file.exists():
        print(f"Missing per-sample file for {v}: {file}")
        continue
    rows = []
    with open(file, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get('lead_name') in CHEST_LEADS:
                rows.append(r)
    variant_dfs[v] = rows

# Utility to compute roc curve for a variant and feature
def compute_roc(rows, true_col, pred_col, true_threshold=None):
    y_true = []
    y_score = []
    for r in rows:
        try:
            tval = float(r.get(true_col, 'nan')) if r.get(true_col) not in (None, '') else np.nan
            sval = float(r.get(pred_col, 'nan')) if r.get(pred_col) not in (None, '') else np.nan
        except Exception:
            tval = np.nan
            sval = np.nan
        if np.isnan(tval) or np.isnan(sval):
            continue
        if true_threshold is not None:
            y_true.append(1 if tval >= true_threshold else 0)
        else:
            y_true.append(int(tval))
        y_score.append(sval)
    fpr, tpr, thr, auc_val = compute_roc_from_scores(np.array(y_true), np.array(y_score))
    return fpr, tpr, thr, auc_val

# Generate QT ROC curves across variants
plt.figure(figsize=(6,6))
for v, df in variant_dfs.items():
    fpr, tpr, thresholds, val_auc = compute_roc(df, 'qt_interval_true', 'qt_interval_pred', true_threshold=QT_THRESHOLD_MS)
    plt.plot(fpr, tpr, label=f"{v} (AUC={val_auc:.3f})")

plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve: QT Prolongation (true QT >= {0} ms)'.format(QT_THRESHOLD_MS))
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'roc_qt_variants.png', dpi=200)
plt.close()

# Generate QRS ROC curves across variants
plt.figure(figsize=(6,6))
for v, df in variant_dfs.items():
    fpr, tpr, thresholds, val_auc = compute_roc(df, 'qrs_duration_true', 'qrs_duration_pred', true_threshold=QRS_THRESHOLD_MS)
    plt.plot(fpr, tpr, label=f"{v} (AUC={val_auc:.3f})")

plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve: QRS Prolongation (true QRS >= {0} ms)'.format(QRS_THRESHOLD_MS))
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'roc_qrs_variants.png', dpi=200)
plt.close()

# Generate sensitivity and specificity vs threshold plot for QT using baseline only
if 'baseline' in variant_dfs:
    rows = variant_dfs['baseline']
    y_true = []
    pred_scores = []
    for r in rows:
        try:
            tval = float(r.get('qt_interval_true'))
            sval = float(r.get('qt_interval_pred'))
        except Exception:
            continue
        y_true.append(1 if tval >= QT_THRESHOLD_MS else 0)
        pred_scores.append(sval)
    y_true = np.array(y_true)
    pred_scores = np.array(pred_scores)

    thresholds = np.linspace(pred_scores.min(), pred_scores.max(), 200)
    sens = []
    spec = []
    for t in thresholds:
        y_pred_bin = (pred_scores >= t).astype(int)
        tp = np.logical_and(y_pred_bin == 1, y_true == 1).sum()
        tn = np.logical_and(y_pred_bin == 0, y_true == 0).sum()
        fp = np.logical_and(y_pred_bin == 1, y_true == 0).sum()
        fn = np.logical_and(y_pred_bin == 0, y_true == 1).sum()
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, sens, label='Sensitivity', color='tab:blue')
    plt.plot(thresholds, spec, label='Specificity', color='tab:orange')
    plt.xlabel('Predicted QT threshold (ms)')
    plt.ylabel('Rate')
    plt.title('Sensitivity & Specificity vs Predicted QT threshold (Baseline)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'sensitivity_specificity_qt_baseline.png', dpi=200)
    plt.close()

print("Saved ROC and sensitivity figures under docs/figures/")
