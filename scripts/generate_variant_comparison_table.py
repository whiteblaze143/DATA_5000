#!/usr/bin/env python3
"""
Generate a LaTeX table for model variant comparisons (per-lead correlations) from
results/eval metrics_summary.json files and best epochs from model training history.
"""
import json
import os
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "eval"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "docs" / "figures"
OUT_FILE = OUT_DIR / "variant_comparison_table.tex"

# Variant names and corresponding model and eval subfolders
VARIANTS = [
    ("baseline", "baseline", "final_exp_baseline"),
    ("hybrid", "hybrid", "final_exp_hybrid"),
    ("physics", "physics", "final_exp_physics"),
]

LEAD_ORDER = ["V1", "V2", "V3", "V5", "V6"]  # only chest leads shown
LEAD_INDICES = {"V1": 6, "V2": 7, "V3": 8, "V5": 10, "V6": 11}

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    means = []
    best_epochs = []

    for display_name, eval_key, model_key in VARIANTS:
        eval_metrics_path = EVAL_DIR / eval_key / "metrics_summary.json"
        hist_path = MODELS_DIR / model_key / "training_history.json"
        agg_path = ROOT / 'results' / 'experiments' / display_name / 'aggregated.json'

        if not eval_metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics_summary.json for variant {eval_key}: {eval_metrics_path}")
        if not hist_path.exists():
            # fallback: try models/run_* directories
            hist_path = MODELS_DIR / f"run_{model_key}" / "training_history.json"

        eval_metrics = load_json(eval_metrics_path)
        # If aggregated.json exists, prefer it to obtain mean and CI per lead
        if agg_path.exists():
            agg = load_json(agg_path)
            corr_vals = [agg['per_lead'][l]['mean_corr'] for l in LEAD_ORDER]
            corr_cis = {l: agg['per_lead'][l]['ci_corr'] for l in LEAD_ORDER}
            mean_corr = float(np.mean(corr_vals))
            # average CI across chest leads for a simple mean CI (for table)
            mean_ci = [float(np.mean([agg['per_lead'][l]['ci_corr'][0] for l in LEAD_ORDER])),
                       float(np.mean([agg['per_lead'][l]['ci_corr'][1] for l in LEAD_ORDER]))]
        else:
            corr = eval_metrics.get("corr_per_lead") or eval_metrics.get("correlation_per_lead")
            corr_vals = [corr[LEAD_INDICES[l]] for l in LEAD_ORDER]
            corr_cis = {l: [math.nan, math.nan] for l in LEAD_ORDER}
            mean_corr = eval_metrics.get("correlation_overall", float(np.mean(corr_vals)))

        rows.append((display_name, corr_vals, corr_cis, mean_corr, mean_ci if agg_path.exists() else None))
        means.append(mean_corr)

        epoch = None
        if hist_path.exists():
            try:
                hist = load_json(hist_path)
                epoch = hist.get("best_epoch")
            except Exception:
                epoch = None
        best_epochs.append(epoch)

    # Construct LaTeX table (include CI when available)
    with open(OUT_FILE, "w") as f:
        f.write("% Auto-generated variant comparison table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Per-Lead Correlation Comparison Across Variants (Chest Leads)}\n")
        f.write("\\label{tab:variant_comparison}\n")
        f.write("\\tiny\n")
        f.write("\\resizebox{0.95\\linewidth}{!}{%\\begin{tabular}{lccc}\\n")
        f.write("\\toprule\n")
        f.write("Lead & Baseline & Hybrid & Physics-Aware \\\\ \n")
        f.write("\\midrule\n")

        for i, lead in enumerate(LEAD_ORDER):
            baseline_val = rows[0][1][i]
            hybrid_val = rows[1][1][i]
            physics_val = rows[2][1][i]
            baseline_ci = rows[0][2].get(lead, [math.nan, math.nan])
            hybrid_ci = rows[1][2].get(lead, [math.nan, math.nan])
            physics_ci = rows[2][2].get(lead, [math.nan, math.nan])

            def fmt(val, ci):
                if ci is None or math.isnan(ci[0]):
                    return f"{val:.3f}"
                return f"{val:.3f} (\\[{ci[0]:.3f},{ci[1]:.3f}\\])"

            f.write(f"{lead} & {fmt(baseline_val, baseline_ci)} & {fmt(hybrid_val, hybrid_ci)} & {fmt(physics_val, physics_ci)} \\\\n")

        f.write("\\midrule\n")

        def fmt_mean(i):
            mean = rows[i][3]
            ci = rows[i][4]
            if ci is None:
                return f"{mean:.3f}"
            return f"{mean:.3f} (\\[{ci[0]:.3f},{ci[1]:.3f}\\])"

        f.write(f"\\textbf{{Mean}} & {fmt_mean(0)} & {fmt_mean(1)} & {fmt_mean(2)} \\\\n")
        f.write(f"Best Epoch & {best_epochs[0]} & {best_epochs[1]} & {best_epochs[2]} \\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\end{table}\n")

    print(f"Wrote {OUT_FILE}")


if __name__ == '__main__':
    main()
