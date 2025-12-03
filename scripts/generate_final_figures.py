#!/usr/bin/env python3
"""
Generate publication-quality figures for final presentation and report.

Run after training completes:
    python scripts/generate_final_figures.py --model_dir models/final_exp_baseline

Generates:
    1. training_curves_final.png - Loss, correlation, LR schedule (4-panel)
    2. reconstruction_samples.png - 3 sample 12-lead ECG comparisons
    3. per_lead_barplot.png - Per-lead correlation with 95% CI
    4. model_comparison.png - Variant comparison (if multiple exist)
    5. information_bottleneck.png - Ground truth correlation heatmap
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
from scipy import stats

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_modules import get_dataloaders
from src.models.unet_1d import UNet1D, UNet1DHybrid
from src.physics import reconstruct_12_leads
from src.utils import evaluate_reconstruction

# Constants
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DL_LEAD_INDICES = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
PHYSICS_LEAD_INDICES = [2, 3, 4, 5]  # III, aVR, aVL, aVF
INPUT_LEAD_INDICES = [0, 1, 9]  # I, II, V4

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#1A1A2E',
    'accent': '#E94D60',
    'accent2': '#0F4C75',
    'accent3': '#3EBAC2',
    'success': '#4EC9B0',
    'warning': '#FFCC00',
    'physics': '#4EC9B0',
    'dl': '#E94D60',
    'input': '#0F4C75',
}


def load_training_history(model_dir):
    """Load training history from JSON."""
    history_path = os.path.join(model_dir, 'training_history.json')
    with open(history_path) as f:
        return json.load(f)


def load_test_results(model_dir):
    """Load test results from JSON."""
    results_path = os.path.join(model_dir, 'test_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def generate_training_curves(history, output_dir, variant_name='baseline'):
    """Generate 4-panel training curves figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Panel 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Correlation curve
    ax = axes[0, 1]
    if 'val_correlation' in history:
        ax.plot(epochs, history['val_correlation'], color=COLORS['success'], linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target (r=0.9)')
        
        # Mark best epoch
        if 'best_epoch' in history:
            best_epoch = history['best_epoch']
            best_corr = history['val_correlation'][best_epoch - 1]
            ax.scatter([best_epoch], [best_corr], color=COLORS['accent'], s=100, zorder=5, 
                       label=f'Best (epoch {best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Validation Correlation (Overall)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.0)
    
    # Panel 3: Per-lead correlation at final epoch
    ax = axes[1, 0]
    if 'val_correlation_per_lead' in history:
        final_corrs = history['val_correlation_per_lead'][-1]
    else:
        final_corrs = [0.0] * 12  # Placeholder
    
    colors = []
    for i in range(12):
        if i in INPUT_LEAD_INDICES:
            colors.append(COLORS['input'])
        elif i in PHYSICS_LEAD_INDICES:
            colors.append(COLORS['physics'])
        else:
            colors.append(COLORS['dl'])
    
    bars = ax.bar(LEAD_NAMES, final_corrs, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lead', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Per-Lead Correlation (Final Epoch)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['input'], label='Input (I, II, V4)'),
        mpatches.Patch(color=COLORS['physics'], label='Physics (III, aVR, aVL, aVF)'),
        mpatches.Patch(color=COLORS['dl'], label='DL (V1, V2, V3, V5, V6)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Learning rate schedule
    ax = axes[1, 1]
    if 'learning_rate' in history:
        ax.plot(epochs, history['learning_rate'], color='purple', linewidth=2)
    else:
        # Estimate from typical ReduceLROnPlateau
        ax.text(0.5, 0.5, 'LR Schedule\nNot Recorded', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{variant_name.upper()} Training Progress', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_curves_final.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_reconstruction_samples(model, test_loader, output_dir, device, n_samples=3):
    """Generate sample reconstruction comparisons."""
    model.eval()
    
    # Get samples
    inputs, targets = next(iter(test_loader))
    inputs = inputs[:n_samples].to(device)
    targets = targets[:n_samples].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        reconstructed = reconstruct_12_leads(inputs, outputs, targets=targets)
    
    targets_np = targets.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    
    # Create figure with all samples
    fig = plt.figure(figsize=(16, 4 * n_samples))
    gs = GridSpec(n_samples, 1, hspace=0.3)
    
    for sample_idx in range(n_samples):
        # Create subplot for this sample
        gs_inner = gs[sample_idx].subgridspec(3, 4, hspace=0.4, wspace=0.3)
        
        for lead_idx, lead_name in enumerate(LEAD_NAMES):
            row = lead_idx // 4
            col = lead_idx % 4
            ax = fig.add_subplot(gs_inner[row, col])
            
            t = np.arange(500) / 500  # First 1 second (500 samples at 500Hz)
            gt = targets_np[sample_idx, lead_idx, :500]
            rc = recon_np[sample_idx, lead_idx, :500]
            
            ax.plot(t, gt, 'b-', linewidth=1.5, alpha=0.8, label='Ground Truth')
            ax.plot(t, rc, 'r--', linewidth=1.5, alpha=0.8, label='Reconstructed')
            
            # Calculate correlation
            corr = np.corrcoef(targets_np[sample_idx, lead_idx], 
                               recon_np[sample_idx, lead_idx])[0, 1]
            
            # Color code by lead type
            if lead_idx in INPUT_LEAD_INDICES:
                title_color = COLORS['input']
                lead_type = '(Input)'
            elif lead_idx in PHYSICS_LEAD_INDICES:
                title_color = COLORS['physics']
                lead_type = '(Physics)'
            else:
                title_color = COLORS['dl']
                lead_type = '(DL)'
            
            ax.set_title(f'{lead_name} {lead_type}\nr={corr:.3f}', fontsize=9, color=title_color)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=7)
            
            if row == 2:
                ax.set_xlabel('Time (s)', fontsize=8)
            if col == 0:
                ax.set_ylabel('Amplitude', fontsize=8)
    
    # Add legend at top
    handles = [
        plt.Line2D([0], [0], color='b', linewidth=2, label='Ground Truth'),
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Reconstructed'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12, 
               bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle(f'Sample ECG Reconstructions ({n_samples} patients)', 
                 fontsize=14, fontweight='bold', y=1.04)
    
    output_path = os.path.join(output_dir, 'reconstruction_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_per_lead_barplot(test_results, output_dir, variant_name='baseline'):
    """Generate per-lead correlation bar plot with error visualization."""
    if test_results is None:
        print("⚠ No test results found, skipping per-lead barplot")
        return
    
    correlations = test_results.get('test_correlation_per_lead', [0]*12)
    maes = test_results.get('test_mae_per_lead', [0]*12)
    snrs = test_results.get('test_snr_per_lead', [0]*12)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors by lead type
    colors = []
    for i in range(12):
        if i in INPUT_LEAD_INDICES:
            colors.append(COLORS['input'])
        elif i in PHYSICS_LEAD_INDICES:
            colors.append(COLORS['physics'])
        else:
            colors.append(COLORS['dl'])
    
    # Panel 1: Correlation
    ax = axes[0]
    bars = ax.bar(LEAD_NAMES, correlations, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Target (r=0.9)')
    ax.set_xlabel('Lead', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Per-Lead Correlation', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    
    # Add value labels
    for bar, val in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Panel 2: MAE
    ax = axes[1]
    bars = ax.bar(LEAD_NAMES, maes, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='Target (MAE<0.05)')
    ax.set_xlabel('Lead', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Per-Lead MAE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Panel 3: SNR
    ax = axes[2]
    bars = ax.bar(LEAD_NAMES, snrs, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='Clinical (>20 dB)')
    ax.set_xlabel('Lead', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('Per-Lead SNR', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Legend for lead types
    legend_elements = [
        mpatches.Patch(color=COLORS['input'], label='Input'),
        mpatches.Patch(color=COLORS['physics'], label='Physics'),
        mpatches.Patch(color=COLORS['dl'], label='Deep Learning'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.08))
    
    plt.suptitle(f'{variant_name.upper()} Per-Lead Performance', fontsize=16, fontweight='bold', y=1.12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'per_lead_barplot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_model_comparison(results_dirs, output_dir):
    """Generate model variant comparison figure."""
    variants = {}
    
    for result_dir in results_dirs:
        if not os.path.exists(result_dir):
            continue
        
        variant_name = os.path.basename(result_dir).replace('final_exp_', '')
        test_results = load_test_results(result_dir)
        
        if test_results:
            variants[variant_name] = test_results
    
    if len(variants) < 2:
        print("⚠ Need at least 2 variants for comparison, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Overall metrics comparison
    ax = axes[0]
    variant_names = list(variants.keys())
    x = np.arange(len(variant_names))
    width = 0.25
    
    overall_corrs = [v['test_correlation_overall'] for v in variants.values()]
    dl_corrs = [np.mean([v['test_correlation_per_lead'][i] for i in DL_LEAD_INDICES]) 
                for v in variants.values()]
    snrs = [v['test_snr_overall'] / 100 for v in variants.values()]  # Scale for visibility
    
    bars1 = ax.bar(x - width, overall_corrs, width, label='Overall r', color=COLORS['accent3'])
    bars2 = ax.bar(x, dl_corrs, width, label='DL Leads r', color=COLORS['dl'])
    bars3 = ax.bar(x + width, snrs, width, label='SNR/100', color=COLORS['success'])
    
    ax.set_xlabel('Model Variant', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([v.upper() for v in variant_names])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    
    # Panel 2: Per DL-lead comparison
    ax = axes[1]
    width = 0.8 / len(variants)
    
    for i, (variant_name, results) in enumerate(variants.items()):
        dl_corrs = [results['test_correlation_per_lead'][idx] for idx in DL_LEAD_INDICES]
        x_pos = np.arange(5) + i * width - (len(variants) - 1) * width / 2
        ax.bar(x_pos, dl_corrs, width, label=variant_name.upper(), alpha=0.8)
    
    ax.set_xlabel('DL Lead', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Per-Lead Comparison (DL Leads Only)', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['V1', 'V2', 'V3', 'V5', 'V6'])
    ax.legend(fontsize=10)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_ground_truth_heatmap(test_loader, output_dir):
    """Generate ground truth inter-lead correlation heatmap."""
    # Collect all target data
    all_targets = []
    for _, targets in test_loader:
        all_targets.append(targets.numpy())
    
    all_targets = np.concatenate(all_targets, axis=0)  # [N, 12, 5000]
    
    # Calculate correlation matrix
    n_samples, n_leads, n_timesteps = all_targets.shape
    corr_matrix = np.zeros((n_leads, n_leads))
    
    for i in range(n_leads):
        for j in range(n_leads):
            # Flatten and calculate correlation
            x = all_targets[:, i, :].flatten()
            y = all_targets[:, j, :].flatten()
            corr_matrix[i, j] = np.corrcoef(x, y)[0, 1]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson Correlation', fontsize=12)
    
    # Add labels
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(12))
    ax.set_xticklabels(LEAD_NAMES, fontsize=10)
    ax.set_yticklabels(LEAD_NAMES, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add correlation values as text
    for i in range(12):
        for j in range(12):
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', 
                    color=color, fontsize=8)
    
    # Highlight input leads
    for idx in INPUT_LEAD_INDICES:
        rect = plt.Rectangle((idx - 0.5, -0.5), 1, 12, fill=False, 
                              edgecolor=COLORS['input'], linewidth=2)
        ax.add_patch(rect)
        rect = plt.Rectangle((-0.5, idx - 0.5), 12, 1, fill=False, 
                              edgecolor=COLORS['input'], linewidth=2)
        ax.add_patch(rect)
    
    ax.set_title('Ground Truth Inter-Lead Correlation\n(Input leads highlighted)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'ground_truth_correlation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")
    
    # Save correlation matrix
    np.save(os.path.join(output_dir, 'ground_truth_correlations.npy'), corr_matrix)


def generate_summary_metrics_figure(all_results, output_dir):
    """Generate a summary figure with key metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create summary table data
    if 'baseline' in all_results:
        results = all_results['baseline']
    else:
        results = list(all_results.values())[0]
    
    # Key metrics
    metrics = [
        ('Overall Correlation', results.get('test_correlation_overall', 0), 'r ≥ 0.90'),
        ('DL Leads Correlation', np.mean([results['test_correlation_per_lead'][i] for i in DL_LEAD_INDICES]), 'r ≥ 0.80'),
        ('Physics Leads Correlation', np.mean([results['test_correlation_per_lead'][i] for i in PHYSICS_LEAD_INDICES]), 'r = 1.00'),
        ('Overall MAE', results.get('test_mae_overall', 0), 'MAE ≤ 0.05'),
        ('Overall SNR (dB)', results.get('test_snr_overall', 0), 'SNR ≥ 20'),
    ]
    
    # Create bar chart
    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    targets = [m[2] for m in metrics]
    
    # Normalize for display (correlation and MAE are 0-1, SNR is different)
    display_values = values.copy()
    display_values[4] = min(values[4] / 100, 1.0)  # Normalize SNR
    
    colors = [COLORS['success'] if i < 3 else COLORS['accent3'] for i in range(5)]
    
    bars = ax.barh(names, display_values, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, val, target in zip(bars, values, targets):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f} ({target})', va='center', fontsize=10)
    
    ax.set_xlim(0, 1.4)
    ax.set_xlabel('Normalized Value', fontsize=12)
    ax.set_title('Summary of Key Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'summary_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate final figures')
    parser.add_argument('--model_dir', type=str, default='models/final_exp_baseline',
                        help='Path to model directory')
    parser.add_argument('--output_dir', type=str, default='docs/figures',
                        help='Output directory for figures')
    parser.add_argument('--data_dir', type=str, default='data/processed_full',
                        help='Path to data directory')
    parser.add_argument('--all_variants', action='store_true',
                        help='Generate comparison across all variants')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load training history
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    history = load_training_history(args.model_dir)
    test_results = load_test_results(args.model_dir)
    variant_name = os.path.basename(args.model_dir).replace('final_exp_', '')
    
    # 1. Training curves
    print("\n[1/6] Training curves...")
    generate_training_curves(history, args.output_dir, variant_name)
    
    # 2. Per-lead barplot
    print("\n[2/6] Per-lead barplot...")
    generate_per_lead_barplot(test_results, args.output_dir, variant_name)
    
    # 3. Load model and generate reconstructions
    print("\n[3/6] Sample reconstructions...")
    if os.path.exists(os.path.join(args.model_dir, 'best_model.pt')):
        # Load model
        model = UNet1D(in_channels=3, out_channels=5, features=64, depth=4, dropout=0.2)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pt')))
        model = model.to(device)
        model.eval()
        
        # Load test data
        _, _, test_loader = get_dataloaders(args.data_dir, batch_size=16, num_workers=2)
        
        generate_reconstruction_samples(model, test_loader, args.output_dir, device)
        
        # 4. Ground truth heatmap
        print("\n[4/6] Ground truth correlation heatmap...")
        generate_ground_truth_heatmap(test_loader, args.output_dir)
    else:
        print("  ⚠ No model found, skipping reconstruction samples")
    
    # 5. Model comparison (if multiple variants)
    print("\n[5/6] Model comparison...")
    if args.all_variants:
        base_dir = os.path.dirname(args.model_dir)
        variant_dirs = [
            os.path.join(base_dir, 'final_exp_baseline'),
            os.path.join(base_dir, 'final_exp_hybrid'),
            os.path.join(base_dir, 'final_exp_physics'),
        ]
        generate_model_comparison(variant_dirs, args.output_dir)
    else:
        print("  ⚠ Use --all_variants to generate comparison")
    
    # 6. Summary metrics
    print("\n[6/6] Summary metrics...")
    if test_results:
        generate_summary_metrics_figure({'baseline': test_results}, args.output_dir)
    
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
