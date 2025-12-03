#!/usr/bin/env python3
"""
Train all ECG model variants with shared data loading.

Loads data ONCE and trains baseline, hybrid, and physics variants sequentially.
This is much more memory-efficient than running separate processes.

Usage:
    python scripts/train_all_variants.py --output_base models/exp
    
This will create:
    - models/exp_baseline/
    - models/exp_hybrid/
    - models/exp_physics/
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_modules import get_dataloaders
from src.models.unet_1d import UNet1D, UNet1DHybrid
from src.utils import set_seed, save_model, evaluate_reconstruction
from src.physics import reconstruct_12_leads, PhysicsAwareLoss

# Lead names for reporting
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DL_LEAD_INDICES = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
DL_LEAD_NAMES = ['V1', 'V2', 'V3', 'V5', 'V6']


# ============================================================================
# FROZEN HYPERPARAMETERS
# ============================================================================
FROZEN_CONFIG = {
    'lr': 3e-4,
    'batch_size': 128,  # Balanced: faster than 64, stable unlike 256
    'epochs': 150,
    'weight_decay': 1e-4,
    'seed': 42,
    'features': 64,
    'depth': 4,
    'dropout': 0.2,
    'num_workers': 4,
    'patience': 20,
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch_standard(model, dataloader, optimizer, criterion, device):
    """Train for one epoch with standard MSE loss."""
    model.train()
    running_loss = 0.0
    chest_leads_indices = [6, 7, 8, 10, 11]
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        chest_targets = targets[:, chest_leads_indices]
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, chest_targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def train_epoch_physics(model, dataloader, optimizer, physics_loss_fn, device):
    """Train for one epoch with physics-aware loss."""
    model.train()
    running_total = 0.0
    chest_leads_indices = [6, 7, 8, 10, 11]
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        chest_targets = targets[:, chest_leads_indices]
        
        optimizer.zero_grad()
        outputs = model(inputs)
        total_loss, _, _ = physics_loss_fn(outputs, chest_targets, inputs, targets)
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_total += total_loss.item()
    
    return running_total / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    chest_leads_indices = [6, 7, 8, 10, 11]
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            chest_targets = targets[:, chest_leads_indices]
            
            outputs = model(inputs)
            loss = criterion(outputs, chest_targets)
            running_loss += loss.item()
            
            reconstructed = reconstruct_12_leads(inputs, outputs, targets=targets)
            metrics = evaluate_reconstruction(targets, reconstructed)
            all_metrics.append(metrics)
    
    avg_metrics = {
        'mae_overall': np.mean([m['mae_overall'] for m in all_metrics]),
        'correlation_overall': np.mean([m['correlation_overall'] for m in all_metrics]),
        'snr_overall': np.mean([m['snr_overall'] for m in all_metrics]),
        'correlation': np.mean([m['correlation'] for m in all_metrics], axis=0),
        'mae': np.mean([m['mae'] for m in all_metrics], axis=0),
        'snr': np.mean([m['snr'] for m in all_metrics], axis=0),
    }
    
    return running_loss / len(dataloader), avg_metrics


def save_sample_reconstructions(model, test_loader, output_dir, device, n_samples=3):
    """Save sample reconstruction visualizations."""
    model.eval()
    chest_leads_indices = [6, 7, 8, 10, 11]
    
    # Get a batch
    inputs, targets = next(iter(test_loader))
    inputs = inputs[:n_samples].to(device)
    targets = targets[:n_samples].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        reconstructed = reconstruct_12_leads(inputs, outputs, targets=targets)
    
    # Convert to numpy
    targets_np = targets.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    
    # Plot each sample
    for idx in range(n_samples):
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, lead_name in enumerate(LEAD_NAMES):
            ax = axes[i]
            t = np.arange(targets_np.shape[2]) / 500  # Time in seconds (500 Hz)
            
            ax.plot(t, targets_np[idx, i], 'b-', alpha=0.7, label='Ground Truth')
            ax.plot(t, recon_np[idx, i], 'r--', alpha=0.7, label='Reconstructed')
            
            # Calculate correlation for this lead
            corr = np.corrcoef(targets_np[idx, i], recon_np[idx, i])[0, 1]
            
            ax.set_title(f'{lead_name} (r={corr:.3f})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'reconstruction_sample_{idx+1}.png'), dpi=150)
        plt.close()
    
    print(f"  ✓ Saved {n_samples} sample reconstructions")


def train_variant(variant_name, model, train_loader, val_loader, test_loader, 
                  output_dir, config, device, physics_loss_fn=None):
    """Train a single variant."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {variant_name.upper()}")
    print(f"{'='*60}")
    print(f"Parameters: {count_parameters(model):,}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    use_physics = physics_loss_fn is not None
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_correlation': [],
        'val_correlation_per_lead': [],
        'learning_rate': [],
        'config': config, 
        'variant': variant_name
    }
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        if use_physics:
            train_loss = train_epoch_physics(model, train_loader, optimizer, physics_loss_fn, device)
        else:
            train_loss = train_epoch_standard(model, train_loader, optimizer, criterion, device)
        
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_correlation'].append(float(metrics['correlation_overall']))
        history['val_correlation_per_lead'].append([float(x) for x in metrics['correlation']])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Step scheduler
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # DL leads correlation
        dl_corr = np.mean([metrics['correlation'][i] for i in DL_LEAD_INDICES])
        
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"r={metrics['correlation_overall']:.4f} | DL_r={dl_corr:.4f} | {epoch_time:.1f}s")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(output_dir, 'best_model.pt'))
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            history['best_val_loss'] = float(best_val_loss)
            history['best_correlation'] = float(metrics['correlation_overall'])
            print(f"  ✓ Best model saved")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Save final model and history
    save_model(model, os.path.join(output_dir, 'final_model.pt'))
    history['total_time_minutes'] = total_time / 60
    history['total_epochs'] = len(history['train_loss'])
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training curves
    save_training_curves(history, output_dir, variant_name)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    test_results = {
        'test_loss': float(test_loss),
        'test_correlation_overall': float(test_metrics['correlation_overall']),
        'test_mae_overall': float(test_metrics['mae_overall']),
        'test_snr_overall': float(test_metrics['snr_overall']),
        'test_correlation_per_lead': [float(x) for x in test_metrics['correlation']],
        'test_mae_per_lead': [float(x) for x in test_metrics['mae']],
        'test_snr_per_lead': [float(x) for x in test_metrics['snr']],
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Save sample reconstructions
    save_sample_reconstructions(model, test_loader, output_dir, device)
    
    # Print detailed results
    dl_corrs = [test_metrics['correlation'][i] for i in DL_LEAD_INDICES]
    print(f"\nTest Results:")
    print(f"  Overall Correlation: {test_metrics['correlation_overall']:.4f}")
    print(f"  Overall MAE:         {test_metrics['mae_overall']:.4f}")
    print(f"  Overall SNR:         {test_metrics['snr_overall']:.2f} dB")
    print(f"  DL Leads Mean r:     {np.mean(dl_corrs):.4f}")
    print(f"\n  Per-Lead Correlation:")
    for i, name in enumerate(LEAD_NAMES):
        marker = "*" if i in DL_LEAD_INDICES else " "
        print(f"    {marker} {name:>4}: r={test_metrics['correlation'][i]:.4f}, "
              f"MAE={test_metrics['mae'][i]:.4f}, SNR={test_metrics['snr'][i]:.1f}dB")
    
    return test_results


def save_training_curves(history, output_dir, variant_name):
    """Save training curves as figures."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'{variant_name.upper()} - Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation curve
    ax = axes[0, 1]
    ax.plot(epochs, history['val_correlation'], 'g-', label='Overall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    ax.set_title(f'{variant_name.upper()} - Validation Correlation')
    ax.grid(True, alpha=0.3)
    if history.get('best_epoch'):
        ax.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best (epoch {history['best_epoch']})")
    ax.legend()
    
    # Per-lead correlation at end
    ax = axes[1, 0]
    final_corrs = history['val_correlation_per_lead'][-1]
    colors = ['green' if i in DL_LEAD_INDICES else 'blue' for i in range(12)]
    bars = ax.bar(LEAD_NAMES, final_corrs, color=colors)
    ax.set_xlabel('Lead')
    ax.set_ylabel('Correlation')
    ax.set_title(f'{variant_name.upper()} - Final Per-Lead Correlation')
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target (0.9)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rate'], 'purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{variant_name.upper()} - Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Training curves saved")


def main():
    parser = argparse.ArgumentParser(description='Train all ECG variants')
    parser.add_argument('--output_base', type=str, default='models/exp',
                        help='Base path for output directories')
    parser.add_argument('--data_dir', type=str, default='data/processed_full',
                        help='Path to data directory')
    parser.add_argument('--variants', type=str, nargs='+', 
                        default=['baseline', 'hybrid', 'physics'],
                        choices=['baseline', 'hybrid', 'physics'],
                        help='Variants to train')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                        help='Physics loss weight')
    
    args = parser.parse_args()
    config = FROZEN_CONFIG.copy()
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # =========================================================================
    # LOAD DATA ONCE
    # =========================================================================
    print("\n" + "="*60)
    print("LOADING DATA (SHARED ACROSS ALL VARIANTS)")
    print("="*60)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # Physics loss (if needed)
    physics_loss_fn = None
    if 'physics' in args.variants:
        norm_params_path = os.path.join(args.data_dir, 'norm_params.pkl')
        physics_loss_fn = PhysicsAwareLoss(norm_params_path, lambda_physics=args.lambda_physics).to(device)
    
    # =========================================================================
    # TRAIN EACH VARIANT
    # =========================================================================
    all_results = {}
    
    for variant in args.variants:
        # Reset seed for fair comparison
        set_seed(config['seed'])
        
        # Create model
        if variant == 'baseline':
            model = UNet1D(
                in_channels=3, out_channels=5,
                features=config['features'], depth=config['depth'], dropout=config['dropout']
            )
            use_physics = False
        elif variant == 'hybrid':
            model = UNet1DHybrid(
                in_channels=3, out_channels=5,
                features=config['features'], depth=config['depth'], dropout=config['dropout'],
                head_hidden_dim=32
            )
            use_physics = False
        elif variant == 'physics':
            model = UNet1D(
                in_channels=3, out_channels=5,
                features=config['features'], depth=config['depth'], dropout=config['dropout']
            )
            use_physics = True
        
        model = model.to(device)
        output_dir = f"{args.output_base}_{variant}"
        
        results = train_variant(
            variant, model, train_loader, val_loader, test_loader,
            output_dir, config, device,
            physics_loss_fn=physics_loss_fn if use_physics else None
        )
        
        all_results[variant] = results
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    # Header
    print(f"{'Variant':<12} {'Overall r':>10} {'DL r':>10} {'SNR (dB)':>10} {'Best Epoch':>12}")
    print("-" * 60)
    
    for variant, results in all_results.items():
        overall_r = results['test_correlation_overall']
        dl_r = np.mean([results['test_correlation_per_lead'][i] for i in DL_LEAD_INDICES])
        snr = results['test_snr_overall']
        
        # Load history to get best epoch
        history_path = f"{args.output_base}_{variant}/training_history.json"
        with open(history_path) as f:
            hist = json.load(f)
        best_epoch = hist.get('best_epoch', 'N/A')
        
        print(f"{variant:<12} {overall_r:>10.4f} {dl_r:>10.4f} {snr:>10.1f} {best_epoch:>12}")
    
    # Per-lead comparison table
    print("\n" + "-"*70)
    print("PER-LEAD CORRELATION COMPARISON (DL Leads)")
    print("-"*70)
    print(f"{'Lead':<8}", end="")
    for variant in all_results.keys():
        print(f"{variant:>12}", end="")
    print()
    
    for i, lead_idx in enumerate(DL_LEAD_INDICES):
        lead_name = LEAD_NAMES[lead_idx]
        print(f"{lead_name:<8}", end="")
        for variant, results in all_results.items():
            corr = results['test_correlation_per_lead'][lead_idx]
            print(f"{corr:>12.4f}", end="")
        print()
    
    # Save combined results
    combined_results = {
        'variants': all_results,
        'config': config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(f"{args.output_base}_all_results.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n✓ All results saved to {args.output_base}_all_results.json")
    
    # Run statistical comparison if we have baseline
    if 'baseline' in all_results and len(all_results) > 1:
        print("\n" + "="*70)
        print("RUNNING STATISTICAL COMPARISON")
        print("="*70)
        
        # Build comparison command
        variants_dirs = [f"{args.output_base}_{v}" for v in all_results.keys() if v != 'baseline']
        if variants_dirs:
            import subprocess
            cmd = [
                'python', 'scripts/compare_variants.py',
                '--baseline', f"{args.output_base}_baseline",
                '--variants', *variants_dirs,
                '--output', f"{args.output_base}_comparison.json"
            ]
            print(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"⚠ Comparison failed: {e}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Output directories:")
    for variant in all_results.keys():
        print(f"  - {args.output_base}_{variant}/")
    print(f"\nNext steps:")
    print(f"  1. Review training curves in each output directory")
    print(f"  2. Check sample reconstructions")
    print(f"  3. Update presentation with new results")


if __name__ == "__main__":
    main()
