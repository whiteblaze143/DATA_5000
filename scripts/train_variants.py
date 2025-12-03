#!/usr/bin/env python3
"""
Training script for ECG lead reconstruction model variants.

Supports three architectures with frozen hyperparameters:
1. baseline: UNet1D shared decoder (17.1M params)
2. hybrid: UNet1DHybrid shared trunk + per-lead heads (17.13M params)
3. physics: UNet1D + PhysicsAwareLoss (17.1M params)

Frozen Hyperparameters (validated via LR sweep on full dataset):
- lr=3e-4, batch_size=64, epochs=150, AdamW, seed=42
- features=64, depth=4, dropout=0.2

Usage:
    python scripts/train_variants.py --variant baseline --output_dir models/exp_baseline
    python scripts/train_variants.py --variant hybrid --output_dir models/exp_hybrid
    python scripts/train_variants.py --variant physics --lambda_physics 0.1 --output_dir models/exp_physics
"""

import os
import sys
import time
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_modules import get_dataloaders
from src.models.unet_1d import UNet1D, UNet1DHybrid
from src.utils import set_seed, save_model, evaluate_reconstruction, plot_reconstruction
from src.physics import reconstruct_12_leads, PhysicsAwareLoss


# ============================================================================
# FROZEN HYPERPARAMETERS (validated via sweep on full dataset)
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
    'patience': 20,  # Early stopping patience
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train ECG Model Variants')
    
    # Required arguments
    parser.add_argument('--variant', type=str, required=True, 
                        choices=['baseline', 'hybrid', 'physics'],
                        help='Model variant to train')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to save outputs')
    
    # Data path (default to full dataset)
    parser.add_argument('--data_dir', type=str, 
                        default='data/processed_full',
                        help='Path to data directory')
    
    # Physics-specific
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                        help='Weight for physics loss (only for physics variant)')
    
    # Optional overrides (for ablations)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override frozen epochs (for ablation only)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override frozen seed (for ablation only)')
    
    return parser.parse_args()


def train_epoch_standard(model, dataloader, optimizer, criterion, device):
    """Train for one epoch with standard MSE loss."""
    model.train()
    running_loss = 0.0
    
    chest_leads_indices = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        chest_targets = targets[:, chest_leads_indices]
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, chest_targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def train_epoch_physics(model, dataloader, optimizer, physics_loss_fn, device):
    """Train for one epoch with physics-aware loss."""
    model.train()
    running_total = 0.0
    running_recon = 0.0
    running_physics = 0.0
    
    chest_leads_indices = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        chest_targets = targets[:, chest_leads_indices]
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Physics-aware loss with full 12-lead targets
        total_loss, recon_loss, phys_loss = physics_loss_fn(
            outputs, chest_targets, inputs, targets
        )
        
        total_loss.backward()
        optimizer.step()
        
        running_total += total_loss.item()
        running_recon += recon_loss.item()
        running_physics += phys_loss.item()
    
    n = len(dataloader)
    return running_total / n, running_recon / n, running_physics / n


def validate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    chest_leads_indices = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            chest_targets = targets[:, chest_leads_indices]
            
            outputs = model(inputs)
            loss = criterion(outputs, chest_targets)
            running_loss += loss.item()
            
            # Reconstruct full 12-lead ECG (use target for physics leads)
            reconstructed = reconstruct_12_leads(inputs, outputs, targets=targets)
            metrics = evaluate_reconstruction(targets, reconstructed)
            all_metrics.append(metrics)
    
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in all_metrics], axis=0),
        'correlation': np.mean([m['correlation'] for m in all_metrics], axis=0),
        'snr': np.mean([m['snr'] for m in all_metrics], axis=0),
        'mae_overall': np.mean([m['mae_overall'] for m in all_metrics]),
        'correlation_overall': np.mean([m['correlation_overall'] for m in all_metrics]),
        'snr_overall': np.mean([m['snr_overall'] for m in all_metrics])
    }
    
    return running_loss / len(dataloader), avg_metrics


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    
    # Use frozen config with optional overrides
    config = FROZEN_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
        print(f"⚠️  Overriding epochs: {config['epochs']}")
    if args.seed is not None:
        config['seed'] = args.seed
        print(f"⚠️  Overriding seed: {config['seed']}")
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # Create model based on variant
    print(f"\n{'='*60}")
    print(f"Training variant: {args.variant.upper()}")
    print(f"{'='*60}")
    
    if args.variant == 'baseline':
        model = UNet1D(
            in_channels=3,
            out_channels=5,
            features=config['features'],
            depth=config['depth'],
            dropout=config['dropout']
        )
        use_physics_loss = False
        
    elif args.variant == 'hybrid':
        model = UNet1DHybrid(
            in_channels=3,
            out_channels=5,
            features=config['features'],
            depth=config['depth'],
            dropout=config['dropout'],
            head_hidden_dim=32
        )
        use_physics_loss = False
        
    elif args.variant == 'physics':
        model = UNet1D(
            in_channels=3,
            out_channels=5,
            features=config['features'],
            depth=config['depth'],
            dropout=config['dropout']
        )
        use_physics_loss = True
        config['lambda_physics'] = args.lambda_physics
    
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Define loss and optimizer
    if use_physics_loss:
        norm_params_path = os.path.join(args.data_dir, 'norm_params.pkl')
        physics_loss_fn = PhysicsAwareLoss(
            norm_params_path, 
            lambda_physics=config['lambda_physics']
        ).to(device)
        print(f"Physics loss weight: {config['lambda_physics']}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'config': config,
        'variant': args.variant,
    }
    if use_physics_loss:
        history['train_recon_loss'] = []
        history['train_physics_loss'] = []
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"lr={config['lr']}, batch={config['batch_size']}, patience={config['patience']}")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        if use_physics_loss:
            train_loss, recon_loss, phys_loss = train_epoch_physics(
                model, train_loader, optimizer, physics_loss_fn, device
            )
            history['train_recon_loss'].append(recon_loss)
            history['train_physics_loss'].append(phys_loss)
        else:
            train_loss = train_epoch_standard(
                model, train_loader, optimizer, criterion, device
            )
        
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append({
            'mae_overall': float(metrics['mae_overall']),
            'correlation_overall': float(metrics['correlation_overall']),
            'snr_overall': float(metrics['snr_overall']),
            'mae': [float(x) for x in metrics['mae']],
            'correlation': [float(x) for x in metrics['correlation']],
        })
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        log_str = (f"Epoch {epoch+1:3d}/{config['epochs']} | "
                   f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                   f"r={metrics['correlation_overall']:.4f} | "
                   f"MAE={metrics['mae_overall']:.4f} | {epoch_time:.1f}s")
        if use_physics_loss:
            log_str += f" | phys={phys_loss:.5f}"
        print(log_str)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(args.output_dir, 'best_model.pt'))
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            history['best_val_loss'] = float(best_val_loss)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {history.get('best_epoch', 'N/A')}")
    
    # Save final model
    save_model(model, os.path.join(args.output_dir, 'final_model.pt'))
    
    # Save history
    history['total_time_minutes'] = total_time / 60
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{args.variant.upper()} - Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation curve
    ax = axes[1]
    corrs = [m['correlation_overall'] for m in history['val_metrics']]
    ax.plot(corrs, label='Correlation', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    ax.set_title(f'{args.variant.upper()} - Validation Correlation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_mae_overall': float(test_metrics['mae_overall']),
        'test_correlation_overall': float(test_metrics['correlation_overall']),
        'test_snr_overall': float(test_metrics['snr_overall']),
        'test_correlation_per_lead': [float(x) for x in test_metrics['correlation']],
        'test_mae_per_lead': [float(x) for x in test_metrics['mae']],
    }
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest Results:")
    print(f"  Loss:        {test_loss:.6f}")
    print(f"  Correlation: {test_metrics['correlation_overall']:.4f}")
    print(f"  MAE:         {test_metrics['mae_overall']:.4f}")
    print(f"  SNR:         {test_metrics['snr_overall']:.2f} dB")
    
    # DL leads only (V1, V2, V3, V5, V6 = indices 6,7,8,10,11)
    dl_corrs = [test_metrics['correlation'][i] for i in [6, 7, 8, 10, 11]]
    print(f"\nDL Leads Correlation: {np.mean(dl_corrs):.4f}")
    for i, idx in enumerate([6, 7, 8, 10, 11]):
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        print(f"  {lead_names[idx]}: r={test_metrics['correlation'][idx]:.4f}")
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
