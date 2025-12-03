#!/usr/bin/env python3
"""
Main training entry point for ECG Lead Reconstruction.
Designed to work on local machine or VM with proper path handling.

Usage:
    # Quick test with synthetic data
    python run_training.py --test_mode
    
    # Train with custom data directory
    python run_training.py --data_dir data/processed --output_dir models/experiment1
    
    # Full training with config file
    python run_training.py --config config.json
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

# Disable torch dynamo to avoid conflicts with transformers
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for VM
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import Config, get_default_config, get_vm_config
from data.data_modules import get_dataloaders
from src.models.unet_1d import UNet1D, UNet1DLeadSpecific
from src.utils import set_seed, save_model, evaluate_reconstruction, plot_reconstruction
from src.physics import reconstruct_12_leads


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train ECG Lead Reconstruction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with synthetic data
    python run_training.py --test_mode
    
    # Custom training
    python run_training.py --data_dir data/processed --epochs 100 --batch_size 64
    
    # Use VM-optimized config
    python run_training.py --vm_mode
    
    # Train with lead-specific decoders (recommended for best performance)
    python run_training.py --model unet_lead_specific --data_dir data/processed_full
        """
    )
    
    # Mode selection
    parser.add_argument('--test_mode', action='store_true',
                       help='Use synthetic test data for pipeline validation')
    parser.add_argument('--vm_mode', action='store_true',
                       help='Use VM-optimized settings (larger batch, more workers)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to data directory (relative or absolute)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Path to save outputs (relative or absolute)')
    
    # Model architecture
    parser.add_argument('--model', type=str, default='unet_1d',
                       choices=['unet_1d', 'unet_lead_specific'],
                       help='Model architecture: unet_1d (shared decoder) or unet_lead_specific (per-lead decoders)')
    parser.add_argument('--features', type=int, default=64,
                       help='Base features for UNet')
    parser.add_argument('--depth', type=int, default=4,
                       help='Depth of UNet')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout probability')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device settings
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    
    return parser.parse_args()


def get_config_from_args(args) -> Config:
    """Create config from command line arguments"""
    # Start with base config
    if args.config:
        config = Config.load(args.config)
    elif args.vm_mode:
        config = get_vm_config()
    else:
        config = get_default_config()
    
    # Override with command line args
    if args.test_mode:
        config.paths.data_dir = 'data/test_data'
    if args.data_dir:
        # Handle relative vs absolute paths
        if os.path.isabs(args.data_dir):
            config.paths.data_dir = args.data_dir
            config.paths.project_root = ''
        else:
            config.paths.data_dir = args.data_dir
    
    if args.output_dir:
        if os.path.isabs(args.output_dir):
            config.paths.output_dir = args.output_dir
        else:
            config.paths.output_dir = args.output_dir
    else:
        # Auto-generate output dir name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config.paths.output_dir = f'models/run_{timestamp}'
    
    # Model config
    config.model.features = args.features
    config.model.depth = args.depth
    config.model.dropout = args.dropout
    
    # Training config
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.patience = args.patience
    config.training.seed = args.seed
    config.training.num_workers = args.num_workers
    
    # Device config
    config.device = args.device
    config.mixed_precision = args.amp
    
    return config


def resolve_path(path: str, project_root: str) -> str:
    """Resolve path to absolute path"""
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    # Chest leads indices: V1, V2, V3, V5, V6
    chest_leads_indices = [6, 7, 8, 10, 11]
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Extract chest leads to predict
        chest_leads_targets = targets[:, chest_leads_indices]
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                chest_leads_outputs = model(inputs)
                loss = criterion(chest_leads_outputs, chest_leads_targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            chest_leads_outputs = model(inputs)
            loss = criterion(chest_leads_outputs, chest_leads_targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    chest_leads_indices = [6, 7, 8, 10, 11]
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            chest_leads_targets = targets[:, chest_leads_indices]
            chest_leads_outputs = model(inputs)
            
            loss = criterion(chest_leads_outputs, chest_leads_targets)
            running_loss += loss.item()
            
            # Reconstruct full 12-lead ECG
            # Pass targets to use stored physics leads (normalized data breaks Einthoven's law)
            reconstructed_12_leads = reconstruct_12_leads(inputs, chest_leads_outputs, targets)
            
            # Evaluate reconstruction
            metrics = evaluate_reconstruction(targets, reconstructed_12_leads)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in all_metrics], axis=0),
        'correlation': np.mean([m['correlation'] for m in all_metrics], axis=0),
        'snr': np.mean([m['snr'] for m in all_metrics], axis=0),
        'mae_overall': np.mean([m['mae_overall'] for m in all_metrics]),
        'correlation_overall': np.mean([m['correlation_overall'] for m in all_metrics]),
        'snr_overall': np.mean([m['snr_overall'] for m in all_metrics])
    }
    
    return running_loss / len(dataloader), avg_metrics


def main():
    """Main training function"""
    args = parse_args()
    config = get_config_from_args(args)
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Resolve paths
    data_dir = resolve_path(config.paths.data_dir, PROJECT_ROOT)
    output_dir = resolve_path(config.paths.output_dir, PROJECT_ROOT)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config for reproducibility
    config.save(os.path.join(output_dir, 'config.json'))
    
    # Print configuration
    print("=" * 60)
    print("ECG Lead Reconstruction Training")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {config.model.name}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Seed: {config.training.seed}")
    
    # Select device
    device = torch.device(config.get_device())
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("=" * 60)
    
    # Get dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )
    
    # Create model based on architecture selection
    print("\nCreating model...")
    model_name = args.model if hasattr(args, 'model') else config.model.name
    
    if model_name == 'unet_lead_specific':
        print("Using UNet1D with Lead-Specific Decoders")
        model = UNet1DLeadSpecific(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            features=config.model.features,
            depth=config.model.depth,
            dropout=config.model.dropout
        )
    else:
        print("Using UNet1D with Shared Decoder")
        model = UNet1D(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            features=config.model.features,
            depth=config.model.depth,
            dropout=config.model.dropout
        )
    
    model = model.to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler() if config.mixed_precision and device.type == 'cuda' else None
    
    # Training loop
    print(f"\nStarting training for {config.training.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    start_time = time.time()
    
    for epoch in range(config.training.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_metrics_history.append(metrics)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.training.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Corr: {metrics['correlation_overall']:.4f} | "
              f"MAE: {metrics['mae_overall']:.4f} | "
              f"SNR: {metrics['snr_overall']:.1f} dB | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(output_dir, 'best_model.pt'))
            patience_counter = 0
            
            # Save example reconstruction
            with torch.no_grad():
                inputs, targets = next(iter(val_loader))
                inputs = inputs.to(device)
                targets = targets.to(device)
                chest_outputs = model(inputs)
                reconstructed = reconstruct_12_leads(inputs, chest_outputs, targets)
                
                plot_reconstruction(
                    targets[0].cpu(),
                    reconstructed[0].cpu(),
                    save_path=os.path.join(output_dir, f'best_reconstruction.png')
                )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Save final model
    save_model(model, os.path.join(output_dir, 'final_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot([m['correlation_overall'] for m in val_metrics_history])
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.title('Validation Correlation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot([m['snr_overall'] for m in val_metrics_history])
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.title('Validation SNR')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_metrics': val_metrics_history,
        'best_val_loss': best_val_loss,
        'total_time_seconds': total_time,
        'epochs_trained': len(train_losses)
    }
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'best_val_loss': float(best_val_loss),
            'total_time_seconds': total_time,
            'epochs_trained': len(train_losses),
            'final_correlation': float(val_metrics_history[-1]['correlation_overall']),
            'final_mae': float(val_metrics_history[-1]['mae_overall']),
            'final_snr': float(val_metrics_history[-1]['snr_overall'])
        }
        json.dump(history_json, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - best_model.pt")
    print(f"  - final_model.pt")
    print(f"  - training_curves.png")
    print(f"  - training_history.json")
    print(f"  - config.json")
    
    # Final metrics summary
    print("\n" + "=" * 60)
    print("Final Metrics (Best Model)")
    print("=" * 60)
    best_idx = val_losses.index(best_val_loss)
    best_metrics = val_metrics_history[best_idx]
    print(f"Validation Loss: {best_val_loss:.4f}")
    print(f"Correlation: {best_metrics['correlation_overall']:.4f}")
    print(f"MAE: {best_metrics['mae_overall']:.4f}")
    print(f"SNR: {best_metrics['snr_overall']:.1f} dB")


if __name__ == "__main__":
    main()
