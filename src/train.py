#!/usr/bin/env python3
# filepath: src/train.py

import os
import sys
import time
import argparse
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
from src.models.unet_1d import UNet1D
from src.utils import set_seed, save_model, evaluate_reconstruction, plot_reconstruction
from src.physics import reconstruct_12_leads

def parse_args():
    parser = argparse.ArgumentParser(description='Train ECG Lead Reconstruction Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--model', type=str, default='unet_1d', choices=['unet_1d'], help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--features', type=int, default=64, help='Base features for UNet')
    parser.add_argument('--depth', type=int, default=4, help='Depth of UNet')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Extract chest leads to predict (V1, V2, V3, V5, V6)
        # Standard order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        chest_leads_indices = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
        chest_leads_targets = targets[:, chest_leads_indices]
        
        # Forward pass
        optimizer.zero_grad()
        chest_leads_outputs = model(inputs)
        
        # Compute loss
        loss = criterion(chest_leads_outputs, chest_leads_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Extract chest leads to predict (V1, V2, V3, V5, V6)
            chest_leads_indices = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
            chest_leads_targets = targets[:, chest_leads_indices]
            
            # Forward pass
            chest_leads_outputs = model(inputs)
            
            # Compute loss
            loss = criterion(chest_leads_outputs, chest_leads_targets)
            running_loss += loss.item()
            
            # Reconstruct full 12-lead ECG
            reconstructed_12_leads = reconstruct_12_leads(inputs, chest_leads_outputs)
            
            # Evaluate reconstruction
            metrics = evaluate_reconstruction(targets, reconstructed_12_leads)
            all_metrics.append(metrics)
    
    # Calculate average metrics
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
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    if args.model == 'unet_1d':
        model = UNet1D(
            in_channels=3,  # I, II, V4
            out_channels=5,  # V1, V2, V3, V5, V6
            features=args.features,
            depth=args.depth,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}")
    print(f"Number of parameters: {num_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    
    # Use SGD optimizer to avoid torch dynamo issues
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_metrics = []
    
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_metrics.append(metrics)
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Corr: {metrics['correlation_overall']:.4f} | "
              f"MAE: {metrics['mae_overall']:.4f} | "
              f"SNR: {metrics['snr_overall']:.2f} dB | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(args.output_dir, 'best_model.pt'))
            patience_counter = 0
            
            # Generate example reconstruction
            with torch.no_grad():
                inputs, targets = next(iter(val_loader))
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                chest_leads_outputs = model(inputs)
                reconstructed_12_leads = reconstruct_12_leads(inputs, chest_leads_outputs)
                
                # Plot first sample
                plot_reconstruction(
                    targets[0].cpu(),
                    reconstructed_12_leads[0].cpu(),
                    save_path=os.path.join(args.output_dir, f'example_epoch_{epoch+1}.png')
                )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Save final model
    save_model(model, os.path.join(args.output_dir, 'final_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot([m['correlation_overall'] for m in val_metrics], label='Correlation')
    plt.plot([m['mae_overall'] for m in val_metrics], label='MAE')
    plt.plot([m['snr_overall'] / 20 for m in val_metrics], label='SNR (scaled)')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.output_dir, 'metric_curves.png'), dpi=300, bbox_inches='tight')
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_metrics': val_metrics
    }
    
    # Save training history
    import pickle
    with open(os.path.join(args.output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()