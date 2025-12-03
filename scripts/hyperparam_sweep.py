#!/usr/bin/env python3
"""
Hyperparameter tuning script for lead-specific model.
Tests different learning rates to find optimal configuration.

Usage:
    python scripts/hyperparam_sweep.py --lrs 1e-4 3e-4 1e-3 --epochs 50
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.train import Trainer
from src.models.unet_1d import UNet1D, UNet1DLeadSpecific
from data.data_modules import ECGReconstructionDataset


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_with_lr(model_type: str, lr: float, epochs: int, output_dir: Path, seed: int = 42):
    """Train model with specific learning rate."""
    
    set_seed(seed)
    
    lr_str = f"{lr:.0e}".replace("e-0", "e-")
    run_name = f"{model_type}_lr{lr_str}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training {model_type} with lr={lr}")
    print(f"{'='*60}")
    
    # Load data
    data_dir = Path("data/processed")
    train_dataset = ECGReconstructionDataset(
        str(data_dir / "train_input.npy"),
        str(data_dir / "train_target.npy"),
        verbose=False
    )
    val_dataset = ECGReconstructionDataset(
        str(data_dir / "val_input.npy"),
        str(data_dir / "val_target.npy"),
        verbose=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "shared":
        model = UNet1D(in_channels=3, out_channels=5, dropout=0.2)
    else:
        model = UNet1DLeadSpecific(in_channels=3, dropout=0.2)
    
    model = model.to(device)
    
    config = Config()
    config.learning_rate = lr
    config.epochs = epochs
    config.output_dir = str(run_dir)
    
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "model_type": model_type,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": 64,
            "seed": seed,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    trainer = Trainer(model, config, device)
    start_time = time.time()
    history = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time
    
    torch.save(model.state_dict(), run_dir / "model.pt")
    
    history["training_time_seconds"] = training_time
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    best_epoch = np.argmin(history["val_loss"]) + 1
    return {
        "learning_rate": lr,
        "model_type": model_type,
        "best_epoch": best_epoch,
        "best_val_loss": min(history["val_loss"]),
        "final_val_loss": history["val_loss"][-1],
        "final_val_corr": history.get("val_correlation", [None])[-1],
        "training_time": training_time
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for ECG reconstruction")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3],
                        help="Learning rates to test")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs per run (use fewer for quick sweep)")
    parser.add_argument("--model", type=str, default="lead_specific", 
                        choices=["shared", "lead_specific", "both"],
                        help="Model to tune")
    parser.add_argument("--output", type=str, default="models/hyperparam_sweep",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Hyperparameter sweep configuration:")
    print(f"  Learning rates: {args.lrs}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model: {args.model}")
    
    model_types = ["shared", "lead_specific"] if args.model == "both" else [args.model]
    
    results = []
    
    for model_type in model_types:
        for lr in args.lrs:
            result = train_with_lr(model_type, lr, args.epochs, output_dir, args.seed)
            results.append(result)
            print(f"✓ {model_type} lr={lr}: Val loss={result['best_val_loss']:.6f}")
    
    # Find best configuration
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("="*60)
    
    for model_type in model_types:
        model_results = [r for r in results if r["model_type"] == model_type]
        best = min(model_results, key=lambda x: x["best_val_loss"])
        
        print(f"\n{model_type.upper()}:")
        print(f"  Best LR: {best['learning_rate']}")
        print(f"  Best Val Loss: {best['best_val_loss']:.6f}")
        print(f"  Best Epoch: {best['best_epoch']}")
        
        print("\n  All results:")
        for r in sorted(model_results, key=lambda x: x["best_val_loss"]):
            print(f"    lr={r['learning_rate']}: loss={r['best_val_loss']:.6f} (ep {r['best_epoch']})")
    
    # Save summary
    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
