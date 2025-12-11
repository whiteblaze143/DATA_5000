#!/usr/bin/env python3
"""Train a multi-label classifier for SNOMED/SCP codes.

Usage:
  python scripts/train_classifier.py --data_dir data/processed_full --labels_dir data/processed_full/labels --output models/classifier_full --epochs 10 --batch_size 64
"""
import argparse
from pathlib import Path
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.diagnosis_modules import get_diagnosis_loaders
from src.models.classifier_1d import Classifier1D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/processed_full')
    p.add_argument('--labels_dir', type=str, default='data/processed_full/labels')
    p.add_argument('--output', type=str, default='models/classifier_full')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--resume', action='store_true', help='Resume training from last checkpoint in output dir')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from (optional)')
    return p.parse_args()


def compute_auc(y_true, y_score):
    # y_true, y_score: 1D numpy arrays
    # Handle edge cases
    mask = (~np.isnan(y_score)) & (~np.isnan(y_true))
    y_true = y_true[mask]
    y_score = y_score[mask]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return float('nan')
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]
    cum_TP = np.cumsum(y_true_sorted)
    cum_FP = np.cumsum(1 - y_true_sorted)
    uniq_scores, idx = np.unique(y_score_sorted, return_index=True)
    tpr = np.concatenate(([0.0], cum_TP[idx - 1] / P if P > 0 else np.zeros_like(idx), [1.0])) if P > 0 else np.array([0.0,1.0])
    fpr = np.concatenate(([0.0], cum_FP[idx - 1] / N if N > 0 else np.zeros_like(idx), [1.0])) if N > 0 else np.array([0.0,1.0])
    order = np.argsort(fpr)
    fpr_s = fpr[order]
    tpr_s = tpr[order]
    auc_val = np.trapz(tpr_s, fpr_s)
    return auc_val


def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_logits = []
    all_true = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            all_logits.append(logits)
            all_true.append(y.numpy())
    all_logits = np.vstack(all_logits)
    all_true = np.vstack(all_true)
    aucs = []
    for c in range(num_classes):
        aucs.append(compute_auc(all_true[:, c], all_logits[:, c]))
    return aucs, all_logits, all_true


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_diagnosis_loaders(args.data_dir, batch_size=args.batch_size)
    # Determine number of classes from labels dir
    labels_json = Path(args.labels_dir) / 'labels.json'
    if labels_json.exists():
        labels = json.loads(labels_json.read_text())
    else:
        raise FileNotFoundError('labels.json not found in labels_dir')
    num_classes = len(labels)

    # infer input channels from a batch
    sample_batch, _ = next(iter(train_loader))
    in_ch = sample_batch.shape[1]
    model = Classifier1D(in_channels=in_ch, num_classes=num_classes)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mean_auc = -1
    metrics = {}
    # Resume support: load checkpoint if requested
    start_epoch = 1
    checkpoint_path = None
    if args.resume:
        # prefer explicit --checkpoint, else output_dir/last_checkpoint.pth
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = output_dir / 'last_checkpoint.pth'
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            ckpt = torch.load(str(checkpoint_path), map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            try:
                optim.load_state_dict(ckpt['optim_state_dict'])
            except Exception:
                print('Unable to load optimizer state (optimizer structure may differ); continuing with fresh optimizer')
            best_mean_auc = ckpt.get('best_mean_auc', best_mean_auc)
            # restore history if present
            prev_history = ckpt.get('history', None)
            if prev_history is not None:
                # ensure metrics variable contains previous
                try:
                    metrics.update(prev_history.get('metrics', {}))
                except Exception:
                    pass
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resuming at epoch {start_epoch}")
        else:
            print('Checkpoint not found; starting from scratch')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        val_aucs, _, _ = evaluate_model(model, val_loader, device, num_classes)
        mean_auc = float(np.nanmean(val_aucs))
        print(f"Epoch {epoch}: mean_val_auc = {mean_auc:.4f}")
        metrics[f'epoch_{epoch}'] = {
            'mean_auc': mean_auc,
            'per_label_auc': val_aucs
        }
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            torch.save(model.state_dict(), output_dir / 'last_model.pt')
        # Save a checkpoint for resume
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'best_mean_auc': best_mean_auc,
            'history': metrics
        }
        torch.save(ckpt, output_dir / 'last_checkpoint.pth')

    json.dump({'best_mean_auc': best_mean_auc}, open(output_dir / 'metrics.json', 'w'))
    print('Training complete; saved model and metrics')


if __name__ == '__main__':
    main()
