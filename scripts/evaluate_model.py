#!/usr/bin/env python3
import argparse
import json
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_1d import UNet1D, UNet1DLeadSpecific, UNet1DHybrid
from src.physics import reconstruct_12_leads
from src.evaluation import calculate_metrics, print_metrics_report


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    # If it's a dict with state_dict key, return state_dict; else return itself
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def build_model(model_type='unet', in_channels=3, out_channels=5, features=64, depth=4, dropout=0.2):
    if model_type == 'unet_leadspecific':
        model = UNet1DLeadSpecific(in_channels=in_channels, out_channels=out_channels, features=features, depth=depth, dropout=dropout)
    elif model_type == 'unet_hybrid':
        model = UNet1DHybrid(in_channels=in_channels, out_channels=out_channels, features=features, depth=depth, dropout=dropout)
    else:
        model = UNet1D(in_channels=in_channels, out_channels=out_channels, features=features, depth=depth, dropout=dropout)
    return model


def infer_and_save(model, device, test_input_path, test_target_path, batch_size, save_dir, save_predictions=False):
    # Load test data
    x = np.load(test_input_path)
    y = np.load(test_target_path)

    N = x.shape[0]
    seq_len = x.shape[2]

    os.makedirs(save_dir, exist_ok=True)
    save_true_path = os.path.join(save_dir, 'test_true.npy')
    save_pred_path = os.path.join(save_dir, 'test_pred.npy')

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            # Model expects [B, C, L]
            out = model(xb)
            # Reconstruct full 12 leads using targets to avoid physics mismatch on normalized data
            yb = torch.tensor(y[start:end], dtype=torch.float32, device=device)
            full_preds = reconstruct_12_leads(xb, out, targets=yb)
            preds.append(full_preds.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Optionally save predictions/true
    if save_predictions:
        np.save(save_true_path, y)
        np.save(save_pred_path, preds)
        print(f'Saved ground truth to: {save_true_path}')
        print(f'Saved predictions to: {save_pred_path}')

    # Compute metrics for chest leads and optionally overall
    metrics = calculate_metrics(y, preds)
    print_metrics_report(metrics)

    # Save aggregated metrics
    metrics_summary = {
        'mae_overall': float(metrics['mae_overall']),
        'correlation_overall': float(metrics['correlation_overall']),
        'snr_overall': float(metrics['snr_overall']),
        'mae_per_lead': metrics['mae'].tolist(),
        'corr_per_lead': metrics['correlation'].tolist(),
        'snr_per_lead': metrics['snr'].tolist()
    }
    metrics_path = os.path.join(save_dir, 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print(f'Metrics summary written to: {metrics_path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate model and optionally save predictions')
    parser.add_argument('--model_path', type=str, default='models/final_exp_baseline/best_model.pt')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'unet_leadspecific', 'unet_hybrid'])
    parser.add_argument('--test_input', type=str, default='data/processed_full/test_input.npy')
    parser.add_argument('--test_target', type=str, default='data/processed_full/test_target.npy')
    parser.add_argument('--save_dir', type=str, default='results/eval')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions and ground truth as .npy files')
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    args = parser.parse_args()

    # Device selection
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')

    # Load model
    model_state = load_checkpoint(args.model_path, device)
    model = build_model(args.model_type, features=args.features, depth=args.depth, dropout=args.dropout)
    model.to(device)
    # Attempt safe load
    try:
        model.load_state_dict(model_state)
        print('Loaded model state_dict')
    except Exception as e:
        print(f'Failed to load state_dict directly: {e}')
        try:
            model = torch.load(args.model_path, map_location=device)
            print('Loaded full model from checkpoint file')
        except Exception as e2:
            print('ERROR: Unable to load model checkpoint:', e2)
            return

    # Run inference and optionally save predictions
    infer_and_save(model, device, args.test_input, args.test_target, args.batch_size, args.save_dir, save_predictions=args.save_predictions)


if __name__ == '__main__':
    main()
