#!/usr/bin/env python3
"""Evaluate classifier on input arrays and save per-sample diagnostic predictions CSV.

Usage example:
  python scripts/evaluate_classifier.py --model_path models/classifier_full/best_model.pt --input data/processed_full/test_input.npy --labels data/processed_full/labels/test_labels.npy --save_dir results/eval/baseline
"""
from pathlib import Path
import argparse
import numpy as np
import csv
import json
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from src.models.classifier_1d import Classifier1D
except Exception:
    Classifier1D = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--labels', type=str, default=None)
    p.add_argument('--labels_json', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='results/eval/classifier')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--lead_indices', type=str, default=None, help='Comma-separated lead indices to select from input if necessary e.g. "0,1,9"')
    p.add_argument('--random', action='store_true', help='Generate random predictions (no model needed)')
    return p.parse_args()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    label_names = None
    if args.labels:
        labels = np.load(args.labels)
    if args.labels_json:
        label_names = json.loads(Path(args.labels_json).read_text())

    # Load input
    X = np.load(args.input)
    # shape [N, leads, samples]
    N = X.shape[0]

    # Load model (unless using random preds)
    # Determine number of classes if possible
    if label_names is not None:
        num_classes = len(label_names)
    elif labels is not None:
        num_classes = labels.shape[1]
    else:
        raise ValueError('labels_json or labels must be provided to determine number of classes')

    if args.random:
        np.random.seed(42)
        probs = np.random.rand(N, num_classes)
    else:
        import torch
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        # Infer model's expected in_channels by reading the saved checkpoint's first conv weight
        ckpt = torch.load(model_path, map_location='cpu')
        model_in_ch = X.shape[1]
        # Try to parse encoder conv shape
        first_conv = 'encoder.0.conv.weight'
        if first_conv in ckpt:
            model_in_ch = ckpt[first_conv].shape[1]
        model = Classifier1D(in_channels=model_in_ch, num_classes=num_classes)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        batch = 64
        preds = []
        with torch.no_grad():
            for i in range(0, N, batch):
                x_np = X[i:i+batch]
                lead_indices = None
                if args.lead_indices:
                    lead_indices = [int(_) for _ in args.lead_indices.split(',')]
                # If the model expects fewer channels than provided, select desired leads
                if model_in_ch != X.shape[1]:
                    if lead_indices is None:
                        lead_indices = [0, 1, 9]
                if lead_indices is not None:
                    x_np = x_np[:, lead_indices, :]
                x = torch.from_numpy(x_np).to(device)
                logits = model(x).cpu().numpy()
                preds.append(logits)
        preds = np.vstack(preds)
        probs = sigmoid(preds)

    # Save CSV
    csv_path = save_dir / 'diagnostic_preds.csv'
    label_names = label_names or [f'label_{i}' for i in range(num_classes)]
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        header = ['sample_idx'] + [f'{l}_pred' for l in label_names]
        if labels is not None:
            header += [f'{l}_true' for l in label_names]
        writer.writerow(header)
        for i in range(N):
            row = [i] + [f'{p:.6f}' for p in probs[i].tolist()]
            if labels is not None:
                row.extend([int(t) for t in labels[i].tolist()])
            writer.writerow(row)

    print(f'Saved predictions CSV to {csv_path}')


if __name__ == '__main__':
    main()
