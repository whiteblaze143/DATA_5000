#!/usr/bin/env python3
import os, json, random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project-relative imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.unet_1d import UNet1D, UNet1DLeadSpecific
from src.physics import reconstruct_12_leads
from src.evaluation import calculate_metrics

LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
CATS = ['Input','Input','Physics','Physics','Physics','Physics','DL','DL','DL','Input','DL','DL']
CHEST_IDX = [6,7,8,10,11]

root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root, 'data', 'processed_full')
shared_dir = os.path.join(root, 'models', 'overnight_full')
leadsp_dir = os.path.join(root, 'models', 'lead_specific_v1')

# Load data
test_input = np.load(os.path.join(data_dir, 'test_input.npy'))
test_target = np.load(os.path.join(data_dir, 'test_target.npy'))
inputs_t = torch.from_numpy(test_input)
targets_t = torch.from_numpy(test_target)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate models
shared = UNet1D(in_channels=3, out_channels=5, features=64, depth=4, dropout=0.2).to(device)
leadspec = UNet1DLeadSpecific(in_channels=3, out_channels=5, features=64, depth=4, dropout=0.2).to(device)

# Load weights
sm = os.path.join(shared_dir, 'best_model.pt')
lm = os.path.join(leadsp_dir, 'best_model.pt')
if not os.path.exists(sm):
    raise FileNotFoundError(f"Missing shared best_model.pt at {sm}")
if not os.path.exists(lm):
    raise FileNotFoundError(f"Missing lead-specific best_model.pt at {lm}")
shared.load_state_dict(torch.load(sm, map_location=device))
leadspec.load_state_dict(torch.load(lm, map_location=device))

shared.eval(); leadspec.eval()
with torch.no_grad():
    inp = inputs_t.to(device)
    tgt = targets_t.to(device)
    out_shared = shared(inp)
    out_leadspec = leadspec(inp)
    rec_shared = reconstruct_12_leads(inp, out_shared, tgt)
    rec_leadspec = reconstruct_12_leads(inp, out_leadspec, tgt)

# Metrics
ms = calculate_metrics(tgt.cpu().numpy(), rec_shared.cpu().numpy())
ml = calculate_metrics(tgt.cpu().numpy(), rec_leadspec.cpu().numpy())

# Build comparison table
comp = []
for i, name in enumerate(LEADS):
    row = {
        'lead': name,
        'category': CATS[i],
        'shared_corr': float(ms['correlation'][i]),
        'shared_mae': float(ms['mae'][i]),
        'shared_snr': float(ms['snr'][i]),
        'leadspec_corr': float(ml['correlation'][i]),
        'leadspec_mae': float(ml['mae'][i]),
        'leadspec_snr': float(ml['snr'][i]),
        'delta_corr': float(ml['correlation'][i] - ms['correlation'][i])
    }
    comp.append(row)

# Convert numpy arrays to lists for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj

out_json = os.path.join(root, 'models', 'model_comparison_metrics.json')
with open(out_json, 'w') as f:
    data = {'comparison': comp, 'shared_overall': convert_to_serializable(ms), 'leadspec_overall': convert_to_serializable(ml)}
    json.dump(data, f, indent=2)

# Markdown report
out_md = os.path.join(root, 'models', 'model_comparison_report.md')
with open(out_md, 'w') as f:
    f.write('# Model Comparison: Shared vs Lead-Specific\n\n')
    f.write('| Lead | Cat | Shared Corr | LeadSpec Corr | ΔCorr | Shared MAE | LeadSpec MAE |\n')
    f.write('|---|---|---:|---:|---:|---:|---:|\n')
    for r in comp:
        f.write(f"| {r['lead']} | {r['category']} | {r['shared_corr']:.4f} | {r['leadspec_corr']:.4f} | {r['delta_corr']:.4f} | {r['shared_mae']:.4f} | {r['leadspec_mae']:.4f} |\n")
    # Summary of DL leads
    s_dl = np.mean([ms['correlation'][i] for i in CHEST_IDX])
    l_dl = np.mean([ml['correlation'][i] for i in CHEST_IDX])
    f.write('\n')
    f.write(f"Average DL leads corr (Shared): {s_dl:.4f}\n")
    f.write(f"Average DL leads corr (LeadSpec): {l_dl:.4f}\n")
    f.write(f"Delta: {l_dl - s_dl:.4f}\n")

# Plot reconstructions for two examples
ex_indices = [0, random.randint(0, inputs_t.shape[0]-1)]
for k, idx in enumerate(ex_indices, start=1):
    true = targets_t[idx].numpy()
    sh = rec_shared[idx].cpu().numpy()
    ls = rec_leadspec[idx].cpu().numpy()
    plt.figure(figsize=(12,8))
    plot_leads = [6,7,8,10,11]
    titles = ['V1','V2','V3','V5','V6']
    for j, li in enumerate(plot_leads, start=1):
        plt.subplot(5,1,j)
        plt.plot(true[li], label='True', lw=1.5)
        plt.plot(sh[li], label='Shared', lw=1)
        plt.plot(ls[li], label='LeadSpec', lw=1)
        plt.title(titles[j-1])
        plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(root, 'models', f'reconstruction_shared_vs_leadspecific_example{k}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

print(out_json)
print(out_md)
