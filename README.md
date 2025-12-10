# 12-Lead ECG Reconstruction from 3 Leads

**DATA 5000 Capstone Project** | Carleton University | December 2025

Damilola Olaiya & Mithun Manivannan

---

## Overview

We reconstruct the full 12-lead ECG from 3 measured leads (I, II, V4) using a hybrid approach:

- **Physics module**: Derives 4 limb leads (III, aVR, aVL, aVF) exactly via Einthoven's and Goldberger's equations
- **1D U-Net**: Learns to reconstruct 5 precordial leads (V1, V2, V3, V5, V6)

Evaluated on PTB-XL (21,837 recordings, 18,885 patients) with strict patient-wise splits.

## Results

| Component | Correlation | Notes |
|-----------|-------------|-------|
| Physics leads (III, aVR, aVL, aVF) | r = 1.000 | Exact by construction |
| Learned leads (V1-V6 excl. V4) | r = 0.846 | V5 best (0.891), V1 hardest (0.818) |
| **Overall** | **r = 0.936** | 17.1M parameters |

**Key finding**: Shared decoder (17.1M params) outperforms lead-specific decoders (40.8M params) with Cohen's d = 0.92. The bottleneck is input information content, not model capacity—V5 reconstructs well because it correlates with input V4 (r=0.79 in ground truth), while V1 is harder (r=0.49).

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Test with synthetic data
python run_training.py --test_mode

# Train on PTB-XL
python scripts/download_ptb_xl.py
python run_training.py --data_dir data/ptb_xl --epochs 150 --batch_size 128 --lr 3e-4
```

## Repository Structure

```
├── run_training.py          # Training entry point
├── src/
│   ├── models/unet_1d.py    # 1D U-Net architecture
│   ├── physics.py           # Einthoven/Goldberger equations
│   ├── train.py             # Training loop
│   └── evaluation.py        # Metrics (correlation, MAE, SNR)
├── data/
│   ├── data_modules.py      # PyTorch DataLoader
│   └── get_data.py          # Data preprocessing
├── scripts/
│   ├── download_ptb_xl.py   # Dataset download
│   ├── quick_eval.py        # Evaluation script
│   └── clinical_features_eval.py  # ECGGenEval-style clinical metrics
├── models/                  # Trained model checkpoints
└── docs/
    ├── PROJECT_REPORT.tex   # Full report (ACM format)
    └── figures/             # Figures for report
```

## Method

**Physics (zero parameters)**:
- Lead III = II − I
- aVR = −(I + II)/2
- aVL = I − II/2  
- aVF = II − I/2

**Deep Learning**: 1D U-Net with 4-level encoder-decoder, 64 base features, skip connections. Trained with MSE loss, AdamW optimizer (lr=3×10⁻⁴), patient-wise train/val/test splits (70/15/15%).

## Clinical Feature Evaluation

Script for ECGGenEval-style clinical feature extraction (QRS duration, PR/QT intervals, heart rate). Requires actual model predictions:

```bash
python scripts/clinical_features_eval.py --y_true path/to/ground_truth.npy --y_pred path/to/predictions.npy
```

## Citation

```bibtex
@misc{olaiya2025ecg,
  author       = {Olaiya, Damilola and Manivannan, Mithun},
  title        = {12-Lead ECG Reconstruction from Reduced Lead Sets},
  year         = {2025},
  institution  = {Carleton University},
  url          = {https://github.com/whiteblaze143/DATA_5000}
}
```

## References

- Wagner et al. (2020). PTB-XL dataset. *Scientific Data*.
- Chen et al. (2024). ECGGenEval: Multi-level ECG evaluation framework.
- Ronneberger et al. (2015). U-Net. *MICCAI*.

## License

MIT
