# ECG Lead Reconstruction

Reconstructing a full 12-lead ECG from reduced leads (I, II, and V4) using physics-informed deep learning.

## Overview
This project implements a hybrid system to reconstruct all 12 standard ECG leads from a reduced set:
- **Physics Component**: Leads I and II are used to deterministically calculate limb leads (III, aVR, aVL, aVF) using Einthoven's and Goldberger's laws
- **Deep Learning Component**: A 1D U-Net reconstructs the remaining precordial leads (V1, V2, V3, V5, V6) from I, II, and V4

## Quick Start

### Local Testing
```bash
# Clone and setup
cd ecg-reconstruction
pip install -r requirements.txt

# Quick test with synthetic data
python run_training.py --test_mode
```

### VM Training (Full Pipeline)
```bash
# SSH into VM
ssh user@vm-address

# Clone repo
git clone <repo-url>
cd ecg-reconstruction

# Make script executable and run
chmod +x train.sh
./train.sh              # Quick test
./train.sh --full       # Full training with PTB-XL
```

## Dataset
We use the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/), which contains 21,837 clinical 12-lead ECG recordings at 500 Hz.

## Project Structure
```
ecg-reconstruction/
├── run_training.py      # Main entry point for training
├── train.sh             # VM training script
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Package configuration
├── data/
│   ├── data_modules.py  # PyTorch Dataset/DataLoader
│   ├── get_data.py      # Data preparation from PTB-XL
│   └── test_data/       # Synthetic test data
├── src/
│   ├── config.py        # Configuration management
│   ├── physics.py       # Einthoven/Goldberger equations
│   ├── train.py         # Training loop (legacy)
│   ├── utils.py         # Utilities and metrics
│   └── models/
│       └── unet_1d.py   # 1D U-Net architecture
├── scripts/             # Utility scripts
├── notebooks/           # Jupyter notebooks
├── models/              # Saved model checkpoints
└── docs/                # Documentation and figures
```

## Training Options

```bash
# Basic usage
python run_training.py --test_mode              # Synthetic data test
python run_training.py --data_dir data/processed  # Custom data path

# Hyperparameters
python run_training.py --epochs 100 --batch_size 64 --lr 0.001

# VM-optimized (larger batch, mixed precision)
python run_training.py --vm_mode --amp

# Full customization
python run_training.py \
    --data_dir data/processed \
    --output_dir models/experiment1 \
    --epochs 100 \
    --batch_size 64 \
    --features 64 \
    --depth 4 \
    --dropout 0.2 \
    --amp
```

## Presentation

A comprehensive LaTeX Beamer presentation is available in `presentation_slides.tex`.

### Compiling the Presentation

**Prerequisites**: LaTeX installation (TeX Live, MiKTeX, or MacTeX)

**Automatic compilation**:

```bash
./compile_presentation.sh
```

**Manual compilation**:

```bash
pdflatex presentation_slides.tex
pdflatex presentation_slides.tex  # Run twice for references
```

The presentation includes:

- Project overview and motivation
- Technical methodology (physics + deep learning)
- Model architecture and training details
- Preliminary results on synthetic data
- Clinical impact and future work

### Presentation Figures

Figures are automatically generated using:

```bash
python scripts/generate_presentation_plots.py
```

This creates visualizations in `docs/figures/` including:

- ECG signal reconstructions
- Performance metrics plots
- Lead comparison examples
