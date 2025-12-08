# 12-Lead ECG Reconstruction from Reduced Lead Sets

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid physics-informed deep learning approach to reconstruct the full 12-lead ECG from only 3 measured leads (I, II, V4).

**DATA 5000 Final Project - Carleton University (December 2025)**

**Team 4:** Damilola Olaiya & Mithun Manivannan

---

## Key Results

| Metric | Value |
|--------|-------|
| **Overall Correlation** | r = 0.936 |
| **Physics Leads (III, aVR, aVL, aVF)** | r = 1.000 (exact) |
| **Learned Leads (V1-V3, V5-V6)** | r = 0.846 |
| **Overall MAE** | 0.012 |
| **Overall SNR** | 63.0 dB |
| **Parameters** | 17.1M |

### Per-Lead Performance

| Lead | Correlation | MAE | SNR (dB) |
|------|-------------|-----|----------|
| V1 | 0.818 | 0.030 | 19.5 |
| V2 | 0.827 | 0.030 | 19.3 |
| V3 | 0.860 | 0.027 | 20.0 |
| V5 | **0.891** | 0.026 | 20.3 |
| V6 | 0.836 | 0.033 | 18.3 |

---

## Approach

Our hybrid approach combines **physics** and **deep learning**:

### Physics Component (Zero Parameters, Exact)
Exploits Einthoven's law and Goldberger's equations to compute 4 limb leads exactly:
- **Lead III** = Lead II − Lead I
- **aVR** = −(Lead I + Lead II) / 2
- **aVL** = Lead I − (Lead II / 2)
- **aVF** = Lead II − (Lead I / 2)

### Deep Learning Component (17.1M Parameters)
1D U-Net reconstructs chest leads (V1, V2, V3, V5, V6) from input leads (I, II, V4).

```
+-------------------------------------------------------------+
|                 INPUT: 3 Measured Leads                     |
|                    Lead I, Lead II, Lead V4                 |
+-----------------------------+-------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
+-------------------+             +---------------------------+
|  PHYSICS MODULE   |             |      1D U-Net             |
|  (Exact Formulas) |             |  (Deep Learning)          |
|                   |             |                           |
|  III, aVR,        |             |  V1, V2, V3, V5, V6       |
|  aVL, aVF         |             |                           |
|                   |             |                           |
|  r = 1.000        |             |  r = 0.846                |
+---------+---------+             +-------------+-------------+
          |                                     |
          +------------------+------------------+
                             |
                             v
+-------------------------------------------------------------+
|              OUTPUT: Complete 12-Lead ECG                   |
+-------------------------------------------------------------+
```

---

## Key Findings

### 1. Physics Guarantees Work
4 of 12 leads reconstructed **perfectly** (r = 1.000) with zero learned parameters.

### 2. Shared Decoder Wins
Counter-intuitively, a shared decoder (17.1M params) **outperforms** lead-specific decoders (40.8M params):
- Cohen's d = 0.92 (large effect)
- p < 0.001

### 3. Information Bottleneck
Performance bounded by ground-truth inter-lead correlations:
- V5 best (r = 0.891): adjacent to input V4 (correlation 0.79)
- V1 hardest (r = 0.818): distant from V4 (correlation 0.49)

**Implication:** Input lead selection matters more than model architecture.

---

## Quick Start

### Installation

```bash
git clone https://github.com/whiteblaze143/DATA_5000.git
cd DATA_5000
pip install -r requirements.txt
```

### Test with Synthetic Data

```bash
python run_training.py --test_mode
```

### Train on PTB-XL

```bash
# Download PTB-XL data first
python scripts/download_ptb_xl.py

# Full training
python run_training.py \
    --data_dir data/ptb_xl \
    --output_dir models/my_experiment \
    --epochs 150 \
    --batch_size 128 \
    --lr 3e-4
```

### Evaluate Trained Model

```bash
python scripts/quick_eval.py --model_path models/final_exp_baseline/best_model.pt
```

---

## Dataset

We use the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/):

| Attribute | Value |
|-----------|-------|
| Total Records | 21,837 |
| Unique Patients | 18,885 |
| Duration | 10 seconds |
| Sampling Rate | 500 Hz |
| Leads | 12 (standard clinical) |

### Data Splits (Patient-Wise)

| Split | Records | Purpose |
|-------|---------|---------|
| Train | 14,363 (70%) | Model training |
| Validation | 1,914 (15%) | Hyperparameter tuning |
| Test | 1,932 (15%) | Final evaluation |

**Important:** We use patient-wise splits to prevent data leakage.

---

## Project Structure

```
DATA_5000/
├── run_training.py          # Main training script
├── train.sh                  # Shell script for VM training
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project configuration
│
├── src/                      # Source code
│   ├── config.py            # Configuration management
│   ├── physics.py           # Einthoven/Goldberger equations
│   ├── train.py             # Training loop
│   ├── evaluation.py        # Metrics (MAE, correlation, SNR)
│   ├── utils.py             # Utilities
│   └── models/
│       ├── unet_1d.py       # 1D U-Net architecture
│       └── baseline.py      # Baseline models
│
├── data/                     # Data handling
│   ├── data_modules.py      # PyTorch Dataset/DataLoader
│   ├── get_data.py          # Data preparation
│   ├── generate_test_data.py # Synthetic data for testing
│   └── README_DATA.md       # Data documentation
│
├── scripts/                  # Utility scripts
│   ├── download_ptb_xl.py   # Download PTB-XL dataset
│   └── quick_eval.py        # Quick model evaluation
│
├── models/                   # Trained models
│   └── final_exp_baseline/
│       ├── best_model.pt    # Best trained model weights
│       └── test_results.json # Evaluation metrics
│
└── docs/                     # Documentation
    ├── PROJECT_REPORT.tex   # LaTeX report source
    ├── PROJECT_REPORT.pdf   # Compiled report
    ├── references.bib       # Bibliography
    └── figures/             # Figures for report
```

---

## Training Options

```bash
python run_training.py \
    --data_dir data/ptb_xl \      # Data directory
    --output_dir models/exp1 \    # Output directory
    --epochs 150 \                # Training epochs
    --batch_size 128 \            # Batch size
    --lr 3e-4 \                   # Learning rate
    --features 64 \               # Base feature count
    --depth 4 \                   # U-Net depth
    --dropout 0.2 \               # Dropout rate
    --amp                         # Mixed precision training
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/PROJECT_REPORT.pdf`](docs/PROJECT_REPORT.pdf) | Full academic report |

### Compiling LaTeX Report

```bash
cd docs && pdflatex PROJECT_REPORT.tex && bibtex PROJECT_REPORT && pdflatex PROJECT_REPORT.tex && pdflatex PROJECT_REPORT.tex
```

---

## References

1. Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." Scientific Data, 2020.
2. Mason, F., et al. "AI-enhanced reconstruction of the 12-lead ECG via 3-leads." npj Digital Medicine, 2024.
3. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.

---

## Citation

```bibtex
@misc{olaiya2025ecg,
  author = {Olaiya, Damilola and Manivannan, Mithun},
  title = {12-Lead ECG Reconstruction from Reduced Lead Sets: A Hybrid Physics-Informed Deep Learning Approach},
  year = {2025},
  institution = {Carleton University},
  howpublished = {\url{https://github.com/whiteblaze143/DATA_5000}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- DATA 5000 course instructors and TAs at Carleton University
- PhysioNet for providing open access to PTB-XL dataset
- PyTorch team for the deep learning framework

