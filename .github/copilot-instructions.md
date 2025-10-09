# ECG Lead Reconstruction - AI Agent Instructions

## Architecture Overview

**Hybrid Physics-Informed Deep Learning Approach:**
- **Physics Component**: Deterministic reconstruction of limb leads (III, aVR, aVL, aVF) from leads I, II using Einthoven's and Goldberger's laws
- **Deep Learning Component**: 1D U-Net reconstructs chest leads (V1, V2, V3, V5, V6) from input leads (I, II, V4)
- **Data Flow**: 3-lead input → 9-lead physics + 5-lead DL → 12-lead output

**Key Files:**
- `src/physics.py`: Physics-based lead calculations (Einthoven's law: III = II - I)
- `src/models/unet_1d.py`: 1D U-Net with Conv1D blocks, downsampling/upsampling for temporal signals
- `src/train.py`: Training loop with physics-informed loss weighting

## Development Workflows

**Data Preparation:**
```bash
# Prepare PTB-XL data (requires ~50GB download)
./scripts/prepare_data.sh /path/to/ptb-xl /path/to/output
# Creates: train/val/test_input.npy, train/val/test_target.npy
```

**Training:**
```bash
# Train baseline model
./scripts/train_baseline.sh
# Uses: data/processed/, outputs to results/baseline/
```

**Presentation:**
```bash
# Compile LaTeX slides (requires pdflatex)
./compile_presentation.sh
# Generates: presentation_slides.pdf with embedded figures
```

**EDA & Visualization:**
```bash
# Generate presentation plots
python scripts/generate_presentation_plots.py
# Creates: docs/figures/ (sample ECGs, correlations, statistics)
```

## Code Patterns & Conventions

**Data Handling:**
- **Shapes**: `[batch, leads, samples]` (e.g., `[32, 12, 5000]` for 12-lead ECG)
- **Normalization**: All signals normalized to [0,1] range
- **Lead Order**: `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]`
- **Input Leads**: Only I, II, V4 (indices 0, 1, 9 in full 12-lead array)

**Model Architecture:**
- **1D Convolutions**: Use `nn.Conv1d` with `kernel_size=3`, `padding=1`
- **U-Net Structure**: Encoder-decoder with skip connections, maxpool/convtranspose for down/up-sampling
- **Dropout**: 0.2 default in all conv blocks for regularization

**Evaluation Metrics:**
- **Primary**: MAE (Mean Absolute Error), Pearson correlation, SNR (Signal-to-Noise Ratio)
- **Physics Leads**: Perfect reconstruction expected (MAE ≈ 0)
- **DL Leads**: Focus on correlation > 0.9, MAE < 0.05 for clinical viability

**Import Structure:**
```python
# Always add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Dependencies & Environment

**Core Stack:**
- **PyTorch 2.1.0**: Deep learning framework
- **NumPy 1.24.3**: Numerical computing
- **WFDB 4.1.0**: ECG data format handling
- **Matplotlib 3.7.2**: Visualization
- **SciPy 1.10.1**: Signal processing

**Pinned Versions:** Use exact versions from `requirements.txt` - no upgrades without testing.

## Common Pitfalls

**Data Shape Confusion:**
- Input: `[batch, 3, samples]` (I, II, V4 only)
- Target: `[batch, 12, samples]` (all leads)
- Never mix lead indices without explicit mapping

**Physics vs DL Leads:**
- Limb leads (III, aVR, aVL, aVF): Computed exactly via physics
- Chest leads (V1-V6): Learned via deep learning
- Different evaluation expectations for each group

**Memory Usage:**
- ECG data is memory-intensive (~100MB per batch)
- Use `float32` consistently, monitor GPU memory
- Batch size 32 is typical maximum

## Testing & Validation

**Reproducibility:**
- Always call `set_seed(42)` before training
- Use deterministic PyTorch settings
- Log all hyperparameters and random seeds

**Model Saving:**
- Use `torch.save(model.state_dict(), path)` (state_dict only)
- Save training curves and validation metrics
- Include model config for reproducibility