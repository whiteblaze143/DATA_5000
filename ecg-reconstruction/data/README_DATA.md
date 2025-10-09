# Data Processing Instructions

This project uses the PTB-XL dataset, a large collection of 12-lead ECGs.

## Data Download
1. Download PTB-XL from PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/
2. Extract to a local directory

## Data Preprocessing
Run the data preparation script:
```bash
./scripts/prepare_data.sh /path/to/ptb-xl /path/to/output
```

This will generate:
- `train_input.npy`: Training input data (leads I, II, V4)
- `train_target.npy`: Training target data (all 12 leads)
- `val_input.npy`: Validation input data
- `val_target.npy`: Validation target data
- `test_input.npy`: Test input data
- `test_target.npy`: Test target data

## Data Format
- Input shape: (N, 3, 5000) - Leads [I, II, V4]
- Target shape: (N, 12, 5000) - All 12 leads [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
- Sampling rate: 500 Hz
- Duration: 10 seconds (5000 samples)
- Values normalized to range [0, 1]

## Patient-wise Split
Data is split by patient ID to prevent leakage:
- Training: ~70% of patients
- Validation: ~10% of patients
- Test: ~20% of patients