# ECG Lead Reconstruction 
Reconstructing a full 12-lead ECG from reduced leads (I, II, and V4). 
## Overview 
This project implements a system to reconstruct all 12 standard ECG leads from a reduced set: - Leads I and II are used to deterministically calculate limb leads (III, aVR, aVL, aVF) using Einthoven's and Goldberger's laws - A deep learning model (1D U-Net) is used to reconstruct the remaining precordial leads (V1, V2, V3, V5, V6) from I, II, and V4 

## Dataset 
We use the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/), which contains 21,837 clinical 12-lead ECG recordings. 

## Getting Started 
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare data: `./scripts/prepare_data.sh /path/to/ptb-xl /path/to/output`
4. Run baseline: `./scripts/train_baseline.sh`
