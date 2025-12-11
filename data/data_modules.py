import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ECGReconstructionDataset(Dataset):
    """
    Dataset for ECG lead reconstruction
    - Inputs: leads I, II, V4
    - Targets: all 12 leads
    """
    def __init__(self, input_path, target_path, labels_path=None, transform=None, verbose=True):
        """
        Args:
            input_path: Path to input data (.npy file)
            target_path: Path to target data (.npy file)
            transform: Optional transform to apply to both input and target
            verbose: Print loading info (default True)
        """
        if verbose:
            print(f"Loading input from: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file not found: {target_path}")
        
        self.inputs = np.load(input_path)
        
        if verbose:
            print(f"Loading target from: {target_path}")
        self.targets = np.load(target_path)
        
        # Ensure data is in float32
        self.inputs = self.inputs.astype(np.float32)
        self.targets = self.targets.astype(np.float32)
        
        self.transform = transform
        # Load labels if provided
        self.labels = None
        if labels_path is not None:
            if os.path.exists(labels_path):
                self.labels = np.load(labels_path)
        
        # Verify shapes
        assert self.inputs.shape[0] == self.targets.shape[0], \
            f"Input and target sample counts don't match: {self.inputs.shape[0]} vs {self.targets.shape[0]}"
        assert self.inputs.shape[2] == self.targets.shape[2], \
            f"Input and target sequence lengths don't match: {self.inputs.shape[2]} vs {self.targets.shape[2]}"
        
        # Print dataset info
        if verbose:
            print("[OK] Dataset loaded successfully:")
            print(f"   - Number of samples: {len(self)}")
            print(f"   - Input shape: {self.inputs.shape}")
            print(f"   - Target shape: {self.targets.shape}")
            print(f"   - Data type: {torch.from_numpy(self.inputs[0]).dtype}")
            print(f"   - Memory (input): {self.inputs.nbytes / 1e9:.2f} GB")
            print(f"   - Memory (target): {self.targets.nbytes / 1e9:.2f} GB")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_signal = self.inputs[idx]
        target_signal = self.targets[idx]
        
        # Apply transforms if any
        if self.transform:
            input_signal, target_signal = self.transform(input_signal, target_signal)
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_signal)
        target_tensor = torch.from_numpy(target_signal)
        
        if self.labels is not None:
            label_tensor = torch.from_numpy(self.labels[idx].astype('float32'))
            return input_tensor, target_tensor, label_tensor
        return input_tensor, target_tensor

def get_dataloaders(data_dir, batch_size=32, num_workers=4, verbose=True, labels_dir=None):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing processed data files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        verbose: Print loading information
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Verify data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please run data preparation first, or use --test_mode for synthetic data."
        )
    
    # Check for required files
    required_files = [
        'train_input.npy', 'train_target.npy',
        'val_input.npy', 'val_target.npy',
        'test_input.npy', 'test_target.npy'
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing data files in {data_dir}: {missing}\n"
            f"Please run data preparation first."
        )
    
    if verbose:
        print(f"\nLoading data from: {data_dir}")
    
    # Create datasets
    train_labels_path = None
    val_labels_path = None
    test_labels_path = None
    if labels_dir is not None:
        train_labels_path = os.path.join(labels_dir, 'train_labels.npy')
        val_labels_path = os.path.join(labels_dir, 'val_labels.npy')
        test_labels_path = os.path.join(labels_dir, 'test_labels.npy')
    train_dataset = ECGReconstructionDataset(
        os.path.join(data_dir, 'train_input.npy'),
        os.path.join(data_dir, 'train_target.npy'),
        labels_path=train_labels_path,
        verbose=verbose
    )
    
    val_dataset = ECGReconstructionDataset(
        os.path.join(data_dir, 'val_input.npy'),
        os.path.join(data_dir, 'val_target.npy'),
        labels_path=val_labels_path,
        verbose=verbose
    )
    
    test_dataset = ECGReconstructionDataset(
        os.path.join(data_dir, 'test_input.npy'),
        os.path.join(data_dir, 'test_target.npy'),
        labels_path=test_labels_path,
        verbose=verbose
    )
    
    # Determine number of workers (0 on Windows for multiprocessing issues)
    if os.name == 'nt' and num_workers > 0:
        import warnings
        warnings.warn("Reducing num_workers to 0 on Windows to avoid multiprocessing issues")
        num_workers = 0
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader