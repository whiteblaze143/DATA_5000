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
    def __init__(self, input_path, target_path, transform=None):
        """
        Args:
            input_path: Path to input data (.npy file)
            target_path: Path to target data (.npy file)
            transform: Optional transform to apply to both input and target
        """
        print(f"Loading input from: {input_path}")
        self.inputs = np.load(input_path)
        
        print(f"Loading target from: {target_path}")
        self.targets = np.load(target_path)
        
        # Ensure data is in float32
        self.inputs = self.inputs.astype(np.float32)
        self.targets = self.targets.astype(np.float32)
        
        self.transform = transform
        
        # Verify shapes
        assert self.inputs.shape[0] == self.targets.shape[0], "Input and target sample counts don't match"
        assert self.inputs.shape[2] == self.targets.shape[2], "Input and target sequence lengths don't match"
        
        # Print dataset info
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
        
        return input_tensor, target_tensor

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing processed data files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = ECGReconstructionDataset(
        os.path.join(data_dir, 'train_input.npy'),
        os.path.join(data_dir, 'train_target.npy')
    )
    
    val_dataset = ECGReconstructionDataset(
        os.path.join(data_dir, 'val_input.npy'),
        os.path.join(data_dir, 'val_target.npy')
    )
    
    test_dataset = ECGReconstructionDataset(
        os.path.join(data_dir, 'test_input.npy'),
        os.path.join(data_dir, 'test_target.npy')
    )
    
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