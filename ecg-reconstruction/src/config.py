#!/usr/bin/env python3
"""
Configuration file for ECG Lead Reconstruction project.
Centralizes all hyperparameters and paths for VM training.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class DataConfig:
    """Data-related configuration"""
    # Lead configuration
    input_leads: List[str] = field(default_factory=lambda: ['I', 'II', 'V4'])
    output_leads: List[str] = field(default_factory=lambda: ['V1', 'V2', 'V3', 'V5', 'V6'])
    all_leads: List[str] = field(default_factory=lambda: [
        'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ])
    
    # Data shape
    sampling_rate: int = 500  # Hz
    signal_length: int = 5000  # samples (10 seconds at 500 Hz)
    
    # Lead indices for extraction
    input_indices: List[int] = field(default_factory=lambda: [0, 1, 9])  # I, II, V4
    chest_leads_indices: List[int] = field(default_factory=lambda: [6, 7, 8, 10, 11])  # V1, V2, V3, V5, V6


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = 'unet_1d'
    in_channels: int = 3  # I, II, V4
    out_channels: int = 5  # V1, V2, V3, V5, V6
    features: int = 64  # Base features
    depth: int = 4  # U-Net depth
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    patience: int = 10  # Early stopping patience
    seed: int = 42
    num_workers: int = 4
    
    # Loss weighting (optional physics-informed loss)
    physics_loss_weight: float = 0.0  # Weight for physics consistency loss
    reconstruction_loss_weight: float = 1.0


@dataclass
class PathConfig:
    """Path configuration - relative to project root"""
    # These will be set relative to project root
    project_root: str = ''
    data_dir: str = 'data/test_data'  # Default to test data
    output_dir: str = 'models/run'
    figures_dir: str = 'docs/figures'
    
    def __post_init__(self):
        """Set project root if not provided"""
        if not self.project_root:
            # Get project root (parent of src directory)
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def get_data_dir(self) -> str:
        return os.path.join(self.project_root, self.data_dir)
    
    def get_output_dir(self) -> str:
        return os.path.join(self.project_root, self.output_dir)
    
    def get_figures_dir(self) -> str:
        return os.path.join(self.project_root, self.figures_dir)


@dataclass
class Config:
    """Main configuration container"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Device configuration
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    mixed_precision: bool = False  # Use AMP for faster training
    
    def get_device(self):
        """Get the appropriate device"""
        import torch
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device
    
    def save(self, path: str):
        """Save config to JSON file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'paths': {k: v for k, v in self.paths.__dict__.items() if k != 'project_root'},
            'device': self.device,
            'mixed_precision': self.mixed_precision
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        if 'data' in config_dict:
            for k, v in config_dict['data'].items():
                setattr(config.data, k, v)
        if 'model' in config_dict:
            for k, v in config_dict['model'].items():
                setattr(config.model, k, v)
        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                setattr(config.training, k, v)
        if 'paths' in config_dict:
            for k, v in config_dict['paths'].items():
                setattr(config.paths, k, v)
        if 'device' in config_dict:
            config.device = config_dict['device']
        if 'mixed_precision' in config_dict:
            config.mixed_precision = config_dict['mixed_precision']
        
        return config


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def get_vm_config() -> Config:
    """Get configuration optimized for VM training with full PTB-XL"""
    config = Config()
    
    # VM-specific paths (adjust for your VM setup)
    config.paths.data_dir = 'data/processed'  # Full PTB-XL processed data
    config.paths.output_dir = 'models/vm_run'
    
    # VM training settings
    config.training.num_workers = 8  # More workers for VM
    config.training.batch_size = 64  # Larger batch if GPU memory allows
    config.training.epochs = 100  # More epochs for full training
    
    # Enable mixed precision for faster training
    config.mixed_precision = True
    
    return config


# Convenience: create default config instance
DEFAULT_CONFIG = get_default_config()
