"""Dataset utilities for SNOMED / diagnostic classification evaluation.

Provides DiagnosisDataset and convenience loaders for train/val/test.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiagnosisDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        # inputs: numpy array [N, leads, samples]
        # labels: numpy array [N, L]
        self.inputs = inputs.astype('float32')
        self.labels = labels.astype('uint8')
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        # PyTorch expects shape [channels, length]
        return torch.from_numpy(x), torch.from_numpy(y.astype('float32'))


def load_npys(data_dir, split='train'):
    data_dir = Path(data_dir)
    inp = np.load(data_dir / f'{split}_input.npy')
    labels = np.load(data_dir / 'labels' / f'{split}_labels.npy')
    return inp, labels


def get_diagnosis_loaders(data_dir, batch_size=32, num_workers=2):
    train_inp, train_labels = load_npys(data_dir, 'train')
    val_inp, val_labels = load_npys(data_dir, 'val')
    test_inp, test_labels = load_npys(data_dir, 'test')

    train_ds = DiagnosisDataset(train_inp, train_labels)
    val_ds = DiagnosisDataset(val_inp, val_labels)
    test_ds = DiagnosisDataset(test_inp, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
