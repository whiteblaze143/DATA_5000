#!/usr/bin/env python3
# filepath: scripts/test_training.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_modules import get_dataloaders
from src.models.unet_1d import UNet1D
from src.utils import set_seed

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Data
    data_dir = 'data/test_data'
    train_loader, _, _ = get_dataloaders(data_dir, batch_size=4)

    # Model
    model = UNet1D(in_channels=3, out_channels=9, features=64, depth=4)
    device = torch.device('cpu')
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()

    # Simple optimizer creation to avoid torch dynamo issues
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # Training loop
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    print("Starting training for 1 epoch...")

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Extract chest leads to predict (V1, V2, V3, V5, V6)
        chest_leads = targets[:, 6:12, :]  # V1-V6 indices in target

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, chest_leads)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        print(f"Batch {num_batches}, Loss: {loss.item():.6f}")

        if num_batches >= 3:  # Just run a few batches for testing
            break

    avg_loss = epoch_loss / num_batches
    print(f"Training completed! Average loss: {avg_loss:.6f}")

    # Save model
    os.makedirs('models/test_run', exist_ok=True)
    torch.save(model.state_dict(), 'models/test_run/model.pth')
    print("Model saved to models/test_run/model.pth")

if __name__ == "__main__":
    main()