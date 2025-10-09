import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    """1D convolutional block with batch norm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DownBlock1D(nn.Module):
    """Downsampling block for U-Net"""
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, dropout=dropout),
            ConvBlock1D(out_channels, out_channels, dropout=dropout)
        )
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x

class UpBlock1D(nn.Module):
    """Upsampling block for U-Net"""
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, dropout=dropout),
            ConvBlock1D(out_channels, out_channels, dropout=dropout)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle odd-length sequences in skip connections
        diff = skip.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, [diff // 2, diff - diff // 2])
        elif diff < 0:
            skip = F.pad(skip, [-diff // 2, -diff + diff // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x

class UNet1D(nn.Module):
    """
    1D U-Net for ECG lead reconstruction
    """
    def __init__(self, in_channels=3, out_channels=5, features=64, depth=4, dropout=0.2):
        """
        Args:
            in_channels: Number of input channels (default: 3 for leads I, II, V4)
            out_channels: Number of output channels (default: 5 for V1, V2, V3, V5, V6)
            features: Number of base features (doubled at each downsampling level)
            depth: Depth of U-Net (number of downsampling operations)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.input_conv = nn.Sequential(
            ConvBlock1D(in_channels, features, dropout=dropout),
            ConvBlock1D(features, features, dropout=dropout)
        )
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(
                DownBlock1D(
                    features * (2**i),
                    features * (2**(i+1)),
                    dropout=dropout
                )
            )
        
        # Middle convolution
        mid_features = features * (2**depth)
        self.middle_conv = nn.Sequential(
            ConvBlock1D(mid_features, mid_features, dropout=dropout),
            ConvBlock1D(mid_features, mid_features, dropout=dropout)
        )
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(
                UpBlock1D(
                    features * (2**(depth-i)),
                    features * (2**(depth-i-1)),
                    dropout=dropout
                )
            )
        
        # Final convolution
        self.output_conv = nn.Conv1d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            
        Returns:
            Output tensor [batch, out_channels, seq_len]
        """
        # Initial convolution
        x = self.input_conv(x)
        
        # Store skip connections
        skips = [x]
        
        # Downsampling path
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
        
        # Remove the last skip connection (bottom of U-Net)
        skips = skips[:-1]
        
        # Middle convolution
        x = self.middle_conv(x)
        
        # Upsampling path
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)
        
        # Final convolution
        x = self.output_conv(x)
        
        return x