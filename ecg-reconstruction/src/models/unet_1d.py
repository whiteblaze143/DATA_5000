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


class LightConvBlock1D(nn.Module):
    """Lightweight 1D conv block for lead-specific decoders (fewer params)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


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


class LeadSpecificDecoder(nn.Module):
    """
    Specialized decoder for a single chest lead.
    Takes bottleneck features + skip connections and produces single lead output.
    
    Architecture designed for specific chest lead positions:
    - V1/V2: Right precordial (need different features than V5/V6)
    - V3: Transitional zone
    - V5/V6: Left precordial (similar to limb leads)
    """
    def __init__(self, features, depth, skip_channels_list, lead_type='default', dropout=0.2):
        """
        Args:
            features: Base feature count
            depth: Number of upsampling stages
            skip_channels_list: List of skip connection channel counts (from encoder)
            lead_type: 'right' (V1,V2), 'transition' (V3), 'left' (V5,V6)
            dropout: Dropout rate
        """
        super().__init__()
        self.depth = depth
        self.lead_type = lead_type
        
        # Lead-type specific kernel sizes (right precordial needs sharper features)
        if lead_type == 'right':
            kernel_sizes = [5, 5, 3, 3]  # Larger receptive field for V1/V2
        elif lead_type == 'transition':
            kernel_sizes = [5, 3, 3, 3]  # Mixed for V3
        else:  # left
            kernel_sizes = [3, 3, 3, 3]  # Standard for V5/V6
        
        # Upsampling path - progressively reduce channels
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        in_ch = features * (2 ** depth)  # Bottleneck channels
        
        for i in range(depth):
            out_ch = features * (2 ** (depth - i - 1))
            skip_ch = skip_channels_list[depth - i - 1]  # Corresponding skip channels
            k = kernel_sizes[i] if i < len(kernel_sizes) else 3
            
            # Upsample
            self.up_convs.append(
                nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            )
            
            # Process with skip connection
            self.up_blocks.append(nn.Sequential(
                LightConvBlock1D(in_ch // 2 + skip_ch, out_ch, kernel_size=k, dropout=dropout),
                LightConvBlock1D(out_ch, out_ch, kernel_size=k, dropout=dropout)
            ))
            
            in_ch = out_ch
        
        # Lead-specific output head with attention-like refinement
        self.refine = nn.Sequential(
            LightConvBlock1D(features, features, kernel_size=3, dropout=dropout),
            nn.Conv1d(features, features // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features // 2, 1, kernel_size=1)
        )
    
    def forward(self, bottleneck, skips):
        """
        Args:
            bottleneck: Bottleneck features [B, C, L]
            skips: List of skip connections (in encoder order, will be reversed)
        Returns:
            Single lead output [B, 1, L]
        """
        x = bottleneck
        reversed_skips = list(reversed(skips))
        
        for i, (up_conv, up_block) in enumerate(zip(self.up_convs, self.up_blocks)):
            x = up_conv(x)
            skip = reversed_skips[i]
            
            # Handle size mismatch
            diff = skip.size(2) - x.size(2)
            if diff > 0:
                x = F.pad(x, [diff // 2, diff - diff // 2])
            elif diff < 0:
                skip = F.pad(skip, [-diff // 2, -diff + diff // 2])
            
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        
        return self.refine(x)

class UNet1D(nn.Module):
    """
    1D U-Net for ECG lead reconstruction (original shared decoder)
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


class UNet1DLeadSpecific(nn.Module):
    """
    1D U-Net with Lead-Specific Decoders for ECG reconstruction.
    
    Architecture:
    - Shared encoder: Efficiently extracts common temporal features from I, II, V4
    - Lead-specific decoders: Specialized pathways for each chest lead position
      - V1, V2: Right precordial (larger kernels for sharp R waves)
      - V3: Transitional zone (mixed characteristics)
      - V5, V6: Left precordial (similar to limb leads)
    
    This design allows each lead to learn position-specific morphological features
    while sharing the computational cost of encoding.
    """
    def __init__(self, in_channels=3, out_channels=5, features=64, depth=4, dropout=0.2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.features = features
        
        # ============ SHARED ENCODER ============
        # Initial convolution
        self.input_conv = nn.Sequential(
            ConvBlock1D(in_channels, features, dropout=dropout),
            ConvBlock1D(features, features, dropout=dropout)
        )
        
        # Downsampling path (shared)
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(
                DownBlock1D(
                    features * (2**i),
                    features * (2**(i+1)),
                    dropout=dropout
                )
            )
        
        # Bottleneck (shared)
        mid_features = features * (2**depth)
        self.bottleneck = nn.Sequential(
            ConvBlock1D(mid_features, mid_features, dropout=dropout),
            ConvBlock1D(mid_features, mid_features, dropout=dropout)
        )
        
        # ============ LEAD-SPECIFIC DECODERS ============
        # Calculate skip connection channels for each level
        skip_channels = [features * (2**i) for i in range(depth)]  # [64, 128, 256, 512]
        
        # Create specialized decoder for each chest lead
        # Lead types based on anatomical position:
        # - V1, V2: Right precordial (anterior right ventricle)
        # - V3: Transition zone
        # - V5, V6: Left precordial (lateral left ventricle)
        
        self.decoder_v1 = LeadSpecificDecoder(features, depth, skip_channels, 'right', dropout)
        self.decoder_v2 = LeadSpecificDecoder(features, depth, skip_channels, 'right', dropout)
        self.decoder_v3 = LeadSpecificDecoder(features, depth, skip_channels, 'transition', dropout)
        self.decoder_v5 = LeadSpecificDecoder(features, depth, skip_channels, 'left', dropout)
        self.decoder_v6 = LeadSpecificDecoder(features, depth, skip_channels, 'left', dropout)
        
        # List of decoders in output order [V1, V2, V3, V5, V6]
        self.decoders = nn.ModuleList([
            self.decoder_v1,
            self.decoder_v2, 
            self.decoder_v3,
            self.decoder_v5,
            self.decoder_v6
        ])
    
    def forward(self, x):
        """
        Forward pass with shared encoder and lead-specific decoders.
        
        Args:
            x: Input tensor [batch, 3, seq_len] with leads I, II, V4
            
        Returns:
            Output tensor [batch, 5, seq_len] with leads V1, V2, V3, V5, V6
        """
        # ===== Shared Encoder =====
        x = self.input_conv(x)
        
        # Store skip connections for decoder
        skips = [x]  # First skip at features resolution
        
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
        
        # Remove bottleneck from skips (it's passed directly)
        skips = skips[:-1]
        
        # Bottleneck processing
        bottleneck = self.bottleneck(x)
        
        # ===== Lead-Specific Decoders =====
        outputs = []
        for decoder in self.decoders:
            lead_output = decoder(bottleneck, skips)
            outputs.append(lead_output)
        
        # Concatenate all lead outputs: [B, 5, L]
        return torch.cat(outputs, dim=1)
    
    def get_encoder_features(self, x):
        """Extract encoder features (for analysis/visualization)"""
        x = self.input_conv(x)
        skips = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
        bottleneck = self.bottleneck(x)
        return bottleneck, skips[:-1]