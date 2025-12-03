import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path

def calculate_limb_leads_numpy(lead_I, lead_II):
    """
    Calculate limb leads (III, aVR, aVL, aVF) from leads I and II using Einthoven's and Goldberger's laws
    
    Args:
        lead_I: Lead I signal [batch, samples] or [samples]
        lead_II: Lead II signal [batch, samples] or [samples]
        
    Returns:
        Dictionary with calculated leads
    """
    # Calculate lead III using Einthoven's law: III = II - I
    lead_III = lead_II - lead_I
    
    # Calculate augmented leads using Goldberger's equations
    # aVR = -(I + II)/2
    lead_aVR = -(lead_I + lead_II) / 2
    
    # aVL = I - II/2
    lead_aVL = lead_I - lead_II / 2
    
    # aVF = II - I/2
    lead_aVF = lead_II - lead_I / 2
    
    return {
        'III': lead_III,
        'aVR': lead_aVR,
        'aVL': lead_aVL,
        'aVF': lead_aVF
    }

def calculate_limb_leads_torch(lead_I, lead_II):
    """
    PyTorch version of calculate_limb_leads
    
    Args:
        lead_I: Lead I signal [batch, 1, samples] or [batch, samples]
        lead_II: Lead II signal [batch, 1, samples] or [batch, samples]
        
    Returns:
        Dictionary with calculated leads
    """
    # Calculate lead III using Einthoven's law: III = II - I
    lead_III = lead_II - lead_I
    
    # Calculate augmented leads using Goldberger's equations
    # aVR = -(I + II)/2
    lead_aVR = -(lead_I + lead_II) / 2
    
    # aVL = I - II/2
    lead_aVL = lead_I - lead_II / 2
    
    # aVF = II - I/2
    lead_aVF = lead_II - lead_I / 2
    
    return {
        'III': lead_III,
        'aVR': lead_aVR,
        'aVL': lead_aVL,
        'aVF': lead_aVF
    }

def reconstruct_12_leads(inputs, outputs, targets=None):
    """
    Reconstruct all 12 leads from inputs (I, II, V4) and outputs (V1, V2, V3, V5, V6)
    
    IMPORTANT: Physics-based reconstruction (Einthoven/Goldberger) only works on RAW 
    voltage data, NOT on globally normalized data. After normalization, each lead has 
    a different offset, breaking the linear relationships.
    
    Args:
        inputs: Input tensor [batch, 3, samples] with leads I, II, V4
        outputs: Output tensor [batch, 5, samples] with leads V1, V2, V3, V5, V6
        targets: Optional target tensor [batch, 12, samples] - if provided, use stored
                 values for physics leads (III, aVR, aVL, aVF) instead of calculating
        
    Returns:
        Full 12-lead ECG [batch, 12, samples]
    """
    batch_size, _, seq_len = inputs.shape
    
    # Extract input leads
    lead_I = inputs[:, 0]  # Lead I
    lead_II = inputs[:, 1]  # Lead II
    lead_V4 = inputs[:, 2]  # Lead V4
    
    if targets is not None:
        # Use stored normalized values for physics leads (correct approach for normalized data)
        lead_III = targets[:, 2]
        lead_aVR = targets[:, 3]
        lead_aVL = targets[:, 4]
        lead_aVF = targets[:, 5]
    else:
        # Calculate limb leads (only valid for raw voltage data!)
        # WARNING: This will produce incorrect results on normalized data
        limb_leads = calculate_limb_leads_torch(lead_I, lead_II)
        lead_III = limb_leads['III']
        lead_aVR = limb_leads['aVR']
        lead_aVL = limb_leads['aVL']
        lead_aVF = limb_leads['aVF']
    
    # Extract predicted chest leads
    lead_V1 = outputs[:, 0]  # V1
    lead_V2 = outputs[:, 1]  # V2
    lead_V3 = outputs[:, 2]  # V3
    lead_V5 = outputs[:, 3]  # V5
    lead_V6 = outputs[:, 4]  # V6
    
    # Stack all leads in standard order
    # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    full_12_leads = torch.stack([
        lead_I, lead_II, lead_III,
        lead_aVR, lead_aVL, lead_aVF,
        lead_V1, lead_V2, lead_V3, lead_V4,
        lead_V5, lead_V6
    ], dim=1)
    
    return full_12_leads


class PhysicsAwareLoss(nn.Module):
    """
    Physics-aware loss combining reconstruction loss with Einthoven/Goldberger constraints.
    
    The key insight: Even though we only predict chest leads (V1-V3, V5-V6), we can still
    encourage the model to learn representations consistent with known ECG physics.
    
    Physics Constraints (after denormalization to raw voltage space):
    1. Einthoven's Law: III = II - I (lead III can be computed from I and II)
    2. Goldberger's Laws:
       - aVR = -(I + II)/2
       - aVL = I - II/2  
       - aVF = II - I/2
    
    These constraints are ALWAYS true for raw ECG voltages. After per-lead normalization,
    the relationships break, so we must denormalize before computing physics losses.
    
    Loss formulation:
    L_total = L_recon + lambda_physics * L_physics
    
    Where L_physics penalizes violations of the physics constraints in raw voltage space.
    
    Rationale for use with DL leads:
    - Even though physics constraints are about limb leads, the model's internal
      representations should be consistent with the full ECG geometry
    - Regularizes the latent space to respect cardiac dipole physics
    - May improve chest lead predictions by encouraging physiologically consistent encodings
    """
    
    def __init__(self, norm_params_path, lambda_physics=0.1, device='cuda'):
        """
        Args:
            norm_params_path: Path to norm_params.pkl with lead_means and lead_stds
            lambda_physics: Weight for physics constraint loss (default: 0.1)
            device: Device to place tensors on
        """
        super().__init__()
        
        self.lambda_physics = lambda_physics
        
        # Load normalization parameters
        with open(norm_params_path, 'rb') as f:
            params = pickle.load(f)
        
        # Register as buffers (move with model, but not trainable)
        # Shape: [12, 1] for broadcasting with [B, 12, L]
        self.register_buffer('lead_means', torch.tensor(params['lead_means'], dtype=torch.float32))
        self.register_buffer('lead_stds', torch.tensor(params['lead_stds'], dtype=torch.float32))
        
        # Lead indices in standard 12-lead order
        # I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
        self.idx_I = 0
        self.idx_II = 1
        self.idx_III = 2
        self.idx_aVR = 3
        self.idx_aVL = 4
        self.idx_aVF = 5
        
        # Reconstruction loss
        self.mse = nn.MSELoss()
    
    def denormalize_lead(self, x_norm, lead_idx):
        """
        Denormalize a single lead from z-score to raw voltage.
        
        Args:
            x_norm: Normalized tensor [B, L] or [B, 1, L]
            lead_idx: Index of the lead (0-11)
            
        Returns:
            Raw voltage tensor with same shape
        """
        mean = self.lead_means[lead_idx].squeeze()  # scalar
        std = self.lead_stds[lead_idx].squeeze()  # scalar
        return x_norm * std + mean
    
    def compute_physics_loss(self, full_12_leads_normalized):
        """
        Compute physics constraint violation loss.
        
        Takes the full 12-lead reconstructed ECG (still normalized), denormalizes,
        then computes how well Einthoven's and Goldberger's laws are satisfied.
        
        Args:
            full_12_leads_normalized: [B, 12, L] tensor of normalized 12-lead ECG
            
        Returns:
            Physics constraint loss (scalar)
        """
        # Denormalize relevant leads
        I_raw = self.denormalize_lead(full_12_leads_normalized[:, self.idx_I], self.idx_I)
        II_raw = self.denormalize_lead(full_12_leads_normalized[:, self.idx_II], self.idx_II)
        III_raw = self.denormalize_lead(full_12_leads_normalized[:, self.idx_III], self.idx_III)
        aVR_raw = self.denormalize_lead(full_12_leads_normalized[:, self.idx_aVR], self.idx_aVR)
        aVL_raw = self.denormalize_lead(full_12_leads_normalized[:, self.idx_aVL], self.idx_aVL)
        aVF_raw = self.denormalize_lead(full_12_leads_normalized[:, self.idx_aVF], self.idx_aVF)
        
        # Compute expected values from physics
        III_expected = II_raw - I_raw
        aVR_expected = -(I_raw + II_raw) / 2
        aVL_expected = I_raw - II_raw / 2
        aVF_expected = II_raw - I_raw / 2
        
        # Physics losses (MSE between actual and expected)
        loss_III = torch.mean((III_raw - III_expected) ** 2)
        loss_aVR = torch.mean((aVR_raw - aVR_expected) ** 2)
        loss_aVL = torch.mean((aVL_raw - aVL_expected) ** 2)
        loss_aVF = torch.mean((aVF_raw - aVF_expected) ** 2)
        
        # Total physics loss
        physics_loss = (loss_III + loss_aVR + loss_aVL + loss_aVF) / 4
        
        return physics_loss
    
    def forward(self, pred, target, inputs=None, full_target=None):
        """
        Compute combined loss: reconstruction + physics constraints.
        
        Args:
            pred: Predicted chest leads [B, 5, L] (V1, V2, V3, V5, V6)
            target: Target chest leads [B, 5, L] (V1, V2, V3, V5, V6)
            inputs: Input leads [B, 3, L] (I, II, V4) - needed for physics
            full_target: Full 12-lead target [B, 12, L] - needed for physics
            
        Returns:
            total_loss, recon_loss, physics_loss (as tuple)
        """
        # Standard reconstruction loss on chest leads
        recon_loss = self.mse(pred, target)
        
        # If physics inputs not provided, return recon loss only
        if inputs is None or full_target is None:
            return recon_loss, recon_loss, torch.tensor(0.0, device=pred.device)
        
        # Reconstruct full 12 leads for physics evaluation
        full_12_leads = reconstruct_12_leads(inputs, pred, targets=full_target)
        
        # Compute physics constraint loss
        physics_loss = self.compute_physics_loss(full_12_leads)
        
        # Combined loss
        total_loss = recon_loss + self.lambda_physics * physics_loss
        
        return total_loss, recon_loss, physics_loss


class PhysicsConsistencyLoss(nn.Module):
    """
    Alternative physics loss that only uses input leads (I, II).
    
    This version doesn't require full 12-lead targets. It computes the
    physics-derived leads from inputs and penalizes inconsistency with
    what the model would implicitly predict.
    
    Simpler than PhysicsAwareLoss but less direct. Useful when full
    12-lead targets are not available at training time.
    """
    
    def __init__(self, norm_params_path, lambda_physics=0.1):
        super().__init__()
        
        self.lambda_physics = lambda_physics
        
        with open(norm_params_path, 'rb') as f:
            params = pickle.load(f)
        
        self.register_buffer('lead_means', torch.tensor(params['lead_means'], dtype=torch.float32))
        self.register_buffer('lead_stds', torch.tensor(params['lead_stds'], dtype=torch.float32))
        
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Simple reconstruction loss with optional physics regularization.
        
        For now, just wraps MSE. Can be extended with additional constraints.
        """
        return self.mse(pred, target)