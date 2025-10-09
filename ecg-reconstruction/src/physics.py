import numpy as np
import torch

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

def reconstruct_12_leads(inputs, outputs):
    """
    Reconstruct all 12 leads from inputs (I, II, V4) and outputs (V1, V2, V3, V5, V6)
    
    Args:
        inputs: Input tensor [batch, 3, samples] with leads I, II, V4
        outputs: Output tensor [batch, 5, samples] with leads V1, V2, V3, V5, V6
        
    Returns:
        Full 12-lead ECG [batch, 12, samples]
    """
    batch_size, _, seq_len = inputs.shape
    
    # Extract input leads
    lead_I = inputs[:, 0]  # Lead I
    lead_II = inputs[:, 1]  # Lead II
    lead_V4 = inputs[:, 2]  # Lead V4
    
    # Calculate limb leads
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