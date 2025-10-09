import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load model checkpoint"""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def evaluate_reconstruction(y_true, y_pred):
    """
    Evaluate reconstruction quality
    
    Args:
        y_true: Ground truth signals [batch, leads, samples]
        y_pred: Predicted signals [batch, leads, samples]
        
    Returns:
        Dictionary of metrics
    """
    # Move to numpy for evaluation
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Get dimensions
    batch_size, n_leads, n_samples = y_true.shape
    
    # Initialize metrics
    metrics = {
        'mae': np.zeros(n_leads),
        'correlation': np.zeros(n_leads),
        'snr': np.zeros(n_leads)
    }
    
    # Calculate metrics per lead
    for lead in range(n_leads):
        # Mean Absolute Error
        metrics['mae'][lead] = np.mean([
            mean_absolute_error(y_true[b, lead], y_pred[b, lead])
            for b in range(batch_size)
        ])
        
        # Pearson correlation
        metrics['correlation'][lead] = np.mean([
            pearsonr(y_true[b, lead], y_pred[b, lead])[0]
            for b in range(batch_size)
        ])
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(y_true[:, lead, :]**2)
        noise_power = np.mean((y_true[:, lead, :] - y_pred[:, lead, :])**2)
        metrics['snr'][lead] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Add overall metrics
    metrics['mae_overall'] = np.mean(metrics['mae'])
    metrics['correlation_overall'] = np.mean(metrics['correlation'])
    metrics['snr_overall'] = np.mean(metrics['snr'])
    
    return metrics

def plot_reconstruction(y_true, y_pred, lead_names=None, sample_idx=0, save_path=None):
    """
    Plot ground truth vs predicted signals
    
    Args:
        y_true: Ground truth signals [batch, leads, samples]
        y_pred: Predicted signals [batch, leads, samples]
        lead_names: List of lead names
        sample_idx: Index of sample to plot
        save_path: Path to save the plot
    """
    # Move to numpy for plotting
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Get dimensions
    n_leads = y_true.shape[1]
    n_samples = y_true.shape[2]
    
    # Default lead names if not provided
    if lead_names is None:
        lead_names = [
            'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        ]
    
    # Create time axis (assuming 500 Hz)
    time = np.arange(n_samples) / 500
    
    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(n_leads):
        ax = axes[i]
        
        # Plot true signal
        ax.plot(time, y_true[sample_idx, i], 'b-', label='True')
        
        # Plot predicted signal
        ax.plot(time, y_pred[sample_idx, i], 'r-', label='Predicted')
        
        # Calculate metrics for this lead
        corr = pearsonr(y_true[sample_idx, i], y_pred[sample_idx, i])[0]
        mae = mean_absolute_error(y_true[sample_idx, i], y_pred[sample_idx, i])
        
        # Add title and labels
        ax.set_title(f'{lead_names[i]} (r={corr:.3f}, MAE={mae:.3f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()