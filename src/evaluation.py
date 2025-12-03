"""
Evaluation metrics for ECG lead reconstruction
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for ECG reconstruction
    
    Args:
        y_true: Ground truth signals [batch, leads, samples]
        y_pred: Predicted signals [batch, leads, samples]
        
    Returns:
        Dictionary with metrics per lead and overall
    """
    batch_size, n_leads, n_samples = y_true.shape
    
    metrics = {
        'mae': np.zeros(n_leads),
        'mse': np.zeros(n_leads),
        'rmse': np.zeros(n_leads),
        'correlation': np.zeros(n_leads),
        'snr': np.zeros(n_leads)
    }
    
    # Calculate per-lead metrics
    for lead in range(n_leads):
        lead_true = y_true[:, lead, :]
        lead_pred = y_pred[:, lead, :]
        
        # MAE
        metrics['mae'][lead] = np.mean([
            mean_absolute_error(lead_true[b], lead_pred[b])
            for b in range(batch_size)
        ])
        
        # MSE and RMSE
        metrics['mse'][lead] = np.mean([
            mean_squared_error(lead_true[b], lead_pred[b])
            for b in range(batch_size)
        ])
        metrics['rmse'][lead] = np.sqrt(metrics['mse'][lead])
        
        # Pearson correlation (handle constant signals gracefully)
        correlations = []
        for b in range(batch_size):
            # Skip if either signal is constant (would cause NaN)
            if np.std(lead_true[b]) < 1e-8 or np.std(lead_pred[b]) < 1e-8:
                continue
            correlations.append(pearsonr(lead_true[b], lead_pred[b])[0])
        metrics['correlation'][lead] = np.mean(correlations) if correlations else 0.0
        
        # SNR
        signal_power = np.mean(lead_true**2)
        noise_power = np.mean((lead_true - lead_pred)**2)
        metrics['snr'][lead] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Overall metrics
    metrics['mae_overall'] = np.mean(metrics['mae'])
    metrics['mse_overall'] = np.mean(metrics['mse'])
    metrics['rmse_overall'] = np.mean(metrics['rmse'])
    metrics['correlation_overall'] = np.mean(metrics['correlation'])
    metrics['snr_overall'] = np.mean(metrics['snr'])
    
    return metrics

def print_metrics_report(metrics, lead_names=None):
    """
    Print a formatted metrics report
    
    Args:
        metrics: Dictionary returned by calculate_metrics
        lead_names: List of lead names
    """
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    print("\n" + "="*60)
    print("ECG Lead Reconstruction Metrics")
    print("="*60)
    
    print(f"{'Lead':<6} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'SNR (dB)':<10}")
    print("-" * 50)
    
    for i, lead_name in enumerate(lead_names):
        print(f"{lead_name:<6} {metrics['mae'][i]:<8.4f} {metrics['rmse'][i]:<8.4f} "
              f"{metrics['correlation'][i]:<8.4f} {metrics['snr'][i]:<10.2f}")
    
    print("-" * 50)
    print(f"{'Overall':<6} {metrics['mae_overall']:<8.4f} {metrics['rmse_overall']:<8.4f} "
          f"{metrics['correlation_overall']:<8.4f} {metrics['snr_overall']:<10.2f}")
    print("="*60)