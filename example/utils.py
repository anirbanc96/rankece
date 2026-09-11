import numpy as np

def phi_func(x, m, A = 0.2):
    """oscillation function: phi(z) = c * sign(sin(2*pi*m*z))"""
    return A * np.sign(np.sin(2 * np.pi * m * x))

def prob_func(x, m, A = 0.2):
    """E[Y|Z=z] = z + h(z), clipped to [0,1]."""
    z = A * (1-x) + x * (1-A)
    return np.clip(z + phi_func(x, m, A), 0.0, 1.0)

def compute_Rn(z, y):
    """Rank estimator R_n = (1/n) sum_{i=1}^{n-1} (y[i]-z(i))(y[i+1]-z(i+1))"""
    n = len(z)
    if n < 2:
        return 0.0
    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]
    y_sorted = y[sort_idx]
    resid = y_sorted - z_sorted
    return np.sum(resid[:-1] * resid[1:]) / n

def compute_binned_ece(z, y, num_bins):
    """Binned l2-ECE: sum_k (mean(y_k) - mean(z_k))^2 * (n_k / n)"""
    bins = np.linspace(0, 1, num_bins + 1)
    bin_idx = np.clip(np.digitize(z, bins) - 1, 0, num_bins - 1)
    binned_ece = 0.0
    for k in range(num_bins):
        mask = bin_idx == k
        if np.sum(mask) < 1:
            continue
        binned_ece += (np.mean(y[mask]) - np.mean(z[mask]))**2 * np.mean(mask)
    return binned_ece