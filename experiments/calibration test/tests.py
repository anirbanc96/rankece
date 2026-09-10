import numpy as np
from scipy.stats import norm

from utils import (
    get_kernel_matrix_fn,
    resolve_band,
    to_two_class,
)


def rank_ece_asymptotic_test(Z, Y, alpha_sig=0.05):

    N = len(Z)

    # Sort predictions and concomitant labels
    sort_idx = np.argsort(Z)
    Z_sorted = Z[sort_idx]
    Y_sorted = Y[sort_idx]

    D = Y_sorted - Z_sorted

    # Consecutive order statistic calculation
    R_N = np.sum(D[:-1] * D[1:]) / N

    # Exact null variance calculation
    var_R_N = np.mean((Z * (1 - Z)) ** 2)

    test_stat = np.sqrt(N) * R_N / np.sqrt(var_R_N)
    p_value = 1 - norm.cdf(test_stat)

    return p_value < alpha_sig


def rank_ece_finite_test(Z, Y, alpha_sig=0.05):

    n = len(Z)

    sort_idx = np.argsort(Z)
    Z_sorted = Z[sort_idx]
    Y_sorted = Y[sort_idx]

    D = Y_sorted - Z_sorted
    hatrankECE = np.sum(D[:-1] * D[1:]) / n

    v = Z_sorted * (1 - Z_sorted)
    sigma2 = np.sum(v[:-1] * v[1:]) / n

    log_term = np.log(2 / alpha_sig)
    threshold = (
        2 * np.sqrt(sigma2 * log_term / n)
        + 2 * log_term / (3 * n)
    )

    return hatrankECE > threshold


def skce_linear_test(Z, Y, alpha_sig=0.05, band=1, kernel="gaussian"):

    n = len(Z)
    out_y, out_fz = to_two_class(Z, Y)
    band = resolve_band(band, out_fz)

    n_pairs = n // 2
    m = 2 * n_pairs  # largest even number <= n

    A_fz = out_fz[0:m:2]      # f(z) for "first" elements of each pair
    B_fz = out_fz[1:m:2]      # f(z) for "second" elements of each pair
    A_diff = (out_y - out_fz)[0:m:2]
    B_diff = (out_y - out_fz)[1:m:2]

    if kernel == "gaussian":
        sq_dists = np.sum((A_fz - B_fz) ** 2, axis=1)
        kernel_vals = np.exp(-sq_dists / (2 * band ** 2))
    elif kernel == "laplace":
        dists = np.sum(np.abs(A_fz - B_fz), axis=1)
        kernel_vals = np.exp(-dists / band)
    else:
        # Custom kernel: assume a matrix-valued callable (X, Y, sigma) -> matrix,
        # and take the diagonal (i.e. pairwise A[i] vs B[i]).
        kernel_matrix_fn = get_kernel_matrix_fn(kernel)
        kernel_vals = np.diag(kernel_matrix_fn(A_fz, B_fz, sigma=band))

    dots = np.sum(A_diff * B_diff, axis=1)
    SKCE_vals = kernel_vals * dots

    SKCE_num = np.mean(SKCE_vals)
    SKCE_den = np.std(SKCE_vals)

    test_stat = np.sqrt(n_pairs) * SKCE_num / SKCE_den
    p_value = 1 - norm.cdf(test_stat)

    return p_value < alpha_sig


def _SKCE_h_matrix(out_y, out_fz, band=1, kernel="gaussian"):

    diff_y = out_y - out_fz                 # (n, 2)
    G = diff_y @ diff_y.T                   # (n, n)

    kernel_matrix_fn = get_kernel_matrix_fn(kernel)
    K = kernel_matrix_fn(out_fz, out_fz, sigma=band)  # (n, n)

    return K * G


def _SKCE_ustat_from_H(n, H):
    """SKCE_hat = (2 / (n(n-1))) * sum_{i<j} H[i,j]."""
    off_diag_sum = np.sum(H) - np.trace(H)
    return off_diag_sum / (n * (n - 1))


def _SKCE_bootstrap_quantile(n, H, alpha_sig=0.05, n_rep=100):
    """
    Bootstrap calibration of n * SKCE_hat following Widmann et.al. (2019):

        T = (2/n) * sum_{i<j} [ H[*i,*j]
                                 - (1/n) sum_k H[*i,k]
                                 - (1/n) sum_k H[k,*j]
                                 + (1/n^2) sum_{k,l} H[k,l] ]

    where *1,...,*n are indices sampled with replacement from {1,...,n},
    and H[k,l] is computed on the ORIGINAL data.

    Returns the (1 - alpha_sig) quantile c of T, used as the rejection
    threshold for n * SKCE_hat.
    """
    H_row_sum = np.sum(H, axis=1)   # includes diagonal
    H_total = np.sum(H)             # sum_{k,l} H[k,l], includes diagonal
    correction = H_total / (n ** 2)

    i_idx, j_idx = np.triu_indices(n, k=1)

    T_vals = np.zeros(n_rep)

    for r in range(n_rep):
        perm = np.random.choice(n, size=n, replace=True)

        H_perm = H[np.ix_(perm, perm)]            # H[*i,*j]
        perm_marg = H_row_sum[perm] / n           # (1/n) sum_k H[*i,k] = (1/n) sum_k H[k,*j] by symmetry

        term1 = H_perm[i_idx, j_idx]
        term2 = perm_marg[i_idx]
        term3 = perm_marg[j_idx]

        T_vals[r] = (2 / n) * np.sum(term1 - term2 - term3 + correction)

    return np.quantile(T_vals, 1 - alpha_sig)


def skce_ustat_test(Z, Y, alpha_sig=0.05, band=1, kernel="gaussian", n_rep=100):

    n = len(Z)
    out_y, out_fz = to_two_class(Z, Y)
    band = resolve_band(band, out_fz)

    H = _SKCE_h_matrix(out_y, out_fz, band=band, kernel=kernel)
    SKCE_stat = n * _SKCE_ustat_from_H(n, H)

    c = _SKCE_bootstrap_quantile(n, H, alpha_sig=alpha_sig, n_rep=n_rep)

    # Reject H0 (calibration) if n * SKCE_hat exceeds the bootstrap threshold
    return SKCE_stat > c