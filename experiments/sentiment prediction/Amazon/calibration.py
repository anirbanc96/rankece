import numpy as np


def rank_ece(y: np.ndarray, z: np.ndarray) -> float:
    
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n < 2:
        raise ValueError("rank_ece requires at least 2 observations.")

    order = np.argsort(z, kind="mergesort")
    y_sorted = y[order]
    z_sorted = z[order]

    resid = y_sorted - z_sorted
    return float(np.mean(resid[:-1] * resid[1:])) * (n-1)/n


def _n_bins(rule: str, n: int) -> int:
    if rule == "fixed10":
        return 10
    if rule == "sqrt":
        return max(1, int(np.floor(np.sqrt(n))))
    if rule == "cube_root":
        return max(1, int(np.floor(n ** (1 / 3))))
    if rule == "n_over_20":
        return max(1, int(np.floor(n / 20)))
    raise ValueError(f"Unknown binning rule: {rule}")


def histogram_l2_ece(y: np.ndarray, z: np.ndarray, rule: str) -> float:
    
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    B = _n_bins(rule, n)

    edges = np.linspace(0.0, 1.0, B + 1)
    bin_idx = np.clip(np.digitize(z, edges[1:-1], right=True), 0, B - 1)

    total = 0.0
    for b in range(B):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ybar = y[mask].mean()
        zbar = z[mask].mean()
        total += (n_b / n) * (ybar - zbar) ** 2
    return float(total)

def debiased_histogram_l2_ece(y: np.ndarray, z: np.ndarray, rule: str) -> float:
    """
    Debiased fixed-width histogram L2-ECE from Kumar et. al. (2019):

        sum_b (n_b / n) * [ (ybar_b - zbar_b)^2 - ybar_b (1 - ybar_b) / (n_b - 1) ]
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    B = _n_bins(rule, n)

    edges = np.linspace(0.0, 1.0, B + 1)
    bin_idx = np.clip(np.digitize(z, edges[1:-1], right=True), 0, B - 1)

    total = 0.0
    for b in range(B):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ybar = y[mask].mean()
        zbar = z[mask].mean()
        term = (ybar - zbar) ** 2
        if n_b >= 2:
            term -= ybar * (1.0 - ybar) / (n_b - 1)
        total += (n_b / n) * term
    return float(total)

import numpy as np


def tcal_histogram_l2_ece(y: np.ndarray, z: np.ndarray, rule: str) -> float:
    """
    T-Cal debiased plug-in statistic for scalar binary squared L2-ECE.

    Computes:
        (1 / n) * sum_b [
            ((sum_{i in b} r_i)**2 - sum_{i in b} r_i**2) / n_b
        ],
    where r_i = y_i - z_i.

    Uses the existing _n_bins(rule, n) helper.
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if y.ndim != 1 or z.ndim != 1 or y.shape != z.shape:
        raise ValueError(
            "y and z must be one-dimensional arrays of equal length."
        )
    if z.size == 0:
        raise ValueError("y and z must be nonempty.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        raise ValueError("y and z must contain only finite values.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y must contain only binary labels 0 and 1.")
    if not np.all((z >= 0.0) & (z <= 1.0)):
        raise ValueError("z must contain probabilities in [0, 1].")

    n = z.size
    B = _n_bins(rule, n)

    if (
        isinstance(B, (bool, np.bool_))
        or not isinstance(B, (int, np.integer))
        or B < 1
    ):
        raise ValueError(
            "_n_bins(rule, n) must return a positive integer."
        )
    B = int(B)

    # Preserve the original right-closed bin convention.
    # Using only interior edges also includes both z=0 and z=1.
    edges = np.linspace(0.0, 1.0, B + 1)
    bin_idx = np.digitize(z, edges[1:-1], right=True)

    r = y - z
    counts = np.bincount(bin_idx, minlength=B)
    sums = np.bincount(bin_idx, weights=r, minlength=B)
    sums_sq = np.bincount(bin_idx, weights=r * r, minlength=B)

    # Singleton bins have no off-diagonal pairs.
    keep = counts >= 1
    contributions = (
        sums[keep] ** 2 - sums_sq[keep]
    ) / counts[keep]

    return float(contributions.sum() / n)

def reference_l2_ece(y: np.ndarray, z: np.ndarray, rule: str) -> float:

    # return debiased_histogram_l2_ece(y, z, rule)
    return tcal_histogram_l2_ece(y, z, rule)
    # return histogram_l2_ece(y, z, rule)
    # return rank_ece(y, z)


# Registry of every estimator to compute in the resampling experiment.
ESTIMATORS = {
    "rank_ece": lambda y, z: rank_ece(y, z),
    "hist_10": lambda y, z: histogram_l2_ece(y, z, "fixed10"),
    "hist_sqrt": lambda y, z: histogram_l2_ece(y, z, "sqrt"),
    "hist_cube_root": lambda y, z: histogram_l2_ece(y, z, "cube_root"),
    "hist_n_over_20": lambda y, z: histogram_l2_ece(y, z, "n_over_20"),
}
