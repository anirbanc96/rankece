import numpy as np
from scipy.spatial.distance import cdist, pdist


def gaussian_kernel_matrix(X, Y, sigma=1):

    sq_dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-sq_dists / (2 * sigma ** 2))


def laplace_kernel_matrix(X, Y, sigma=1):

    dists = cdist(X, Y, metric="cityblock")
    return np.exp(-dists / sigma)


_KERNEL_MATRICES = {
    "gaussian": gaussian_kernel_matrix,
    "laplace": laplace_kernel_matrix,
}


def get_kernel_matrix_fn(kernel):

    if callable(kernel):
        return kernel
    try:
        return _KERNEL_MATRICES[kernel]
    except KeyError:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Choose from {list(_KERNEL_MATRICES)} "
            f"or pass a callable with signature (X, Y, sigma)."
        )


def median_bandwidth(X):
    
    dists = pdist(X, metric="euclidean")

    if dists.size == 0:
        return 1.0

    med = np.median(dists)

    # Avoid degenerate (zero) bandwidth
    if med == 0:
        med = 1.0

    return med


def resolve_band(band, X):
    
    if band == "median":
        return median_bandwidth(X)
    return band


def to_two_class(Z, Y):
    
    out_y = np.column_stack([1.0-Y, Y])
    out_fz = np.column_stack([1.0-Z, Z])
    return out_y, out_fz


def gen_data(n, rho, prob_fn=None):
    
    Z = np.random.beta(rho, 1 - rho, size=n)

    if prob_fn is None:
        p = Z
    else:
        p = prob_fn(Z)
        p = np.clip(p, 0.0, 1.0)

    Y = np.random.binomial(1, p).astype(float)

    return Z, Y


H1_PROB_FUNCS = {
    "p = Z":              lambda Z: Z,
    "p = Z - Z^15":       lambda Z: Z - Z ** 15,
    "p = Z - Z^25":        lambda Z: Z - Z ** 25,
    "p = Z - Z^4":        lambda Z: Z - Z**4,
    "p = sin":            lambda Z: Z + 0.25 * np.sign(np.sin(10 * np.pi * Z)),
}