import numpy as np
import matplotlib.pyplot as plt

def linear(z, slope = 1, constant = 0):
    return np.clip(slope * z + constant, 0, 1)

def staircase(z, num_jumps = 1, constant = 0):
    return np.clip(np.floor(z * num_jumps) / (num_jumps+1) + constant, 0, 1)

def quadratic(z, quadratic_coef = 1, linear_coef = 0, constant = 0):
    return np.clip(quadratic_coef * (z**2) + linear_coef * z + constant, 0, 1)

def gaussian(z, mu = 0, sigma = 1):
    return np.clip((np.exp(-(z-mu)**2/(2*sigma**2)))/np.sqrt(2 * np.pi * sigma ** 2), 0, 1)

def sawtooth(z, amplitude = 1, k = 1, constant = 0):
    return np.clip(amplitude * ((k * z) % 1 - constant), 0, 1)

def spike(z, k = 10):
    h = z.copy()
    centers = np.linspace(0.01, 0.99, k)
    # Alternate heights between +0.3 and -0.3
    heights = [0.3, -0.3] * 5

    for c, amp in zip(centers, heights):
        h += amp * np.exp(-((z - c)**2) / (2 * 0.02**2))

    return np.clip(h, 0, 1)


def random_fourier(z, K=10, seed=0, amplitude=0.4):
    rng = np.random.default_rng(seed)

    # Generate coefficients first
    a_coeffs = [rng.normal(scale=1/k) for k in range(1, K + 1)]
    b_coeffs = [rng.normal(scale=1/k) for k in range(1, K + 1)]

    def _eval(x):
        y = np.zeros_like(x, dtype=float)
        for k, (a, b) in enumerate(zip(a_coeffs, b_coeffs), start=1):
            y += a * np.sin(2 * np.pi * k * x)
            y += b * np.cos(2 * np.pi * k * x)
        return y

    return np.clip(0.5 + amplitude * _eval(np.asarray(z)), 0, 1)


def weierstrass(z, terms=8):
    def _eval(x):
        y = np.zeros_like(x, dtype=float)
        for k in range(terms):
            y += (0.5**k) * np.cos((3**k) * np.pi * x)
        return y

    h = 0.15 + 0.5 * np.asarray(z) + 0.25 * _eval(np.asarray(z))
    return np.clip(h, 0, 1)

def chirp(z):
    return np.clip(
        0.15 + 0.7*z
        + 0.5*np.sin(40*np.pi*z**2),
        0,
        1,
    )