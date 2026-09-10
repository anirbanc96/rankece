from utils import uniform_sampler
from utils import run_experiment
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from utils import *
from probability_functions import *

BIN_SCHEDULES = {
    r"$\ell_2$-binECE  $K=10$":         lambda N: bins_fixed(N, B=10),
    r"$\ell_2$-binECE  $K=\sqrt{n}$":   bins_sqrt,
    r"$\ell_2$-binECE  $K=n^{1/3}$":    bins_cbrt,
    r"$\ell_2$-binECE  $K=n/20$":        lambda N: bins_n_over_k(N, k = 20),
}

functions = [
    (lambda z: linear(z, slope = 0.5, constant = 0.3), "Linear"),
    (lambda z: quadratic(z, quadratic_coef = 0.8, linear_coef = -0.2, constant = 0.2), "Quadratic"),
    (lambda z: staircase(z, num_jumps = 10, constant = 0.05), "Staircase"),
    (lambda z: gaussian(z, mu = 0.5, sigma = 0.5), "Gaussian"),
    (lambda z: sawtooth(z, amplitude = 0.75, k = 10, constant = -0.1), "Sawtooth"),
    (lambda z: spike(z, k = 10), "Spikes"),
    (lambda z: random_fourier(z, K=10, amplitude=0.25, seed=42), "Random Fourier"),
    (lambda z: weierstrass(z, terms = 10), "Weierstrass"),
    (lambda z: chirp(z), "Increasing Frequency"),
]



for (func, name) in functions:

    results = run_experiment(func, bin_schedules=BIN_SCHEDULES, z_sampler = uniform_sampler)
    plot_single_experiment(results, title = name, bin_schedules=BIN_SCHEDULES, fig_size=(3, 3.5), save_path=f"plots/{name.replace(' ', '_').lower()}.pdf")