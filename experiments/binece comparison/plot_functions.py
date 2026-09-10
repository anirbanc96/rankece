from probability_functions import weierstrass
from probability_functions import random_fourier
import matplotlib.pyplot as plt
import math
from probability_functions import *
from utils import *

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

HIGHLIGHT_NAMES = {"Quadratic", "Spikes", "Increasing Frequency"}

highlight_functions = [f for f in functions if f[1] in HIGHLIGHT_NAMES]
other_functions = [f for f in functions if f[1] not in HIGHLIGHT_NAMES]

p_range = np.linspace(0, 1, 10000)


def plot_functions_grid(funcs, ncols, save_path):
    n = len(funcs)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False
    )

    for ax, (func, name) in zip(axes.flat, funcs):
        print(name, true_ece(func))
        ax.plot(p_range, func(p_range), color="black", lw=2)
        ax.set_title(name, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Z", fontsize=8)
        ax.set_ylabel(r"$\mathbb{E}[Y|Z]$", fontsize=8)

    # Hide any unused subplots
    for ax in axes.flat[len(funcs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    fig.savefig(save_path, dpi=1200)
    plt.show()
    plt.close(fig)


print(f"Visualizing {len(highlight_functions)} highlighted Calibration Gap Functions...")
plot_functions_grid(highlight_functions, ncols=3, save_path="prob_functions_highlight.pdf")

print(f"Visualizing the remaining {len(other_functions)} Calibration Gap Functions...")
plot_functions_grid(other_functions, ncols=3, save_path="prob_functions_other.pdf")
