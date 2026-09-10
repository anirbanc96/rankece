import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

A = 0.2
TRUE_ECE = 0.04

n_fixed = 2000
m_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
num_runs = 50
bin_type = [10, r"$\sqrt{n}$", r"$n/10$"]
bin_counts = [10, int(np.sqrt(n_fixed)), int(n_fixed/10)]

print("=== Experiment 1: Fixed n={}, varying frequency m ===".format(n_fixed))

Rn_means = []
Rn_stds = []
Rn_std_error = []
binned_means = {b: [] for b in bin_counts}
binned_stds = {b: [] for b in bin_counts}
binned_std_error = {b: [] for b in bin_counts}

for m in tqdm(m_values):
    rn_vals = []
    bin_vals = {b: [] for b in bin_counts}

    for run in range(num_runs):
        np.random.seed(run)
        x = np.random.uniform(0, 1, n_fixed)
        z = A * (1-x) + x * (1-A)
        p = prob_func(x, m, A = A)
        y = np.random.binomial(1, p)

        rn_vals.append(compute_Rn(z, y)/TRUE_ECE)
        for b in bin_counts:
            bin_vals[b].append(compute_binned_ece(z, y, b)/TRUE_ECE)
            # bin_vals[b].append(compute_binned_ece(z, p, b))

    Rn_means.append(np.mean(rn_vals))
    Rn_stds.append(np.std(rn_vals))
    Rn_std_error.append(np.std(rn_vals) / np.sqrt(num_runs))
    for b in bin_counts:
        binned_means[b].append(np.mean(bin_vals[b]))
        binned_stds[b].append(np.std(bin_vals[b]))
        binned_std_error[b].append(np.std(bin_vals[b]) / np.sqrt(num_runs))


fig, axes = plt.subplots(1, 1, figsize=(4, 3))

ax = axes
ax.axhline(1, color='black', ls='--', lw=2)
ax.plot(m_values, Rn_means, 'o-', color='C0', lw=2, label=r'rankECE')
ax.fill_between(m_values,
                np.array(Rn_means) - np.array(Rn_std_error),
                np.array(Rn_means) + np.array(Rn_std_error), alpha=0.15, color='C0')
colors_bin = ['C1', 'C2', 'C3']
for b_count, b_type_label, col in zip(bin_counts, bin_type, colors_bin):
    ax.plot(m_values, binned_means[b_count], 's-', color=col, lw=1.5, label=r"$\ell_2-$binECE" f' (K={b_type_label})')
    ax.fill_between(m_values,
                    np.array(binned_means[b_count]) - np.array(binned_std_error[b_count]),
                    np.array(binned_means[b_count]) + np.array(binned_std_error[b_count]), alpha=0.1, color=col)

ax.set_xlabel('m', fontsize=13)
ax.set_ylabel('Ratio', fontsize=13)
ax.set_title(f'Varying Frequency', fontsize=13)
ax.legend(fontsize=10, loc = 'lower left')
ax.set_xscale('log')
ax.grid(alpha=0.3)

fig.savefig("Over_m.pdf", dpi = 1200)