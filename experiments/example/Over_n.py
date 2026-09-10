import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

A = 0.2
TRUE_ECE = 0.04
colors_bin = ['C1', 'C2', 'C3']

m_fixed = 100
n_values = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
bin_type = [10, r"$\sqrt{n}$", r"$n/10$"]
num_bin_type = len(bin_type)
num_runs = 50

Rn_means2 = []
Rn_stds2 = []
Rn_std_error2 = []


# Index 0: fixed 10 bins
# Index 1: sqrt(n) bins
# Index 2: n/10 bins
binned_means2 = {b: [] for b in range(num_bin_type)}
binned_stds2 = {b: [] for b in range(num_bin_type)}
binned_std_error2 = {b: [] for b in range(num_bin_type)}

np.random.seed(42)

for n in tqdm(n_values):
    rn_vals = []

    current_bin_counts = [10, int(np.sqrt(n)), int(n/10)]
    current_bin_counts = [max(1, bc) for bc in current_bin_counts]

    bin_vals_for_this_n_by_type = [[], [], []]

    for run in range(num_runs):
        np.random.seed(run)
        x = np.random.uniform(0, 1, n)
        z = A * (1-x) + x * (1-A)
        p = prob_func(x, m_fixed, A = A)
        y = np.random.binomial(1, p)

        rn_vals.append(compute_Rn(z, y)/TRUE_ECE)

        # Append for each binning strategy type
        for i in range(len(current_bin_counts)):
            bin_vals_for_this_n_by_type[i].append(compute_binned_ece(z, y, current_bin_counts[i])/TRUE_ECE)


    Rn_means2.append(np.mean(rn_vals))
    Rn_stds2.append(np.std(rn_vals))
    Rn_std_error2.append(np.std(rn_vals) / np.sqrt(num_runs))

    # Store the aggregated results for this n, for each binning strategy type
    for i in range(len(current_bin_counts)):
        binned_means2[i].append(np.mean(bin_vals_for_this_n_by_type[i]))
        binned_stds2[i].append(np.std(bin_vals_for_this_n_by_type[i]))
        binned_std_error2[i].append(np.std(bin_vals_for_this_n_by_type[i]) / np.sqrt(num_runs))


fig, axes = plt.subplots(1, 1, figsize=(4, 3))
ax = axes
ax.axhline(1, color='black', ls='--', lw=2)
ax.plot(n_values, Rn_means2, 'o-', color='C0', lw=2, label=f'rankECE')
ax.fill_between(n_values,
                np.array(Rn_means2) - np.array(Rn_std_error2),
                np.array(Rn_means2) + np.array(Rn_std_error2), alpha=0.15, color='C0')

# Loop through the results for each binning strategy type
for i, (b_type_label, col) in enumerate(zip(bin_type, colors_bin)):
    ax.plot(n_values, binned_means2[i], 's-', color=col, lw=1.5, label=r'$\ell_2-$binECE' f' (K={b_type_label})')
    ax.fill_between(n_values,
                    np.array(binned_means2[i]) - np.array(binned_std_error2[i]),
                    np.array(binned_means2[i]) + np.array(binned_std_error2[i]), alpha=0.1, color=col)

ax.set_xlabel('n', fontsize=13)
ax.set_ylabel('Ratio', fontsize=13)
ax.set_title(f'Varying Sample Size', fontsize=13)
ax.legend(fontsize=10, bbox_to_anchor = (0.3, 0.4))
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("Over_n.pdf", dpi = 1200)

plt.show()
plt.close()