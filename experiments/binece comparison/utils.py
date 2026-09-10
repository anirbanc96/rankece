import numpy as np
import matplotlib.pyplot as plt


def uniform_sampler(N, rng, lo=0.0, hi=1.0):
    return rng.uniform(lo, hi, size=N)

def generate_data(z_sampler, prob_fn, N, rng):
    
    z    = z_sampler(N, rng)
    prob = np.clip(prob_fn(z), 0.0, 1.0)
    y    = rng.binomial(1, prob).astype(float)
    return z, y

def true_ece(func):
    x = np.linspace(0, 1, 500000)
    return np.mean((func(x) - x)**2)

def binned_ece(z, y, num_bins):

    N      = len(z)
    n_bins = max(1, int(num_bins))

    idx = np.clip(
        np.floor(z * n_bins).astype(int), 0, n_bins - 1
    )

    count = np.bincount(idx, minlength=n_bins).astype(float)   # n_b
    y_sum = np.bincount(idx, weights=y, minlength=n_bins)
    z_sum = np.bincount(idx, weights=z, minlength=n_bins)

    nonzero = count > 0
    denom   = np.where(nonzero, count, 1.0)   # avoid /0; zero bins contribute 0
    y_bar   = y_sum / denom
    z_bar   = z_sum / denom

    ece = float(np.dot(count[nonzero], (y_bar - z_bar)[nonzero] ** 2) / N)
    return ece

def compute_rank_ece(z, y):
    
    N = len(z)
    if N < 2:
        return float("nan")
    order = np.argsort(z)
    D     = y[order] - z[order]
    return float(np.dot(D[:-1], D[1:]) / N)


def bins_fixed(N, B=10):
    return max(1, int(B))


def bins_sqrt(N):
    return max(1, int(round(N ** 0.5)))


def bins_cbrt(N):
    return max(1, int(round(N ** (1.0 / 3.0))))


def bins_n_over_k(N, k = 10):
    return max(1, N // k)

def run_experiment(prob_fn, bin_schedules, z_sampler, z_lo=0.0, z_hi=1.0, sample_sizes=None, n_trials=100, seed=42):
    
    if sample_sizes is None:
        sample_sizes = [100, 250, 500, 750, 1_000, 2_500, 5_000, 7_500, 10_000, 25_000, 50_000, 75_000, 1_00_000]

    # Ground-truth ECE via large-MC

    rng = np.random.default_rng(seed)

    true_ece_val = true_ece(prob_fn)

    results = {
        "true_ece":    true_ece_val,
        "sample_sizes": list(sample_sizes),
        "rank": {"mean": [], "se": []},
        "bin":  {name: {"mean": [], "se": []} for name in bin_schedules},
    }

    for N in sample_sizes:
        rank_arr = np.empty(n_trials)
        bin_arrs = {name: np.empty(n_trials) for name in bin_schedules}

        for t in range(n_trials):
            z, y = generate_data(z_sampler, prob_fn, N, rng)
            rank_arr[t] = compute_rank_ece(z, y)/true_ece_val
            for name, sched_fn in bin_schedules.items():
                bin_arrs[name][t] = binned_ece(z, y, sched_fn(N))/true_ece_val

        results["rank"]["mean"].append(float(np.mean(rank_arr)))
        results["rank"]["se"].append(
            float(np.std(rank_arr, ddof=1) / np.sqrt(n_trials))
        )

        for name in bin_schedules:
            arr = bin_arrs[name]
            results["bin"][name]["mean"].append(float(np.mean(arr)))
            results["bin"][name]["se"].append(
                float(np.std(arr, ddof=1) / np.sqrt(n_trials))
            )

    return results


def plot_single_experiment(results, title, bin_schedules, RANK_COLOR="crimson", BIN_COLORS=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"],fig_size = None, save_path = None):

    Ns = np.array(results["sample_sizes"])
    x = np.arange(len(Ns))   # Equally spaced x positions
    true_ece = results["true_ece"]

    fig, ax = plt.subplots(figsize=fig_size)

    # Track data extent to adapt y-limits if data exceeds the default range
    data_min = np.inf
    data_max = -np.inf

    # True ECE reference line
    # ax.axhline(1.0, color="black", lw=2.0, ls="--", zorder=6,label=r"True $\ell_2$-ECE" + f"= {true_ece:.4f}",)
    ax.axhline(1.0, color="black", lw=2.0, ls="--")

    # rankECE
    mu = np.array(results["rank"]["mean"])
    se = np.array(results["rank"]["se"])

    ax.plot(x, mu,color=RANK_COLOR,lw=2.0,marker="o",ms=3.5,label="rankECE",zorder=5,)

    ax.fill_between(x, mu - se, mu + se,color=RANK_COLOR,alpha=0.18,zorder=4,)

    data_min = min(data_min, np.min(mu - se))
    data_max = max(data_max, np.max(mu + se))

    # binECE schedules
    for (name, _), col in zip(bin_schedules.items(), BIN_COLORS):
        mu = np.array(results["bin"][name]["mean"])
        se = np.array(results["bin"][name]["se"])

        ax.plot(x, mu,color=col,lw=1.8,marker="s",ms=3.0,label=name,zorder=3,)

        ax.fill_between(x, mu - se, mu + se,color=col,alpha=0.12,zorder=2,)

        data_min = min(data_min, np.min(mu - se))
        data_max = max(data_max, np.max(mu + se))

    # Label ticks with the actual sample sizes
    logNs = np.log10(Ns)
    ticks = np.where(np.isclose(logNs, np.round(logNs)))[0]

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{int(np.log10(Ns[i]))}$" for i in ticks])

    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Ratio")
    # Default y-limits, widened only if the data extends beyond them
    y_lo, y_hi = 0.5, 1.5
    if data_min < y_lo:
        y_lo = data_min-0.1
    if data_max > y_hi:
        y_hi = data_max+0.1
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc = "best")
    if save_path is not None:
        fig.savefig(save_path, dpi = 1200, bbox_inches = "tight")
    return fig, ax

