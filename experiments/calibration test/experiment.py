import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from utils import gen_data, H1_PROB_FUNCS
from tests import (
    rank_ece_asymptotic_test,
    rank_ece_finite_test,
    skce_linear_test,
    skce_ustat_test,
)

DEFAULT_N_VALUES   = [100, 200, 300, 400, 500]
DEFAULT_RHO_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_ALPHA_SIG  = 0.05
DEFAULT_N_REPS     = 200
DEFAULT_N_JOBS     = -1
DEFAULT_RHO_FIXED  = 0.1
DEFAULT_N_FIXED    = 200
DEFAULT_H1_NAME    = "p = Z - Z^2"
DEFAULT_SAVE_DIR   = "results"

def _slugify(name):
    return (
        name.replace(" ", "")
            .replace("=", "")
            .replace("^", "pow")
            .replace("*", "x")
            .replace("/", "_")
            .replace("+", "plus")
            .replace("-", "minus")
            .replace("(", "")
            .replace(")", "")
    )

TESTS = {
    "rankECE (A)": dict(
        fn=rank_ece_asymptotic_test,
        kwargs={},
    ),
    "rankECE (F)": dict(
        fn=rank_ece_finite_test,
        kwargs={},
    ),
    "SKCE-L (G)": dict(
        fn=skce_linear_test,
        kwargs=dict(kernel="gaussian", band="median"),
    ),
    "SKCE-L (L)": dict(
        fn=skce_linear_test,
        kwargs=dict(kernel="laplace", band="median"),
    ),
    "SKCE-U (G)": dict(
        fn=skce_ustat_test,
        kwargs=dict(kernel="gaussian", band="median", n_rep=100),
    ),
    "SKCE-U (L)": dict(
        fn=skce_ustat_test,
        kwargs=dict(kernel="laplace", band="median", n_rep=100),
    ),
}


def _single_rep(test_fn, n, rho, prob_fn, alpha_sig, test_kwargs, seed):


    np.random.seed(seed)

    Z, Y = gen_data(n, rho, prob_fn=prob_fn)
    try:
        reject = test_fn(Z, Y, alpha_sig=alpha_sig, **test_kwargs)
    except Exception:
        # e.g. degenerate variance -> count as non-rejection
        reject = False
    return int(bool(reject))


def run_simulation(test_fn, x_values, x_param, fixed_value,
                    prob_fn=None, n_reps=DEFAULT_N_REPS,
                    alpha_sig=DEFAULT_ALPHA_SIG,
                    n_jobs=DEFAULT_N_JOBS, **test_kwargs):

    np.random.seed(42)

    if x_param not in ("n", "rho"):
        raise ValueError("x_param must be 'n' or 'rho'")

    means = np.zeros(len(x_values))
    ses = np.zeros(len(x_values))

    for j, x in enumerate(x_values):
        if x_param == "n":
            n, rho = int(x), fixed_value
        else:
            n, rho = fixed_value, x

        desc = f"{x_param}={x} (fixed {'rho' if x_param == 'n' else 'n'}={fixed_value})"

        seeds = np.random.randint(0, 2**31 - 1, size=n_reps)

        with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
            outcomes = parallel(
                delayed(_single_rep)(
                    test_fn, n, rho, prob_fn, alpha_sig, test_kwargs, int(seed)
                )
                for seed in tqdm(seeds, desc=desc, leave=False)
            )

        outcomes = np.asarray(outcomes, dtype=float)
        p_hat = outcomes.mean()

        means[j] = p_hat
        ses[j] = outcomes.std()/np.sqrt(n_reps)

    return means, ses


def plot_comparison(results_dict, x_values, xlabel, ylabel, title,
                     alpha_sig=None, hline_at_alpha=False, legend_position = None, anchor = None, save_path=None):

    fig, ax = plt.subplots(figsize=(4, 3))

    x_values = np.asarray(x_values)

    for name, (means, ses) in results_dict.items():
        line, = ax.plot(x_values, means, marker="o", label=name)
        ax.fill_between(
            x_values, means - ses, means + ses,
            color=line.get_color(), alpha=0.2,
        )

    if hline_at_alpha and alpha_sig is not None:
        ax.axhline(alpha_sig, color="black", linestyle="--",
                    linewidth=1, label=f"alpha = {alpha_sig}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    if legend_position is None:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.legend(loc=legend_position, bbox_to_anchor=anchor, fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()

def _save_results(results, path):
    """Pickle `results` (a dict) to `path`, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {path}")


def run_power_vs_n(n_values=DEFAULT_N_VALUES,
                    rho_fixed=DEFAULT_RHO_FIXED,
                    h1_name=DEFAULT_H1_NAME,
                    n_reps=DEFAULT_N_REPS,
                    alpha_sig=DEFAULT_ALPHA_SIG,
                    n_jobs=DEFAULT_N_JOBS,
                    tests=None,
                    plot=True,
                    legend_position = None,
                    anchor = None,
                    save_dir=DEFAULT_SAVE_DIR):

    tests = tests or TESTS
    prob_fn = H1_PROB_FUNCS[h1_name]
    results = {}

    for name, spec in tests.items():
        print(f"[Power vs n: {h1_name}] Running: {name}")
        results[name] = run_simulation(
            spec["fn"], n_values, x_param="n", fixed_value=rho_fixed,
            prob_fn=prob_fn, n_reps=n_reps, alpha_sig=alpha_sig,
            n_jobs=n_jobs, **spec["kwargs"]
        )

    tag = _slugify(h1_name)

    _save_results(
        dict(results=results, n_values=n_values, rho_fixed=rho_fixed,
             h1_name=h1_name, alpha_sig=alpha_sig, n_reps=n_reps),
        os.path.join(save_dir, f"power_vs_n_{tag}_with_rho_{rho_fixed}.pkl"),
    )

    if plot:
        if h1_name == "p = Z":
            plot_comparison(
            results, n_values,
            xlabel="n", ylabel="Rejection rate",
            title=f"Type-I Error",
            alpha_sig=alpha_sig, hline_at_alpha=True,
            legend_position = legend_position,
            anchor = anchor,
            save_path=os.path.join(save_dir, f"power_vs_n_{tag}_with_rho_{rho_fixed}.pdf"),
            )
        else:
            plot_comparison(
                results, n_values,
                xlabel="n", ylabel="Rejection rate",
                title=f"Power",
                alpha_sig=alpha_sig, hline_at_alpha=False,
                legend_position = legend_position,
                anchor = anchor,
                save_path=os.path.join(save_dir, f"power_vs_n_{tag}_with_rho_{rho_fixed}.pdf"),
            )

    return results


def run_power_vs_rho(rho_values=DEFAULT_RHO_VALUES,
                      n_fixed=DEFAULT_N_FIXED,
                      h1_name=DEFAULT_H1_NAME,
                      n_reps=DEFAULT_N_REPS,
                      alpha_sig=DEFAULT_ALPHA_SIG,
                      n_jobs=DEFAULT_N_JOBS,
                      tests=None,
                      plot=True,
                      legend_position = None,
                      save_dir=DEFAULT_SAVE_DIR):

    tests = tests or TESTS
    prob_fn = H1_PROB_FUNCS[h1_name]
    results = {}

    for name, spec in tests.items():
        print(f"[Power vs rho: {h1_name}] Running: {name}")
        results[name] = run_simulation(
            spec["fn"], rho_values, x_param="rho", fixed_value=n_fixed,
            prob_fn=prob_fn, n_reps=n_reps, alpha_sig=alpha_sig,
            n_jobs=n_jobs, **spec["kwargs"]
        )

    tag = _slugify(h1_name)

    _save_results(
        dict(results=results, rho_values=rho_values, n_fixed=n_fixed,
             h1_name=h1_name, alpha_sig=alpha_sig, n_reps=n_reps),
        os.path.join(save_dir, f"power_vs_rho_{tag}_with_n_{n_fixed}.pkl"),
    )

    if plot:
        if h1_name == "p = Z":
            plot_comparison(
                results, rho_values,
                xlabel=r"$\rho$", ylabel="Rejection rate",
                title=f"Type-I Error",
                alpha_sig=alpha_sig, hline_at_alpha=True,
                legend_position = legend_position,
                save_path=os.path.join(save_dir, f"power_vs_rho_{tag}_with_n_{n_fixed}.pdf"),
            )
        else:
            plot_comparison(
            results, rho_values,
            xlabel=r"$\rho$", ylabel="Rejection rate",
            title=f"Power",
            alpha_sig=alpha_sig, hline_at_alpha=False,
            legend_position = legend_position,
            save_path=os.path.join(save_dir, f"power_vs_rho_{tag}_with_n_{n_fixed}.pdf"),
            )

    return results

run_power_vs_n(rho_fixed = 0.3, h1_name = "p = Z")
run_power_vs_rho(n_fixed = 100, h1_name = "p = Z")

run_power_vs_n(rho_fixed = 0.3 ,h1_name = "p = sin", legend_position = "center right", anchor = (0.99, 0.55))
run_power_vs_rho(n_fixed = 100, h1_name = "p = sin")
run_power_vs_rho(n_fixed = 200, h1_name = "p = sin")

run_power_vs_n(rho_fixed = 0.3, h1_name = "p = Z - Z^15", legend_position="center right", anchor = (0.99, 0.6))
run_power_vs_rho(n_fixed = 100, h1_name = "p = Z - Z^15")
run_power_vs_rho(n_fixed = 200, h1_name = "p = Z - Z^15")

# run_power_vs_n(rho_fixed = 0.4, h1_name = "p = Z - Z^25", legend_position="center right", anchor = (0.99, 0.4))

run_power_vs_n(rho_fixed = 0.15, h1_name = "p = Z - Z^4", legend_position="lower right", anchor = (0.99, 0.05))
run_power_vs_rho(n_fixed = 100, h1_name = "p = Z - Z^4")