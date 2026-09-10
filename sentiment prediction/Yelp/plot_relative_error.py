from config import N_REPEATS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

ESTIMATOR_LABELS = {
    "rank_ece": "rankECE",
    "hist_10": r"$\ell_2$-binECE (K=10)",
    "hist_sqrt": r"$\ell_2$-binECE (K=$\sqrt{n}$)",
    "hist_cube_root": r"$\ell_2$-binECE (K=$n^{1/3}$)",
    "hist_n_over_20": r"$\ell_2$-binECE (K=$n/20$)",
}

MIN_REFERENCE_FOR_RELATIVE_ERROR = 5e-4


def plot_relative_error(name: str, df: pd.DataFrame):
    reference = df["reference"].iloc[0]
    if reference < MIN_REFERENCE_FOR_RELATIVE_ERROR:
        print(
            f"[skip] {name}: reference L2-ECE={reference:.2e} is too close to "
            f"zero for a stable relative-error plot "
            f"(threshold={MIN_REFERENCE_FOR_RELATIVE_ERROR:.0e})."
        )
        return

    fig, ax = plt.subplots(figsize=(4, 3))

    for est_name, label in ESTIMATOR_LABELS.items():
        rel_err = df[est_name] / reference
        grouped = rel_err.groupby(df["n"])
        mean = grouped.mean()
        std = grouped.std()/np.sqrt(N_REPEATS)

        line, = ax.plot(mean.index, mean.values, marker="o", label=label)
        ax.fill_between(
            mean.index,
            mean.values - std.values,
            mean.values + std.values,
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

    ax.axhline(
        1.0, color="black", linestyle="--", linewidth=1.5,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"n")
    ax.set_ylabel("Ratio")
    # ax.set_title(
    #     f"Relative estimator error vs. sample size — {name}\n"
    #     f"(reference L2-ECE = {reference:.5f}; shaded band = ±1 SD across repetitions)"
    # )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.3)
    fig.tight_layout()

    out_path = config.PLOTS_DIR / f"{name}_relative_error_Yelp.pdf"
    fig.savefig(out_path, dpi=1200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    result_files = sorted(
        p for p in config.RESULTS_DIR.glob("*_results.csv")
        if p.name != "all_results.csv"
    )
    if not result_files:
        print("No *_results.csv files found in results/ -- run an experiment script first.")
        return

    for path in result_files:
        name = path.stem[: -len("_results")]
        df = pd.read_csv(path)
        plot_relative_error(name, df)


if __name__ == "__main__":
    main()
