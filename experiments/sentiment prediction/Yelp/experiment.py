import numpy as np
import pandas as pd
from tqdm import tqdm

import calibration
import config


def compute_reference(y, z) -> float:
    return calibration.reference_l2_ece(y, z, config.REFERENCE_BINNING_RULE)
    # return calibration.reference_l2_ece(y, z, "cube_root")


def run_experiment_for_model(name, y, z, rng) -> pd.DataFrame:
    n_pool = len(z)
    reference = compute_reference(y, z)
    print(
        f"[{name}] reference L2-ECE ({config.REFERENCE_BINNING_RULE}, "
        f"n={n_pool:,}) = {reference:.6f}"
    )

    sample_sizes = [s for s in config.SAMPLE_SIZES if s < n_pool]
    dropped = sorted(set(config.SAMPLE_SIZES) - set(sample_sizes))
    if dropped:
        print(f"[{name}] skipping sample sizes >= pool size ({n_pool:,}): {dropped}")

    rows = []
    for n in sample_sizes:
        for rep in tqdm(range(config.N_REPEATS), desc=f"{name} n={n}"):
            idx = rng.choice(n_pool, size=n, replace=False)
            y_s, z_s = y[idx], z[idx]
            row = {"model": name, "n": n, "rep": rep, "reference": reference}
            for est_name, est_fn in calibration.ESTIMATORS.items():
                row[est_name] = est_fn(y_s, z_s)
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(config.RANDOM_SEED)
    all_results = []

    for name in config.MODELS:
        score_path = config.SCORES_DIR / f"{name}.csv"
        if not score_path.exists():
            print(f"[skip] no scores found for {name} (run score_models.py first)")
            continue

        df = pd.read_csv(score_path)
        y, z = df["y"].to_numpy(dtype=np.float64), df["z"].to_numpy(dtype=np.float64)
        result = run_experiment_for_model(name, y, z, rng)
        all_results.append(result)

        out_path = config.RESULTS_DIR / f"{name}_results.csv"
        result.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(config.RESULTS_DIR / "all_results.csv", index=False)
        print(f"Saved {config.RESULTS_DIR / 'all_results.csv'}")


if __name__ == "__main__":
    main()
