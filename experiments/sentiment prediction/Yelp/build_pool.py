import numpy as np
import pandas as pd

import config

COLUMN_NAMES = ["label", "title", "text"]


def _load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, dtype={"label": int})
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    # Amazon/Yelp polarity convention: label 1 = negataive, label 2 = positive.
    df["y"] = (df["label"] == 2).astype(int)
    return df[["text", "y"]]


def build_pool() -> pd.DataFrame:
    inner = config.DATA_DIR / config.DATASET_INNER_DIR[config.DATASET]
    test_df = _load_csv(inner / "test.csv")

    if config.POOL_MODE == "test_only":
        pool = test_df
    elif config.POOL_MODE == "train_test_pool":
        train_df = _load_csv(inner / "train.csv")
        pool = pd.concat([train_df, test_df], ignore_index=True)
    else:
        raise ValueError(f"Unknown POOL_MODE: {config.POOL_MODE}")

    pool = pool.reset_index(drop=True)

    if config.MAX_POOL_SIZE is not None and len(pool) > config.MAX_POOL_SIZE:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = rng.choice(len(pool), size=config.MAX_POOL_SIZE, replace=False)
        pool = pool.iloc[idx].reset_index(drop=True)

    print(
        f"Pool built: {len(pool):,} rows "
        f"(dataset={config.DATASET}, mode={config.POOL_MODE}, "
        f"positive rate={pool['y'].mean():.3f})"
    )
    return pool


if __name__ == "__main__":
    pool = build_pool()
    out_path = config.DATA_DIR / "pool.parquet"
    pool.to_parquet(out_path, index=False)
    print(f"Saved pool to {out_path}")
