## Pipeline

Run these in order:

```bash
python download_data.py    # downloads + extracts amazon/yelp_review_polarity_csv.tgz
python build_pool.py       # assembles the (text, label) pool per config.POOL_MODE
python score_models.py     # runs every pretrained model in config.MODELS
python experiment.py       # computes reference L2-ECE + resampling experiment
python plot_relative_error.py         # one PNG per model in results/plots/
```

## Key knobs (all in `config.py`)

- `DATASET`: `"amazon_polarity"` (larger) or `"yelp_polarity"` (smaller, faster).
- `POOL_MODE`: `"test_only"` (original test samples) or `"train_test_pool"`
  (more samples)
- `MAX_POOL_SIZE`: For computational efficiency, subsample from the origianl dataset.
- `MODELS`: pretrained models used for the experiment.
- `SAMPLE_SIZES` / `N_REPEATS`: resampling experiment settings. 

## Project structure

```
rank_ece/
├── requirements.txt
├── config.py
├── download_data.py
├── build_pool.py
├── calibration.py
├── score_models.py
├── experiment.py
├── plotting.py
├── data/                 # downloaded CSVs + pool.parquet
└── results/
    ├── scores/            # {model_name}.csv  -> columns [y, z]
    ├── {model_name}_results.csv
    ├── all_results.csv
    └── plots/{model_name}.png
```
