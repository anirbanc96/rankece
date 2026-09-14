# rankECE

Code for [A Ranking Approach for Measuring Calibration](https://arxiv.org/abs/2609.13100).

The repository contains experiments comparing rankECE with binned calibration error estimates, sentiment prediction experiments using pretrained models, and tests for perfect calibration.

## Directory structure

```text
.
├── example/                 # Example varying frequency m and sample size n
│   ├── Over_m.py
│   ├── Over_n.py
│   ├── plot_functions.py
│   └── utils.py
├── binece comparison/       # Synthetic comparisons of rankECE and binECE
│   ├── experiments.py
│   ├── probability_functions.py
│   ├── plot_functions.py
│   ├── utils.py
│   └── plots/
├── calibration test/        # Calibration test power and timing comparisons
│   ├── experiment.py
│   ├── tests.py
│   ├── timing.py
│   ├── utils.py
│   └── results/
├── sentiment prediction/    # Comparing rankECE and binECE on sentiment prediction tasks
│   ├── Amazon/
│   └── Yelp/
├── manuscript/              # Paper source, compiled PDF, and figures
├── LICENSE
└── README.md
```

Both sentiment directories have the following structure (`data/` and score/result CSVs are generated when running the scripts):

```text
Amazon/ or Yelp/
├── config.py                # Dataset, models, device, and experiment settings
├── requirements.txt
├── download_data.py         # Download and extract the review dataset
├── build_pool.py            # Prepare review texts and binary sentiment labels
├── score_models.py          # Compute pretrained model probabilities
├── calibration.py           # rankECE, binECE, and reference estimators
├── experiment.py            # Compare estimates over repeated subsamples
├── plot_relative_error.py   # Plot estimator-to-reference ratios
├── data/                    # Downloaded data and pool.parquet
└── results/
    ├── scores/              # One prediction CSV per model
    ├── <model>_results.csv
    ├── all_results.csv
    └── plots/               # PDF figures
```

## Running the experiments

Create and activate an environment from the repository root, then install the dependencies for the synthetic experiments and calibration tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy matplotlib tqdm joblib reportlab
```

Start each subsection from the repository root with this environment active. Settings are defined in the scripts; reduce sample sizes or repetitions for shorter runs.

### Example with varying frequency m and sample size n

```bash
cd example
python Over_m.py
python Over_n.py
python plot_functions.py
```

These vary the frequency `m` and sample size `n`, and plot ratio of rankECE and binECE to ECE. PDFs are saved in `example/`. Adjust `m_values`, `n_values`, and `num_runs` in the corresponding scripts.

### Synthetic rankECE–binECE comparisons

```bash
cd "binece comparison"
python experiments.py
python plot_functions.py
```

Comparison PDFs are saved in `plots/`; function plots are saved in the current directory. Choose probability functions and binning rules in `experiments.py`.

### Calibration tests

```bash
cd "calibration test"
python experiment.py
python timing.py
```

These compare rankECE and kernel-based calibration tests (SKCE). Power/type-I error plots and simulation results (`.pkl`), plus timing tables (`.csv` and `.pdf`), are saved in `results/`. Adjust the defaults and experiment calls in `experiment.py`, and the settings in the main block of `timing.py`.

### Sentiment prediction: Amazon and Yelp

Both datasets use the same workflow and dependencies to evaluate four pretrained sentiment models. From the repository root:

```bash
python -m pip install -r "sentiment prediction/Amazon/requirements.txt"
cd "sentiment prediction/Amazon"  # Use "sentiment prediction/Yelp" for Yelp.
python download_data.py
python build_pool.py
python score_models.py
python experiment.py
python plot_relative_error.py
```

In each dataset's `config.py`, set `DEVICE` to `"mps"`, `"cuda"`, or `"cpu"` (default: `"mps"`, with CPU fallback), and adjust `MODELS`, batch sizes, `MAX_POOL_SIZE`, `SAMPLE_SIZES`, or `N_REPEATS` as needed. Defaults use up to 50,000 reviews and 200 repetitions; Amazon uses the test split, while Yelp pools train and test. Keep sample sizes below the pool size.

The first run downloads the data and model weights. Outputs are `data/pool.parquet`, prediction CSVs in `results/scores/`, experiment CSVs in `results/`, and PDFs in `results/plots/`. Repeat the commands in the other dataset directory for the second experiment. 
