from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
SCORES_DIR = RESULTS_DIR / "scores"
PLOTS_DIR = RESULTS_DIR / "plots"

for _d in (DATA_DIR, RESULTS_DIR, SCORES_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


DATASET = "yelp_polarity"  # "amazon_polarity" or "yelp_polarity"

DATASET_URLS = {
    # "amazon_polarity": "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz",
    "yelp_polarity": "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz",
}

DATASET_INNER_DIR = {
    # "amazon_polarity": "amazon_review_polarity_csv",
    "yelp_polarity": "yelp_review_polarity_csv",
}

POOL_MODE = "train_test_pool"  # change to "train_test_pool" for more samples

MAX_POOL_SIZE = 50000

RANDOM_SEED = 42

# pretrained models
MODELS = {
    "distilbert_sst2": {
        "type": "hf",
        "hf_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "batch_size": 64,
        "max_length": 256,
    },
    "bert_sst2": {
        "type": "hf",
        "hf_name": "textattack/bert-base-uncased-SST-2",
        "batch_size": 32,
        "max_length": 256,
    },
    "roberta_twitter_sentiment": {
        "type": "hf",
        "hf_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "batch_size": 32,
        "max_length": 256,
    },
    "bert_multilingual_stars": {
        "type": "hf",
        "hf_name": "nlptown/bert-base-multilingual-uncased-sentiment",
        "batch_size": 32,
        "max_length": 256,
    },
}


DEVICE = "mps"

# reference l2-ece computed using debiased bin ece
REFERENCE_BINNING_RULE = "sqrt"  # one of: "fixed10", "sqrt", "cube_root", "n_over_20"


SAMPLE_SIZES = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
N_REPEATS = 200
