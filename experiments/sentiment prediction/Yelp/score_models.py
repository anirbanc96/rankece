import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from build_pool import build_pool


def _get_device():
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _score_hf_model(texts, model_cfg, device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_cfg["hf_name"])
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg["hf_name"])
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    pos_idx = None
    if num_labels == 2:
        id2label = {k: v.lower() for k, v in model.config.id2label.items()}
        pos_idx = next((k for k, v in id2label.items() if "pos" in v), 1)

    batch_size = model_cfg["batch_size"]
    max_length = model_cfg["max_length"]
    probs = np.empty(len(texts), dtype=np.float64)

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=model_cfg["hf_name"]):
            batch = texts[start:start + batch_size]
            enc = tok(
                list(batch), padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            softmax = torch.softmax(model(**enc).logits, dim=-1)

            if num_labels == 2:
                p_pos = softmax[:, pos_idx]
            else:
                # Multi-class heads (e.g. 1-5 star ratings, or
                # negative/neutral/positive): treat the top half of classes
                # as "positive".
                cutoff = (num_labels + 1) // 2
                p_pos = softmax[:, cutoff:].sum(dim=-1)

            probs[start:start + len(batch)] = p_pos.detach().cpu().numpy()

    del model
    return probs


def main():
    pool_path = config.DATA_DIR / "pool.parquet"
    pool = pd.read_parquet(pool_path) if pool_path.exists() else build_pool()

    texts = pool["text"].tolist()
    y = pool["y"].to_numpy()
    device = _get_device()
    print(f"Using device: {device}")

    for name, model_cfg in config.MODELS.items():
        out_path = config.SCORES_DIR / f"{name}.csv"
        if out_path.exists():
            print(f"[skip] {name} already scored.")
            continue

        print(f"Scoring model: {name}")
        if model_cfg["type"] == "hf":
            z = _score_hf_model(texts, model_cfg, device)
        else:
            raise ValueError(f"Unknown model type for {name}: {model_cfg['type']}")

        z = np.clip(z, 1e-6, 1 - 1e-6)
        pd.DataFrame({"y": y, "z": z}).to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(z):,} rows)")


if __name__ == "__main__":
    main()
