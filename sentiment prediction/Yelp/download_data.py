import tarfile

import requests
from tqdm import tqdm

import config


def download_file(url: str, dest_path):
    if dest_path.exists():
        print(f"[skip] {dest_path.name} already downloaded.")
        return
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with open(tmp_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    tmp_path.rename(dest_path)


def extract(archive_path, dest_dir):
    inner_dir = dest_dir / config.DATASET_INNER_DIR[config.DATASET]
    if inner_dir.exists():
        print(f"[skip] {inner_dir.name} already extracted.")
        return
    print(f"Extracting {archive_path.name} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def main():
    url = config.DATASET_URLS[config.DATASET]
    archive_path = config.DATA_DIR / url.split("/")[-1]
    download_file(url, archive_path)
    extract(archive_path, config.DATA_DIR)
    inner = config.DATA_DIR / config.DATASET_INNER_DIR[config.DATASET]
    print(f"Done. train.csv / test.csv are under: {inner}")


if __name__ == "__main__":
    main()
