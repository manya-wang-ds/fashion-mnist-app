"""Evaluate the committed CNN checkpoint on the official Fashion-MNIST test set.

Images go through the same transform as the Streamlit app (src.predict_cnn.preprocess_image:
pixel / 255, no other normalization). The official idx files are downloaded from
github.com/zalandoresearch/fashion-mnist into data/fashion/ if they are not already there.

Usage:
    python scripts/evaluate_cnn.py [--data-dir data/fashion]
"""
import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict_cnn import load_model, preprocess_image  # noqa: E402


BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
IMAGES_FILE = "t10k-images-idx3-ubyte.gz"
LABELS_FILE = "t10k-labels-idx1-ubyte.gz"
LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def download(data_dir):
    """Fetch the official test-set idx files if they are missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (IMAGES_FILE, LABELS_FILE):
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {BASE_URL + name}")
            urllib.request.urlretrieve(BASE_URL + name, path)


def read_idx(path, magic, expected_shape):
    """Read a gzipped idx file and check its header against the expected shape."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    header_size = 4 * (1 + len(expected_shape))
    header = np.frombuffer(data[:header_size], dtype=">i4")
    if header[0] != magic or tuple(header[1:]) != expected_shape:
        raise ValueError(f"Unexpected idx header in {path}: {header.tolist()}")
    return np.frombuffer(data[header_size:], dtype=np.uint8).reshape(expected_shape)


def per_class_metrics(y_true, y_pred, num_classes):
    """Return per-class precision, recall, F1 and support from a confusion matrix."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    precision = np.divide(true_positive, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(num_classes), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(num_classes), where=denominator > 0)
    return precision, recall, f1, support


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "fashion")
    args = parser.parse_args()

    download(args.data_dir)
    images = read_idx(args.data_dir / IMAGES_FILE, 2051, (10000, 28, 28))
    labels = read_idx(args.data_dir / LABELS_FILE, 2049, (10000,)).astype(np.int64)

    inputs = torch.cat([preprocess_image(image) for image in images])
    model = load_model()
    with torch.no_grad():
        predictions = torch.cat(
            [model(batch).argmax(dim=1) for batch in inputs.split(1000)]
        ).numpy()

    accuracy = float((predictions == labels).mean())
    precision, recall, f1, support = per_class_metrics(labels, predictions, len(LABELS))

    print(f"Test images: {len(labels)}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {f1.mean():.4f}")
    print()
    print(f"{'Class':<12} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>8}")
    for name, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        print(f"{name:<12} {p:>9.4f} {r:>9.4f} {f:>9.4f} {s:>8d}")


if __name__ == "__main__":
    main()
