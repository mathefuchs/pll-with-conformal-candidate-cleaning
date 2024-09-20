""" Configurations. """

from glob import glob
from typing import Dict, Tuple

# Dataset kind
DATASET_KIND = {
    "rl": 0,
    "mnist": 1,
    "cifar": 2,
}

# Data splits
SPLIT_IDX = {
    "train": 0,
    "test": 1,
}

# Data
SELECTED_DATASETS: Dict[str, Tuple[int, str]] = {
    # Real-world datasets
    "bird-song": (0, "rl"),
    "lost": (1, "rl"),
    "mir-flickr": (2, "rl"),
    "msrc-v2": (3, "rl"),
    "soccer": (4, "rl"),
    "yahoo-news": (5, "rl"),
    # Supervised datasets
    "mnist": (6, "mnist"),
    "fmnist": (7, "mnist"),
    "kmnist": (8, "mnist"),
    "svhn": (9, "mnist"),
    "cifar10": (10, "cifar"),
    "cifar100": (11, "cifar"),
}

# All real-world datasets
REAL_WORLD_DATA = list(sorted(
    glob("data/realworld-datasets/*.mat")
))
REAL_WORLD_DATA_LABELS = [
    path.split("/")[-1].split(".")[0] for path in REAL_WORLD_DATA
]
REAL_WORLD_LABEL_TO_PATH = dict(zip(REAL_WORLD_DATA_LABELS, REAL_WORLD_DATA))
