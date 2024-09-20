""" Module for loading data. """

import pickle
from typing import List

import numpy as np
import torch
import torchvision
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.classifier_base import ClassifierBase
from partial_label_learning.config import REAL_WORLD_LABEL_TO_PATH


class Dataset:
    """ A dataset. """

    def __init__(
        self, x_full: np.ndarray, y_full: np.ndarray, y_true: np.ndarray,
    ) -> None:
        self.x_full = x_full
        self.y_full = y_full
        self.y_true = y_true

    def copy(self) -> "Dataset":
        """ Copies the dataset.

        Returns:
            Dataset: The copy.
        """

        return Dataset(
            self.x_full.copy(), self.y_full.copy(), self.y_true.copy(),
        )

    def create_data_split(
        self, rng: np.random.Generator, train_frac: float = 0.8,
    ) -> "Datasplit":
        """ Creates a random data split. """

        train_size = int(train_frac * self.x_full.shape[0])
        train_size -= train_size % 16  # Make multiple of 16
        train_ind = rng.choice(
            self.x_full.shape[0], size=train_size,
            replace=False, shuffle=False,
        )
        test_ind = np.setdiff1d(np.arange(self.x_full.shape[0]), train_ind)
        rng.shuffle(train_ind)
        rng.shuffle(test_ind)
        return Datasplit(
            x_train=self.x_full[train_ind].copy(),
            x_test=self.x_full[test_ind].copy(),
            y_train=self.y_full[train_ind].copy(),
            y_true_train=self.y_true[train_ind].copy(),
            y_true_test=self.y_true[test_ind].copy(),
        )


class Datasplit:
    """ A data split. """

    def __init__(
        self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray,
        y_true_train: np.ndarray, y_true_test: np.ndarray,
    ) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_true_train = y_true_train
        self.y_true_test = y_true_test

    def copy(self) -> "Datasplit":
        """ Copies the datasplit.

        Returns:
            Datasplit: The copy.
        """

        return Datasplit(
            self.x_train.copy(), self.x_test.copy(), self.y_train.copy(),
            self.y_true_train.copy(), self.y_true_test.copy(),
        )

    def augment_targets_uniform(
        self,
        rng: np.random.Generator,
        partial_rate: float,
        is_cifar100: bool,
    ) -> "Datasplit":
        """ Augments a supervised dataset with random label candidates. """

        # Get classes that can cooccurr together
        num_class = self.y_train.shape[1]
        if is_cifar100:
            coocc_mask = _cifar_100_coocc_mask()
        else:
            coocc_mask = np.ones((num_class, num_class))

        # Compute flip probability
        flip_probs = np.zeros_like(self.y_train, dtype=float)
        for i, y_true in enumerate(self.y_true_train):
            flip_probs[i] = partial_rate * coocc_mask[y_true]
            flip_probs[i, y_true] = 1
        sample = rng.random(flip_probs.shape)
        y_train_copy = (1 * (flip_probs > sample)).copy()
        return Datasplit(
            self.x_train, self.x_test, y_train_copy,
            self.y_true_train, self.y_true_test,
        )

    def augment_targets_instance_dependent(
        self, model: ClassifierBase, device: torch.device,
    ) -> "Datasplit":
        """ Augments a supervised dataset with instance-dependent noise. """

        # Create optimizer
        optim = torch.optim.SGD(model.parameters())
        model.train()

        # Accelerate training
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)

        # Prepare data
        x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        data_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=256, shuffle=True,
        )

        # Training loop
        loss_fn = nn.MSELoss()
        for _ in range(100):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                with ctx:
                    probs, _, _ = model(x_batch)
                    loss = loss_fn(probs, y_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()

        # Inference
        model.eval()
        inference_loader = DataLoader(
            TensorDataset(torch.tensor(self.x_train, dtype=torch.float32)),
            batch_size=256, shuffle=False,
        )
        with torch.no_grad():
            all_results = []
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(device)
                all_results.append(model(x_batch)[0].cpu().numpy())
            train_probs = np.vstack(all_results)

        # Determine augmentation probabilities
        train_false_probs = (1 - self.y_train) * train_probs
        max_false_probs = np.max(train_false_probs, axis=1, keepdims=True)
        train_false_probs /= np.where(
            max_false_probs > 1e-10, max_false_probs, 1.0)
        mean_false_probs = np.mean(
            train_false_probs, axis=1, keepdims=True)
        train_false_probs = 0.8 * train_false_probs / np.where(
            mean_false_probs > 1e-10, mean_false_probs, 1.0)
        train_false_probs = np.clip(train_false_probs, 0.0, 1.0)

        # Augmentation
        sampler = torch.distributions.binomial.Binomial(
            total_count=1, probs=torch.tensor(train_false_probs))
        sample = sampler.sample()
        y_train_copy = self.y_train.copy()
        y_train_copy[sample == 1] = 1

        return Datasplit(
            self.x_train, self.x_test, y_train_copy,
            self.y_true_train, self.y_true_test,
        )


class Experiment:
    """ An experiment. """

    def __init__(
        self, dataset_name: str, dataset_kind: str,
        seed: int, datasplit: Datasplit,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_kind = dataset_kind
        self.seed = seed
        self.datasplit = datasplit

    def __str__(self) -> str:
        return f"{self.dataset_name}_{self.seed}"


def get_rl_dataset(dataset_name: str) -> Dataset:
    """ Retrieves a real-world dataset. """

    # Coerce data into dense array
    def coerce(data) -> np.ndarray:
        try:
            return data.toarray()
        except:  # pylint: disable=bare-except
            return data

    # Extract raw data
    raw_mat_data = loadmat(REAL_WORLD_LABEL_TO_PATH[dataset_name])
    x_raw = coerce(raw_mat_data["data"])
    y_partial_raw = coerce(raw_mat_data["partial_target"].transpose())
    y_true_raw = np.argmax(
        coerce(raw_mat_data["target"].transpose()), axis=1)

    # Number of classes representing 99% of all occurrences
    num_classes = int(np.where(np.cumsum(
        np.array(list(reversed(list(np.sort(np.count_nonzero(
            coerce(raw_mat_data["target"].transpose()), axis=0
        )))))) / y_true_raw.shape[0]) > 0.99
    )[0].min())
    num_classes = min(num_classes + 1, int(y_partial_raw.shape[1]))
    classes_in_use = set(map(int, np.sort(np.argsort(
        np.count_nonzero(y_partial_raw, axis=0))[-num_classes:])))

    # Collect all relevant data
    x_list = []
    y_partial_list = []
    y_true_list: List[int] = []
    mask = np.array(list(sorted(list(classes_in_use))))
    for x_row, y_partial_row, y_true_row in zip(
        x_raw, y_partial_raw, y_true_raw,
    ):
        if int(y_true_row) in classes_in_use:
            x_list.append(x_row)
            y_partial_list.append(y_partial_row[mask])
            y_true_list.append(int(np.where(
                mask == int(y_true_row))[0][0]))
    x_arr = np.array(x_list)
    y_partial_arr = np.array(y_partial_list)
    y_true_arr = np.array(y_true_list)

    # Normalize
    x_arr_min = np.min(x_arr, axis=0, keepdims=True)
    x_arr_max = np.max(x_arr, axis=0, keepdims=True)
    diff = x_arr_max - x_arr_min
    diff = np.where(diff < 1e-10, 1., diff)
    x_arr = (x_arr - x_arr_min) / diff
    x_arr = x_arr[:, x_arr.std(axis=0) > 1e-10].copy()

    # Store dataset
    return Dataset(x_arr, y_partial_arr, y_true_arr)


def get_torch_dataset(dataset_name: str) -> Datasplit:
    """ Retrieves a dataset from Pytorch. """

    # MNIST
    if dataset_name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root="data", train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(
            root="data", train=False, download=True)
        x_train = (train_dataset.data.numpy()[:, None, :, :] / 255).copy()
        x_test = (test_dataset.data.numpy()[:, None, :, :] / 255).copy()
        y_train_true = train_dataset.targets.numpy().copy()
        y_test_true = test_dataset.targets.numpy().copy()

    # FMNIST
    elif dataset_name == "fmnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root="data", train=True, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(
            root="data", train=False, download=True)
        x_train = (train_dataset.data.numpy()[:, None, :, :] / 255).copy()
        x_test = (test_dataset.data.numpy()[:, None, :, :] / 255).copy()
        y_train_true = train_dataset.targets.numpy().copy()
        y_test_true = test_dataset.targets.numpy().copy()

    # KMNIST
    elif dataset_name == "kmnist":
        train_dataset = torchvision.datasets.KMNIST(
            root="data", train=True, download=True)
        test_dataset = torchvision.datasets.KMNIST(
            root="data", train=False, download=True)
        x_train = (train_dataset.data.numpy()[:, None, :, :] / 255).copy()
        x_test = (test_dataset.data.numpy()[:, None, :, :] / 255).copy()
        y_train_true = train_dataset.targets.numpy().copy()
        y_test_true = test_dataset.targets.numpy().copy()

    # CIFAR-10
    elif dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="data", train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True)
        x_train = (np.transpose(train_dataset.data, (0, 3, 1, 2)) / 255).copy()
        x_test = (np.transpose(test_dataset.data, (0, 3, 1, 2)) / 255).copy()
        y_train_true = np.array(train_dataset.targets, dtype=int).copy()
        y_test_true = np.array(test_dataset.targets, dtype=int).copy()
        mu = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(x_train, axis=(0, 2, 3), keepdims=True)
        x_train = (x_train - mu) / std
        x_test = (x_test - mu) / std

    # CIFAR-100
    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="data", train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            root="data", train=False, download=True)
        x_train = (np.transpose(train_dataset.data, (0, 3, 1, 2)) / 255).copy()
        x_test = (np.transpose(test_dataset.data, (0, 3, 1, 2)) / 255).copy()
        y_train_true = np.array(train_dataset.targets, dtype=int).copy()
        y_test_true = np.array(test_dataset.targets, dtype=int).copy()
        mu = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(x_train, axis=(0, 2, 3), keepdims=True)
        x_train = (x_train - mu) / std
        x_test = (x_test - mu) / std

    # SVHN
    elif dataset_name == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            root="data", split="train", download=True)
        test_dataset = torchvision.datasets.SVHN(
            root="data", split="test", download=True)
        x_train = (train_dataset.data / 255).copy()
        x_test = (test_dataset.data / 255).copy()
        y_train_true = train_dataset.labels.copy()
        y_test_true = test_dataset.labels.copy()
        mu = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(x_train, axis=(0, 2, 3), keepdims=True)
        x_train = (x_train - mu) / std
        x_test = (x_test - mu) / std
    else:
        raise ValueError(f"Invalid dataset name '{dataset_name}'.")

    # Extract targets
    l_classes = np.unique(y_train_true).shape[0]
    y_train = np.zeros((x_train.shape[0], l_classes), dtype=int)
    for i, y_val in enumerate(y_train_true):
        y_train[i, y_val] = 1

    # Create dataset
    return Datasplit(
        x_train, x_test, y_train, y_train_true, y_test_true,
    )


def _cifar_100_coocc_mask() -> np.ndarray:
    """ Returns all classes that can co-occur together. """

    with open("data/cifar-100-python/meta", "rb") as file:
        metadata = pickle.load(file, encoding="latin1")["fine_label_names"]
    idx_to_class = list(metadata)
    class_to_idx = {lbl: i for i, lbl in enumerate(idx_to_class)}
    super_classes = [
        ["beaver", "dolphin", "otter", "seal", "whale"],
        ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        ["orchid", "poppy", "rose", "sunflower", "tulip"],
        ["bottle", "bowl", "can", "cup", "plate"],
        ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        ["clock", "keyboard", "lamp", "telephone", "television"],
        ["bed", "chair", "couch", "table", "wardrobe"],
        ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        ["fox", "porcupine", "possum", "raccoon", "skunk"],
        ["bear", "leopard", "lion", "tiger", "wolf"],
        ["bridge", "castle", "house", "road", "skyscraper"],
        ["cloud", "forest", "mountain", "plain", "sea"],
        ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        ["crab", "lobster", "snail", "spider", "worm"],
        ["baby", "boy", "girl", "man", "woman"],
        ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    ]
    coocc_mask = np.zeros((100, 100), dtype=int)
    for super_cls in super_classes:
        for item1 in super_cls:
            for item2 in super_cls:
                idx1 = class_to_idx[item1]
                idx2 = class_to_idx[item2]
                coocc_mask[idx1, idx2] = 1
                coocc_mask[idx2, idx1] = 1
    return coocc_mask
