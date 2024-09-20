""" Create data for PLL experiments. """

import numpy as np
import torch
from joblib import Parallel, delayed

from models.model_util import create_model, get_model_arch
from partial_label_learning.config import SELECTED_DATASETS
from partial_label_learning.data import (Experiment, get_rl_dataset,
                                         get_torch_dataset)


def create_experiment_data(
    dataset_name: str, dataset_kind: str, seed: int,
):
    """ Create experiment data. """

    # Init random generator
    torch.manual_seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))

    # Load dataset
    if dataset_kind == "rl":
        dataset = get_rl_dataset(dataset_name)
        datasplit = dataset.create_data_split(rng)
    elif dataset_kind in ("mnist", "cifar"):
        datasplit = get_torch_dataset(dataset_name)
    else:
        raise ValueError()

    # Augment dataset
    if dataset_kind == "mnist":
        # Instance-dependent noise for MNIST datasets
        arch = get_model_arch(dataset_name)
        model, device = create_model(
            arch, datasplit.y_train.shape[1], datasplit.x_train.shape[1:])
        datasplit = datasplit.augment_targets_instance_dependent(model, device)
    elif dataset_kind == "cifar":
        # Uniform noise using hierarchie
        datasplit = datasplit.augment_targets_uniform(
            rng, partial_rate=0.1, is_cifar100=dataset_name == "cifar100")

    # Save experiment
    exp = Experiment(
        dataset_name, dataset_kind, seed, datasplit)
    torch.save(exp, f"./experiments/{exp}.pt")
    avg_candidates = float(np.mean(
        np.count_nonzero(exp.datasplit.y_train, axis=1)))
    frac_supervised = np.count_nonzero(np.count_nonzero(
        exp.datasplit.y_train, axis=1) == 1) / exp.datasplit.y_train.shape[0]
    print(
        f"Successfully wrote '{exp}.pt' with {avg_candidates:.3f} "
        f"avg. candidates and {frac_supervised:.3f} supervised instances."
    )


if __name__ == "__main__":
    # Make sure all data is downloaded
    for n, (_, k) in SELECTED_DATASETS.items():
        if k != "rl":
            get_torch_dataset(n)

    # Create experiment data
    Parallel(n_jobs=2)(
        delayed(create_experiment_data)(
            dataset_name, dataset_kind, seed,
        )
        for dataset_name, (_, dataset_kind) in SELECTED_DATASETS.items()
        for seed in range(5)
    )
