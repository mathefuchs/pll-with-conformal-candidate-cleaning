""" Module for creating models. """

import random
from math import prod
from typing import Tuple

import torch

from models.classifier_base import ClassifierBase
from models.lenet import LeNet
from models.mlp import MLP, MLPFeature
from models.resnet import ResNetClassifier
from partial_label_learning.config import SELECTED_DATASETS


def get_model_arch(dataset_name: str, algo_name: str = "") -> str:
    """ Get model arch. """

    if dataset_name not in SELECTED_DATASETS:
        raise ValueError("Invalid dataset.")

    if SELECTED_DATASETS[dataset_name][0] < 9:
        # MLP for small datasets
        if "valen" in algo_name or "pico" in algo_name:
            return "mlpfeat"
        return "mlp"
    if 9 <= SELECTED_DATASETS[dataset_name][0] < 12:
        return "resnet"  # ResNet for color images

    raise ValueError("Unreachable.")


def create_model(
    arch: str, num_class: int, input_shape: Tuple,
) -> Tuple[ClassifierBase, torch.device]:
    """ Create a model with the given architecture. """

    if arch == "mlp":
        # MLP with flattened input
        m_features = prod(input_shape)
        model: ClassifierBase = MLP(m_features, num_class)
    elif arch == "mlpfeat":
        # MLP with latent features
        m_features = prod(input_shape)
        model: ClassifierBase = MLPFeature(m_features, num_class)
    elif arch == "lenet":
        # LeNet architecture
        in_channels = input_shape[0]
        model: ClassifierBase = LeNet(num_class, in_channels)
    elif arch == "resnet":
        # ResNet-18 architecture
        in_channels = input_shape[0]
        model: ClassifierBase = ResNetClassifier(num_class, in_channels)
    else:
        raise ValueError(f"Invalid architecture '{arch}'.")

    if torch.cuda.is_available():
        cuda_idx = random.randrange(torch.cuda.device_count())
        device = torch.device(f"cuda:{cuda_idx}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model, device
