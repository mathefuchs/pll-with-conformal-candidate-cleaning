""" Module for base PLL classifier. """

from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from models.classifier_base import ClassifierBase
from partial_label_learning.result import SplitResult


class PllBaseClassifier(ABC):
    """ Base PLL classifier. """

    def __init__(
        self, rng: np.random.Generator, debug: bool,
        model: ClassifierBase, device: torch.device,
        is_small_scale_dataset: bool,
    ) -> None:
        self.rng = rng
        self.debug = debug
        self.model = model
        self.device = device
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.num_epoch = 200
        self.batch_size = 16 if is_small_scale_dataset else 256
        self.max_lr = 0.01
        self.weight_decay = 1e-4

    @abstractmethod
    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs.

        Args:
            inputs (np.ndarray): The inputs.
            partial_targets (np.ndarray): The partial targets.

        Returns:
            SplitResult: The disambiguated targets.
        """

        raise NotImplementedError()

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        raise NotImplementedError()
