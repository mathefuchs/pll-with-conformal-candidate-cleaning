""" Module for base classifier. """

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class ClassifierBase(ABC, nn.Module):
    """ Base classifier. """

    @abstractmethod
    def forward(
        self, inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass. Returns (probabilities, logits, features).

        Args:
            inputs (torch.Tensor): The inputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Returns (probabilities, logits, features).
        """

        raise NotImplementedError()
