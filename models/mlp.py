""" Multi-Layer Perceptron. """

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.classifier_base import ClassifierBase


class MLP(ClassifierBase):
    """ Standard MLP classifier. """

    def __init__(
        self, m_features: int, l_classes: int,
    ) -> None:
        super().__init__()
        self.m_features = m_features
        self.mlp = nn.Sequential(
            nn.Linear(m_features, 300), nn.ReLU(),  # Layer 1-2
            nn.BatchNorm1d(300), nn.Linear(300, 300), nn.ReLU(),  # Layer 2-3
            nn.BatchNorm1d(300), nn.Linear(300, 300), nn.ReLU(),  # Layer 3-4
            nn.BatchNorm1d(300), nn.Linear(300, l_classes),  # Layer 4-5
        )

    def forward(
        self, inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = inputs.view(-1, self.m_features)
        logits = self.mlp(inputs)
        return F.softmax(logits, dim=1), logits, None


class MLPFeature(ClassifierBase):
    """ MLP with latent feature representation. """

    def __init__(self, m_features: int, l_classes: int):
        super().__init__()
        self.m_features = m_features
        self.encoder = nn.Sequential(
            nn.Linear(m_features, 300), nn.ReLU(),  # Layer 1-2
            nn.BatchNorm1d(300), nn.Linear(300, 300), nn.ReLU(),  # Layer 2-3
            nn.BatchNorm1d(300), nn.Linear(300, 300), nn.ReLU(),  # Layer 3-4
            nn.BatchNorm1d(300), nn.Linear(300, 128),  # Layer 4-5
        )
        self.fc = nn.Linear(128, l_classes)

    def forward(
        self, inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = inputs.view(-1, self.m_features)
        feat = self.encoder(inputs)
        logits = self.fc(feat)
        return F.softmax(logits, dim=1) + 1e-10, logits, F.normalize(feat, dim=1)
