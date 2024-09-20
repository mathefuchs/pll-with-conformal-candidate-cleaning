""" Module for simple LeNet architecture. """

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.classifier_base import ClassifierBase


class LeNet(ClassifierBase):
    """ LeNet architecture. """

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),  # x input channels
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Conv2d(6, 16, kernel_size=5),  # 6 input, 16 output channels
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(
        self, inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features(inputs)
        feat = feat.view(-1, 16 * 4 * 4)
        inputs = self.classifier(feat)
        return F.softmax(inputs, dim=1) + 1e-10, inputs, F.normalize(feat, dim=1)
