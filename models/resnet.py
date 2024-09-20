""" Module for ResNet models. """

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.classifier_base import ClassifierBase


class ResNet9(nn.Module):
    """ ResNet module. """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = self._conv_block(in_channels, 64)
        self.conv2 = self._conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(
            self._conv_block(128, 128), self._conv_block(128, 128))
        self.conv3 = self._conv_block(128, 256, pool=True)
        self.conv4 = self._conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(
            self._conv_block(512, 512), self._conv_block(512, 512))
        self.conv5 = self._conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(
            self._conv_block(1028, 1028), self._conv_block(1028, 1028))
        self.classifier = nn.Sequential(nn.MaxPool2d(2), nn.Flatten())

    def _conv_block(self, in_channels: int, out_channels: int, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out


class ResNetClassifier(ClassifierBase):
    """ ResNet classifier. """

    def __init__(self, num_class: int, in_channels: int):
        super().__init__()
        self.encoder = ResNet9(in_channels)
        self.head = nn.Linear(1028, 128)
        self.fc = nn.Linear(1028, num_class)

    def forward(
        self, inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(inputs)
        feat_c = self.head(feat)
        logits = self.fc(feat)
        return F.softmax(logits, dim=1) + 1e-10, logits, F.normalize(feat_c, dim=1)
