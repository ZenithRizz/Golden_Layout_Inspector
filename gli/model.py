"""Small CNN defect classifier (Hailo-friendly: Conv/BN/ReLU/MaxPool/GAP/Linear only).

Input: N x 3 x 64 x 64, raw 0-255 values (captured gray, golden gray, |diff|).
Normalisation is inside the graph so the NPU can take uint8 frames directly.
"""
import torch
import torch.nn as nn

from .vision import CLASSES


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class DefectNet(nn.Module):
    def __init__(self, n_classes=len(CLASSES), width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(_block(3, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 4 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        x = x * (1.0 / 255.0)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
