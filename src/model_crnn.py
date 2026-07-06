from __future__ import annotations
"""CRNN + CTC model for recognizing complete license plate strings.

The CNN compresses the plate image height while preserving a left-to-right
feature sequence along width. The bidirectional LSTM models character order,
and the final linear layer predicts a character-or-blank class per time step
for CTC decoding.
"""

import torch
from torch import nn

from plate_chars import NUM_CLASSES


class CRNNLicensePlate(nn.Module):
    """Convolutional recurrent recognizer for cropped plate images."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return CTC logits in ``[time, batch, classes]`` format."""

        features = self.cnn(images)
        if features.size(2) != 1:
            raise RuntimeError(f"Expected CNN feature height 1, got {tuple(features.shape)}")
        # Convert CNN output from [N, C, 1, W] to [N, W, C] so each width
        # position becomes one time step for the LSTM.
        features = features.squeeze(2).permute(0, 2, 1)
        sequence_features, _ = self.rnn(features)
        logits = self.classifier(sequence_features)
        return logits.permute(1, 0, 2)


if __name__ == "__main__":
    model = CRNNLicensePlate()
    dummy = torch.randn(2, 1, 48, 160)
    output = model(dummy)
    print("output shape:", output.shape)
    assert output.dim() == 3
    assert output.size(1) == dummy.size(0)
