# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch import flatten
from torch.nn import functional as F
import timm


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class CNN(nn.Module):
    def __init__(self, nclass=1, **kwargs):
        super(CNN, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,  # 400
                hop_length=40,  # 160
                window_fn=torch.hamming_window,
                n_mels=40,
            ),
        )

        self.backbone = timm.create_model(
            "resnet50",
            pretrained=False,
            num_classes=nclass,
        )
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, nclass),
        )

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
        # print(x.shape)
        x = x.unsqueeze(1)
        x = torch.cat([x, x, x], dim=1)
        cl = self.backbone(x)
        return cl


if __name__ == "__main__":
    model = CNN(in_features=1, nclass=1)
    print(model)
    x = torch.Tensor(np.random.rand(32, 32000))
    y = model(x)
    print(y.shape)
