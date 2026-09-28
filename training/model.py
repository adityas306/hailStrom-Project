import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMNowcaster(nn.Module):
    """5-channel ConvLSTM nowcaster.

    Input:  [B, T, 5, H, W]
    Output: [B, 6, 1, H, W]
    """

    def __init__(self, input_channels=5, hidden_channels=32, output_channels=1, future_steps=6):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.future_steps = future_steps
        self.encoder = ConvLSTMCell(input_channels, hidden_channels)
        self.prediction_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Input must be [B,T,C,H,W]")
        batch, _, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")

        h = torch.zeros(batch, self.hidden_channels, height, width, device=x.device)
        c = torch.zeros_like(h)

        for t in range(x.size(1)):
            h, c = self.encoder(x[:, t], h, c)

        predictions = []
        for _ in range(self.future_steps):
            prediction = self.prediction_head(h)
            predictions.append(prediction)
            # Autoregressive feedback: feed the predicted radar field into channel 0.
            next_input = torch.zeros(
                batch, self.input_channels, height, width, device=x.device
            )
            next_input[:, 0:1] = prediction
            h, c = self.encoder(next_input, h, c)

        return torch.stack(predictions, dim=1)

