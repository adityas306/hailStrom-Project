import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def resize(frame, size=64):
    x = torch.from_numpy(frame[None, None]).float()
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def make_five_channels(frame, previous=None):
    radar = resize(frame)
    gy, gx = np.gradient(radar)
    smooth = radar * 0.85 + 0.15 * np.mean(radar)
    temporal = np.zeros_like(radar) if previous is None else radar - previous
    channels = np.stack([
        radar,
        np.clip((gx + 1) / 2, 0, 1),
        smooth,
        np.clip((gy + 1) / 2, 0, 1),
        np.clip(temporal + 0.5, 0, 1),
    ]).astype(np.float32)
    return channels


class WeatherDataset(Dataset):
    def __init__(self, root_dir, input_frames=4, output_frames=6, augment=False):
        self.root_dir = root_dir
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.augment = augment
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
        self.files = sorted(os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".npy"))
        self.sequence_length = input_frames + output_frames

    def __len__(self):
        return max(0, len(self.files) - self.sequence_length + 1)

    def __getitem__(self, index):
        files = self.files[index:index + self.sequence_length]
        raw = [np.load(f).astype(np.float32) for f in files]
        frames = []
        previous = None
        for frame in raw:
            channels = make_five_channels(frame, previous)
            frames.append(channels)
            previous = resize(frame)
        frames = np.stack(frames)
        if self.augment:
            noise = np.random.normal(0, 0.008, frames.shape).astype(np.float32)
            frames = np.clip(frames + noise, 0, 1)
        inputs = torch.from_numpy(frames[:self.input_frames]).float()
        targets = torch.from_numpy(frames[self.input_frames:, 0:1]).float()
        return inputs, targets
