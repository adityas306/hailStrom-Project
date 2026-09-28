import numpy as np

CHANNELS = 5
GRID_SIZE = 64


def _resize_2d(data, height=GRID_SIZE, width=GRID_SIZE):
    """Small dependency-free bilinear resize for weather grids."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.shape == (height, width):
        return arr.astype(np.float32)

    y = np.linspace(0, arr.shape[0] - 1, height)
    x = np.linspace(0, arr.shape[1] - 1, width)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, arr.shape[1] - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, arr.shape[0] - 1)
    xw = x - x0
    yw = y - y0

    top = arr[y0[:, None], x0[None, :]] * (1 - xw)[None, :] + arr[y0[:, None], x1[None, :]] * xw[None, :]
    bottom = arr[y1[:, None], x0[None, :]] * (1 - xw)[None, :] + arr[y1[:, None], x1[None, :]] * xw[None, :]
    return (top * (1 - yw)[:, None] + bottom * yw[:, None]).astype(np.float32)


def _channel(value, height=GRID_SIZE, width=GRID_SIZE):
    if value is None:
        return np.zeros((height, width), dtype=np.float32)
    return np.nan_to_num(_resize_2d(value, height, width), nan=0.0, posinf=1.0, neginf=0.0)


def create_empty_frame(height=GRID_SIZE, width=GRID_SIZE):
    return np.zeros((CHANNELS, height, width), dtype=np.float32)


def fuse_weather_data(dwr_data, insat_data, lightning_data, height=GRID_SIZE, width=GRID_SIZE):
    frame = create_empty_frame(height, width)
    frame[0] = _channel(dwr_data.get("reflectivity"), height, width)
    frame[1] = _channel(dwr_data.get("velocity"), height, width)
    frame[2] = _channel(insat_data.get("ir"), height, width)
    frame[3] = _channel(insat_data.get("wv"), height, width)
    frame[4] = _channel(lightning_data.get("density"), height, width)
    return frame


def create_sequence(frames):
    if not frames:
        raise ValueError("No weather frames supplied")
    sequence = np.stack(frames, axis=0)
    if sequence.ndim != 4 or sequence.shape[1] != CHANNELS:
        raise ValueError(f"Expected [T,5,H,W], got {sequence.shape}")
    return sequence.astype(np.float32)


def create_model_tensor(frames):
    return create_sequence(frames)[None, ...]
