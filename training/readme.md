# StormCast India — Training Workflow & Validation Notes

This directory contains the training-side workflow for the ConvLSTM nowcaster.

## 1. Training files

| File | Purpose |
|---|---|
| `config.py` | Training paths and hyperparameters |
| `dataset.py` | Loads temporal `.npy` sequences |
| `model.py` | ConvLSTM training model |
| `train.py` | Training loop and checkpoint creation |
| `validate.py` | Validation MSE calculation |
| `download_imd.py` | IMD data download helper |
| `download_imd_timeseries.py` | Timestamped radar download workflow |
| `extract_imd.py` | Radar extraction helper |
| `extract_imd_frames.py` | Converts radar volumes into frames |
| `split_data.py` | Data splitting utility |
| `check_timestamps.py` | Timestamp inspection utility |

> The current repository does **not** include `prepare_sequences.py`. Do not run that command unless you add that preprocessing script separately.

## 2. Expected temporal dataset

The model learns from sequences rather than isolated radar files.

The general structure is:

```text
data/
├── train/
│   ├── frame_0001.npy
│   ├── frame_0002.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each frame should represent a consistent spatial grid and channel layout.

The training dataset creates:

```text
X = [T_input, C, H, W]
Y = [T_output, C, H, W]
```

The DataLoader adds the batch dimension:

```text
[B, T_input, C, H, W]
```

## 3. Environment setup

From the project root:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\activate
```

Install the backend/training dependencies listed by the project.

For a PyTorch training environment, the core packages are:

```powershell
pip install torch torchvision torchaudio
pip install numpy scipy xarray netCDF4
```

Additional radar packages may be required by the extraction scripts:

```powershell
pip install open-radar-data xradar
```

## 4. Data preparation

### Download timestamped data

```powershell
python training\download_imd_timeseries.py
```

### Extract frames

```powershell
python training\extract_imd_frames.py
```

Inspect timestamps before training:

```powershell
python training\check_timestamps.py
```

If you use your own preprocessing pipeline, ensure the final `.npy` frames follow the same shape and normalization expected by `training/dataset.py`.

## 5. Run training

```powershell
python training\split_data.py
python training\train.py
```

The training script:

1. Finds available training frames.
2. Builds temporal sequences.
3. Creates a deterministic validation split when necessary.
4. Selects CUDA if available.
5. Trains the ConvLSTM.
6. Computes validation MSE.
7. Saves the best checkpoint.

The checkpoint path is configured in `training/config.py`.

## 6. Validate

```powershell
python training\validate.py
```

The current validation script reports:

```text
Validation MSE: <value>
```

MSE is useful as a basic regression metric, but it is not enough to demonstrate operational weather-nowcasting skill.

For a stronger evaluation, add event-based metrics appropriate to the target.

## 7. Recommended validation strategy

Avoid random leakage between neighboring timestamps.

For operational nowcasting, prefer a chronological split such as:

```text
Earlier dates  → training
Later dates    → validation
Held-out dates → test
```

Also test events separately, for example:

- ordinary rainfall
- convective storms
- high-reflectivity cells
- hail-producing events
- intense short-duration rainfall
- lightning-rich storms

## 8. Recommended metrics

For continuous forecast fields:

- MSE
- MAE
- RMSE
- correlation
- skill against a persistence baseline

For event/hazard products:

- Precision
- Recall
- F1
- CSI / Critical Success Index
- POD / Probability of Detection
- FAR / False Alarm Ratio
- Brier score for probabilities
- reliability/calibration curves

For spatial forecasts, evaluate both location and intensity errors.

## 9. Persistence baseline

A useful baseline is persistence:

```text
future field = latest observed field
```

A trained model should be compared against this baseline. A lower validation MSE than another model does not automatically mean better operational performance; the target, event threshold and evaluation period matter.

## 10. Checkpoint compatibility

The backend expects:

```text
5 input channels
6 future steps
```

The model architecture and hyperparameters used during training must match the inference model in:

```text
backend/app/model/nowcaster.py
```

After training, verify:

```text
backend/app/models/convlstm_nowcaster.pth
```

or set `MODEL_PATH` in the backend environment.

## 11. Small bundled dataset

The repository includes sample/processed data so the pipeline can be exercised locally.

A small dataset is useful for:

- testing code
- verifying tensor shapes
- checking model loading
- demonstrating the UI

It should **not** be presented as sufficient evidence of operational forecast accuracy.

For a real deployment, train on a substantially larger, timestamp-consistent, representative archive.

## 12. Common training errors

### `num_samples=0`

Usually means there are not enough usable frames to create a sequence.

Check:

```powershell
dir data\train
```

and:

```powershell
python training\check_timestamps.py
```

### Shape mismatch

Inspect a frame:

```python
import numpy as np
x = np.load("data/train/frame_0001.npy")
print(x.shape)
```

Ensure every frame uses a compatible shape.

### Checkpoint not found during backend startup

Train first, then copy/configure the checkpoint path:

```text
backend/app/models/convlstm_nowcaster.pth
```

## 13. Reproducibility

For experiments, record:

- dataset date range
- source radar/site
- preprocessing version
- normalization statistics
- input/output frame counts
- grid resolution
- model architecture
- learning rate
- batch size
- number of epochs
- random seed
- validation/test periods
- checkpoint filename

This makes model comparisons auditable.
