# StormCast India — Model Integration & Inference Guide

This document explains how the PyTorch nowcasting model is connected to the FastAPI backend and how the frontend receives the forecast/hazard response.

## 1. End-to-end inference flow

```text
DWR + INSAT + Lightning
        ↓
Source adapters
        ↓
Quality control / normalization
        ↓
Temporal fusion
        ↓
Model tensor [B, T, C, H, W]
        ↓
ConvLSTMNowcaster
        ↓
6 future forecast steps
        ↓
Hazard post-processing
        ↓
FastAPI /api/hazards
        ↓
React dashboard
```

The current backend uses five model input channels:

| Channel | Meaning |
|---|---|
| 0 | DWR reflectivity |
| 1 | DWR radial velocity / derived motion |
| 2 | INSAT infrared (IR) |
| 3 | INSAT water vapour (WV) / cloud feature |
| 4 | Lightning density |

The expected model input is:

```text
[B, T, 5, H, W]
```

where `B` = batch size, `T` = historical time steps, and `H/W` = spatial grid.

The bundled backend currently builds a 12-frame input sequence and predicts 6 future steps.

## 2. Model implementation

The backend model is in:

```text
backend/app/model/nowcaster.py
```

The main class is:

```python
ConvLSTMNowcaster
```

Expected input:

```python
[B, T, 5, H, W]
```

Expected output:

```python
[B, 6, 1, H, W]
```

The output is normalized to approximately `[0, 1]` through a sigmoid prediction head.

## 3. Checkpoint location

By default, the backend looks for:

```text
backend/app/models/convlstm_nowcaster.pth
```

You can override the location with:

```env
MODEL_PATH=models/convlstm_nowcaster.pth
```

The checkpoint must contain a compatible PyTorch `state_dict`. The training script saves a dictionary containing:

```python
{
    "model_state_dict": ...,
    "input_channels": 5,
    "input_frames": ...,
    "output_frames": 6,
    "hidden_channels": ...,
    "grid_size": ...,
    "validation_loss": ...
}
```

The inference service accepts either this dictionary or a raw `state_dict`.

## 4. Loading the model

`NowcasterService` automatically:

1. Selects CUDA when available, otherwise CPU.
2. Creates the `ConvLSTMNowcaster`.
3. Reads `MODEL_PATH`.
4. Loads the checkpoint.
5. Switches the model to evaluation mode.

Health information is available at:

```text
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "data_mode": "demo"
}
```

## 5. Inference endpoint

The main forecast endpoint is:

```text
GET /api/hazards?lat=26.8467&lon=80.9462
```

The backend:

1. Fetches DWR, INSAT and lightning inputs.
2. Normalizes each source.
3. Fuses the sources into model frames.
4. Builds a 12-step tensor.
5. Runs PyTorch inference.
6. Extracts the forecast around the requested location.
7. Converts the model signal into the dashboard hazard fields.
8. Returns JSON.

## 6. Important distinction: model output vs hazard products

The current ConvLSTM predicts a future spatial field. The backend then derives dashboard indicators such as:

- storm probability
- hail probability
- cloudburst probability
- lightning probability
- lightning density
- downburst wind estimate
- arrival time
- confidence

These derived values are application-level post-processing. They should not be treated as independently trained hazard models unless dedicated hazard labels and calibration models are added.

For a production SIH deployment, train/calibrate each hazard output against appropriate observations where possible.

## 7. Data alignment requirements

Before inference, multi-source observations should be:

1. Quality controlled.
2. Converted to a common timestamp.
3. Reprojected to a common coordinate system.
4. Resampled to a common 1–3 km grid.
5. Normalized using the same statistics used during training.
6. Ordered consistently by time and channel.

Do not train or infer directly from mixed-resolution, misaligned raw products.

## 8. Training/inference consistency

The following must remain identical between training and inference:

- channel order
- normalization ranges/statistics
- grid orientation
- grid dimensions
- temporal ordering
- number of input frames
- model architecture
- checkpoint architecture parameters

If any of these change, retrain or convert the checkpoint accordingly.

## 9. Replacing the model

To integrate another PyTorch model, keep the service contract simple:

```python
class MyNowcaster:
    def predict_tensor(self, tensor):
        # tensor: [B, T, 5, H, W]
        # return: [B, 6, 1, H, W]
        ...
```

Then replace the service initialization in:

```text
backend/app/model/nowcaster.py
```

without changing the API contract.

## 10. Production checklist

Before calling the model production-ready:

- [ ] Real DWR data is available at inference time.
- [ ] Real INSAT data is available at inference time.
- [ ] Lightning data source is validated.
- [ ] Timestamps are synchronized.
- [ ] All inputs use the training normalization.
- [ ] Model checkpoint is trained on representative Indian weather events.
- [ ] Validation/test sets are temporally separated from training.
- [ ] Hazard thresholds are calibrated using observations.
- [ ] Missing-source behavior is tested.
- [ ] CPU and GPU inference are tested.
- [ ] `/api/health` reports the correct model status.
- [ ] Forecast uncertainty is exposed to users.
- [ ] Predictions are logged with source timestamps and model version.

## 11. Common errors

### Checkpoint not found

```text
Warning: model checkpoint not found
```

Run the training workflow first and verify `MODEL_PATH`.

### Shape mismatch

Typical causes:

- wrong number of channels
- wrong number of spatial dimensions
- different model architecture
- incompatible checkpoint

Expected model input:

```text
[B, T, 5, H, W]
```

### Model loads but results are poor

Check:

- normalization mismatch
- training data quality
- temporal gaps
- radar/satellite reprojection
- class/target definition
- validation methodology

A successfully loaded checkpoint does not by itself establish forecast skill.
