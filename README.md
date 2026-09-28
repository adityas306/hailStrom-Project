# 🌩️ StormCast India

### AI-Assisted Convective-Storm Nowcasting Dashboard | SIH 2026 | Problem Statement: SIH26084

StormCast India is an AI-assisted weather nowcasting platform designed to provide **location-specific 0–6 hour convective-storm forecasts**. The system combines radar, satellite and lightning-inspired weather features into a common tensor and uses a **PyTorch ConvLSTM model** to generate future storm-risk information.

> **Current project status:** The repository includes a fully runnable **DEMO mode** with deterministic synthetic DWR, INSAT and lightning fields. It is intended for development, demonstration and hackathon presentation. The bundled demo feeds must be replaced with operational meteorological data sources before the system is used for real-world warnings.

---

## ✨ Key Features

- 🗺️ Interactive React + Leaflet weather map
- 🔎 Indian city/location search
- 📍 Map-click based weather and hazard inspection
- 🌧️ 0–6 hour forecast workflow
- ⚡ Storm, hail, cloudburst and lightning probabilities
- 💨 Estimated downburst wind and storm-arrival time
- 🤖 PyTorch ConvLSTM inference
- 🌦️ Five-channel weather-fusion pipeline
- 🔌 FastAPI REST API
- 📡 WebSocket-based forecast updates
- 🏙️ Local dataset containing 200 Indian locations
- 🐳 Docker Compose support
- 🧪 Training and validation scripts included

---

## 🎯 Problem Statement

Convective storms such as severe thunderstorms, hail, lightning, downburst winds and cloudbursts can develop rapidly and at highly localized scales. Conventional weather prediction products may not provide the required spatial and temporal detail for short-term local decision support.

StormCast India addresses this challenge through a **high-resolution, AI-assisted nowcasting workflow** that can fuse multiple weather observations and produce localized hazard information for the next few hours.

---

## 💡 Proposed Solution

The platform follows this pipeline:

```text
Weather Data Sources
        ↓
Data Quality / Pre-processing
        ↓
Temporal & Spatial Alignment
        ↓
Normalization
        ↓
5-Channel Weather Fusion
        ↓
PyTorch ConvLSTM Nowcaster
        ↓
0–6 Hour Future Fields
        ↓
Hazard Probability & Risk Classification
        ↓
Interactive Web Dashboard
```

### Five-channel input concept

| Channel | Feature |
|---|---|
| 1 | DWR Reflectivity |
| 2 | DWR Radial Velocity / Motion Feature |
| 3 | INSAT Infrared (IR) |
| 4 | INSAT Water Vapour (WV) / Cloud Feature |
| 5 | Lightning Density |

The intended operational pipeline is:

```text
Source QC → Temporal Alignment → Reprojection → 1–3 km Grid
→ Normalization → ConvLSTM → Calibration → Hazard Products
```

---

## 🤖 AI / ML Model

The project uses a PyTorch ConvLSTM architecture for spatiotemporal forecasting.

### Model contract

```text
Input : [Batch, Time, 5, 64, 64]
Output: [Batch, 6, 1, 64, 64]
```

Current training configuration:

- Input frames: `4`
- Future frames: `6`
- Input channels: `5`
- Hidden channels: `32`
- Grid size: `64 × 64`
- Batch size: `4`
- Epochs: `8`
- Learning rate: `0.001`

The trained checkpoint is stored at:

```text
backend/app/models/convlstm_nowcaster.pth
```

### Important ML note

The included training workflow can create five derived channels from the bundled radar-frame data so that the complete model pipeline can be demonstrated. These derived channels are **not equivalent to a real operational DWR + INSAT + lightning multi-source dataset**. A production model should be trained on properly aligned multi-source meteorological observations.

---

## 🏗️ System Architecture

```text
┌──────────────────────────────┐
│       React + Leaflet        │
│        Frontend (Vite)       │
└──────────────┬───────────────┘
               │ HTTP / WebSocket
               ▼
┌──────────────────────────────┐
│          FastAPI             │
│      StormCast India API     │
├──────────────────────────────┤
│ Location Services            │
│ Weather Fusion               │
│ Normalization                │
│ Hazard Prediction            │
│ WebSocket Updates            │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│     PyTorch ConvLSTM         │
│  [B,T,5,64,64] → 6 outputs   │
└──────────────┬───────────────┘
               ▲
               │
┌──────────────┴───────────────┐
│ Weather Adapters             │
│ DWR | INSAT | Lightning      │
└──────────────────────────────┘
```

---

## 📁 Project Structure

```text
StormCast_India/
│
├── backend/
│   ├── app/
│   │   ├── adapters/
│   │   │   ├── dwr.py
│   │   │   ├── insat.py
│   │   │   └── lightning.py
│   │   ├── data/
│   │   │   └── locations.json
│   │   ├── model/
│   │   │   └── nowcaster.py
│   │   ├── models/
│   │   │   └── convlstm_nowcaster.pth
│   │   ├── preprocessing/
│   │   │   ├── fusion.py
│   │   │   └── normalize.py
│   │   └── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   └── .env.example
│
├── training/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── validate.py
│   ├── download_imd.py
│   ├── download_imd_timeseries.py
│   ├── extract_imd.py
│   ├── extract_imd_frames.py
│   ├── split_data.py
│   └── check_timestamps.py
│
├── data_schema/
│   └── example_forecast.json
│
├── sample_data/
├── docker-compose.yml
├── MODEL_INTEGRATION.md
└── README.md
```

---

## 🛠️ Technology Stack

### Frontend

- React 18
- Vite
- React Leaflet
- Leaflet
- JavaScript / CSS

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- NumPy
- HTTPX
- Python-dotenv

### AI / ML

- PyTorch
- ConvLSTM
- NumPy-based preprocessing

### Deployment / DevOps

- Docker
- Docker Compose
- Vercel-compatible frontend deployment
- FastAPI-compatible backend deployment

---

# 🚀 Run Locally

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd StormCast_India
```

---

## 2. Backend Setup

From the project root:

### Windows PowerShell

```powershell
python -m venv backend/.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Set demo mode:

```powershell
$env:DATA_MODE="demo"
```

Start FastAPI:

```powershell
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Backend will be available at:

```text
http://localhost:8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

---

## 3. Frontend Setup

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

If the backend is running somewhere else, create/update `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

For a deployed backend:

```env
VITE_API_URL=https://your-backend-domain.example.com
```

---

# 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status and version |
| GET | `/api/health` | Backend/model health |
| GET | `/api/hazards?lat=<lat>&lon=<lon>` | Location-specific hazard forecast |
| GET | `/api/location/search?q=<query>` | Search Indian locations |
| GET | `/api/location/reverse?lat=<lat>&lon=<lon>` | Reverse location lookup |
| GET | `/api/sources` | Current weather-source status |
| WS | `/ws/weather` | Live forecast updates |

### Example

```text
GET /api/hazards?lat=26.8467&lon=80.9462
```

The response contains information such as:

- Storm probability
- Hail probability
- Cloudburst probability
- Lightning probability
- Lightning density
- Downburst wind estimate
- Estimated arrival time
- Confidence
- Future forecast steps
- Model-loaded status
- Input tensor shape
- Data mode

---

# ⚠️ Risk Classification

Storm probability is currently converted into three dashboard risk levels:

```text
Probability >= 0.75  → HIGH
Probability >= 0.45  → MODERATE
Otherwise             → LOW
```

These thresholds are part of the current application logic and should be recalibrated and validated against appropriate meteorological observations before operational use.

---

# 📡 Data Mode

The current repository runs weather adapters in **DEMO mode** by default.

```env
DATA_MODE=demo
```

The demo adapters generate deterministic, spatially structured fields for:

- DWR reflectivity
- DWR velocity
- INSAT IR
- INSAT WV
- Lightning density

The backend also exposes the configured mode through `/api/health` and `/api/sources`.

### Production integration

The backend `.env.example` contains placeholders for operational source configuration:

```env
MOSDAC_API_URL=
MOSDAC_API_KEY=
DWR_API_URL=
DWR_API_KEY=
LIGHTNING_API_URL=
LIGHTNING_API_KEY=
```

These values are placeholders only. The actual data-provider access, authentication, licensing and adapter implementation must be configured separately.

---

# 🧠 Train the Model

The repository contains a reproducible training pipeline under `training/`.

The training configuration expects temporal `.npy` frames under:

```text
data/train/
```

The training script can fall back to:

```text
data/processed/radar/
```

when the training directory does not yet contain enough frames.

### Train

From the project root:

```powershell
python training\train.py
```

The script trains the ConvLSTM and saves the best checkpoint to:

```text
backend/app/models/convlstm_nowcaster.pth
```

### Validate

```powershell
python training\validate.py
```

The validation script reports the validation Mean Squared Error (MSE).

> For meaningful meteorological performance evaluation, use a sufficiently large, timestamped and properly aligned dataset covering representative weather regimes. The bundled/demo data is not a substitute for such a corpus.

---

# 🐳 Run with Docker

The repository includes Dockerfiles for both services and a root `docker-compose.yml`.

From the project root:

```powershell
docker compose up --build
```

Services:

```text
Backend  → http://localhost:8000
Frontend → http://localhost:5173
```

The Compose configuration currently sets:

```text
DATA_MODE=demo
UPDATE_SECONDS=10
```

---

# 🌐 Deployment

A practical deployment architecture is:

```text
Frontend → Vercel
Backend  → Render / Docker-compatible service
Model    → Loaded by FastAPI backend
```

### Frontend

Build command:

```bash
npm run build
```

Output directory:

```text
dist
```

Set:

```env
VITE_API_URL=https://YOUR-BACKEND-URL
```

### Backend

Use the `backend/` directory and its Dockerfile on a Docker-compatible hosting service.

The backend needs the model checkpoint:

```text
backend/app/models/convlstm_nowcaster.pth
```

After deployment, verify:

```text
/
/api/health
/docs
```

If the frontend uses the WebSocket endpoint, the deployed HTTPS API should be converted to the secure WebSocket scheme (`wss`).

---

# 🔐 Configuration

### Backend `.env`

Use `backend/.env.example` as the starting point:

```env
DATA_MODE=demo
UPDATE_SECONDS=30
MODEL_PATH=models/convlstm_nowcaster.pth
MOSDAC_API_URL=
MOSDAC_API_KEY=
DWR_API_URL=
DWR_API_KEY=
LIGHTNING_API_URL=
LIGHTNING_API_KEY=
```

### Frontend `.env`

```env
VITE_API_URL=http://localhost:8000
```

Do not commit private API keys or credentials to GitHub.

---

# 🧪 Development Notes

### Model loading

The backend loads the ConvLSTM checkpoint from the configured model location. `/api/health` reports whether the model is loaded and which PyTorch device is being used.

### Location search

The application first checks the bundled Indian location dataset. If a local match is not found, the backend can attempt an online OpenStreetMap Nominatim search. Reverse geocoding also uses Nominatim with a local-city fallback when the online service is unavailable.

### WebSocket

The `/ws/weather` endpoint periodically sends forecast updates. The current demo WebSocket stream uses the Lucknow coordinate as its demonstration cell.

---

# 📊 Example Forecast Schema

A sample forecast structure is available at:

```text
data_schema/example_forecast.json
```

This can be used as a reference while integrating frontend, backend and model outputs.

---

# 🔭 Future Scope

- Integrate operational DWR radar feeds
- Integrate INSAT-3D/3DR/3DS products through an approved data-access workflow
- Integrate a reliable lightning observation source
- Improve temporal alignment and reprojection
- Expand from demo-derived channels to true multi-source training tensors
- Train on large historical storm-event datasets
- Add model calibration and uncertainty estimation
- Validate against independent observations and event-based metrics
- Improve storm-cell tracking and trajectory estimation
- Add alerting and role-based access for operational users
- Add historical forecast replay and verification dashboards

---

# ⚠️ Disclaimer

StormCast India is an **AI-assisted research/demo system** in its current repository form. The bundled weather adapters operate in DEMO mode and do not represent live operational observations. Forecast probabilities and hazard values must not be treated as official weather warnings.

Before operational deployment, the system requires validated real-time data feeds, robust quality control, scientific calibration, independent verification and appropriate operational safeguards.

---

# 👥 Intended Users

The platform can be adapted for:

- Disaster-management teams
- Emergency-response organizations
- Smart-city systems
- Aviation and transport operations
- Agriculture and infrastructure monitoring
- Weather-research teams
- Academic and hackathon demonstrations

---

# 🏆 Smart India Hackathon

**Project:** StormCast India  
**Problem Statement:** SIH26084  
**Focus:** AI-assisted short-term convective weather nowcasting

The project demonstrates how multi-source weather features, spatiotemporal deep learning and an interactive geospatial interface can be combined into a single nowcasting workflow.

---

## 📄 Additional Documentation

- `MODEL_INTEGRATION.md` — model integration guidance
- `training/readme.md` — training-related notes
- `data_schema/example_forecast.json` — example forecast schema
- `backend/.env.example` — backend configuration template
- `frontend/.env.example` — frontend API configuration template

---


**StormCast India — AI-assisted convective-storm nowcasting for a safer and more responsive India.** 🌩️🇮🇳
