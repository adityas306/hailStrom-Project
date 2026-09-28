import asyncio
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.dwr import DWRAdapter
from app.adapters.insat import INSATAdapter
from app.adapters.lightning import LightningAdapter
from app.model.nowcaster import nowcaster
from app.preprocessing.fusion import fuse_weather_data, create_model_tensor
from app.preprocessing.normalize import (
    normalize_reflectivity,
    normalize_velocity,
    normalize_satellite,
    normalize_lightning,
)

BASE_DIR = Path(__file__).resolve().parent
LOCATIONS_FILE = BASE_DIR / "data" / "locations.json"

app = FastAPI(title="StormCast India", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dwr = DWRAdapter()
insat = INSATAdapter()
lightning = LightningAdapter()


def load_locations():
    try:
        return __import__("json").loads(LOCATIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

LOCATIONS = load_locations()


@app.get("/")
async def root():
    return {"name": "StormCast India", "status": "running", "version": "1.1.0"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": nowcaster.model_loaded,
        "device": str(nowcaster.device),
        "data_mode": os.getenv("DATA_MODE", "demo"),
    }


async def create_weather_frame(lat: float, lon: float, step: int = 0):
    dwr_data, insat_data, lightning_data = await asyncio.gather(
        dwr.fetch(lat, lon, step=step),
        insat.fetch(lat, lon, step=step),
        lightning.fetch(lat, lon, step=step),
    )

    dwr_data["reflectivity"] = normalize_reflectivity(dwr_data["reflectivity"])
    dwr_data["velocity"] = normalize_velocity(dwr_data["velocity"])
    insat_data["ir"] = normalize_satellite(insat_data["ir"])
    insat_data["wv"] = normalize_satellite(insat_data["wv"])
    lightning_data["density"] = normalize_lightning(lightning_data["density"])

    return fuse_weather_data(dwr_data, insat_data, lightning_data)


async def create_model_input(lat: float, lon: float, time_steps: int = 12):
    frames = await asyncio.gather(*[
        create_weather_frame(lat, lon, step=i) for i in range(time_steps)
    ])
    return create_model_tensor(frames)


def safe_probability(value):
    return float(np.clip(value, 0.0, 1.0))


async def predict_location(lat: float, lon: float):
    model_input = await create_model_input(lat, lon, time_steps=12)
    prediction = nowcaster.predict_tensor(model_input)
    future = prediction[0, :, 0]

    # Use the forecast value around the clicked pixel/centre, while retaining
    # a small spatial aggregate so a single noisy pixel does not dominate.
    centre = future[:, future.shape[1] // 2, future.shape[2] // 2]
    spatial_mean = future.mean(axis=(1, 2))
    storm = safe_probability(0.65 * float(centre.mean()) + 0.35 * float(spatial_mean.mean()))

    hail = safe_probability(storm * 0.72)
    cloudburst = safe_probability(storm * 0.80)
    lightning_probability = safe_probability(storm * 0.92)
    lightning_density = round(lightning_probability * 12.0, 2)
    downburst = round(15.0 + storm * 85.0, 1)
    confidence = safe_probability(0.55 + 0.35 * abs(storm - 0.5) * 2 + (0.10 if nowcaster.model_loaded else 0.0))
    arrival = int(max(5, round(120 - storm * 100)))

    return {
        "storm_probability": storm,
        "hail_probability": hail,
        "cloudburst_probability": cloudburst,
        "lightning_probability": lightning_probability,
        "lightning_density_per_100km2": lightning_density,
        "downburst_wind_kmh": downburst,
        "arrival_minutes": arrival,
        "confidence": confidence,
        "future_steps": future.mean(axis=(1, 2)).tolist(),
        "model_loaded": nowcaster.model_loaded,
        "input_shape": list(model_input.shape),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def risk_from_probability(probability):
    if probability >= 0.75:
        return "HIGH"
    if probability >= 0.45:
        return "MODERATE"
    return "LOW"


@app.get("/api/hazards")
async def hazards(lat: float, lon: float):
    forecast = await predict_location(lat, lon)
    return {
        "location": {"lat": lat, "lon": lon},
        "risk": risk_from_probability(forecast["storm_probability"]),
        "grid_resolution_km": 3,
        "forecast_horizon_hours": 6,
        "data_mode": os.getenv("DATA_MODE", "demo"),
        **forecast,
    }


@app.get("/api/location/search")
async def location_search(q: str = Query(..., min_length=1, max_length=100)):
    query = q.strip().lower()
    local = [x for x in LOCATIONS if query in x.get("name", "").lower()]
    if local:
        return {"results": local[:10], "source": "local"}

    # Optional online fallback. If internet is unavailable, the local dataset
    # still keeps the application usable.
    try:
        async with httpx.AsyncClient(timeout=4.0, headers={"User-Agent": "StormCastIndia/1.1"}) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "jsonv2", "limit": 8, "countrycodes": "in"},
            )
            response.raise_for_status()
            results = [
                {"name": item.get("display_name", q), "lat": float(item["lat"]), "lon": float(item["lon"])}
                for item in response.json()
                if "lat" in item and "lon" in item
            ]
            return {"results": results, "source": "nominatim"}
    except Exception:
        return {"results": [], "source": "unavailable"}


@app.get("/api/location/reverse")
async def location_reverse(lat: float, lon: float):
    try:
        async with httpx.AsyncClient(timeout=4.0, headers={"User-Agent": "StormCastIndia/1.1"}) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
            )
            response.raise_for_status()
            item = response.json()
            address = item.get("address", {})
            parts = [
                address.get("city") or address.get("town") or address.get("village") or address.get("municipality"),
                address.get("state"),
            ]
            name = ", ".join(dict.fromkeys([p for p in parts if p]))
            return {"name": name or item.get("display_name", "Selected Location"), "display_name": item.get("display_name")}
    except Exception:
        # Offline fallback: nearest known city.
        nearest = None
        best = float("inf")
        for item in LOCATIONS:
            d = (float(item["lat"]) - lat) ** 2 + (float(item["lon"]) - lon) ** 2
            if d < best:
                best, nearest = d, item
        if nearest:
            return {"name": f"Near {nearest['name']}", "display_name": f"Near {nearest['name']}"}
        return {"name": "Selected Location", "display_name": "Selected Location"}


@app.get("/api/sources")
async def sources():
    return {
        "data_mode": os.getenv("DATA_MODE", "demo"),
        "sources": [
            {"name": "DWR Radar", "status": "DEMO" if os.getenv("DATA_MODE", "demo") == "demo" else "CONFIGURED"},
            {"name": "INSAT-3D/3DR/3DS", "status": "DEMO" if os.getenv("DATA_MODE", "demo") == "demo" else "CONFIGURED"},
            {"name": "Lightning Network", "status": "DEMO" if os.getenv("DATA_MODE", "demo") == "demo" else "CONFIGURED"},
        ],
    }


@app.websocket("/ws/weather")
async def weather_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            now = datetime.now(timezone.utc)
            lat, lon = 26.8467, 80.9462
            forecast = await predict_location(lat, lon)
            probability = forecast["storm_probability"]
            payload = {
                "type": "forecast_update",
                "timestamp": now.isoformat(),
                "data_mode": os.getenv("DATA_MODE", "demo"),
                "cells": [{
                    "lat": lat,
                    "lon": lon,
                    "storm_probability": probability,
                    "hail_probability": forecast["hail_probability"],
                    "cloudburst_probability": forecast["cloudburst_probability"],
                    "lightning_density": forecast["lightning_density_per_100km2"],
                    "arrival_minutes": forecast["arrival_minutes"],
                }],
                "storm_track": [
                    {"lat": lat - 0.1, "lon": lon - 0.2},
                    {"lat": lat, "lon": lon},
                    {"lat": lat + 0.1, "lon": lon + 0.2},
                ],
            }
            await websocket.send_json(payload)
            await asyncio.sleep(int(os.getenv("UPDATE_SECONDS", "30")))
    except WebSocketDisconnect:
        print("WebSocket disconnected")
