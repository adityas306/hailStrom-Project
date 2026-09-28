import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.model.nowcaster import nowcaster
from app.adapters.dwr import DWRAdapter
from app.adapters.insat import INSATAdapter
from app.adapters.lightning import LightningAdapter


# =========================================================
# BASE / LOCATIONS
# =========================================================

BASE = Path(__file__).resolve().parent

locations = json.loads(
    (BASE / "data" / "locations.json").read_text(
        encoding="utf-8"
    )
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="StormCast India API",
    version="1.0.0",
    description=(
        "Real-time-ready 0–6 hour "
        "convective storm nowcasting prototype."
    )
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATA ADAPTERS
# =========================================================

dwr = DWRAdapter()
insat = INSATAdapter()
lightning = LightningAdapter()


# =========================================================
# RISK LABEL
# =========================================================

def risk_label(p: float) -> str:

    if p >= 0.75:
        return "HIGH"

    if p >= 0.45:
        return "MODERATE"

    return "LOW"


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "name": "StormCast India",
        "status": "running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "mode": os.getenv(
            "DATA_MODE",
            "demo"
        ),
        "time": datetime.now(
            timezone.utc
        ).isoformat()
    }


# =========================================================
# LOCATION SEARCH
# =========================================================

@app.get("/api/location/search")
async def location_search(
    q: str = Query(
        min_length=1,
        max_length=80
    )
):

    ql = q.lower().strip()

    matches = [
        x
        for x in locations
        if ql in x["name"].lower()
    ]

    return {
        "results": matches[:10]
    }


# =========================================================
# REVERSE LOCATION
#
# LAT/LON -> CITY / DISTRICT / STATE
# =========================================================

@app.get("/api/location/reverse")
async def location_reverse(
    lat: float,
    lon: float
):

    try:

        url = (
            "https://nominatim.openstreetmap.org/"
            "reverse"
        )

        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10,
            "addressdetails": 1,
        }

        headers = {
            "User-Agent":
                "StormCastIndia/1.0"
        }

        async with httpx.AsyncClient(
            timeout=10
        ) as client:

            response = await client.get(
                url,
                params=params,
                headers=headers
            )

        response.raise_for_status()

        data = response.json()

        address = data.get(
            "address",
            {}
        )

        # City
        city = (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("village")
            or address.get("county")
        )

        # District
        district = (
            address.get("state_district")
            or address.get("county")
        )

        # State
        state = address.get(
            "state"
        )

        # Country
        country = address.get(
            "country"
        )

        # =================================================
        # BUILD DISPLAY NAME
        # =================================================

        if city and state:

            name = f"{city}, {state}"

        elif district and state:

            name = f"{district}, {state}"

        elif city:

            name = city

        elif state:

            name = state

        else:

            name = data.get(
                "display_name",
                "Selected Location"
            )

        return {
            "name": name,
            "city": city,
            "district": district,
            "state": state,
            "country": country,
            "display_name":
                data.get("display_name"),
            "lat": lat,
            "lon": lon
        }

    except Exception as e:

        print(
            "Reverse geocoding error:",
            e
        )

        return {
            "name": "Selected Location",
            "lat": lat,
            "lon": lon
        }


# =========================================================
# HAZARDS
# =========================================================

@app.get("/api/hazards")
async def hazards(
    lat: float,
    lon: float
):

    # Selected location -> model
    forecast = nowcaster.predict(
        lat,
        lon
    )

    return {

        "location": {
            "lat": lat,
            "lon": lon
        },

        "grid_resolution_km": 3,

        "forecast_horizon_hours": 6,

        "risk": risk_label(
            forecast[
                "storm_probability"
            ]
        ),

        **forecast
    }


# =========================================================
# SOURCES
# =========================================================

@app.get("/api/sources")
async def sources():

    return {

        "sources": [

            await dwr.fetch(),

            await insat.fetch(),

            await lightning.fetch()

        ]

    }


# =========================================================
# DASHBOARD
# =========================================================

@app.get("/api/dashboard")
async def dashboard(
    lat: float = 26.8467,
    lon: float = 80.9462
):

    forecast = nowcaster.predict(
        lat,
        lon
    )

    return {

        "hazard": {

            "lat": lat,

            "lon": lon,

            "risk": risk_label(
                forecast[
                    "storm_probability"
                ]
            ),

            **forecast

        },

        "sources": {

            "dwr":
                await dwr.fetch(),

            "insat":
                await insat.fetch(),

            "lightning":
                await lightning.fetch()

        }

    }


# =========================================================
# WEBSOCKET WEATHER
# =========================================================

@app.websocket("/ws/weather")
async def weather_socket(
    websocket: WebSocket
):

    await websocket.accept()

    try:

        while True:

            now = datetime.now(
                timezone.utc
            )

            cells = []

            # =================================================
            # CURRENT DEMO GRID
            # =================================================

            grid_points = [

                (26.70, 80.75),
                (26.70, 80.95),
                (26.70, 81.15),

                (26.85, 80.75),
                (26.85, 80.95),
                (26.85, 81.15),

                (27.00, 80.75),
                (27.00, 80.95),
                (27.00, 81.15)

            ]

            for lat, lon in grid_points:

                f = nowcaster.predict(
                    lat,
                    lon
                )

                cells.append({

                    "lat": lat,

                    "lon": lon,

                    "storm_probability":
                        f[
                            "storm_probability"
                        ],

                    "hail_probability":
                        f[
                            "hail_probability"
                        ],

                    "cloudburst_probability":
                        f[
                            "cloudburst_probability"
                        ],

                    "lightning_density":
                        f[
                            "lightning_density_per_100km2"
                        ],

                    "arrival_minutes":
                        f[
                            "arrival_minutes"
                        ]

                })


            # =================================================
            # WEBSOCKET PAYLOAD
            # =================================================

            payload = {

                "type":
                    "forecast_update",

                "timestamp":
                    now.isoformat(),

                "data_mode":
                    os.getenv(
                        "DATA_MODE",
                        "demo"
                    ),

                "grid_resolution_km":
                    3,

                "cells":
                    cells,

                "storm_track": [

                    {
                        "lat": 26.72,
                        "lon": 80.55
                    },

                    {
                        "lat": 26.79,
                        "lon": 80.72
                    },

                    {
                        "lat": 26.85,
                        "lon": 80.95
                    },

                    {
                        "lat": 26.91,
                        "lon": 81.18
                    }

                ]

            }


            await websocket.send_json(
                payload
            )


            await asyncio.sleep(
                int(
                    os.getenv(
                        "UPDATE_SECONDS",
                        "10"
                    )
                )
            )


    except WebSocketDisconnect:

        print(
            "Weather WebSocket disconnected"
        )