from datetime import datetime, timezone
import hashlib
import numpy as np


class DWRAdapter:
    name = "DWR Radar"

    def __init__(self, height=96, width=96):
        self.height = height
        self.width = width

    def _rng(self, lat, lon, step):
        key = f"dwr:{lat:.4f}:{lon:.4f}:{step}:{datetime.now(timezone.utc).strftime('%Y%m%d%H') }".encode()
        seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
        return np.random.default_rng(seed)

    async def fetch(self, lat=None, lon=None, step=0):
        rng = self._rng(lat or 0, lon or 0, step)
        y, x = np.mgrid[0:self.height, 0:self.width]
        cx = self.width * (0.35 + 0.08 * np.sin(step / 2))
        cy = self.height * (0.45 + 0.07 * np.cos(step / 3))
        blob = np.exp(-(((x - cx) / 20) ** 2 + ((y - cy) / 17) ** 2))
        reflectivity = np.clip(rng.normal(18, 4, (self.height, self.width)) + blob * 48, 0, 70).astype(np.float32)
        velocity = (rng.normal(0, 4, (self.height, self.width)) + (x - cx) * 0.18 * blob).astype(np.float32)
        return {"source": self.name, "status": "DEMO", "observed_at": datetime.now(timezone.utc).isoformat(), "reflectivity": reflectivity, "velocity": velocity}
