from datetime import datetime, timezone
import numpy as np


class LightningAdapter:
    name = "Lightning Network"

    def __init__(self, height=96, width=96):
        self.height = height
        self.width = width

    async def fetch(self, lat=None, lon=None, step=0):
        y, x = np.mgrid[0:self.height, 0:self.width]
        cx = self.width * (0.36 + 0.08 * np.sin(step / 2))
        cy = self.height * (0.45 + 0.06 * np.cos(step / 3))
        storm = np.exp(-(((x - cx) / 18) ** 2 + ((y - cy) / 16) ** 2))
        density = np.clip(storm * 14 + np.random.default_rng(step + 41).normal(0, 0.35, (self.height, self.width)), 0, 20).astype(np.float32)
        return {"source": self.name, "status": "DEMO", "observed_at": datetime.now(timezone.utc).isoformat(), "density": density}
