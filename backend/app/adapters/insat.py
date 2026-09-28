from datetime import datetime, timezone
import numpy as np


class INSATAdapter:
    name = "INSAT-3D/3DR/3DS"

    def __init__(self, height=96, width=96):
        self.height = height
        self.width = width

    async def fetch(self, lat=None, lon=None, step=0):
        y, x = np.mgrid[0:self.height, 0:self.width]
        cx = self.width * (0.38 + 0.08 * np.sin(step / 2))
        cy = self.height * (0.44 + 0.06 * np.cos(step / 3))
        cloud = np.exp(-(((x - cx) / 23) ** 2 + ((y - cy) / 20) ** 2))
        ir = np.clip(285 - cloud * 80, 180, 330).astype(np.float32)
        wv = np.clip(275 - cloud * 60, 180, 330).astype(np.float32)
        return {"source": self.name, "status": "DEMO", "observed_at": datetime.now(timezone.utc).isoformat(), "ir": ir, "wv": wv}
