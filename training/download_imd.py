from pathlib import Path
from open_radar_data import DATASETS


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "radar"
RAW_DIR.mkdir(parents=True, exist_ok=True)


FILES = [
    "IMD/JPR220822135253-IMD-B.nc",
    "IMD/JPR220822135253-IMD-B.nc.1",
    "IMD/JPR220822135253-IMD-B.nc.2",
    "IMD/JPR220822135253-IMD-B.nc.3",
    "IMD/JPR220822135253-IMD-B.nc.4",
    "IMD/JPR220822135253-IMD-B.nc.5",
    "IMD/JPR220822135253-IMD-B.nc.6",
    "IMD/JPR220822135253-IMD-B.nc.7",
    "IMD/JPR220822135253-IMD-B.nc.8",
    "IMD/JPR220822135253-IMD-B.nc.9",
]


for name in FILES:

    print("Downloading:", name)

    downloaded = DATASETS.fetch(name)

    downloaded = Path(downloaded)

    output_name = name.split("/")[-1]

    output_path = RAW_DIR / output_name

    output_path.write_bytes(
        downloaded.read_bytes()
    )

    print("Saved:", output_path)


print()
print("IMD radar download complete.")