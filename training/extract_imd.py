import os
import numpy as np
import xarray as xr

RAW_DIR = "../data/raw/radar"
OUTPUT_DIR = "../data/processed/radar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted([
    os.path.join(RAW_DIR, f)
    for f in os.listdir(RAW_DIR)
    if f.endswith(".nc") or ".nc." in f
])

print("Found files:", len(files))

frames = []

for file in files:

    print("\nProcessing:", os.path.basename(file))

    try:
        ds = xr.open_dataset(file, engine="netcdf4")

        if "Z" not in ds:
            print("Skipping: Z not found")
            ds.close()
            continue

        # Get timestamp
        if "esStartTime" in ds:
            timestamp = ds["esStartTime"].values
        else:
            timestamp = None

        dbz = ds["Z"].values

        dbz = np.squeeze(dbz)

        if dbz.ndim != 2:
            print("Skipping unexpected shape:", dbz.shape)
            ds.close()
            continue

        # Convert invalid values
        dbz = np.nan_to_num(
            dbz,
            nan=-10.0,
            posinf=70.0,
            neginf=-10.0
        )

        # Reflectivity range
        dbz = np.clip(dbz, -10.0, 70.0)

        # Normalize 0-1
        dbz = (dbz + 10.0) / 80.0

        dbz = dbz.astype(np.float32)

        frames.append({
            "timestamp": timestamp,
            "data": dbz
        })

        print("Timestamp:", timestamp)
        print("Shape:", dbz.shape)

        ds.close()

    except Exception as e:
        print("ERROR:", e)


# Sort by timestamp
frames.sort(key=lambda x: x["timestamp"])


print("\nSaving sorted frames...")

# Remove old npy files
for file in os.listdir(OUTPUT_DIR):
    if file.endswith(".npy"):
        os.remove(os.path.join(OUTPUT_DIR, file))


for i, frame in enumerate(frames):

    output_file = os.path.join(
        OUTPUT_DIR,
        f"frame_{i:04d}.npy"
    )

    np.save(output_file, frame["data"])

    print(
        f"{i:04d} | "
        f"{frame['timestamp']} | "
        f"{frame['data'].shape}"
    )


print("\n==============================")
print("EXTRACTION COMPLETE")
print("==============================")
print("Total frames:", len(frames))
print("==============================")