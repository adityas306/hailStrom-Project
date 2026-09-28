from pathlib import Path
import numpy as np
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "radar"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "radar"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Find radar volumes
# --------------------------------------------------

volumes = sorted(
    [p for p in RAW_DIR.iterdir() if p.is_dir()]
)

print("Found volumes:", len(volumes))


# --------------------------------------------------
# Process each volume
# --------------------------------------------------

for volume_dir in volumes:

    timestamp = volume_dir.name

    sweep_file = volume_dir / f"JPR{timestamp}-IMD-B.nc"

    if not sweep_file.exists():
        print("Missing:", sweep_file)
        continue

    print()
    print("=" * 60)
    print("Reading:", sweep_file)
    print("=" * 60)

    try:

        # --------------------------------------------------
        # Open NetCDF file
        # --------------------------------------------------

        ds = xr.open_dataset(
            sweep_file,
            engine="netcdf4"
        )

        print(ds)

    except Exception as e:

        print("ERROR opening file:")
        print(e)

        continue


    # --------------------------------------------------
    # Find reflectivity variable
    # --------------------------------------------------

    print("Variables:")
    print(list(ds.data_vars))


    if "DBZH" not in ds:

        print("DBZH not found!")

        ds.close()
        continue


    # --------------------------------------------------
    # Read reflectivity
    # --------------------------------------------------

    dbz = ds["DBZH"].values

    print("Original shape:", dbz.shape)
    print("Original dtype:", dbz.dtype)


    # --------------------------------------------------
    # Remove unnecessary dimensions
    # --------------------------------------------------

    dbz = np.squeeze(dbz)

    print("Squeezed shape:", dbz.shape)


    # --------------------------------------------------
    # Convert to float32
    # --------------------------------------------------

    dbz = dbz.astype(
        np.float32,
        copy=False
    )


    # --------------------------------------------------
    # NaN handling
    # --------------------------------------------------

    dbz = np.nan_to_num(
        dbz,
        nan=-10.0,
        posinf=70.0,
        neginf=-10.0
    )


    # --------------------------------------------------
    # Clip reflectivity
    # --------------------------------------------------

    dbz = np.clip(
        dbz,
        -10.0,
        70.0
    )


    # --------------------------------------------------
    # Normalize
    #
    # -10 dBZ -> 0
    #  70 dBZ -> 1
    # --------------------------------------------------

    dbz = (
        dbz + 10.0
    ) / 80.0


    dbz = dbz.astype(
        np.float32
    )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_file = (
        PROCESSED_DIR /
        f"frame_{timestamp}.npy"
    )

    np.save(
        output_file,
        dbz
    )


    print(
        "Saved:",
        output_file
    )

    print(
        "Final shape:",
        dbz.shape
    )

    print(
        "Min:",
        dbz.min()
    )

    print(
        "Max:",
        dbz.max()
    )


    ds.close()


# --------------------------------------------------
# Done
# --------------------------------------------------

print()
print("=" * 60)
print("FRAME EXTRACTION COMPLETE")
print("=" * 60)