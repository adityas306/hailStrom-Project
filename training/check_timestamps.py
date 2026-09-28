from pathlib import Path
import xarray as xr

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# data/raw/radar
RAW_DIR = BASE_DIR / "data" / "raw" / "radar"

print("Project root :", BASE_DIR)
print("Radar folder :", RAW_DIR)
print("=" * 70)

# --------------------------------------------------
# CHECK RADAR DIRECTORY
# --------------------------------------------------
if not RAW_DIR.exists():
    print("❌ Radar directory nahi mili!")
    print("Expected path:")
    print(RAW_DIR)
    print("\nFolder create karne ke liye:")
    print("mkdir data\\raw\\radar")
    raise SystemExit(1)

# --------------------------------------------------
# FIND NETCDF FILES
# --------------------------------------------------
files = sorted([
    f for f in RAW_DIR.iterdir()
    if f.is_file() and (
        f.name.endswith(".nc") or ".nc." in f.name
    )
])

print("Total files:", len(files))
print("=" * 70)

if len(files) == 0:
    print("\n⚠️ Koi NetCDF radar file nahi mili.")
    print("Radar files yahan honi chahiye:")
    print(RAW_DIR)

# --------------------------------------------------
# CHECK EACH FILE
# --------------------------------------------------
for file in files:

    print("\nFILE:", file.name)
    print("-" * 70)

    try:
        ds = xr.open_dataset(
            file,
            engine="netcdf4"
        )

        # ------------------------------------------
        # Dataset information
        # ------------------------------------------
        print("Dimensions:")
        print(ds.dims)

        print("\nVariables:")
        print(list(ds.data_vars))

        # ------------------------------------------
        # Start time
        # ------------------------------------------
        if "esStartTime" in ds:
            print("\nesStartTime:")
            print(ds["esStartTime"].values)

        else:
            print("\nesStartTime: NOT FOUND")

        # ------------------------------------------
        # Radar radial time
        # ------------------------------------------
        if "radialTime" in ds:

            radial = ds["radialTime"].values

            print("\nradialTime shape:", radial.shape)

            if radial.size > 0:
                print("radialTime first:", radial.flat[0])
                print("radialTime last :", radial.flat[-1])

            else:
                print("radialTime is empty")

        else:
            print("\nradialTime: NOT FOUND")

        # ------------------------------------------
        # Close dataset
        # ------------------------------------------
        ds.close()

    except Exception as e:
        print("ERROR:", repr(e))

print("\n" + "=" * 70)
print("TIMESTAMP CHECK COMPLETE")