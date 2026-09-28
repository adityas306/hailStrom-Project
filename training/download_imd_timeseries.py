from pathlib import Path
import re
import time

from open_radar_data import DATASETS


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "radar"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MAX_RETRIES = 5
RETRY_DELAY = 5


# ============================================================
# GET IMD FILE REGISTRY
# ============================================================

print()
print("=" * 70)
print("STORMCAST INDIA - IMD RADAR DOWNLOADER")
print("=" * 70)

print()
print("Searching IMD radar files...")

registry = DATASETS.registry_files


# Only Jaipur IMD radar
imd_files = [
    f
    for f in registry
    if f.startswith("IMD/JPR")
    and "IMD-B.nc" in f
]


print(
    f"Found {len(imd_files)} IMD radar files in registry."
)


# ============================================================
# GROUP SWEEPS BY TIMESTAMP
# ============================================================

pattern = re.compile(
    r"^(IMD/JPR\d{12}-IMD-B\.nc)(?:\.(\d+))?$"
)

volumes = {}


for filename in imd_files:

    match = pattern.match(filename)

    if not match:
        continue

    base_file = match.group(1)

    # Example:
    # JPR220822135253
    timestamp = base_file.split("/")[-1][3:15]

    if timestamp not in volumes:
        volumes[timestamp] = []

    volumes[timestamp].append(filename)


print(
    f"Found {len(volumes)} radar volumes."
)


# ============================================================
# SORT SWEEPS
# ============================================================

def sweep_number(filename):

    if filename.endswith(".nc"):
        return 0

    try:
        return int(
            filename.split(".nc.")[-1]
        ) + 1

    except ValueError:
        return 999


# ============================================================
# DOWNLOAD
# ============================================================

total_success = 0
total_failed = 0


for timestamp in sorted(volumes):

    files = sorted(
        volumes[timestamp],
        key=sweep_number
    )

    volume_dir = RAW_DIR / timestamp

    volume_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print("=" * 70)
    print("VOLUME:", timestamp)
    print("SWEEPS:", len(files))
    print("=" * 70)


    for remote_file in files:

        filename = remote_file.split("/")[-1]

        output_file = volume_dir / filename


        # ----------------------------------------------------
        # Already downloaded
        # ----------------------------------------------------

        if output_file.exists():

            size_mb = (
                output_file.stat().st_size
                / (1024 * 1024)
            )

            print(
                f"[SKIP] {filename} "
                f"({size_mb:.2f} MB already exists)"
            )

            total_success += 1

            continue


        # ----------------------------------------------------
        # Download with retry
        # ----------------------------------------------------

        success = False


        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            print()
            print(
                f"[DOWNLOAD] {filename}"
            )

            print(
                f"Attempt {attempt}/{MAX_RETRIES}"
            )


            try:

                downloaded = Path(
                    DATASETS.fetch(
                        remote_file
                    )
                )


                # Copy downloaded file
                output_file.write_bytes(
                    downloaded.read_bytes()
                )


                # Verify file exists
                if output_file.exists():

                    size_mb = (
                        output_file.stat().st_size
                        / (1024 * 1024)
                    )


                    print(
                        f"[SUCCESS] {filename}"
                    )

                    print(
                        f"Saved: {output_file}"
                    )

                    print(
                        f"Size: {size_mb:.2f} MB"
                    )


                    success = True
                    total_success += 1

                    break


            except Exception as e:

                print()
                print(
                    f"[ERROR] Download failed:"
                )

                print(e)


                if attempt < MAX_RETRIES:

                    print(
                        f"Retrying in "
                        f"{RETRY_DELAY} seconds..."
                    )

                    time.sleep(
                        RETRY_DELAY
                    )

                else:

                    print(
                        f"[FAILED] "
                        f"{filename}"
                    )

                    total_failed += 1


        # ----------------------------------------------------
        # Continue with next file
        # ----------------------------------------------------

        if not success:

            print(
                "Continuing with next sweep..."
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(
    f"Successful / existing files : "
    f"{total_success}"
)

print(
    f"Failed files                : "
    f"{total_failed}"
)

print(
    f"Data directory              : "
    f"{RAW_DIR}"
)

print("=" * 70)