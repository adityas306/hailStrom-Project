from pathlib import Path
import shutil


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "radar"
)

TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
TEST_DIR = BASE_DIR / "data" / "test"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

TRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND FRAMES
# ============================================================

files = sorted(
    SOURCE_DIR.glob("*.npy")
)

print("Source directory:")
print(SOURCE_DIR)

print()
print("Total frames:", len(files))


if len(files) == 0:

    raise RuntimeError(
        "\nNo .npy files found!\n"
        f"Expected frames in:\n{SOURCE_DIR}\n\n"
        "Run extract_imd.py first."
    )


# ============================================================
# CHECK MINIMUM DATA
# ============================================================

if len(files) < 8:

    raise RuntimeError(
        "\nNot enough radar frames.\n"
        f"Found: {len(files)}\n"
        "Minimum required for 4 input + 4 output = 8 frames."
    )


# ============================================================
# SPLIT
# ============================================================

n = len(files)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)


train_files = files[
    :train_end
]

val_files = files[
    train_end:val_end
]

test_files = files[
    val_end:
]


print()
print("Train frames:", len(train_files))
print("Validation frames:", len(val_files))
print("Test frames:", len(test_files))


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_frames(
    files,
    destination
):

    for file in files:

        shutil.copy2(
            file,
            destination / file.name
        )


# ============================================================
# COPY DATA
# ============================================================

copy_frames(
    train_files,
    TRAIN_DIR
)

copy_frames(
    val_files,
    VAL_DIR
)

copy_frames(
    test_files,
    TEST_DIR
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("Data splitting complete.")

print()
print("Train directory:")
print(TRAIN_DIR)

print("Validation directory:")
print(VAL_DIR)

print("Test directory:")
print(TEST_DIR)