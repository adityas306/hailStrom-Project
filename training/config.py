from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RADAR_DIR = DATA_DIR / "processed" / "radar"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

INPUT_FRAMES = 4
OUTPUT_FRAMES = 6
INPUT_CHANNELS = 5
HIDDEN_CHANNELS = 32
BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 0.001
GRID_SIZE = 64
MODEL_DIR = BASE_DIR / "backend" / "app" / "models"
MODEL_PATH = MODEL_DIR / "convlstm_nowcaster.pth"
