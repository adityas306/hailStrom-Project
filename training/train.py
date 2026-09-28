from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import *
from dataset import WeatherDataset
from model import ConvLSTMNowcaster


def main():
    files = sorted(TRAIN_DIR.glob('*.npy'))
    if len(files) < INPUT_FRAMES + OUTPUT_FRAMES:
        # Fall back to processed radar frames so the included sample data can train.
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        for src in sorted(RADAR_DIR.glob('*.npy')):
            target = TRAIN_DIR / src.name
            if not target.exists():
                target.write_bytes(src.read_bytes())
        files = sorted(TRAIN_DIR.glob('*.npy'))

    if len(files) < INPUT_FRAMES + OUTPUT_FRAMES:
        raise RuntimeError(f'Need at least {INPUT_FRAMES + OUTPUT_FRAMES} frames; found {len(files)}')

    dataset = WeatherDataset(TRAIN_DIR, INPUT_FRAMES, OUTPUT_FRAMES, augment=True)
    if len(dataset) == 0:
        raise RuntimeError('No usable temporal sequences found.')

    # With a small bundled dataset, use a deterministic train/validation split of sequences.
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    if n_train == 0:
        n_train, n_val = 1, 0
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)) if n_val else (dataset, None)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_set else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMNowcaster(input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, future_steps=OUTPUT_FRAMES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = train_loss
        if val_loader:
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    val_loss += criterion(model(inputs.to(device)), targets.to(device)).item()
            val_loss /= max(1, len(val_loader))

        print(f'Epoch {epoch+1}/{EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f}')
        if val_loss <= best:
            best = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_channels': INPUT_CHANNELS,
                'input_frames': INPUT_FRAMES,
                'output_frames': OUTPUT_FRAMES,
                'hidden_channels': HIDDEN_CHANNELS,
                'grid_size': GRID_SIZE,
                'validation_loss': best,
            }, MODEL_PATH)

    print(f'Checkpoint saved: {MODEL_PATH}')


if __name__ == '__main__':
    main()
