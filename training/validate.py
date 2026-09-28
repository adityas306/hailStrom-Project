from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from dataset import WeatherDataset
from model import ConvLSTMNowcaster


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = VAL_DIR if len(list(VAL_DIR.glob('*.npy'))) >= INPUT_FRAMES + OUTPUT_FRAMES else TRAIN_DIR
    dataset = WeatherDataset(data_dir, INPUT_FRAMES, OUTPUT_FRAMES)
    if len(dataset) == 0:
        raise RuntimeError('No validation sequence available.')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ConvLSTMNowcaster(input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, future_steps=OUTPUT_FRAMES).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    criterion = torch.nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            total += criterion(model(x.to(device)), y.to(device)).item()
    print(f'Validation MSE: {total / max(1, len(loader)):.6f}')


if __name__ == '__main__':
    main()
