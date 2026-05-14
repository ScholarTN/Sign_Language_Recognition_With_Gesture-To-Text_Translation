import os, random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from PIL import Image

# ---------------- CONFIG ----------------
IMG_SIZE = 48
BATCH_SIZE = 16
EPOCHS = 3
IMAGES_PER_CLASS = 200
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "asl_data" / "asl_alphabet_train" / "asl_alphabet_train"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUT_DIR / "asl_cnn.pth"

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)

class_indices = defaultdict(list)
for idx, (_, label) in enumerate(full_ds.samples):
    class_indices[label].append(idx)

subset_indices = []
for label, indices in class_indices.items():
    subset_indices += random.sample(indices, min(IMAGES_PER_CLASS, len(indices)))

subset = Subset(full_ds, subset_indices)

val_size = int(0.15 * len(subset))
train_size = len(subset) - val_size
train_set, val_set = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

print("Train:", train_size)
print("Val:", val_size)

# ---------------- MODEL ----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN ----------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if train:
            optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return loss_sum / total, correct / total

print("\nTraining...")
for epoch in range(EPOCHS):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader, False)
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {tr_acc*100:.2f}% | "
          f"Val Acc: {val_acc*100:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved to", MODEL_PATH)