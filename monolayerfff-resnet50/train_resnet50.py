import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import pandas as pd
from PIL import Image

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 20
PATIENCE = 3
NUM_CLASSES = 3

CHECKPOINT_PATH = "checkpoints/best_resnet50.pth"

# =========================
# DATASET (CSV BASED)
# =========================
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# LOAD DATA
# =========================
train_dataset = CSVDataset("data/train.csv", transform=train_transform)
val_dataset = CSVDataset("data/val.csv", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier (with dropout for regularization)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================
# TRAIN + EVAL FUNCTIONS
# =========================
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    return total_loss, acc


def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

# =========================
# TRAINING LOOP
# =========================
os.makedirs("checkpoints", exist_ok=True)

best_acc = 0.0
early_stop_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_acc = evaluate()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f}")
    print(f"Val Acc   : {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc + 1e-4:
        best_acc = val_acc
        early_stop_counter = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc
        }, CHECKPOINT_PATH)

        print(f"✅ New best model saved (Val Acc: {best_acc:.4f})")

    else:
        early_stop_counter += 1
        print(f"No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")

    # Early stopping
    if early_stop_counter >= PATIENCE:
        print("🛑 Early stopping triggered.")
        break

print(f"\nBest Validation Accuracy: {best_acc:.4f}")
print("Training complete.")
