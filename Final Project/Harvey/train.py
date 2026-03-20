"""
Meningioma Brain Tumor Detection - Custom CNN from Scratch
Binary classification: meningioma (1) vs. notumor (0)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(DATA_DIR, "Training")
TEST_DIR    = os.path.join(DATA_DIR, "Testing")

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 1e-3
VAL_SPLIT   = 0.15          # 15% of training set used for validation
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Using device: {DEVICE}")


# ─────────────────────────── TRANSFORMS ───────────────────────
# Training: augmentation to improve generalisation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet channel stats
                         std=[0.229, 0.224, 0.225]),    # used purely for normalisation
])

# Validation / Test: no augmentation, just resize + normalise
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────── DATASETS ─────────────────────────
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)

# Class mapping: {'meningioma': 0, 'notumor': 1} - ImageFolder sorts alphabetically
# We'll print it so it's explicit
print(f"Class mapping: {full_train_dataset.class_to_idx}")
# meningioma=0 → label 0 (tumor), notumor=1 → label 1 (no tumor)

n_val   = int(len(full_train_dataset) * VAL_SPLIT)
n_train = len(full_train_dataset) - n_val
train_dataset, val_dataset = random_split(
    full_train_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

# Apply eval transforms to the validation split
val_dataset.dataset = datasets.ImageFolder(TRAIN_DIR, transform=eval_transforms)

test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_dataset)}")


if __name__ == '__main__':

    class ConvBlock(nn.Module):
        """Conv2d → BatchNorm → ReLU → MaxPool"""
        def __init__(self, in_channels, out_channels, pool=True):
            super().__init__()
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)


    class TumorCNN(nn.Module):
        """
        Custom CNN for binary tumor detection.
        Input:  (B, 3, 224, 224)
        Output: (B, 1) logit
        """
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(3,   32),   # → (B, 32, 112, 112)
                ConvBlock(32,  64),   # → (B, 64,  56,  56)
                ConvBlock(64,  128),  # → (B, 128, 28,  28)
                ConvBlock(128, 256),  # → (B, 256, 14,  14)
                ConvBlock(256, 256),  # → (B, 256,  7,   7)
            )
            self.gap = nn.AdaptiveAvgPool2d(1)   # Global Average Pooling → (B, 256, 1, 1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1),   # Single logit for BCEWithLogitsLoss
            )

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)
            x = self.classifier(x)
            return x


    model = TumorCNN().to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


    # ─────────────────────────── TRAINING ─────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)


    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = model(images)
                loss    = criterion(outputs, labels)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds       = (torch.sigmoid(outputs) >= 0.5).float()
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        return total_loss / total, correct / total


    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print("\n" + "="*60)
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss,   val_acc   = run_epoch(val_loader,   train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_model.pth"))

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
              + (" ← best" if val_acc == best_val_acc else ""))

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed/60:.1f} min | Best val acc: {best_val_acc:.4f}")


    # ─────────────────────────── EVALUATION ───────────────────────
    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_model.pth"), map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            preds   = (torch.sigmoid(outputs) >= 0.5).float().cpu().squeeze(1)
            all_preds.extend(preds.numpy().astype(int))
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = [k for k, v in sorted(test_dataset.class_to_idx.items(), key=lambda x: x[1])]
    print(f"\nTest set results (class order: {class_names})")
    print(classification_report(all_labels, all_preds, target_names=class_names))


    # ─────────────────────────── PLOTS ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=axes[2], colorbar=False)
    axes[2].set_title("Confusion Matrix (Test Set)")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "results.png"), dpi=150)
    plt.show()
    print("Saved results.png")
