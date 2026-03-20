"""
Grad-CAM heatmap visualisation for the trained TumorCNN.
Run after training is complete:  python gradcam.py
Loads best_model.pth and visualises on a sample of test images.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
TEST_DIR   = os.path.join(DATA_DIR, "Testing")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pth")

IMG_SIZE   = 224
NUM_IMAGES = 12        # how many images to visualise in the grid
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────── MODEL DEFINITION ─────────────────
# (must match train.py exactly)
class ConvBlock(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ─────────────────────────── GRAD-CAM ─────────────────────────
class GradCAM:
    """
    Hooks into the target layer, runs a forward+backward pass,
    and produces a (H, W) heatmap in [0, 1].
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        # forward hook — saves the output feature maps
        target_layer.register_forward_hook(self._save_activation)
        # backward hook — saves the gradients flowing back
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        input_tensor: (1, 3, H, W) on DEVICE, requires_grad not needed on input.
        Returns: numpy array (H, W) in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)           # forward pass
        self.model.zero_grad()
        output.backward(torch.ones_like(output))    # backward w.r.t. the single logit

        # Global-average-pool the gradients over spatial dims → importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)                        # keep only positive influence

        # Upsample to input image size
        cam = torch.nn.functional.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def overlay_heatmap(img_tensor, cam, alpha=0.45):
    """
    img_tensor : (3, H, W) normalised tensor
    cam        : (H, W) array in [0, 1]
    Returns    : (H, W, 3) uint8 RGB overlay
    """
    # Denormalise image back to [0, 1]
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img  = (img * std + mean).clip(0, 1)

    # Colour the CAM with a jet colourmap
    heatmap = cm.jet(cam)[:, :, :3]               # (H, W, 3) in [0, 1]

    # Blend
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)
    return overlay


# ─────────────────────────── MAIN ─────────────────────────────
if __name__ == '__main__':
    # Load model
    model = TumorCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Using device: {DEVICE}")

    # Attach Grad-CAM to the last ConvBlock (index 4 in model.features)
    gradcam = GradCAM(model, target_layer=model.features[4])

    # Load test dataset (no augmentation)
    eval_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)
    class_names  = [k for k, v in sorted(test_dataset.class_to_idx.items(), key=lambda x: x[1])]
    print(f"Classes: {class_names}")   # ['meningioma', 'notumor']

    # Pick a random balanced sample
    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    selected = indices[:NUM_IMAGES]

    # ── Plot grid ──────────────────────────────────────────────
    cols = 4
    rows = NUM_IMAGES // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Grad-CAM Heatmaps — Meningioma Detection\n"
                 "(red = high activation, blue = low activation)", fontsize=13)

    for ax, idx in zip(axes.flat, selected):
        img_tensor, label = test_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        # Generate heatmap
        cam = gradcam.generate(input_tensor)

        # Predict
        with torch.no_grad():
            logit = model(input_tensor)
            prob  = torch.sigmoid(logit).item()
            pred  = int(prob >= 0.5)

        overlay = overlay_heatmap(img_tensor, cam)

        true_label = class_names[label]
        pred_label = class_names[pred]
        correct    = (pred == label)

        ax.imshow(overlay)
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}  ({prob:.2f})",
            color='green' if correct else 'red',
            fontsize=9
        )
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "gradcam_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved heatmap grid to {out_path}")
