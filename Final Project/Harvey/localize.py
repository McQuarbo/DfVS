"""
Tumor Size & Location Estimation (direct image processing — no Grad-CAM)
=========================================================================
Only runs on images correctly predicted as meningioma (true positives).

Pipeline applied directly to the MRI:
  1. Grayscale + Gaussian blur      (spatial filtering)
  2. Otsu threshold + morphological closing → brain mask  (skull strip)
  3. Percentile threshold within brain → hyper-intense candidate mask
  4. Morphological opening           (remove noise)
  5. Morphological closing           (fill tumor holes)
  6. Contour detection               (find tumor boundary)
  7. Moment-based feature extraction (centroid, area, bounding box)

Run after training:  python localize.py
Requires:  pip install opencv-python
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torchvision import datasets, transforms

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
TEST_DIR   = os.path.join(DATA_DIR, "Testing")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pth")

IMG_SIZE        = 224
NUM_IMAGES      = 8       # how many true-positive images to display
SEED            = 42
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholding: keep pixels brighter than this percentile within the brain mask
BRIGHT_PERCENTILE = 88    # top 12 % brightest pixels → likely tumor

# Morphological kernels (ellipse structuring elements)
OPEN_KERNEL   = 3         # opening  — erode noise
CLOSE_KERNEL  = 11        # closing  — fill gaps

random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────── MODEL ────────────────────────────
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


# ─────────────────────────── IMAGE PROCESSING ─────────────────
def make_brain_mask(gray):
    """
    Step 1-2: Extract the brain region by exploiting the dark gap between
    the skull ring and the brain visible in MRI scans.

      - Otsu threshold: skull + brain both appear bright → both become white
      - Morphological opening (kernel > skull ring thickness):
          Erosion destroys thin structures (the skull ring) while the larger
          brain blob shrinks but survives. Dilation then restores the brain
          to roughly its original size. The skull ring is gone.
      - Largest remaining blob = the brain
      - Morphological closing: fills dark internal holes (ventricles, sulci)
          so the brain mask is one solid region
      - Small erosion (5px): conservative margin to avoid any residual skull

    Returns a binary mask (H, W) uint8.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold — skull and brain are both bright relative to background
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Opening: kernel (15px) > skull ring thickness → skull ring is erased,
    # brain blob survives
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # Keep only the largest remaining blob — the brain
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(brain_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Closing: fills dark ventricles/sulci so the mask is solid
    k_close    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, k_close)

    # Small erosion for a safe margin from any residual skull boundary
    k_erode    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brain_mask = cv2.erode(brain_mask, k_erode, iterations=1)

    return brain_mask


def segment_tumor(gray, brain_mask):
    """
    Step 3-6: Find hyper-intense (bright) regions within the brain.

      3. Gaussian blur — spatial filtering to smooth local noise
      4. Percentile threshold — keep only the top N% brightest brain pixels
         (meningiomas are hyper-intense relative to surrounding tissue)
      5. Morphological opening  — erode + dilate to remove speckle noise
      6. Morphological closing  — dilate + erode to fill holes in the region
      7. Contour detection      — find the tumor boundary

    Returns:
      tumor_mask : binary (H, W) uint8
      contours   : list of OpenCV contours
      blurred    : the smoothed grayscale image (for display)
    """
    # Step 3 — Gaussian blur (spatial filtering)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Step 4 — Percentile threshold within the brain mask only
    brain_pixels = blurred[brain_mask > 0]
    if len(brain_pixels) == 0:
        return np.zeros_like(gray), [], blurred
    threshold = np.percentile(brain_pixels, BRIGHT_PERCENTILE)
    _, bright = cv2.threshold(blurred, int(threshold), 255, cv2.THRESH_BINARY)

    # Restrict to brain region
    bright = cv2.bitwise_and(bright, bright, mask=brain_mask)

    # Step 5 — Morphological opening (remove isolated noise)
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    opened = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k_open)

    # Step 6 — Morphological closing (fill holes in tumor body)
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    tumor_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    # Step 7 — Contour detection (tumor boundary)
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return tumor_mask, contours, blurred


def extract_features(contours, img_area):
    """
    Step 7: From candidate contours pick the most tumor-like one and
    compute spatial features via image moments.

      - Filter out tiny specks (< 80 px)
      - Score each contour by  area × circularity
          circularity = 4π·area / perimeter²  (1.0 = perfect circle)
        Thin skull strips score near 0; compact round blobs score high.
      - Centroid  (cx, cy)  — first-order image moments
      - Area in pixels and as % of total image area
      - Bounding box (x, y, w, h)
      - Equivalent circular diameter

    Returns a dict, or None if no valid contour found.
    """
    if not contours:
        return None

    # Filter by minimum area
    valid = [c for c in contours if cv2.contourArea(c) >= 80]
    if not valid:
        return None

    def circularity(c):
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        return (4 * np.pi * a / (p * p)) if p > 0 else 0

    # Best contour = largest area weighted by circularity
    # Favours compact blobs over thin strips of skull
    best    = max(valid, key=lambda c: cv2.contourArea(c) * circularity(c))
    area_px = cv2.contourArea(best)

    M  = cv2.moments(best)
    cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
    cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

    x, y, w, h = cv2.boundingRect(best)
    diameter   = 2 * np.sqrt(area_px / np.pi)

    return {
        'contour'  : best,
        'area_px'  : area_px,
        'area_pct' : 100.0 * area_px / img_area,
        'centroid' : (cx, cy),
        'bbox'     : (x, y, w, h),
        'diameter' : diameter,
    }


def denorm(tensor):
    """Denormalise a (3, H, W) tensor → (H, W, 3) uint8 RGB."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def annotate(img_bgr, features, brain_mask):
    """Draw brain boundary, tumor contour, bounding box, centroid, and text."""
    out = img_bgr.copy()

    # Brain outline (white, thin)
    brain_contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, brain_contours, -1, (200, 200, 200), 1)

    if features is None:
        return out

    cx, cy     = features['centroid']
    x, y, w, h = features['bbox']
    contour    = features['contour']

    # Tumor boundary (red)
    cv2.drawContours(out, [contour], -1, (0, 0, 220), 2)

    # Bounding box (orange)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 140, 255), 1)

    # Centroid cross (yellow)
    cv2.drawMarker(out, (cx, cy), (0, 255, 255),
                   cv2.MARKER_CROSS, 14, 2)

    # Measurements overlay
    lines = [
        f"Area: {features['area_pct']:.1f}% of image",
        f"Diam: {features['diameter']:.1f} px",
        f"Centre: ({cx}, {cy})",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (5, 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────── MAIN ─────────────────────────────
if __name__ == '__main__':
    model = TumorCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded  |  device: {DEVICE}")

    eval_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)
    class_names  = [k for k, v
                    in sorted(test_dataset.class_to_idx.items(), key=lambda x: x[1])]
    meningioma_idx = class_names.index('meningioma')

    # ── Find all true positives (true=meningioma, pred=meningioma)
    print("Scanning test set for true positives...")
    true_positives = []
    with torch.no_grad():
        for idx, (img_tensor, label) in enumerate(test_dataset):
            if label != meningioma_idx:
                continue
            prob = torch.sigmoid(
                model(img_tensor.unsqueeze(0).to(DEVICE))
            ).item()
            if prob >= 0.5:
                true_positives.append((idx, prob))

    print(f"Found {len(true_positives)} true positive meningioma images")
    random.shuffle(true_positives)
    selected = true_positives[:NUM_IMAGES]

    # ── Figure: 4 columns (original | brain mask | tumor mask | annotated)
    fig, axes = plt.subplots(len(selected), 4,
                             figsize=(16, len(selected) * 3.5))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        'Original MRI',
        'Brain Mask\n(Otsu + morphological close)',
        'Tumor Mask\n(percentile threshold + open/close)',
        'Annotated\n(boundary + bounding box)',
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight='bold')

    img_area = IMG_SIZE * IMG_SIZE

    print(f"\n{'Idx':<6} {'Conf':>6} {'Area%':>7} {'Diam(px)':>9} {'Centroid':>14}")
    print("-" * 50)

    for row, (idx, prob) in enumerate(selected):
        img_tensor, _ = test_dataset[idx]
        img_rgb       = denorm(img_tensor)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Classical pipeline
        brain_mask              = make_brain_mask(gray)
        tumor_mask, contours, _ = segment_tumor(gray, brain_mask)
        features                = extract_features(contours, img_area)

        # Annotated image
        img_bgr       = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr = annotate(img_bgr, features, brain_mask)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_ylabel(f"Conf: {prob:.2f}", fontsize=8,
                                rotation=0, labelpad=55, va='center')
        axes[row, 1].imshow(brain_mask, cmap='gray')
        axes[row, 2].imshow(tumor_mask, cmap='gray')
        axes[row, 3].imshow(annotated_rgb)

        if features:
            axes[row, 3].set_xlabel(
                f"Area: {features['area_pct']:.1f}%  |  "
                f"Diam: {features['diameter']:.1f}px  |  "
                f"Centre: {features['centroid']}",
                fontsize=7
            )
            print(f"{idx:<6} {prob:>6.3f} {features['area_pct']:>6.1f}% "
                  f"{features['diameter']:>9.1f} {str(features['centroid']):>14}")
        else:
            print(f"{idx:<6} {prob:>6.3f}   N/A       N/A            N/A")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Meningioma Localisation — Direct Image Processing Pipeline\n"
        "Grayscale → Gaussian Blur → Otsu (brain mask) → "
        "Percentile Threshold → Morphological Open/Close → Contour Features",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "localization_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved to {out_path}")
