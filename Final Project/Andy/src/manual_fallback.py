import matplotlib.pyplot as plt
import numpy as np

from .utils import order_corners


def manual_select_card_corners(image: np.ndarray) -> np.ndarray:
    rgb = image[:, :, ::-1] if image.ndim == 3 else image
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb)
    plt.title("Click the 4 Wise card corners: TL, TR, BR, BL")
    pts = plt.ginput(4, timeout=0)
    plt.close()

    if len(pts) != 4:
        raise RuntimeError("Manual corner selection failed: 4 points were not selected.")

    return order_corners(np.array(pts, dtype=np.float32))
