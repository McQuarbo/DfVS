import cv2
import numpy as np


def rectify_perspective(image_bgr, card_corners, cfg):
    out_w = cfg["rectification"]["output_card_width_px"]
    out_h = cfg["rectification"]["output_card_height_px"]

    pts = card_corners.astype(np.float32)
    edge_w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2.0
    edge_h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2.0

    # Keep destination orientation aligned to observed card orientation.
    if edge_w >= edge_h:
        dst_w, dst_h = out_w, out_h
    else:
        dst_w, dst_h = out_h, out_w

    dst = np.array(
        [
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(pts, dst)

    H, W = image_bgr.shape[:2]
    src_corners = np.array(
        [[[0, 0]], [[W - 1, 0]], [[W - 1, H - 1]], [[0, H - 1]]],
        dtype=np.float32,
    )
    warped_corners = cv2.perspectiveTransform(src_corners, M).reshape(-1, 2)

    min_xy = warped_corners.min(axis=0)
    max_xy = warped_corners.max(axis=0)

    tx = -min_xy[0] if min_xy[0] < 0 else 0.0
    ty = -min_xy[1] if min_xy[1] < 0 else 0.0

    T = np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    M_adj = T @ M

    out_w = int(np.ceil(max_xy[0] + tx)) + 1
    out_h = int(np.ceil(max_xy[1] + ty)) + 1
    out_w = max(out_w, int(dst_w + tx) + 1)
    out_h = max(out_h, int(dst_h + ty) + 1)

    rectified = cv2.warpPerspective(image_bgr, M_adj, (out_w, out_h))

    return {
        "image": rectified,
        "matrix": M_adj,
        "card_rectified_size": (int(dst_w), int(dst_h)),
    }


def compute_scale_from_card(card_corners, cfg):
    pts = card_corners.astype(np.float32)

    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])

    edge_a = (w1 + w2) / 2.0
    edge_b = (h1 + h2) / 2.0

    # Map the longer observed edge to physical card width regardless of rotation.
    card_width_px = max(edge_a, edge_b)
    card_height_px = min(edge_a, edge_b)

    scale_x = cfg["card"]["real_width_mm"] / card_width_px
    scale_y = cfg["card"]["real_height_mm"] / card_height_px
    mm_per_pixel = (scale_x + scale_y) / 2.0

    return {
        "card_width_px": float(card_width_px),
        "card_height_px": float(card_height_px),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "mm_per_pixel": float(mm_per_pixel),
    }
