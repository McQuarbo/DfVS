import cv2
import numpy as np

from src.utils import order_corners


def detect_reference_card(image_bgr, pre, cfg):
    color_candidate = None

    if cfg["card_detection"]["use_color_first"]:
        found, corners, debug_mask, metrics = detect_wise_card_by_color(image_bgr, cfg)
        if found:
            area_ok = metrics["area_fraction"] <= cfg["card_detection"].get("max_green_area_frac", 0.08)
            aspect_ok = abs(metrics["detected_aspect_ratio"] - cfg["card"]["true_aspect"]) <= cfg["card_detection"]["aspect_tol"]
            if area_ok and aspect_ok:
                return {
                    "found": True,
                    "corners": corners,
                    "method": "automatic_color",
                    "debug_mask": debug_mask,
                    "metrics": metrics,
                }

            # Keep a relaxed auto candidate before dropping to manual.
            color_candidate = {
                "found": True,
                "corners": corners,
                "method": "automatic_color_relaxed",
                "debug_mask": debug_mask,
                "metrics": metrics,
            }

    found, corners, debug_mask, metrics = detect_card_by_edges(image_bgr, pre, cfg)
    if found:
        return {
            "found": True,
            "corners": corners,
            "method": "automatic_edge",
            "debug_mask": debug_mask,
            "metrics": metrics,
        }

    if color_candidate is not None:
        m = color_candidate["metrics"]
        relaxed_area_ok = m["area_fraction"] <= cfg["card_detection"].get("relaxed_max_green_area_frac", 0.14)
        relaxed_aspect_ok = abs(m["detected_aspect_ratio"] - cfg["card"]["true_aspect"]) <= cfg["card_detection"].get(
            "relaxed_aspect_tol", cfg["card_detection"]["aspect_tol"] + 0.20
        )
        if relaxed_area_ok and relaxed_aspect_ok:
            return color_candidate

    return {
        "found": False,
        "corners": None,
        "method": "failed",
        "debug_mask": None,
        "metrics": {
            "detected_aspect_ratio": 0.0,
            "area_fraction": 0.0,
            "rectangularity": 0.0,
            "edge_support": 0.0,
        },
    }


def detect_wise_card_by_color(image_bgr, cfg):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array(cfg["card_detection"]["green_hsv_lower"], dtype=np.uint8)
    upper = np.array(cfg["card_detection"]["green_hsv_upper"], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_corners = None
    best_metrics = None
    true_aspect = cfg["card"]["true_aspect"]
    H, W = image_bgr.shape[:2]
    img_area = H * W

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["card_detection"]["min_green_area"]:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        pts = order_corners(pts)

        w1 = np.linalg.norm(pts[1] - pts[0])
        w2 = np.linalg.norm(pts[2] - pts[3])
        h1 = np.linalg.norm(pts[3] - pts[0])
        h2 = np.linalg.norm(pts[2] - pts[1])
        w = (w1 + w2) / 2.0
        h = (h1 + h2) / 2.0

        if w < 1 or h < 1:
            continue

        aspect = max(w, h) / min(w, h)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_area = abs(cv2.contourArea(box))
        rectangularity = area / max(box_area, 1.0)

        area_fraction = area / img_area
        aspect_score = max(0.0, 1.0 - abs(aspect - true_aspect) / 0.4)
        rect_score = max(0.0, min(1.0, rectangularity))

        score = 2.5 * aspect_score + 1.5 * rect_score + 0.2 * min(1.0, area_fraction / 0.03)

        if score > best_score:
            best_score = score
            best_corners = pts
            best_metrics = {
                "detected_aspect_ratio": aspect,
                "area_fraction": area_fraction,
                "rectangularity": rectangularity,
                "edge_support": 1.0,
            }

    if best_corners is None:
        return False, None, mask, None

    return True, best_corners, mask, best_metrics


def detect_card_by_edges(image_bgr, pre, cfg):
    edge = pre["edge"]
    H, W = edge.shape[:2]
    img_area = H * W

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bw = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    bw = cv2.dilate(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_corners = None
    best_metrics = None
    true_aspect = cfg["card"]["true_aspect"]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_fraction = area / img_area

        if area_fraction < cfg["card_detection"]["edge_min_area_frac"]:
            continue
        if area_fraction > cfg["card_detection"]["edge_max_area_frac"]:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        pts = order_corners(pts)

        w1 = np.linalg.norm(pts[1] - pts[0])
        w2 = np.linalg.norm(pts[2] - pts[3])
        h1 = np.linalg.norm(pts[3] - pts[0])
        h2 = np.linalg.norm(pts[2] - pts[1])
        w = (w1 + w2) / 2.0
        h = (h1 + h2) / 2.0
        if w < 1 or h < 1:
            continue

        aspect = max(w, h) / min(w, h)
        if abs(aspect - true_aspect) > cfg["card_detection"]["aspect_tol"]:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_area = abs(cv2.contourArea(box))
        rectangularity = area / max(box_area, 1.0)

        score = (
            2.0 * max(0.0, 1.0 - abs(aspect - true_aspect) / 0.4)
            + 1.2 * max(0.0, min(1.0, rectangularity))
            + 0.2 * min(1.0, area_fraction / 0.03)
        )

        if score > best_score:
            best_score = score
            best_corners = pts
            best_metrics = {
                "detected_aspect_ratio": aspect,
                "area_fraction": area_fraction,
                "rectangularity": rectangularity,
                "edge_support": 0.8,
            }

    if best_corners is None:
        return False, None, bw, None

    return True, best_corners, bw, best_metrics
