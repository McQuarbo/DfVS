from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

from src.utils import ensure_dir, save_image
from src.preprocess import preprocess_image
from src.card_detection import detect_reference_card
from src.manual import manual_select_card_corners
from src.rectification import rectify_perspective, compute_scale_from_card
from src.segmentation import segment_object, segment_object_original
from src.measurement import measure_phone
from src.reliability import compute_reliability
from src.visualization import make_card_detection_overlay, make_final_overlay


def read_image_any_format(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _with_object_profile(cfg, object_name=None):
    cfg_run = deepcopy(cfg)

    objects = cfg_run.get("objects", {})
    active_name = object_name or cfg_run.get("default_object", "phone")
    if active_name not in objects:
        known = ", ".join(sorted(objects.keys())) if objects else "none"
        raise ValueError(f"Unknown object profile '{active_name}'. Available: {known}")

    profile = objects[active_name]
    cfg_run["active_object"] = active_name
    cfg_run["active_object_profile"] = profile
    cfg_run["segmentation"]["phone_target_aspect"] = float(profile["target_aspect"])
    cfg_run["segmentation"]["target_aspects"] = [float(profile["target_aspect"])]
    return cfg_run


def _measurement_quality(measurement, cfg):
    if measurement["object_area_pixels"] <= 0:
        return -1.0

    width = float(measurement["width_mm"])
    height = float(measurement["height_mm"])
    short_mm = min(width, height)
    long_mm = max(width, height)
    aspect = long_mm / max(short_mm, 1e-6)

    priors = cfg["segmentation"].get("target_aspects")
    if isinstance(priors, (list, tuple)) and len(priors) > 0:
        aspect_score = max(max(0.0, 1.0 - abs(aspect - float(p)) / 1.2) for p in priors)
    else:
        target_aspect = float(cfg["segmentation"]["phone_target_aspect"])
        aspect_score = max(0.0, 1.0 - abs(aspect - target_aspect) / 1.0)

    profile = cfg.get("active_object_profile", {})
    short_lo, short_hi = profile.get("short_mm_range", [20.0, 120.0])
    long_lo, long_hi = profile.get("long_mm_range", [60.0, 320.0])

    short_score = max(0.0, min(1.0, (short_mm - short_lo) / max(short_hi - short_lo, 1e-6)))
    long_score = max(0.0, min(1.0, (long_mm - long_lo) / max(long_hi - long_lo, 1e-6)))
    border_score = 0.0 if measurement["touches_border"] else 1.0
    fill_score = max(0.0, min(1.0, (measurement["fill_ratio"] - 0.35) / (0.95 - 0.35)))

    return 2.0 * aspect_score + 1.4 * short_score + 1.2 * long_score + 0.8 * fill_score + 0.8 * border_score


def _looks_implausible(measurement, cfg):
    if measurement["object_area_pixels"] <= 0:
        return True

    short_mm = min(float(measurement["width_mm"]), float(measurement["height_mm"]))
    long_mm = max(float(measurement["width_mm"]), float(measurement["height_mm"]))
    aspect = long_mm / max(short_mm, 1e-6)

    profile = cfg.get("active_object_profile", {})
    short_lo, _ = profile.get("short_mm_range", [15.0, 220.0])
    long_lo, _ = profile.get("long_mm_range", [40.0, 520.0])
    min_aspect = float(profile.get("min_aspect", 1.15))
    max_aspect = float(profile.get("max_aspect", 6.0))

    if short_mm < short_lo * 0.45 or long_mm < long_lo * 0.45:
        return True
    if aspect < min_aspect * 0.7 or aspect > max_aspect * 1.3:
        return True

    return False


def _build_manual_card(corners, image_bgr, cfg):
    pts = corners.astype(np.float32)
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)

    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])
    cw = (w1 + w2) / 2.0
    ch = (h1 + h2) / 2.0
    detected_aspect = max(cw, ch) / max(1.0, min(cw, ch))

    poly_area = abs(cv2.contourArea(pts))
    area_fraction = poly_area / max(img_area, 1.0)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box_area = abs(cv2.contourArea(box))
    rectangularity = poly_area / max(box_area, 1.0)

    return {
        "found": True,
        "corners": pts,
        "method": "manual",
        "debug_mask": None,
        "metrics": {
            "detected_aspect_ratio": float(detected_aspect),
            "area_fraction": float(area_fraction),
            "rectangularity": float(rectangularity),
            "edge_support": 1.0,
        },
    }


def _build_card_exclusion_mask(card_corners, rect_info, rectified_shape):
    """Create a filled mask of the reference card region in rectified space."""
    H, W = rectified_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    # In rectified space the card occupies the top-left rectangle whose
    # size is encoded in rect_info["card_rectified_size"].
    cw, ch = rect_info["card_rectified_size"]
    # Pad a bit to cover edge bleed.
    pad = 10
    pts = np.array([[0, 0], [cw + pad, 0], [cw + pad, ch + pad], [0, ch + pad]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def _build_card_exclusion_mask_original(card_corners, image_shape):
    """Create a filled mask of the reference card region in original image space."""
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = card_corners.reshape(-1, 2).astype(np.int32)
    # Expand slightly to cover edge bleed.
    center = pts.mean(axis=0)
    expanded = ((pts - center) * 1.08 + center).astype(np.int32)
    cv2.fillConvexPoly(mask, expanded, 255)
    return mask


def run_single_image(image_path: Path, cfg, allow_manual_fallback=True, force_manual=False, object_name=None):
    cfg = _with_object_profile(cfg, object_name)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    for p in cfg["paths"].values():
        if isinstance(p, Path) and p.name in ["results", "intermediate", "overlays", "tables"]:
            ensure_dir(p)

    image_bgr = read_image_any_format(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    stem = image_path.stem

    pre = preprocess_image(image_bgr, cfg)

    if cfg["save"]["intermediate"]:
        save_image(cfg["paths"]["intermediate"] / f"{stem}_gray.png", pre["gray"])
        save_image(cfg["paths"]["intermediate"] / f"{stem}_edge.png", pre["edge"])

    card = detect_reference_card(image_bgr, pre, cfg)

    if force_manual:
        if not allow_manual_fallback:
            raise RuntimeError("Cannot force manual mode when GUI/manual fallback is disabled.")

        manual_corners = manual_select_card_corners(image_bgr)
        if manual_corners is None:
            raise RuntimeError("Manual corner selection was cancelled.")

        card = _build_manual_card(manual_corners, image_bgr, cfg)

    if not card["found"] and allow_manual_fallback:
        manual_corners = manual_select_card_corners(image_bgr)
        if manual_corners is not None:
            card = _build_manual_card(manual_corners, image_bgr, cfg)

    if not card["found"]:
        raise RuntimeError("Reference card detection failed and no manual fallback was provided.")

    card_overlay = make_card_detection_overlay(image_bgr, card)
    if cfg["save"]["intermediate"]:
        save_image(cfg["paths"]["intermediate"] / f"{stem}_card_detection.png", card_overlay)
        if card["debug_mask"] is not None:
            save_image(cfg["paths"]["intermediate"] / f"{stem}_card_mask.png", card["debug_mask"])

    scale = compute_scale_from_card(card["corners"], cfg)

    rect = rectify_perspective(image_bgr, card["corners"], cfg)
    rectified_bgr = rect["image"]

    rect_w, rect_h = rect["card_rectified_size"]
    card_long_px = float(max(rect_w, rect_h))
    card_short_px = float(min(rect_w, rect_h))
    rectified_mm_per_pixel = 0.5 * (
        cfg["card"]["real_width_mm"] / card_long_px
        + cfg["card"]["real_height_mm"] / card_short_px
    )

    if cfg["save"]["intermediate"]:
        save_image(cfg["paths"]["intermediate"] / f"{stem}_rectified.png", rectified_bgr)

    # Build an exclusion mask covering the detected reference card so
    # segmentation never confuses the card with the target object.
    card_excl_rect = _build_card_exclusion_mask(
        card["corners"], rect, rectified_bgr.shape[:2]
    )
    card_excl_orig = _build_card_exclusion_mask_original(
        card["corners"], image_bgr.shape[:2]
    )

    segmentation = segment_object(rectified_bgr, cfg, card_mask=card_excl_rect)
    measurement = measure_phone(segmentation["clean_mask"], rectified_mm_per_pixel, cfg)

    invalid_rectified_mask = measurement["object_area_pixels"] == 0
    if measurement["bbox"] is not None:
        _, _, bw, bh = measurement["bbox"]
        Hm, Wm = segmentation["clean_mask"].shape[:2]
        bbox_cover = (bw * bh) / max(Hm * Wm, 1)
        if measurement["touches_border"] and bbox_cover > 0.80 and measurement["fill_ratio"] > 0.55:
            invalid_rectified_mask = True

    if invalid_rectified_mask or _looks_implausible(measurement, cfg):
        alt_segmentation = segment_object_original(image_bgr, cfg, card_mask=card_excl_orig)
        alt_measurement = measure_phone(alt_segmentation["clean_mask"], scale["mm_per_pixel"], cfg)

        rect_score = _measurement_quality(measurement, cfg)
        alt_score = _measurement_quality(alt_measurement, cfg)

        if alt_score > rect_score:
            segmentation = alt_segmentation
            measurement = alt_measurement
            rectified_bgr = image_bgr

    if cfg["save"]["intermediate"]:
        save_image(cfg["paths"]["intermediate"] / f"{stem}_raw_mask.png", segmentation["raw_mask"])
        save_image(cfg["paths"]["intermediate"] / f"{stem}_clean_mask.png", segmentation["clean_mask"])

    reliability = compute_reliability(card, scale, segmentation, measurement, cfg)

    final_overlay = make_final_overlay(rectified_bgr, segmentation["clean_mask"], measurement, reliability)
    overlay_path = cfg["paths"]["overlays"] / f"{stem}_overlay.png"
    save_image(overlay_path, final_overlay)

    return {
        "card": card,
        "scale": scale,
        "rectification": rect,
        "segmentation": segmentation,
        "measurement": measurement,
        "reliability": reliability,
        "paths": {
            "overlay": str(overlay_path)
        }
    }
