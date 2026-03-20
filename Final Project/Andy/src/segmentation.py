import cv2
import numpy as np


def get_target_aspects(cfg):
    vals = cfg["segmentation"].get("target_aspects")
    if isinstance(vals, (list, tuple)) and len(vals) > 0:
        return [float(v) for v in vals]
    return [float(cfg["segmentation"].get("phone_target_aspect", 2.0))]


def aspect_match_score(aspect, cfg, tol=1.2):
    priors = get_target_aspects(cfg)
    return max(max(0.0, 1.0 - abs(float(aspect) - p) / tol) for p in priors)


def segment_object(rectified_bgr, cfg, card_mask=None):
    method = cfg["segmentation"]["method"].lower()

    if method == "hsv":
        raw = segment_by_hsv(rectified_bgr)
    else:
        raw = segment_by_adaptive_threshold(rectified_bgr, cfg)

    # Exclude the detected reference card region from the mask so the
    # segmentation stage never confuses the card with the target object.
    if card_mask is not None:
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(card_mask))

    clean = clean_mask(raw)
    selected = select_best_component(clean, cfg)

    multi_cue = segment_by_multi_cue(rectified_bgr, cfg, allow_border=False)
    if card_mask is not None:
        multi_cue = cv2.bitwise_and(multi_cue, cv2.bitwise_not(card_mask))
    selected = choose_better_mask(selected, multi_cue, cfg)

    profile = cfg.get("active_object_profile", {})

    # Profile-driven candidate paths.
    if profile.get("use_color_outline"):
        color_candidate = segment_color_outline(rectified_bgr, cfg, allow_border=False)
        selected = choose_better_mask(selected, color_candidate, cfg)

    if profile.get("use_card_outline"):
        card_candidate = segment_card_like_outline(rectified_bgr, cfg, allow_border=True)
        if card_mask is not None:
            card_candidate = cv2.bitwise_and(card_candidate, cv2.bitwise_not(card_mask))
        cq = mask_quality_score(card_candidate, cfg)
        if cq > 1.25 and mask_geometry_plausible(card_candidate, cfg):
            selected = extract_largest_component(card_candidate)
        else:
            selected = choose_higher_quality_mask(selected, card_candidate, cfg, bias_to_b=0.10)

    # Edge-rectangle candidate: always try for non-phone profiles, fallback for phone.
    edge_candidate = segment_by_edge_rect(rectified_bgr, cfg, allow_border=False)
    if card_mask is not None:
        edge_candidate = cv2.bitwise_and(edge_candidate, cv2.bitwise_not(card_mask))
    edge_score = mask_quality_score(edge_candidate, cfg)
    selected_score = mask_quality_score(selected, cfg)
    if edge_score > selected_score + 0.35:
        selected = extract_largest_component(edge_candidate)

    if np.count_nonzero(selected) == 0:
        rescued = rescue_fragmented_mask(clean)
        selected = select_best_component(rescued, cfg, allow_border=True)

    metrics = compute_segmentation_metrics(clean, selected)

    return {
        "raw_mask": raw,
        "clean_mask": selected,
        "metrics": metrics,
    }


# Backward-compatible alias.
segment_phone = segment_object


def segment_object_original(image_bgr, cfg, card_mask=None):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if card_mask is not None:
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(card_mask))

    clean = clean_mask_no_fill(raw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    selected = select_best_component(clean, cfg, allow_border=False)
    if np.count_nonzero(selected) == 0:
        selected = select_best_component(clean, cfg, allow_border=True)

    multi_cue = segment_by_multi_cue(image_bgr, cfg, allow_border=True)
    if card_mask is not None:
        multi_cue = cv2.bitwise_and(multi_cue, cv2.bitwise_not(card_mask))
    selected = choose_better_mask(selected, multi_cue, cfg)

    profile = cfg.get("active_object_profile", {})

    if profile.get("use_color_outline"):
        color_candidate = segment_color_outline(image_bgr, cfg, allow_border=True)
        selected = choose_better_mask(selected, color_candidate, cfg)

    if profile.get("use_card_outline"):
        card_candidate = segment_card_like_outline(image_bgr, cfg, allow_border=True)
        if card_mask is not None:
            card_candidate = cv2.bitwise_and(card_candidate, cv2.bitwise_not(card_mask))
        cq = mask_quality_score(card_candidate, cfg)
        if cq > 1.25 and mask_geometry_plausible(card_candidate, cfg):
            selected = extract_largest_component(card_candidate)
        else:
            selected = choose_higher_quality_mask(selected, card_candidate, cfg, bias_to_b=0.10)

    edge_candidate = segment_by_edge_rect(image_bgr, cfg, allow_border=True)
    if card_mask is not None:
        edge_candidate = cv2.bitwise_and(edge_candidate, cv2.bitwise_not(card_mask))
    edge_score = mask_quality_score(edge_candidate, cfg)
    selected_score = mask_quality_score(selected, cfg)
    if edge_score > selected_score + 0.30:
        selected = extract_largest_component(edge_candidate)

    metrics = compute_segmentation_metrics(clean, selected)

    return {
        "raw_mask": raw,
        "clean_mask": selected,
        "metrics": metrics,
    }


# Backward-compatible alias.
segment_phone_original = segment_object_original


def segment_by_adaptive_threshold(image_bgr, cfg):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    block_size = cfg["segmentation"]["adaptive_block_size"]
    if block_size % 2 == 0:
        block_size += 1

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        cfg["segmentation"]["adaptive_C"],
    )

    return bw


def _auto_gamma(gray):
    mean = float(np.mean(gray)) / 255.0
    mean = max(mean, 1e-3)
    gamma = np.log(0.5) / np.log(mean)
    gamma = float(np.clip(gamma, 0.65, 1.6))

    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, lut)


def _enhance_gray_for_segmentation(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement: gamma correction + local histogram equalization (CLAHE).
    gray = _auto_gamma(gray)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def segment_by_multi_cue(image_bgr, cfg, allow_border=False):
    gray = _enhance_gray_for_segmentation(image_bgr)

    # Global and local thresholding cues.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    block_size = int(cfg["segmentation"].get("adaptive_block_size", 51))
    if block_size % 2 == 0:
        block_size += 1
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(cfg["segmentation"].get("adaptive_C", 8)),
    )

    # Gradient cue (Sobel magnitude).
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag8 = cv2.convertScaleAbs(mag)
    _, sobel_bw = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fused = cv2.bitwise_or(otsu, adaptive)
    fused = cv2.bitwise_or(fused, sobel_bw)

    # Morphology to stabilize masks.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k_open)
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k_close)

    fused = clean_mask_no_fill(fused)
    fused = fill_holes(fused)

    selected = select_best_component(fused, cfg, allow_border=allow_border)
    if np.count_nonzero(selected) == 0 and allow_border:
        selected = select_best_component(fused, cfg, allow_border=True)

    return selected


def segment_by_hsv(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask1 = cv2.inRange(v, 0, 150)
    mask2 = cv2.inRange(s, 0, 80)

    bw = cv2.bitwise_and(mask1, mask2)
    return bw


def segment_by_edge_rect(image_bgr, cfg, allow_border=False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    v = float(np.median(gray))
    lo = int(max(20, 0.66 * v))
    hi = int(min(220, 1.33 * v))
    edge = cv2.Canny(gray, lo, hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    edge = cv2.dilate(edge, kernel, iterations=1)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    img_area = float(H * W)

    best_score = -1.0
    best_box = None

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < cfg["segmentation"]["min_component_area"]:
            continue

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 2 or rh < 2:
            continue

        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        aspect_score = aspect_match_score(aspect, cfg, tol=1.2)

        box = cv2.boxPoints(rect).astype(np.float32)
        box_area = abs(cv2.contourArea(box))
        if box_area < 1.0:
            continue

        area_frac = box_area / img_area
        if area_frac < 0.01 or area_frac > 0.75:
            continue

        rectangularity = area / box_area
        area_score = max(0.0, 1.0 - abs(area_frac - 0.22) / 0.22)

        x, y, w, h = cv2.boundingRect(box.astype(np.int32))
        touches_border = (x <= 1 or y <= 1 or (x + w) >= W - 1 or (y + h) >= H - 1)
        if touches_border and not allow_border:
            continue

        border_penalty = 0.35 if touches_border else 0.0
        score = 2.2 * aspect_score + 1.1 * max(0.0, min(1.0, rectangularity)) + 0.7 * area_score - border_penalty

        if score > best_score:
            best_score = score
            best_box = box

    out = np.zeros_like(gray, dtype=np.uint8)
    if best_box is not None:
        cv2.fillConvexPoly(out, best_box.astype(np.int32), 255)
    return out


def segment_color_outline(image_bgr, cfg, allow_border=False):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Green glass/body + gold foil neck ranges.
    green = cv2.inRange(hsv, np.array([28, 25, 20], dtype=np.uint8), np.array([100, 255, 255], dtype=np.uint8))
    gold = cv2.inRange(hsv, np.array([8, 35, 35], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))

    # Include darker/saturated pixels to keep bottle silhouette through labels/shadows.
    h, s, v = cv2.split(hsv)
    dark_sat = cv2.inRange(s, 40, 255)
    dark_val = cv2.inRange(v, 0, 165)
    dark = cv2.bitwise_and(dark_sat, dark_val)

    color_mask = cv2.bitwise_or(green, gold)
    color_mask = cv2.bitwise_or(color_mask, dark)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k_open)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k_close)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 40, 130)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    fused = cv2.bitwise_or(color_mask, edge)
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    fused = clean_mask_no_fill(fused)

    selected = select_best_component(fused, cfg, allow_border=allow_border)
    if np.count_nonzero(selected) == 0 and allow_border:
        selected = segment_by_edge_rect(image_bgr, cfg, allow_border=True)

    return selected


def segment_card_like_outline(image_bgr, cfg, allow_border=False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edge = cv2.Canny(gray, 40, 130)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    img_area = float(H * W)

    best_score = -1.0
    best_poly = None

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < max(1200.0, cfg["segmentation"]["min_component_area"]):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
            poly = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            poly = cv2.boxPoints(rect).astype(np.float32)

        poly_area = abs(cv2.contourArea(poly))
        if poly_area < 1.0:
            continue

        area_frac = poly_area / img_area
        if area_frac < 0.015 or area_frac > 0.85:
            continue

        x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
        touches_border = (x <= 1 or y <= 1 or (x + w) >= W - 1 or (y + h) >= H - 1)
        if touches_border and not allow_border:
            continue
        if touches_border and area_frac > 0.55:
            continue

        rect = cv2.minAreaRect(poly.astype(np.float32))
        rw, rh = rect[1]
        if rw < 2 or rh < 2:
            continue

        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        aspect_score = aspect_match_score(aspect, cfg, tol=0.75)

        rectangularity = area / poly_area
        rect_score = max(0.0, min(1.0, rectangularity))

        # Prefer large card-like rectangles.
        area_score = max(0.0, 1.0 - abs(area_frac - 0.30) / 0.35)
        border_penalty = 0.25 if touches_border else 0.0
        score = 2.4 * aspect_score + 1.1 * rect_score + 1.0 * area_score - border_penalty

        if score > best_score:
            best_score = score
            best_poly = poly

    out = np.zeros_like(gray, dtype=np.uint8)
    if best_poly is not None:
        cv2.fillConvexPoly(out, best_poly.astype(np.int32), 255)

    return out


def clean_mask(mask):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_close)

    h, w = out.shape[:2]
    flood = out.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    out = cv2.bitwise_or(out, flood_inv)

    return out


def clean_mask_no_fill(mask):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_close)
    return out


def choose_better_mask(mask_a, mask_b, cfg):
    score_a = mask_quality_score(mask_a, cfg)
    score_b = mask_quality_score(mask_b, cfg)

    primary_weak = score_a < 2.3
    if primary_weak and score_b > score_a + 0.12:
        return extract_largest_component(mask_b)
    return extract_largest_component(mask_a)


def choose_higher_quality_mask(mask_a, mask_b, cfg, bias_to_b=0.0):
    score_a = mask_quality_score(mask_a, cfg)
    score_b = mask_quality_score(mask_b, cfg)

    if score_b + float(bias_to_b) >= score_a:
        return extract_largest_component(mask_b)
    return extract_largest_component(mask_a)


def mask_geometry_plausible(mask, cfg):
    if np.count_nonzero(mask) == 0:
        return False

    m = extract_largest_component(mask)
    if np.count_nonzero(m) == 0:
        return False

    H, W = m.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return False

    x, y, w, h, area = stats[1]
    area_frac = area / max(float(H * W), 1.0)
    touches_border = (x <= 1 or y <= 1 or (x + w) >= W - 1 or (y + h) >= H - 1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if rw < 2 or rh < 2:
        return False

    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    profile = cfg.get("active_object_profile", {})
    min_aspect = float(profile.get("min_aspect", 1.15))
    max_aspect = float(profile.get("max_aspect", 6.0))

    if aspect < min_aspect * 0.65 or aspect > max_aspect * 1.45:
        return False
    if area_frac < 0.015 or area_frac > 0.70:
        return False
    if touches_border and area_frac > 0.40:
        return False

    return True


def extract_largest_component(mask):
    if np.count_nonzero(mask) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    return (labels == best_label).astype(np.uint8) * 255


def mask_quality_score(mask, cfg):
    if np.count_nonzero(mask) == 0:
        return -1.0

    m = extract_largest_component(mask)
    if np.count_nonzero(m) == 0:
        return -1.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return -1.0

    x, y, w, h, area = stats[1]
    H, W = m.shape[:2]

    touches_border = (x <= 1 or y <= 1 or (x + w) >= W - 1 or (y + h) >= H - 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if rw < 2 or rh < 2:
        return -1.0

    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    aspect_score = aspect_match_score(aspect, cfg, tol=1.2)

    area_frac = area / max(float(H * W), 1.0)
    area_score = max(0.0, 1.0 - abs(area_frac - 0.10) / 0.15)

    fill = area / max(float(rw * rh), 1.0)
    fill_score = max(0.0, min(1.0, (fill - 0.35) / (0.95 - 0.35)))

    profile = cfg.get("active_object_profile", {})
    relax = profile.get("relax_border", False)
    if relax:
        border_penalty = 0.15 if touches_border else 0.0
    else:
        border_penalty = 0.35 if touches_border else 0.0
    return 2.0 * aspect_score + 1.0 * area_score + 0.8 * fill_score - border_penalty


def select_best_component(binary_mask, cfg, allow_border=False):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    best_score = -1.0
    best_mask = np.zeros_like(binary_mask)
    H, W = binary_mask.shape[:2]

    profile = cfg.get("active_object_profile", {})
    relax_constraints = profile.get("relax_border", False)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < cfg["segmentation"]["min_component_area"]:
            continue

        touches_border = (x <= 1 or y <= 1 or (x + w) >= W - 1 or (y + h) >= H - 1)
        if touches_border and not allow_border:
            if not relax_constraints:
                continue

        component = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 2 or rh < 2:
            continue

        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        aspect_score = aspect_match_score(aspect, cfg, tol=1.2)

        perimeter = max(cv2.arcLength(cnt, True), 1.0)

        border_component = np.zeros_like(component)
        border_component[0, :] = component[0, :]
        border_component[-1, :] = component[-1, :]
        border_component[:, 0] = np.maximum(border_component[:, 0], component[:, 0])
        border_component[:, -1] = np.maximum(border_component[:, -1], component[:, -1])
        border_pixels = float(np.count_nonzero(border_component))
        border_contact = border_pixels / perimeter

        if allow_border and touches_border:
            if relax_constraints:
                if border_contact > 0.35:
                    continue
            else:
                if border_contact > 0.22:
                    continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1.0)

        area_score = min(1.0, area / 40000.0)

        if relax_constraints:
            border_penalty = 0.08 if touches_border else 0.0
            border_penalty += min(0.15, 0.6 * border_contact)
        else:
            border_penalty = 0.25 if touches_border else 0.0
            border_penalty += min(0.35, 1.2 * border_contact)
        
        score = 1.7 * aspect_score + 1.0 * solidity + 1.0 * area_score - border_penalty

        if score > best_score:
            best_score = score
            best_mask = component

    return best_mask


def rescue_fragmented_mask(mask):
    h, w = mask.shape[:2]
    k = max(31, int(round(min(h, w) * 0.16)))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = fill_holes(out)
    return out


def compute_segmentation_metrics(pre_clean_mask, final_mask):
    total_area = np.count_nonzero(pre_clean_mask)
    final_area = np.count_nonzero(final_mask)

    if total_area == 0:
        largest_fraction = 0.0
    else:
        largest_fraction = final_area / total_area

    filled = fill_holes(final_mask)
    hole_area = np.count_nonzero(filled) - np.count_nonzero(final_mask)
    hole_fraction = hole_area / max(np.count_nonzero(filled), 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pre_clean_mask, connectivity=8)
    if num_labels <= 1:
        small_blob_fraction = 0.0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        small_blob_fraction = float(np.mean(areas < 1000))

    perimeter = cv2.countNonZero(cv2.Canny(final_mask, 50, 150))
    area = np.count_nonzero(final_mask)
    jaggedness = perimeter / max(np.sqrt(area), 1.0)

    return {
        "largest_component_fraction": float(largest_fraction),
        "hole_fraction": float(hole_fraction),
        "small_blob_fraction": float(small_blob_fraction),
        "jaggedness": float(jaggedness),
    }


def fill_holes(mask):
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, flood_inv)
