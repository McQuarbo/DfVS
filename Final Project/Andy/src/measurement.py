import cv2
import numpy as np


def get_target_aspects(cfg):
    vals = cfg["segmentation"].get("target_aspects")
    if isinstance(vals, (list, tuple)) and len(vals) > 0:
        return [float(v) for v in vals]
    return [float(cfg["segmentation"].get("phone_target_aspect", 2.0))]


def nearest_aspect_prior(aspect, cfg):
    priors = get_target_aspects(cfg)
    return min(priors, key=lambda p: abs(float(aspect) - p))


def measure_phone(mask, mm_per_pixel, cfg):
    result = {
        "width_mm": float("nan"),
        "height_mm": float("nan"),
        "area_mm2": float("nan"),
        "touches_border": True,
        "object_area_pixels": 0,
        "fill_ratio": 0.0,
        "aspect_outlier_score": 1.0,
        "bbox": None,
    }

    if np.count_nonzero(mask) == 0:
        return result

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return result

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1

    x, y, w, h, area = stats[best_label]
    component = np.uint8(labels == best_label) * 255

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    if rw < 1 or rh < 1:
        return result

    pts = cnt.reshape(-1, 2).astype(np.float32)
    theta = np.deg2rad(float(rect[2]))
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

    proj_u = pts[:, 0] * u[0] + pts[:, 1] * u[1]
    proj_v = pts[:, 0] * v[0] + pts[:, 1] * v[1]
    
    # Profile-driven trimming percentiles.
    profile = cfg.get("active_object_profile", {})
    trim_lo, trim_hi = profile.get("trim_percentiles", [1, 99])
    
    p1_lo, p1_hi = np.percentile(proj_u, [trim_lo, trim_hi])
    p2_lo, p2_hi = np.percentile(proj_v, [trim_lo, trim_hi])

    span1 = float(max(1.0, p1_hi - p1_lo))
    span2 = float(max(1.0, p2_hi - p2_lo))

    # Keep the estimate stable by avoiding larger spans than the enclosing rectangle.
    long_span = min(max(span1, span2), max(rw, rh))
    short_span = min(min(span1, span2), min(rw, rh))

    observed_aspect = long_span / max(short_span, 1.0)
    target_aspect = nearest_aspect_prior(observed_aspect, cfg)
    if observed_aspect < target_aspect:
        desired_short = long_span / target_aspect
        min_allowed = short_span * 0.88
        short_span = max(desired_short, min_allowed)

    rw_robust = long_span
    rh_robust = short_span

    H, W = mask.shape[:2]
    touches_border = (x <= 1 or y <= 1 or (x + w) >= W - 1 or (y + h) >= H - 1)

    dim1_mm = rw_robust * mm_per_pixel
    dim2_mm = rh_robust * mm_per_pixel

    width_mm = min(dim1_mm, dim2_mm)
    height_mm = max(dim1_mm, dim2_mm)

    filled = np.zeros_like(component)
    cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
    area_filled = float(np.count_nonzero(filled))
    area_hull = float(cv2.contourArea(hull))
    area_box = float(max(rw_robust * rh_robust, 1.0))
    if (area_filled / area_box) < 0.15:
        area_geom = max(area_hull, 1.0)
    else:
        area_geom = max(area_filled, min(area_hull, area_filled * 1.10), 1.0)

    bbox_area = max(rw_robust * rh_robust, 1.0)
    fill_ratio = area_geom / bbox_area

    aspect = max(rw_robust, rh_robust) / max(1.0, min(rw_robust, rh_robust))
    aspect_outlier_score = min(1.0, abs(aspect - target_aspect) / 1.0)

    result.update({
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "area_mm2": float(area_geom * (mm_per_pixel ** 2)),
        "touches_border": bool(touches_border),
        "object_area_pixels": int(round(area_geom)),
        "fill_ratio": float(fill_ratio),
        "aspect_outlier_score": float(aspect_outlier_score),
        "bbox": (int(x), int(y), int(w), int(h)),
    })

    return result
