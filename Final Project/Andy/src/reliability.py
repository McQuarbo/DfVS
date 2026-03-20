def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def compute_reliability(card, scale, segmentation, measurement, cfg):
    if card["found"]:
        m = card["metrics"]
        true_aspect = cfg["card"]["true_aspect"]

        s_aspect = clamp01(1 - abs(m["detected_aspect_ratio"] - true_aspect) / 0.25)
        s_area = clamp01((m["area_fraction"] - 0.005) / (0.03 - 0.005))
        s_rect = clamp01((m["rectangularity"] - 0.60) / (0.95 - 0.60))
        s_edge = clamp01(m["edge_support"])

        C = 0.35 * s_aspect + 0.25 * s_area + 0.20 * s_rect + 0.20 * s_edge
    else:
        C = 0.0

    scale_err = abs(scale["scale_x"] - scale["scale_y"]) / max((scale["scale_x"] + scale["scale_y"]) / 2.0, 1e-9)
    s_scale = clamp01(1 - scale_err / 0.12)
    s_size = clamp01((scale["card_width_px"] - 80) / (500 - 80))
    R = 0.75 * s_scale + 0.25 * s_size

    sm = segmentation["metrics"]
    s_dom = clamp01(sm["largest_component_fraction"])
    s_holes = clamp01(1 - sm["hole_fraction"] / 0.15)
    s_noise = clamp01(1 - sm["small_blob_fraction"] / 0.5)
    s_smooth = clamp01(1 - sm["jaggedness"] / 25)
    G = 0.40 * s_dom + 0.20 * s_holes + 0.20 * s_noise + 0.20 * s_smooth

    s_border = 0.0 if measurement["touches_border"] else 1.0
    s_size = clamp01((measurement["object_area_pixels"] - 5000) / (50000 - 5000))
    s_fill = clamp01((measurement["fill_ratio"] - 0.4) / (0.95 - 0.4))
    s_shape = clamp01(1 - measurement["aspect_outlier_score"])
    M = 0.35 * s_border + 0.25 * s_size + 0.20 * s_fill + 0.20 * s_shape

    score = 100 * (0.35 * C + 0.25 * R + 0.25 * G + 0.15 * M)

    if (not card["found"]) or scale_err > 0.20 or measurement["touches_border"]:
        level = "Low"
    elif score >= cfg["reliability"]["high_threshold"]:
        level = "High"
    elif score >= cfg["reliability"]["medium_threshold"]:
        level = "Medium"
    else:
        level = "Low"

    reasons = []
    if scale_err > 0.12:
        reasons.append("scale mismatch elevated")
    if segmentation["metrics"]["small_blob_fraction"] > 0.3:
        reasons.append("segmentation noisy")
    if segmentation["metrics"]["hole_fraction"] > 0.1:
        reasons.append("mask contains holes")
    if measurement["touches_border"]:
        reasons.append("object touches border")
    if not card["found"]:
        reasons.append("card detection failed")

    if not reasons:
        reason = "pipeline internally consistent"
    else:
        reason = "; ".join(reasons)

    return {
        "score": float(score),
        "level": level,
        "reason": reason,
        "details": {
            "card_score": C,
            "rectification_score": R,
            "segmentation_score": G,
            "measurement_score": M,
            "scale_error": scale_err,
        }
    }
