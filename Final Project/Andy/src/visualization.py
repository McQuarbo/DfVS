import cv2

from src.utils import draw_polygon, rgb_text_box


def make_card_detection_overlay(image_bgr, card):
    if not card["found"]:
        return image_bgr.copy()
    return draw_polygon(image_bgr, card["corners"], color=(0, 255, 0), thickness=2)


def make_final_overlay(rectified_bgr, mask, measurement, reliability):
    overlay = rectified_bgr.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    if measurement["bbox"] is not None:
        x, y, w, h = measurement["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    lines = [
        f"Width: {measurement['width_mm']:.2f} mm",
        f"Height: {measurement['height_mm']:.2f} mm",
        f"Area: {measurement['area_mm2']:.2f} mm^2",
        f"Confidence: {reliability['level']} ({reliability['score']:.1f})",
        f"Reason: {reliability['reason']}",
    ]
    overlay = rgb_text_box(overlay, lines, origin=(20, 45))
    return overlay
