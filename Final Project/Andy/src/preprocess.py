import cv2


def preprocess_image(image_bgr, cfg):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), cfg["preprocess"]["gaussian_sigma"])

    if cfg["preprocess"]["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    edge = cv2.Canny(gray, 50, 150)

    return {
        "gray": gray,
        "edge": edge,
    }
