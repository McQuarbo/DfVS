import cv2
import numpy as np


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def draw_polygon(image, corners, color=(0, 255, 0), thickness=2):
    img = image.copy()
    corners = corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, color, thickness)
        cv2.circle(img, p1, 5, (0, 0, 255), -1)
    return img


def save_image(path, image):
    cv2.imwrite(str(path), image)


def rgb_text_box(img, lines, origin=(20, 40)):
    out = img.copy()
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_h = 28

    widths = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        widths.append(w)

    box_w = max(widths) + 20
    box_h = line_h * len(lines) + 20

    cv2.rectangle(out, (x - 10, y - 30), (x - 10 + box_w, y - 30 + box_h), (0, 0, 0), -1)

    yy = y
    for line in lines:
        cv2.putText(out, line, (x, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yy += line_h

    return out
