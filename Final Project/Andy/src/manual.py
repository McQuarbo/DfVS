import cv2
import numpy as np

from src.utils import order_corners

_clicked_points = []


def _mouse_callback(event, x, y, flags, param):
    global _clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = float(param.get("scale", 1.0))
        ox = int(round(x / scale))
        oy = int(round(y / scale))
        if len(_clicked_points) < 4:
            _clicked_points.append((ox, oy))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _clicked_points:
            _clicked_points.pop()


def _prepare_display(image_bgr, max_w=1600, max_h=1000):
    h, w = image_bgr.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        disp = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        disp = image_bgr.copy()
    return disp, scale


def _refine_points_to_local_corners(image_bgr, pts, radius=36):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    refined = []

    for (x, y) in pts:
        x = int(round(float(x)))
        y = int(round(float(y)))
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(w, x + radius + 1)
        y1 = min(h, y + radius + 1)

        roi = gray[y0:y1, x0:x1]
        best = (x, y)

        if roi.size >= 25:
            corners = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=24,
                qualityLevel=0.02,
                minDistance=4,
                blockSize=5,
                useHarrisDetector=True,
                k=0.04,
            )

            if corners is not None and len(corners) > 0:
                cand = corners.reshape(-1, 2)
                cand[:, 0] += x0
                cand[:, 1] += y0
                d2 = (cand[:, 0] - x) ** 2 + (cand[:, 1] - y) ** 2
                i = int(np.argmin(d2))
                best = (int(round(cand[i, 0])), int(round(cand[i, 1])))

        refined.append(best)

    return np.array(refined, dtype=np.float32)


def manual_select_card_corners(image_bgr):
    global _clicked_points
    _clicked_points = []

    disp, scale = _prepare_display(image_bgr)
    window_name = "Manual fallback: click 4 Wise card corners"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, disp)
        cv2.setMouseCallback(window_name, _mouse_callback, {"scale": scale})
    except cv2.error:
        # Fallback for environments where OpenCV highgui is not available.
        from src.manual_fallback import manual_select_card_corners as mpl_manual_select

        return mpl_manual_select(image_bgr)

    while True:
        temp = disp.copy()
        for p in _clicked_points:
            dp = (int(round(p[0] * scale)), int(round(p[1] * scale)))
            cv2.circle(temp, dp, 5, (0, 0, 255), -1)

        cv2.putText(
            temp,
            "Click 4 corners. Auto-confirms on 4th click.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            temp,
            f"Selected: {len(_clicked_points)}/4 | ENTER/SPACE confirm | U/R/Right-click edit | ESC cancel",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, temp)
        key = cv2.waitKeyEx(20)

        if len(_clicked_points) == 4:
            break

        if key in (13, 10, 32):
            if len(_clicked_points) == 4:
                break
        elif key in (8, ord("u"), ord("U")):
            if _clicked_points:
                _clicked_points.pop()
        elif key in (ord("r"), ord("R")):
            _clicked_points = []
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    pts = np.array(_clicked_points, dtype=np.float32)
    pts = _refine_points_to_local_corners(image_bgr, pts)
    return order_corners(pts)
