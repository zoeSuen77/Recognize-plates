from __future__ import annotations

import random
import cv2
import numpy as np
from PIL import Image


def parse_ccpd_points(points_text: str) -> list[tuple[float, float]]:
    """Parse CCPD points field into a list of (x, y) tuples.

    CCPD format: "x1&y1_x2&y2_x3&y3_x4&y4" (4 points).
    Returns empty list if parsing fails.
    """
    if not points_text:
        return []
    try:
        pts = []
        for pair in points_text.split("_"):
            if not pair or "&" not in pair:
                return []
            x_text, y_text = pair.split("&")
            pts.append((float(x_text), float(y_text)))
        return pts
    except (ValueError, IndexError):
        return []


def order_plate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Order 4 points to: top-left, top-right, bottom-right, bottom-left.

    Uses the standard sum/difference heuristic, which is more stable for
    quadrilateral perspective warps than assuming the input point order.
    """
    if len(points) < 4:
        return points

    pts = np.array(points[:4], dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[int(np.argmin(sums))]
    bottom_right = pts[int(np.argmax(sums))]
    top_right = pts[int(np.argmin(diffs))]
    bottom_left = pts[int(np.argmax(diffs))]
    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    if len({tuple(point) for point in ordered.tolist()}) != 4:
        # Degenerate point layouts are rare but possible after bad parsing.
        pts_by_y = sorted(points[:4], key=lambda point: (point[1], point[0]))
        top = sorted(pts_by_y[:2], key=lambda point: point[0])
        bottom = sorted(pts_by_y[2:], key=lambda point: point[0])
        return [top[0], top[1], bottom[1], bottom[0]]

    return [tuple(map(float, point)) for point in ordered]


def warp_plate_perspective(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image | None:
    """Apply perspective correction to straighten a tilted license plate.

    Args:
        image: Full vehicle image (PIL RGB).
        points: 4 CCPD corner points.

    Returns:
        Warped front-facing plate image, or None if correction fails.
    """
    if len(points) < 4:
        return None
    try:
        ordered = order_plate_points(points)
        src = np.array(ordered, dtype=np.float32)

        top_w = float(np.linalg.norm(src[1] - src[0]))
        bot_w = float(np.linalg.norm(src[2] - src[3]))
        out_w = max(int(round(top_w)), int(round(bot_w)))

        left_h = float(np.linalg.norm(src[3] - src[0]))
        right_h = float(np.linalg.norm(src[2] - src[1]))
        out_h = max(int(round(left_h)), int(round(right_h)))

        if out_w < 10 or out_h < 10:
            return None

        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        img_np = np.array(image)
        warped = cv2.warpPerspective(img_np, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
        return Image.fromarray(warped)
    except Exception:
        return None


def augment_plate_image(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Apply lightweight data augmentation to a cropped/warped plate image.

    Augmentations: brightness, contrast, slight blur, slight rotation, JPEG noise.
    """
    if rng is None:
        rng = random.Random()

    img = np.array(image).astype(np.float32)

    # Brightness
    img *= 1.0 + rng.uniform(-0.15, 0.15)
    # Contrast
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = mean + (1.0 + rng.uniform(-0.15, 0.15)) * (img - mean)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Slight Gaussian blur
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Slight rotation
    if rng.random() < 0.3:
        angle = rng.uniform(-2.0, 2.0)
        h, w = img.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # JPEG compression noise
    if rng.random() < 0.2:
        quality = rng.randint(65, 95)
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return Image.fromarray(img)
