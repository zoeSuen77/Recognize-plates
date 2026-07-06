from __future__ import annotations
"""YOLO-only plate recognition route: plate detection plus character detection.

This file is useful for visualizing and analyzing individual character boxes,
but its final plate accuracy is lower than the CRNN route because character
boxes are trained from pseudo labels and can suffer from extra/missing boxes.
"""

import argparse
from itertools import combinations
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from detect_plate_yolo import (
    DEFAULT_DETECTOR_PATH,
    Detection,
    clip_bbox,
    load_yolo_model,
)
from plate_chars import INDEX_TO_CHAR
from plate_rules import (
    allowed_chars_for_position,
    expected_plate_lengths,
    infer_plate_type,
    score_plate_candidate,
)
from utils import MODELS_DIR, RESULTS_DIR, resolve_project_path


DEFAULT_CHAR_DETECTOR_PATH = MODELS_DIR / "char_detector.pt"
YOLO_RECOGNITION_DIR = RESULTS_DIR / "yolo_recognition"
DEFAULT_MAX_CHARS = 8
STANDARD_PLATE_LENGTHS = (7, 8)


def load_char_model(char_path: Path):
    """Load the trained YOLO character detector."""

    return load_yolo_model(char_path)


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU for duplicate-character suppression."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)
    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _dedupe_priority(det: dict, plate_width: int, use_plate_rules: bool) -> float:
    """Score a detection for deduplication priority (higher = keep over others)."""
    priority = float(det["confidence"])
    if use_plate_rules and plate_width > 0 and det["x_center"] <= plate_width * 0.25:
        # Province char (left-most) gets a boost
        from plate_chars import PROVINCES
        if det["char"] in PROVINCES:
            priority += 0.4
        elif det["char"] in "ABCDEFGHJKLMNPQRSTUVWXYZ":
            priority += 0.2
    return priority


def _remove_duplicate_centers(detections: list[dict], plate_width: int, use_plate_rules: bool) -> list[dict]:
    """Remove detections that likely refer to the same character position."""

    if not detections:
        return []

    min_center_gap = max(2.0, plate_width / 40.0)
    selected: list[dict] = []
    for det in sorted(detections, key=lambda item: _dedupe_priority(item, plate_width, use_plate_rules), reverse=True):
        overlaps_existing = False
        for kept in selected:
            center_gap = abs(det["x_center"] - kept["x_center"])
            if center_gap < min_center_gap or _bbox_iou(det["bbox_xyxy"], kept["bbox_xyxy"]) > 0.45:
                overlaps_existing = True
                break
        if not overlaps_existing:
            selected.append(det)
    return sorted(selected, key=lambda item: item["x_center"])


def _candidate_score(candidate: tuple[dict, ...], plate_width: int, use_plate_rules: bool) -> float:
    """Score a candidate set of detections for plausibility as a complete plate."""
    ordered = sorted(candidate, key=lambda item: item["x_center"])
    confidences = [item["confidence"] for item in ordered]
    score = sum(confidences) / max(len(confidences), 1)

    # Prefer standard lengths
    if len(ordered) == 7:
        score += 0.05
    elif len(ordered) == 8:
        score += 0.03

    # Even spacing bonus
    if len(ordered) >= 3:
        gaps = [right["x_center"] - left["x_center"] for left, right in zip(ordered, ordered[1:])]
        positive_gaps = [gap for gap in gaps if gap > 0]
        if positive_gaps:
            mean_gap = sum(positive_gaps) / len(positive_gaps)
            gap_deviation = sum(abs(gap - mean_gap) for gap in positive_gaps) / (len(positive_gaps) * max(mean_gap, 1.0))
            score -= 0.12 * gap_deviation

    # Plate rules
    if use_plate_rules:
        candidate_text = "".join(d["char"] for d in ordered)
        text_score = score_plate_candidate(candidate_text, ordered)
        score += 0.5 * text_score

    return score


def postprocess_char_detections(
    detections: list[dict],
    plate_width: int,
    max_chars: int = DEFAULT_MAX_CHARS,
    use_plate_rules: bool = True,
) -> list[dict]:
    """Filter, order, and trim YOLO character detections for Chinese plates.

    Uses plate_rules module for smarter candidate selection.
    """
    if max_chars <= 0:
        raise ValueError("--max-chars must be positive")
    if not detections:
        return []

    deduped = _remove_duplicate_centers(detections, plate_width, use_plate_rules)
    if not deduped:
        return []

    if len(deduped) <= max_chars:
        ordered = sorted(deduped, key=lambda item: item["x_center"])
        if use_plate_rules and len(ordered) > 1:
            candidate_text = "".join(d["char"] for d in ordered)
            plate_type = infer_plate_type(candidate_text)
            expected = expected_plate_lengths(plate_type)
            allowed_len = min(max_chars, max(expected))

            # If we have more than expected, trim or select
            if len(ordered) > allowed_len:
                # Use combination scoring
                target_lengths = [l for l in expected if l <= max_chars and l <= len(ordered)]
                if not target_lengths:
                    target_lengths = [min(max_chars, len(ordered))]
                best = _select_best_combination(deduped, plate_width, target_lengths, max_chars, use_plate_rules)
                return best if best else ordered[:allowed_len]

            # Try to fix first char if it's not a province
            if len(ordered) >= 2 and ordered[0]["char"] not in "京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新":
                from plate_chars import PROVINCES
                # Check if there's a province char later that should be first
                for i, d in enumerate(ordered):
                    if d["char"] in PROVINCES and d["x_center"] < plate_width * 0.4:
                        # Move it to front
                        province_det = ordered.pop(i)
                        ordered.insert(0, province_det)
                        break
            return ordered[:max_chars]

    # Too many detections — select best combination
    target_lengths = [l for l in STANDARD_PLATE_LENGTHS if l <= max_chars and l <= len(deduped)]
    if not target_lengths:
        target_lengths = [max_chars]

    best = _select_best_combination(deduped, plate_width, target_lengths, max_chars, use_plate_rules)
    if best:
        return best

    # Fallback
    return sorted(deduped, key=lambda item: item["x_center"])[:max_chars]


def _select_best_combination(
    detections: list[dict],
    plate_width: int,
    target_lengths: list[int],
    max_chars: int,
    use_plate_rules: bool,
) -> list[dict] | None:
    """Select the best combination of detections for a complete plate."""
    pool_size = min(len(detections), max(max(target_lengths) + 4, max_chars + 2))
    candidate_pool = sorted(detections, key=lambda item: item["confidence"], reverse=True)[:pool_size]

    best_candidate: tuple[dict, ...] | None = None
    best_score = float("-inf")

    for target_length in sorted(target_lengths, reverse=True):
        for candidate in combinations(candidate_pool, target_length):
            score = _candidate_score(candidate, plate_width, use_plate_rules)
            if score > best_score:
                best_score = score
                best_candidate = candidate

    if best_candidate is None:
        return None
    return sorted(best_candidate, key=lambda item: item["x_center"])


def detect_raw_chars(char_model, plate_image: Image.Image, conf: float = 0.25) -> list[dict]:
    """Detect raw characters on a cropped plate image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        plate_image.save(tmp_path)

    try:
        results = char_model(tmp_path, conf=conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        cls_ids = boxes.cls.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        xyxy = boxes.xyxy.detach().cpu().tolist()
        xywh = boxes.xywh.detach().cpu().tolist()

        detections = []
        for cls_id, confidence, xyw, xyyx in zip(cls_ids, confs, xywh, xyxy):
            class_idx = int(cls_id)
            char = INDEX_TO_CHAR.get(class_idx, "?")
            detections.append({
                "cls_id": class_idx,
                "char": char,
                "confidence": float(confidence),
                "x_center": float(xyw[0]),
                "bbox_xyxy": list(xyyx),
            })

        detections.sort(key=lambda d: d["x_center"])
        return detections
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def detect_chars(
    char_model,
    plate_image: Image.Image,
    conf: float = 0.25,
    max_chars: int = DEFAULT_MAX_CHARS,
    use_plate_rules: bool = True,
) -> list[dict]:
    """Detect and postprocess character boxes on a cropped plate image."""

    raw_detections = detect_raw_chars(char_model, plate_image, conf)
    return postprocess_char_detections(raw_detections, plate_image.width, max_chars, use_plate_rules)


def recognize_plate_yolo(
    image_path: Path,
    plate_detector,
    char_detector,
    conf: float = 0.25,
    max_chars: int = DEFAULT_MAX_CHARS,
    use_plate_rules: bool = True,
    save_visualization: Path | None = None,
) -> tuple[str, list[dict], Detection]:
    """Full pipeline: plate detect → crop → char detect → sort → plate string."""
    from detect_plate_yolo import detect_best_plate

    detection = detect_best_plate(plate_detector, image_path, conf)
    image = Image.open(image_path).convert("RGB")
    bbox = clip_bbox(detection.bbox, image.width, image.height)
    plate_img = image.crop(bbox)
    char_detections = detect_chars(char_detector, plate_img, conf, max_chars, use_plate_rules)
    plate_text = "".join(d["char"] for d in char_detections)

    # Apply plate type rules for final validation
    if plate_text and use_plate_rules:
        plate_type = infer_plate_type(plate_text)
        expected = expected_plate_lengths(plate_type)
        if len(plate_text) not in expected:
            # Try trimming or padding isn't useful, just flag
            pass

    if save_visualization:
        save_visualization.parent.mkdir(parents=True, exist_ok=True)
        draw = ImageDraw.Draw(image)

        draw.rectangle(bbox, outline="red", width=4)
        label_text = f"{plate_text}  {detection.confidence:.3f}" if plate_text else f"no chars  {detection.confidence:.3f}"
        draw.text((bbox[0], max(0, bbox[1] - 18)), label_text, fill="red")

        pw = bbox[2] - bbox[0]
        ph = bbox[3] - bbox[1]
        for cd in char_detections:
            xc_norm = cd["x_center"] / plate_img.width
            char_left = bbox[0] + int(xc_norm * pw - pw / len(char_detections) / 2)
            char_right = bbox[0] + int(xc_norm * pw + pw / len(char_detections) / 2)
            char_top = bbox[1]
            char_bottom = bbox[3]
            draw.rectangle((char_left, char_top, char_right, char_bottom), outline="lime", width=2)

        image.save(save_visualization)

    return plate_text, char_detections, detection


def default_visual_output(image_path: Path) -> Path:
    """Default output path for YOLO-only recognition visualizations."""

    return YOLO_RECOGNITION_DIR / f"{image_path.stem}_yolo_result.jpg"


def main() -> None:
    """Command-line entry point for the YOLO-only recognition route."""

    parser = argparse.ArgumentParser(
        description="Recognize a license plate using YOLO plate detection + YOLO character detection v2. "
        "Uses plate_rules for smarter character selection."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--plate-detector", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--char-detector", type=Path, default=DEFAULT_CHAR_DETECTOR_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--use-plate-rules", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    image_path = resolve_project_path(args.image)
    plate_detector_path = resolve_project_path(args.plate_detector)
    char_detector_path = resolve_project_path(args.char_detector)
    output_path = resolve_project_path(args.output) if args.output else default_visual_output(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not plate_detector_path.exists():
        raise FileNotFoundError(f"Plate detector not found: {plate_detector_path}")
    if not char_detector_path.exists():
        raise FileNotFoundError(f"Char detector not found: {char_detector_path}.")

    print(f"Loading plate detector: {plate_detector_path}")
    plate_model = load_yolo_model(plate_detector_path)
    print(f"Loading char detector: {char_detector_path}")
    char_model = load_char_model(char_detector_path)

    plate_text, char_detections, detection = recognize_plate_yolo(
        image_path, plate_model, char_model, args.conf, args.max_chars, args.use_plate_rules, output_path,
    )

    print(f"\n{'='*50}")
    print(f"detected plate:   {plate_text}")
    print(f"plate bbox:       ({int(detection.bbox[0])}, {int(detection.bbox[1])}, "
          f"{int(detection.bbox[2])}, {int(detection.bbox[3])})")
    print(f"detection conf:   {detection.confidence:.4f}")
    print(f"char detections:  {len(char_detections)}")
    for cd in char_detections:
        print(f"  [{cd['char']}] cls_id={cd['cls_id']}  conf={cd['confidence']:.3f}  x_center={cd['x_center']:.1f}")
    print(f"visualization:    {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
