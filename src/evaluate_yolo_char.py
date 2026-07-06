from __future__ import annotations
"""Evaluate the YOLO character-detection recognition path.

This is the comparison route against the recommended YOLO plate detector +
CRNN recognizer. It crops a plate either from labels.csv geometry or from a
trained plate detector, runs the character detector, applies post-processing,
and writes row-level predictions for error analysis.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

from detect_plate_yolo import clip_bbox, detect_best_plate, load_yolo_model
from recognize_plate_yolo import DEFAULT_MAX_CHARS, detect_raw_chars, postprocess_char_detections
from utils import CCPD_LABELS_PATH, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, missing_labels_message, resolve_project_path


DEFAULT_CHAR_DETECTOR_PATH = MODELS_DIR / "char_detector.pt"


def parse_pair(text: str) -> tuple[int, int]:
    """Parse a CCPD coordinate pair like ``317&389`` into integer x/y."""

    x_text, y_text = text.split("&")
    return int(x_text), int(y_text)


def crop_from_ccpd_fields(image: Image.Image, bbox: str, points: str, padding: int = 4) -> Image.Image:
    """Crop a plate region from stored CCPD bbox/points fields."""

    width, height = image.size
    boxes: list[tuple[int, int]] = []
    if points:
        boxes.extend(parse_pair(item) for item in points.split("_") if item)
    if bbox:
        left_top, right_bottom = bbox.split("_")
        boxes.extend([parse_pair(left_top), parse_pair(right_bottom)])
    if not boxes:
        return image
    xs = [x for x, _ in boxes]
    ys = [y for _, y in boxes]
    left = max(0, min(xs) - padding)
    top = max(0, min(ys) - padding)
    right = min(width, max(xs) + padding)
    bottom = min(height, max(ys) + padding)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def resolve_image_path(filename: str, labels_path: Path) -> Path:
    """Resolve source image paths stored in ``labels.csv``."""

    path = Path(filename)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return labels_path.parent / path


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two plate strings."""

    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[-1]


def character_accuracy(truth: str, prediction: str) -> float:
    """Approximate character-level accuracy from normalized edit distance."""

    max_len = max(len(truth), len(prediction), 1)
    distance = edit_distance(truth, prediction)
    return max(0.0, 1.0 - distance / max_len)


def summarize_predictions(labels: list[str], predictions: list[str]) -> dict[str, float]:
    """Aggregate metrics for the YOLO character-recognition route."""

    if not labels:
        return {
            "character_accuracy": 0.0,
            "full_plate_accuracy": 0.0,
            "avg_edit_distance": 0.0,
            "normalized_edit_distance": 0.0,
        }
    distances = [edit_distance(label, pred) for label, pred in zip(labels, predictions)]
    normalizers = [max(len(label), len(pred), 1) for label, pred in zip(labels, predictions)]
    return {
        "character_accuracy": sum(character_accuracy(label, pred) for label, pred in zip(labels, predictions)) / len(labels),
        "full_plate_accuracy": sum(label == pred for label, pred in zip(labels, predictions)) / len(labels),
        "avg_edit_distance": sum(distances) / len(distances),
        "normalized_edit_distance": sum(distance / norm for distance, norm in zip(distances, normalizers)) / len(distances),
    }


def classify_error(label: str, prediction: str) -> str:
    """Classify a wrong plate by length mismatch or same-length substitution."""

    if label == prediction:
        return "correct"
    if len(prediction) < len(label):
        return "missing_chars"
    if len(prediction) > len(label):
        return "extra_chars"
    return "same_length_classification"


def avg_confidence(detections: list[dict]) -> float:
    """Average confidence over retained character detections."""

    if not detections:
        return 0.0
    return sum(float(det["confidence"]) for det in detections) / len(detections)


def crop_plate_for_eval(
    image_path: Path,
    original: Image.Image,
    row: dict[str, str],
    plate_model,
    conf: float,
):
    """Choose the evaluation crop source: stored labels or a YOLO plate model."""

    if plate_model is None:
        return crop_from_ccpd_fields(original, row.get("bbox", ""), row.get("points", "")), "", ""

    detection = detect_best_plate(plate_model, image_path, conf)
    bbox = clip_bbox(detection.bbox, original.width, original.height)
    return original.crop(bbox), f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}", f"{detection.confidence:.6f}"


def main() -> None:
    """Command-line entry point for YOLO character route evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-based character detection on the CCPD test split. "
        "Can use labels.csv plate boxes or a YOLO plate detector, then runs char_detector to recognize characters."
    )
    parser.add_argument("--labels", type=Path, default=CCPD_LABELS_PATH)
    parser.add_argument("--plate-model", "--plate-detector", dest="plate_model", type=Path, default=None)
    parser.add_argument("--char-detector", "--char-model", dest="char_detector", type=Path, default=DEFAULT_CHAR_DETECTOR_PATH)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--use-plate-rules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples for quick evaluation.")
    args = parser.parse_args()

    labels_path = resolve_project_path(args.labels)
    char_detector_path = resolve_project_path(args.char_detector)
    plate_model_path = resolve_project_path(args.plate_model) if args.plate_model else None

    if not labels_path.exists():
        raise FileNotFoundError(missing_labels_message(labels_path))
    if plate_model_path is not None and not plate_model_path.exists():
        raise FileNotFoundError(f"Plate detector not found: {plate_model_path}")
    if not char_detector_path.exists():
        raise FileNotFoundError(
            f"Char detector not found: {char_detector_path}. "
            "Train it first with: python src/train_yolo_char.py"
        )

    plate_model = None
    if plate_model_path is not None:
        print(f"Loading plate detector: {plate_model_path}")
        plate_model = load_yolo_model(plate_model_path)

    print(f"Loading char detector: {char_detector_path}")
    char_model = load_yolo_model(char_detector_path)

    # Read split rows
    rows: list[dict[str, str]] = []
    with labels_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == args.split:
                if args.max_samples and len(rows) >= args.max_samples:
                    break
                rows.append(row)

    crop_source = "plate detector" if plate_model is not None else "labels.csv bbox"
    print(
        f"Evaluating on {len(rows)} {args.split} samples "
        f"(conf={args.conf}, max_chars={args.max_chars}, rules={args.use_plate_rules}, crop={crop_source})"
    )
    print()

    predictions: list[str] = []
    labels: list[str] = []
    results_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{len(rows)}]")

        try:
            source = resolve_image_path(row["filename"], labels_path)
            with Image.open(source) as img:
                original = img.convert("RGB")
            plate_img, plate_bbox, plate_conf = crop_plate_for_eval(source, original, row, plate_model, args.conf)
            ground_truth = row["plate_number"]
        except Exception as exc:
            print(f"  [skip] {row.get('filename', '<unknown>')}: {exc}")
            continue

        # Detect characters
        raw_dets = detect_raw_chars(char_model, plate_img, args.conf)
        char_dets = postprocess_char_detections(raw_dets, plate_img.width, args.max_chars, args.use_plate_rules)
        raw_text = "".join(d["char"] for d in raw_dets)
        pred_text = "".join(d["char"] for d in char_dets)
        error_type = classify_error(ground_truth, pred_text)

        labels.append(ground_truth)
        predictions.append(pred_text)
        results_rows.append({
            "image_path": row["filename"],
            "label": ground_truth,
            "prediction": pred_text,
            "correct": "1" if ground_truth == pred_text else "0",
            "num_detected_chars": str(len(raw_dets)),
            "avg_char_conf": f"{avg_confidence(char_dets):.6f}",
            "error_type": error_type,
            "raw_prediction": raw_text,
            "postprocessed_prediction": pred_text,
            "plate_bbox": plate_bbox,
            "plate_conf": plate_conf,
        })

    print()
    metrics = summarize_predictions(labels, predictions)

    print(f"{'='*55}")
    print(f"  YOLO+YOLO Character Detection Evaluation")
    print(f"{'='*55}")
    print(f"  character_accuracy:   {metrics['character_accuracy']:.4f}")
    print(f"  full_plate_accuracy:  {metrics['full_plate_accuracy']:.4f}")
    print(f"  avg_edit_distance:    {metrics['avg_edit_distance']:.4f}")
    print(f"  normalized_edit_dist: {metrics['normalized_edit_distance']:.4f}")
    print(f"  total_samples:        {len(labels)}")
    print(f"{'='*55}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "yolo_char_predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "label",
                "prediction",
                "correct",
                "num_detected_chars",
                "avg_char_conf",
                "error_type",
                "raw_prediction",
                "postprocessed_prediction",
                "plate_bbox",
                "plate_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
