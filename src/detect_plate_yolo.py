from __future__ import annotations
"""Run the YOLO plate detector on one full vehicle image.

This module is the detector half of the final pipeline. It loads the trained
YOLO weight, selects the highest-confidence plate box, saves a cropped plate,
and writes a visualization image with the detected bounding box.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from utils import MODELS_DIR, RESULTS_DIR, SELF_PHOTOS_DIR, resolve_project_path


DEFAULT_DETECTOR_PATH = MODELS_DIR / "plate_detector.pt"


@dataclass
class Detection:
    """YOLO detection result used by downstream cropping and visualization."""

    bbox: tuple[float, float, float, float]
    confidence: float


def load_yolo_model(detector_path: Path):
    """Load an Ultralytics YOLO model from a local ``.pt`` file."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Missing dependency: ultralytics. Install it with: pip install ultralytics") from exc
    return YOLO(str(detector_path))


def detect_best_plate(model, image_path: Path, conf: float = 0.25) -> Detection:
    """Return the highest-confidence plate detection for one image."""

    results = model(str(image_path), conf=conf, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no results.")

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError(f"No license plate detected in: {image_path}")

    confidences = boxes.conf.detach().cpu()
    best_index = int(confidences.argmax().item())
    bbox_values = boxes.xyxy[best_index].detach().cpu().tolist()
    return Detection(
        bbox=(float(bbox_values[0]), float(bbox_values[1]), float(bbox_values[2]), float(bbox_values[3])),
        confidence=float(confidences[best_index].item()),
    )


def clip_bbox(bbox: tuple[float, float, float, float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Clip a float YOLO bbox to valid integer image coordinates."""

    x1, y1, x2, y2 = bbox
    left = max(0, min(image_width, int(round(x1))))
    top = max(0, min(image_height, int(round(y1))))
    right = max(0, min(image_width, int(round(x2))))
    bottom = max(0, min(image_height, int(round(y2))))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid detected bbox after clipping: {(left, top, right, bottom)}")
    return left, top, right, bottom


def crop_plate(image_path: Path, detection: Detection, output_path: Path) -> tuple[int, int, int, int]:
    """Save the detected plate crop and return the clipped bbox."""

    image = Image.open(image_path).convert("RGB")
    bbox = clip_bbox(detection.bbox, image.width, image.height)
    plate = image.crop(bbox)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plate.save(output_path)
    return bbox


def save_detection_visualization(image_path: Path, bbox: tuple[int, int, int, int], confidence: float, output_path: Path) -> None:
    """Save the original image with the detected plate rectangle drawn on top."""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=4)
    draw.text((bbox[0], max(0, bbox[1] - 18)), f"license_plate {confidence:.3f}", fill="red")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def default_crop_output(image_path: Path) -> Path:
    """Default output path for the cropped plate image."""

    return SELF_PHOTOS_DIR / "cropped" / f"{image_path.stem}_plate{image_path.suffix}"


def default_visual_output(image_path: Path) -> Path:
    """Default output path for the detection visualization."""

    return RESULTS_DIR / "detections" / f"{image_path.stem}_detected.jpg"


def main() -> None:
    """Command-line entry point for one-image plate detection and cropping."""

    parser = argparse.ArgumentParser(description="Detect one license plate with YOLO and save the cropped plate image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    image_path = resolve_project_path(args.image)
    detector_path = resolve_project_path(args.detector)
    output_path = resolve_project_path(args.output) if args.output else default_crop_output(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector model not found: {detector_path}")

    model = load_yolo_model(detector_path)
    detection = detect_best_plate(model, image_path, args.conf)
    clipped_bbox = crop_plate(image_path, detection, output_path)
    visual_output = default_visual_output(image_path)
    save_detection_visualization(image_path, clipped_bbox, detection.confidence, visual_output)

    print(f"bbox: {clipped_bbox}")
    print(f"confidence: {detection.confidence:.4f}")
    print(f"cropped plate saved to: {output_path}")
    print(f"detection visualization saved to: {visual_output}")


if __name__ == "__main__":
    main()
