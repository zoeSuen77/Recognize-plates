from __future__ import annotations
"""End-to-end full-vehicle license plate recognition pipeline.

This script connects the two strongest modules in the project:
YOLO finds the plate in a full vehicle photo, then CRNN + CTC recognizes the
cropped plate as a complete string. It is the best file to read when reviewing
how the final system works from input image to visualized output.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from detect_plate_yolo import DEFAULT_DETECTOR_PATH, crop_plate, detect_best_plate, load_yolo_model
from utils import DEFAULT_MODEL_PATH, RESULTS_DIR, get_device, resolve_project_path


def load_recognizer(model_path: Path, device, color_mode: str | None = None):
    """Load the trained CRNN recognizer and restore its color-mode setting."""

    import torch

    from model_crnn import CRNNLicensePlate
    from plate_chars import NUM_CLASSES

    checkpoint = torch.load(model_path, map_location=device)
    resolved_color_mode = color_mode or checkpoint.get("color_mode", "grayscale")
    input_channels = 1 if resolved_color_mode == "grayscale" else 3
    model = CRNNLicensePlate(num_classes=NUM_CLASSES, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, resolved_color_mode


def default_output_path(image_path: Path) -> Path:
    """Default path for the final full-image visualization."""

    return RESULTS_DIR / "detections" / f"{image_path.stem}_result.jpg"


def default_crop_path(image_path: Path) -> Path:
    """Default path for the intermediate plate crop."""

    return RESULTS_DIR / "detections" / f"{image_path.stem}_plate_crop.jpg"


def save_result_visualization(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    prediction: str,
    confidence: float,
    output_path: Path,
) -> None:
    """Draw the detected bbox and recognized text on the original image."""

    from predict import load_font

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=4)
    font = load_font(24)
    label = f"{prediction}  {confidence:.3f}" if prediction else f"license_plate  {confidence:.3f}"
    text_y = max(0, bbox[1] - 32)
    draw.text((bbox[0], text_y), label, fill="red", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def recognize_cropped_plate(model, crop_path: Path, color_mode: str, device) -> tuple[str, list[float]]:
    """Run CRNN prediction on a cropped plate image."""

    from predict import build_transform, predict_tensor

    image = Image.open(crop_path).convert("RGB")
    image_tensor = build_transform(color_mode)(image).unsqueeze(0)
    return predict_tensor(model, image_tensor, device)


def main() -> None:
    """Command-line entry point for YOLO detection plus CRNN recognition."""

    parser = argparse.ArgumentParser(description="Detect a plate with YOLO, crop it, then recognize it with the existing CRNN model.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--recognizer", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--color-mode", choices=["grayscale", "rgb"], default=None)
    args = parser.parse_args()

    image_path = resolve_project_path(args.image)
    detector_path = resolve_project_path(args.detector)
    recognizer_path = resolve_project_path(args.recognizer)
    output_path = resolve_project_path(args.output) if args.output else default_output_path(image_path)
    crop_output = default_crop_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector model not found: {detector_path}")
    if not recognizer_path.exists():
        raise FileNotFoundError(f"Recognizer model not found: {recognizer_path}")

    device = get_device(args.device)
    detector = load_yolo_model(detector_path)
    detection = detect_best_plate(detector, image_path, args.conf)
    clipped_bbox = crop_plate(image_path, detection, crop_output)

    # The recognizer is loaded after detection so this script can fail early on
    # missing detector/input issues before initializing the heavier CRNN stack.
    recognizer, color_mode = load_recognizer(recognizer_path, device, args.color_mode)
    prediction, confidences = recognize_cropped_plate(recognizer, crop_output, color_mode, device)
    save_result_visualization(image_path, clipped_bbox, prediction, detection.confidence, output_path)

    print(f"bbox: {clipped_bbox}")
    print(f"detection confidence: {detection.confidence:.4f}")
    print(f"cropped plate saved to: {crop_output}")
    print(f"predicted plate: {prediction}")
    print("character confidences:", ", ".join(f"{char}:{conf:.3f}" for char, conf in zip(prediction, confidences)))
    print(f"result visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
