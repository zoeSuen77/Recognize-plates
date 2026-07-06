from __future__ import annotations
"""Single-image prediction utility for the CRNN recognizer.

Use this script for quick checks on a cropped plate image, or for a random
sample from the CCPD test split when no image path is supplied.
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from load_data import CCPDPlateDataset, ResizePad, crop_from_ccpd_fields
from metrics import greedy_decode_with_confidence
from model_crnn import CRNNLicensePlate
from plate_chars import NUM_CLASSES
from utils import CCPD_LABELS_PATH, DEFAULT_MODEL_PATH, RESULTS_DIR, get_device, missing_labels_message, resolve_project_path


def build_transform(color_mode: str):
    """Build the same resize/normalize transform used during CRNN training."""

    channels = 1 if color_mode == "grayscale" else 3
    return transforms.Compose(
        [
            ResizePad((160, 48)),
            transforms.Grayscale(num_output_channels=channels) if color_mode == "grayscale" else transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,) * channels, std=(0.5,) * channels),
        ]
    )


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a Chinese-capable font for result visualization."""

    for candidate in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    print("Chinese font not found. Using PIL default font; Chinese text may not render correctly.")
    return ImageFont.load_default()


def save_result_image(image_path: Path, prediction: str, output_path: Path) -> None:
    """Append predicted text below an image and save it."""

    image = Image.open(image_path).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 42), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, image.height + 6), f"Pred: {prediction}", fill="red", font=load_font(24))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def predict_tensor(model, image_tensor: torch.Tensor, device: torch.device) -> tuple[str, list[float]]:
    """Run one CRNN forward pass and return decoded text with confidences."""

    with torch.no_grad():
        decoded = greedy_decode_with_confidence(model(image_tensor.to(device)))[0]
    return str(decoded["text"]), list(decoded["confidences"])


def main() -> None:
    """Command-line entry point for one-image CRNN prediction."""

    parser = argparse.ArgumentParser(description="Predict one image or a random CCPD test image.")
    parser.add_argument("--image", type=Path, default=None, help="Path to a cropped plate image or your own pre-cropped photo.")
    parser.add_argument("--labels", type=Path, default=CCPD_LABELS_PATH, help="Used when --image is omitted.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--color-mode", choices=["grayscale", "rgb"], default=None)
    parser.add_argument("--no-crop", action="store_true", help="For CCPD test-set prediction, use full image instead of filename-derived crop.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    args.model = resolve_project_path(args.model)
    args.labels = resolve_project_path(args.labels)
    if args.image is not None:
        args.image = resolve_project_path(args.image)
    if args.output is not None:
        args.output = resolve_project_path(args.output)

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}. Run python src/train.py first.")
    device = get_device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    color_mode = args.color_mode or checkpoint.get("color_mode", "grayscale")
    crop_plate = checkpoint.get("crop_plate", True) and not args.no_crop
    input_channels = 1 if color_mode == "grayscale" else 3

    model = CRNNLicensePlate(num_classes=NUM_CLASSES, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = build_transform(color_mode)

    true_plate = None
    bbox = points = ""
    if args.image is None:
        if not args.labels.exists():
            raise FileNotFoundError(missing_labels_message(args.labels))
        dataset = CCPDPlateDataset(args.labels, split="test", color_mode=color_mode, crop_plate=crop_plate)
        random.seed(args.seed)
        sample_row = random.choice(dataset.samples)
        image_path = resolve_project_path(sample_row["filename"])
        if not image_path.exists():
            image_path = dataset.labels_path.parent / sample_row["filename"]
        true_plate = sample_row["plate_number"]
        bbox = sample_row.get("bbox", "")
        points = sample_row.get("points", "")
    else:
        image_path = resolve_project_path(args.image)

    image = Image.open(image_path).convert("RGB")
    if args.image is None and crop_plate:
        # Random CCPD samples are full images, so crop by stored geometry before
        # applying the recognizer transform.
        image = crop_from_ccpd_fields(image, bbox, points)
    image_tensor = transform(image).unsqueeze(0)
    prediction, confidences = predict_tensor(model, image_tensor, device)

    print(f"image: {image_path}")
    if true_plate is not None:
        print(f"true plate: {true_plate}")
    print(f"predicted plate: {prediction}")
    print("character confidences:", ", ".join(f"{char}:{conf:.3f}" for char, conf in zip(prediction, confidences)))
    if args.image is not None:
        print("Note: if this is a full vehicle photo, crop the plate region first or add a detector before recognition.")

    output_path = args.output or RESULTS_DIR / f"{image_path.stem}_prediction.jpg"
    save_result_image(image_path, prediction, output_path)
    print(f"Result image saved to: {output_path}")


if __name__ == "__main__":
    main()
