from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from license_plate_dataset import ResizePad
from metrics import greedy_decode_with_confidence
from model_crnn import CRNNLicensePlate
from plate_chars import NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "license_plate_crnn_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_to_tensor(path: Path, color_mode: str) -> torch.Tensor:
    from torchvision import transforms

    channels = 1 if color_mode == "grayscale" else 3
    transform = transforms.Compose(
        [
            ResizePad((160, 48)),
            transforms.Grayscale(num_output_channels=channels) if color_mode == "grayscale" else transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,) * channels, std=(0.5,) * channels),
        ]
    )
    return transform(Image.open(path)).unsqueeze(0)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    print("Chinese font not found. Using PIL default font; Chinese text may not render correctly.")
    return ImageFont.load_default()


def save_result_image(image_path: Path, text: str, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 42), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(24)
    draw.text((8, image.height + 6), f"Pred: {text}", fill="red", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a cropped Chinese license plate image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--color-mode", choices=["grayscale", "rgb"], default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    device = get_device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    color_mode = args.color_mode or checkpoint.get("color_mode", "grayscale")
    input_channels = 1 if color_mode == "grayscale" else 3
    model = CRNNLicensePlate(num_classes=NUM_CLASSES, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tensor = image_to_tensor(args.image, color_mode).to(device)
    with torch.no_grad():
        decoded = greedy_decode_with_confidence(model(tensor))[0]
    prediction = str(decoded["text"])
    confidences = decoded["confidences"]
    print(f"Predicted plate: {prediction}")
    print("Character confidences:", ", ".join(f"{char}:{conf:.3f}" for char, conf in zip(prediction, confidences)))

    output_path = args.output or RESULTS_DIR / f"{args.image.stem}_prediction.jpg"
    save_result_image(args.image, prediction, output_path)
    print(f"Result image saved to: {output_path}")


if __name__ == "__main__":
    main()

