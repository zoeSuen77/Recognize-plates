from __future__ import annotations
"""Dataset and DataLoader utilities for CRNN + CTC training.

The recognizer is trained on cropped plate regions, but the master CSV stores
paths to the original CCPD images. This module resolves image paths, crops the
plate from filename-derived geometry, applies a fixed-size transform, and
packs variable-length CTC targets into batches.
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from plate_chars import encode_plate
from utils import CCPD_LABELS_PATH, PROJECT_ROOT, missing_labels_message, resolve_project_path


class ResizePad:
    """Resize an image into a fixed canvas while preserving aspect ratio."""

    def __init__(self, size: tuple[int, int], fill: int = 255) -> None:
        self.width, self.height = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = min(self.width / image.width, self.height / image.height)
        new_w = max(1, int(round(image.width * scale)))
        new_h = max(1, int(round(image.height * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new(image.mode, (self.width, self.height), self.fill)
        canvas.paste(image, ((self.width - new_w) // 2, (self.height - new_h) // 2))
        return canvas


def parse_pair(text: str) -> tuple[int, int]:
    """Parse a CCPD coordinate pair like ``317&389`` into integer x/y."""

    x_text, y_text = text.split("&")
    return int(x_text), int(y_text)


def crop_from_ccpd_fields(image: Image.Image, bbox: str, points: str, padding: int = 4) -> Image.Image:
    """Crop a plate region using CCPD points and bbox fields.

    Points are preferred because they describe all four plate corners. The
    bbox is also included so the crop remains valid when points are absent or
    slightly noisy. A small padding keeps border characters from being clipped.
    """

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
    """Resolve a CSV filename that may be absolute, project-relative, or CSV-relative."""

    path = Path(filename)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return labels_path.parent / path


class CCPDPlateDataset(Dataset):
    """PyTorch dataset returning plate tensors and CTC targets."""

    def __init__(
        self,
        labels_path: Path = CCPD_LABELS_PATH,
        split: str = "train",
        image_size: tuple[int, int] = (160, 48),
        color_mode: str = "grayscale",
        max_samples: int | None = None,
        crop_plate: bool = True,
    ) -> None:
        self.labels_path = resolve_project_path(labels_path)
        self.split = split
        self.image_size = image_size
        self.color_mode = color_mode
        self.crop_plate = crop_plate
        if not self.labels_path.exists():
            raise FileNotFoundError(missing_labels_message(self.labels_path))

        channels = 1 if color_mode == "grayscale" else 3
        self.transform = transforms.Compose(
            [
                ResizePad(image_size),
                transforms.Grayscale(num_output_channels=channels) if color_mode == "grayscale" else transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,) * channels, std=(0.5,) * channels),
            ]
        )
        self.samples = self._read_labels(max_samples)

    def _read_labels(self, max_samples: int | None) -> list[dict[str, str]]:
        """Read only the requested split and validate plate labels early."""

        samples: list[dict[str, str]] = []
        with self.labels_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"filename", "plate_number", "split"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{self.labels_path} must contain columns: {sorted(required)}")
            for row in reader:
                if row["split"] != self.split:
                    continue
                encode_plate(row["plate_number"])
                samples.append(row)
                if max_samples and len(samples) >= max_samples:
                    break
        if not samples:
            raise ValueError(f"No samples found for split={self.split!r} in {self.labels_path}")
        return samples

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample, crop the plate if enabled, and encode its label."""

        row = self.samples[index]
        image_path = resolve_image_path(row["filename"], self.labels_path)
        image = Image.open(image_path).convert("RGB")
        if self.crop_plate:
            image = crop_from_ccpd_fields(image, row.get("bbox", ""), row.get("points", ""))
        encoded = encode_plate(row["plate_number"])
        return {
            "image": self.transform(image),
            "target": torch.tensor(encoded, dtype=torch.long),
            "target_length": torch.tensor(len(encoded), dtype=torch.long),
            "plate_number": row["plate_number"],
            "filename": row["filename"],
            "image_path": str(image_path),
        }


def ctc_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length CTC targets into one flat target tensor."""

    return {
        "images": torch.stack([item["image"] for item in batch]),
        "targets": torch.cat([item["target"] for item in batch]),
        "target_lengths": torch.stack([item["target_length"] for item in batch]),
        "plate_numbers": [item["plate_number"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def build_dataloader(
    labels_path: Path = CCPD_LABELS_PATH,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    color_mode: str = "grayscale",
    max_samples: int | None = None,
    crop_plate: bool = True,
) -> DataLoader:
    """Build a DataLoader with the project-standard CRNN preprocessing."""

    dataset = CCPDPlateDataset(
        labels_path=labels_path,
        split=split,
        color_mode=color_mode,
        max_samples=max_samples,
        crop_plate=crop_plate,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=ctc_collate_fn)


def main() -> None:
    """Small smoke test for checking tensor shapes and decoded labels."""

    parser = argparse.ArgumentParser(description="Smoke test real CCPD Dataset/DataLoader.")
    parser.add_argument("--labels", type=Path, default=CCPD_LABELS_PATH)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-crop", action="store_true", help="Load full CCPD image instead of cropped plate region.")
    args = parser.parse_args()

    loader = build_dataloader(args.labels, args.split, args.batch_size, crop_plate=not args.no_crop)
    batch = next(iter(loader))
    print("images:", batch["images"].shape)
    print("targets:", batch["targets"].shape)
    print("target_lengths:", batch["target_lengths"].tolist())
    print("plate_numbers:", batch["plate_numbers"])


if __name__ == "__main__":
    main()
