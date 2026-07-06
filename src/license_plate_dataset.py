from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from plate_chars import encode_plate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "license_plates" / "labels.csv"


class ResizePad:
    def __init__(self, size: tuple[int, int], fill: int = 255) -> None:
        self.width, self.height = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = min(self.width / image.width, self.height / image.height)
        new_w = max(1, int(round(image.width * scale)))
        new_h = max(1, int(round(image.height * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new(image.mode, (self.width, self.height), self.fill)
        left = (self.width - new_w) // 2
        top = (self.height - new_h) // 2
        canvas.paste(image, (left, top))
        return canvas


class LicensePlateDataset(Dataset):
    def __init__(
        self,
        labels_path: Path = DEFAULT_LABELS,
        split: str | None = None,
        image_size: tuple[int, int] = (160, 48),
        color_mode: str = "grayscale",
        max_samples: int | None = None,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.split = split
        self.image_size = image_size
        self.color_mode = color_mode
        channels = 1 if color_mode == "grayscale" else 3
        self.transform = transforms.Compose(
            [
                ResizePad(image_size),
                transforms.Grayscale(num_output_channels=channels) if color_mode == "grayscale" else transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,) * channels, std=(0.5,) * channels),
            ]
        )
        self.samples = self._read_labels(max_samples=max_samples)

    def _read_labels(self, max_samples: int | None) -> list[dict[str, str]]:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        samples: list[dict[str, str]] = []
        with self.labels_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"image_path", "label", "split"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"labels.csv must contain columns: {sorted(required)}")
            for row in reader:
                if self.split and row["split"] != self.split:
                    continue
                encode_plate(row["label"])
                samples.append(row)
                if max_samples and len(samples) >= max_samples:
                    break
        if not samples:
            raise ValueError(f"No samples found for split={self.split!r}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        image = Image.open(image_path)
        encoded = encode_plate(sample["label"])
        return {
            "image": self.transform(image),
            "encoded_label": torch.tensor(encoded, dtype=torch.long),
            "label_length": torch.tensor(len(encoded), dtype=torch.long),
            "original_label": sample["label"],
            "image_path": str(image_path),
        }


def license_plate_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "targets": torch.cat([item["encoded_label"] for item in batch]),
        "target_lengths": torch.stack([item["label_length"] for item in batch]),
        "labels": [item["original_label"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def build_dataloader(
    labels_path: Path,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    color_mode: str = "grayscale",
    max_samples: int | None = None,
) -> DataLoader:
    dataset = LicensePlateDataset(labels_path, split=split, color_mode=color_mode, max_samples=max_samples)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=license_plate_collate_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test license plate Dataset and DataLoader.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    loader = build_dataloader(args.labels, args.split, args.batch_size)
    batch = next(iter(loader))
    print("images:", batch["images"].shape)
    print("targets:", batch["targets"].shape)
    print("target_lengths:", batch["target_lengths"].tolist())
    print("labels:", batch["labels"])


if __name__ == "__main__":
    main()

