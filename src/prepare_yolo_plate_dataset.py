from __future__ import annotations
"""Convert CCPD plate boxes into a YOLO license-plate detection dataset.

The CRNN recognizer expects cropped plates. To handle full vehicle photos, this
script builds the detector-side dataset: one image plus one normalized YOLO
label per CCPD sample. It uses only train/val rows so the test split remains
reserved for evaluation.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

from PIL import Image

from utils import CCPD_LABELS_PATH, DATA_DIR, PROJECT_ROOT, resolve_project_path


DEFAULT_OUTPUT_DIR = DATA_DIR / "yolo_plate"


def parse_pair(text: str) -> tuple[int, int]:
    """Parse a CCPD coordinate pair like ``317&389`` into integer x/y."""

    x_text, y_text = text.split("&")
    return int(x_text), int(y_text)


def resolve_image_path(filename: str, labels_path: Path) -> Path:
    """Resolve source image paths stored in ``labels.csv``."""

    path = Path(filename)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return labels_path.parent / path


def parse_bbox(bbox: str) -> tuple[int, int, int, int]:
    """Parse CCPD bbox text into ``left, top, right, bottom`` pixel coordinates."""

    left_top, right_bottom = bbox.split("_")
    x1, y1 = parse_pair(left_top)
    x2, y2 = parse_pair(right_bottom)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def bbox_to_yolo(bbox: str, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """Convert a CCPD pixel bbox into normalized YOLO center/size values."""

    x1, y1, x2, y2 = parse_bbox(bbox)
    x1 = max(0, min(image_width, x1))
    x2 = max(0, min(image_width, x2))
    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {(x1, y1, x2, y2)}")

    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2
    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def link_or_copy_image(source: Path, destination: Path, copy_images: bool) -> str:
    """Place an image in the YOLO tree using symlink by default or copy on demand."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if not destination.exists():
            destination.unlink()
        else:
            return "exists"
    elif destination.exists():
        return "exists"
    if copy_images:
        shutil.copy2(source, destination)
        return "copied"
    try:
        os.symlink(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def safe_image_name(index: int, source: Path) -> str:
    """Create a deterministic filename that avoids collisions across CCPD folders."""

    return f"{index:08d}_{source.name}"


def read_split_rows(labels_path: Path) -> dict[str, list[dict[str, str]]]:
    """Read train/val rows from the master labels CSV."""

    rows = {"train": [], "val": []}
    with labels_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"filename", "split", "bbox"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{labels_path} must contain columns: {sorted(required)}")
        for row in reader:
            split = row["split"]
            if split in rows:
                rows[split].append(row)
    return rows


def apply_limit(rows_by_split: dict[str, list[dict[str, str]]], limit: int | None) -> dict[str, list[dict[str, str]]]:
    """Apply a debug sample limit while preserving a train/val ratio."""

    if limit is None:
        return rows_by_split
    if limit <= 0:
        raise ValueError("--limit must be a positive integer")

    train_count = len(rows_by_split["train"])
    val_count = len(rows_by_split["val"])
    total = train_count + val_count
    if total == 0:
        return rows_by_split

    if limit == 1 or val_count == 0:
        return {"train": rows_by_split["train"][: min(train_count, limit)], "val": []}

    train_limit = min(train_count, max(1, round(limit * train_count / total)))
    val_limit = min(val_count, max(1, limit - train_limit))
    while train_limit + val_limit > limit:
        if train_limit > 1:
            train_limit -= 1
        else:
            val_limit -= 1
    return {
        "train": rows_by_split["train"][:train_limit],
        "val": rows_by_split["val"][:val_limit],
    }


def write_plate_yaml(output_dir: Path) -> Path:
    """Write the Ultralytics dataset config for one ``license_plate`` class."""

    yaml_path = output_dir / "plate.yaml"
    yaml_text = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: license_plate\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def prepare_dataset(labels_path: Path, output_dir: Path, limit: int | None, copy_images: bool) -> None:
    """Generate YOLO image/label folders and the accompanying ``plate.yaml``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows_by_split = apply_limit(read_split_rows(labels_path), limit)
    stats = {
        "train": 0,
        "val": 0,
        "linked": 0,
        "copied": 0,
        "exists": 0,
        "skipped": 0,
    }

    global_index = 0
    for split, rows in rows_by_split.items():
        for row in rows:
            global_index += 1
            try:
                source = resolve_image_path(row["filename"], labels_path)
                with Image.open(source) as image:
                    image_width, image_height = image.size
                x_center, y_center, width, height = bbox_to_yolo(row["bbox"], image_width, image_height)
            except Exception as exc:
                # Skip only the bad row. A single missing image or invalid bbox
                # should not invalidate a large training dataset generation job.
                stats["skipped"] += 1
                print(f"[skip] {row.get('filename', '<unknown>')}: {exc}")
                continue

            image_name = safe_image_name(global_index, source)
            image_output = output_dir / "images" / split / image_name
            label_output = output_dir / "labels" / split / f"{Path(image_name).stem}.txt"
            action = link_or_copy_image(source, image_output, copy_images)
            label_output.write_text(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n", encoding="utf-8")
            stats[action] += 1
            stats[split] += 1

    yaml_path = write_plate_yaml(output_dir)
    print(f"YOLO dataset saved to: {output_dir}")
    print(f"plate.yaml saved to: {yaml_path}")
    print(f"train labels: {stats['train']}")
    print(f"val labels: {stats['val']}")
    print(f"images linked: {stats['linked']}")
    print(f"images copied: {stats['copied']}")
    print(f"images already existed: {stats['exists']}")
    print(f"skipped rows: {stats['skipped']}")
    if copy_images:
        print("[info] Images were copied (--copy-images). No external drive dependency.")
    else:
        print("[info] Images were symlinked. The external drive (/Volumes/ksid/ccpd_data) MUST remain mounted during YOLO training.")
        print("[info] Run with --copy-images to copy images and eliminate external drive dependency.")


def main() -> None:
    """Command-line entry point for detector dataset generation."""

    parser = argparse.ArgumentParser(description="Convert CCPD labels.csv into a YOLO plate-detection dataset.")
    parser.add_argument("--labels", type=Path, default=CCPD_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional total train+val sample limit for quick debugging.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    args = parser.parse_args()

    labels_path = resolve_project_path(args.labels)
    output_dir = resolve_project_path(args.output)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    prepare_dataset(labels_path, output_dir, args.limit, args.copy_images)


if __name__ == "__main__":
    main()
