from __future__ import annotations
"""Create the master CCPD labels file used by the whole project.

CCPD stores useful annotations directly in each image filename. This script
parses those filenames into a normalized CSV with image path, plate text,
dataset split, bounding box, and four corner points. Later training and
evaluation scripts all depend on this file instead of reparsing filenames.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from plate_chars import is_valid_plate
from utils import CCPD_LABELS_PATH, CCPD_RAW_DIR, PROJECT_ROOT, ensure_dirs, resolve_project_path

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


@dataclass
class CCPDRecord:
    """One parsed CCPD image annotation before split assignment."""

    filename: str
    plate_number: str
    bbox: str
    points: str


def parse_pair(text: str) -> tuple[int, int]:
    """Parse a CCPD coordinate pair like ``317&389`` into integer x/y."""

    x_text, y_text = text.split("&")
    return int(x_text), int(y_text)


def parse_ccpd_filename(image_path: Path) -> CCPDRecord:
    """Extract plate text and geometry fields from a CCPD filename.

    CCPD filenames are dash-separated. The fields used here are:
    bbox at index 2, four plate points at index 3, and encoded plate
    character indices at index 4.
    """

    parts = image_path.stem.split("-")
    if len(parts) < 5:
        raise ValueError("filename does not match CCPD dash-separated format")

    bbox_text = parts[2]
    points_text = parts[3]
    label_text = parts[4]

    left_top, right_bottom = bbox_text.split("_")
    parse_pair(left_top)
    parse_pair(right_bottom)
    for point in points_text.split("_"):
        parse_pair(point)

    indices = [int(item) for item in label_text.split("_")]
    if len(indices) < 7:
        raise ValueError(f"expected at least 7 plate indices, got {len(indices)}")
    try:
        plate = PROVINCES[indices[0]] + ALPHABETS[indices[1]] + "".join(ADS[index] for index in indices[2:7])
    except IndexError as exc:
        raise ValueError(f"plate index out of range: {indices}") from exc
    if not is_valid_plate(plate):
        raise ValueError(f"parsed illegal plate number: {plate}")

    try:
        filename = str(image_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        filename = str(image_path.resolve())
    return CCPDRecord(filename=filename, plate_number=plate, bbox=bbox_text, points=points_text)


def collect_images(raw_dir: Path) -> list[Path]:
    """Recursively collect image files from the raw CCPD directory."""

    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in raw_dir.rglob("*") if path.suffix.lower() in suffixes)


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """Compute train/val/test counts while keeping tiny samples usable."""

    if total <= 0:
        return 0, 0, 0
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if total >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        test_count = max(1, total - train_count - val_count)
        while train_count + val_count + test_count > total:
            train_count -= 1
    elif total == 2:
        train_count, val_count, test_count = 1, 0, 1
    else:
        train_count, val_count, test_count = 1, 0, 0
    return train_count, val_count, test_count


def split_for_index(index: int, train_count: int, val_count: int) -> str:
    """Map a shuffled sample index to train, val, or test."""

    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def main() -> None:
    """Command-line entry point for generating ``data/ccpd/labels.csv``."""

    parser = argparse.ArgumentParser(description="Parse real CCPD filenames and create data/ccpd/labels.csv.")
    parser.add_argument("--raw-dir", type=Path, default=CCPD_RAW_DIR, help="Directory containing original CCPD images.")
    parser.add_argument("--output", type=Path, default=CCPD_LABELS_PATH, help="Output CSV path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional small real-CCPD subset for quick tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ensure_dirs()
    raw_dir = resolve_project_path(args.raw_dir)
    output = resolve_project_path(args.output)
    if not raw_dir.exists():
        raise FileNotFoundError(f"CCPD raw directory not found: {raw_dir}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1")

    image_paths = collect_images(raw_dir)
    random.seed(args.seed)
    random.shuffle(image_paths)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    train_count, val_count, _ = split_counts(len(image_paths), args.train_ratio, args.val_ratio)
    rows: list[dict[str, str]] = []
    skipped = 0
    for index, image_path in enumerate(image_paths):
        try:
            record = parse_ccpd_filename(image_path)
        except Exception as exc:
            # Keep dataset preparation robust: a few malformed filenames should
            # not stop a full CCPD parse job that may contain hundreds of
            # thousands of images.
            skipped += 1
            print(f"[skip] {image_path.name}: {exc}")
            continue
        rows.append(
            {
                "filename": record.filename,
                "plate_number": record.plate_number,
                "split": split_for_index(index, train_count, val_count),
                "bbox": record.bbox,
                "points": record.points,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "plate_number", "split", "bbox", "points"])
        writer.writeheader()
        writer.writerows(rows)

    split_summary = {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]}
    print(f"Found images: {len(image_paths)}")
    print(f"Parsed labels: {len(rows)}")
    print(f"Skipped images: {skipped}")
    print(f"Split counts: {split_summary}")
    print(f"Labels saved to: {output}")
    print("Images were not copied; labels.csv records original paths and split values.")


if __name__ == "__main__":
    main()
