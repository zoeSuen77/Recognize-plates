from __future__ import annotations
"""Build the YOLO character-detection dataset from CCPD plate labels.

CCPD does not provide true character bounding boxes, so this script creates
pseudo boxes from each plate string after cropping or perspective-warping the
plate. This dataset powers the comparison route, not the recommended CRNN
recognizer.
"""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from load_data import crop_from_ccpd_fields, resolve_image_path as _resolve_image_path
from plate_chars import CHAR_TO_INDEX, PLATE_CHARS
from plate_geometry import augment_plate_image, parse_ccpd_points, warp_plate_perspective
from utils import CCPD_LABELS_PATH, DATA_DIR, PROJECT_ROOT, resolve_project_path


DEFAULT_OUTPUT_DIR = DATA_DIR / "yolo_chars"

# Build class-name list for chars.yaml — YOLO class ID = index in PLATE_CHARS (0..64)
CLASS_NAMES = list(PLATE_CHARS)
NUM_CLASSES = len(CLASS_NAMES)  # 65

# Layout constants for pseudo char boxes. They intentionally leave border
# padding because the warped plate crop often contains rivets, shadows, or trim.
LEFT_MARGIN_RATIO = 0.04
RIGHT_MARGIN_RATIO = 0.04
TOP_MARGIN_RATIO = 0.10
BOTTOM_MARGIN_RATIO = 0.10
CHAR_GAP_RATIO = 0.015


def safe_image_name(index: int, source: Path) -> str:
    """Create a deterministic filename that avoids collisions across folders."""

    return f"{index:08d}_{source.stem}{source.suffix}"


def apply_limit(
    rows_by_split: dict[str, list[dict[str, str]]], limit: int | None
) -> dict[str, list[dict[str, str]]]:
    """Apply a train/val sample limit while keeping the test rows available."""

    if limit is None:
        return rows_by_split
    if limit <= 0:
        raise ValueError("--limit must be a positive integer")
    test_rows = rows_by_split.get("test", [])
    train_count = len(rows_by_split["train"])
    val_count = len(rows_by_split["val"])
    total = train_count + val_count
    if total == 0:
        return {"train": [], "val": [], "test": test_rows}
    if limit == 1 or val_count == 0:
        return {"train": rows_by_split["train"][: min(train_count, limit)], "val": [], "test": test_rows}
    train_limit = min(train_count, max(1, round(limit * train_count / total)))
    val_limit = min(val_count, max(1, limit - train_limit))
    while train_limit + val_limit > limit:
        if train_limit > 1:
            train_limit -= 1
        else:
            val_limit -= 1
    return {"train": rows_by_split["train"][:train_limit], "val": rows_by_split["val"][:val_limit], "test": test_rows}


def make_pseudo_char_boxes(
    plate_number: str,
    width: int,
    height: int,
) -> list[tuple[int, float, float, float, float]]:
    """Generate pseudo character bounding boxes for a plate image.

    Returns list of (class_id, x_center, y_center, box_w, box_h)
    with values normalized to [0, 1].

    Supports 7-char (standard) and 8-char (new energy) plates with
    appropriate margins and gaps.
    """
    n = len(plate_number)
    if n == 0:
        return []

    if width <= 0 or height <= 0:
        return []

    # Province and city-code characters are usually a touch wider/looser than
    # later alphanumeric characters. New-energy plates have one extra character,
    # so a separate weight vector keeps labels from becoming too narrow.
    if n == 7:
        weights = [1.08, 1.02, 0.98, 0.98, 0.98, 0.98, 0.98]
    elif n == 8:
        weights = [1.04, 0.98, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    else:
        weights = [1.0] * n

    available_w = width * (1.0 - LEFT_MARGIN_RATIO - RIGHT_MARGIN_RATIO)
    total_gap_w = width * CHAR_GAP_RATIO * max(n - 1, 0)
    char_total_w = max(1.0, available_w - total_gap_w)
    unit_w = char_total_w / max(sum(weights), 1.0)

    available_h = height * (1.0 - TOP_MARGIN_RATIO - BOTTOM_MARGIN_RATIO)
    char_h = available_h

    boxes = []
    x_cursor = width * LEFT_MARGIN_RATIO
    for i, ch in enumerate(plate_number):
        current_w = unit_w * weights[i]
        if ch not in CHAR_TO_INDEX:
            x_cursor += current_w + width * CHAR_GAP_RATIO
            continue
        cls_id = CHAR_TO_INDEX[ch]
        x_center = x_cursor + current_w / 2.0
        y_center = height * TOP_MARGIN_RATIO + char_h / 2.0
        x_cursor += current_w + width * CHAR_GAP_RATIO

        boxes.append((
            cls_id,
            x_center / width,
            y_center / height,
            current_w / width,
            char_h / height,
        ))

    return boxes


def prepare_dataset(
    labels_path: Path,
    output_dir: Path,
    limit: int | None,
    augment: bool,
    augment_copies: int,
    seed: int,
) -> None:
    """Generate cropped plate images, pseudo character labels, and chars.yaml."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Read rows
    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    with labels_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"filename", "plate_number", "split", "bbox"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{labels_path} must contain columns: {sorted(required)}")
        for row in reader:
            split = row["split"]
            if split in rows_by_split:
                rows_by_split[split].append(row)

    rows_by_split = apply_limit(rows_by_split, limit)

    stats = {"train": 0, "val": 0, "test": 0, "skipped": 0, "total_chars": 0, "warped": 0, "augmented": 0}
    global_index = 0

    for split in ["train", "val", "test"]:
        rows = rows_by_split.get(split, [])
        for row in rows:
            global_index += 1
            try:
                source = _resolve_image_path(row["filename"], labels_path)
                with Image.open(source) as img:
                    original = img.convert("RGB")
                plate_number = row["plate_number"]
                if not plate_number:
                    raise ValueError("Empty plate_number")

                # Step 1: Try perspective correction using points
                points_text = row.get("points", "")
                if points_text:
                    pts = parse_ccpd_points(points_text)
                    plate_img = warp_plate_perspective(original, pts)
                    if plate_img is not None:
                        stats["warped"] += 1
                    else:
                        plate_img = crop_from_ccpd_fields(original, row.get("bbox", ""), points_text)
                else:
                    plate_img = crop_from_ccpd_fields(original, row.get("bbox", ""), "")

            except Exception as exc:
                # Skip only this row so large dataset generation can continue
                # when a source image or geometry field is malformed.
                stats["skipped"] += 1
                print(f"[skip] {row.get('filename', '<unknown>')}: {exc}")
                continue

            pw, ph = plate_img.size
            if pw < 10 or ph < 10:
                stats["skipped"] += 1
                continue

            # Generate char boxes
            boxes = make_pseudo_char_boxes(plate_number, pw, ph)
            if not boxes:
                stats["skipped"] += 1
                continue

            label_lines = [f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n" for cls_id, xc, yc, bw, bh in boxes]

            # Save original
            image_name = safe_image_name(global_index, source)
            img_out = output_dir / "images" / split / image_name
            label_out = output_dir / "labels" / split / f"{Path(image_name).stem}.txt"
            plate_img.save(img_out)
            label_out.write_text("".join(label_lines), encoding="utf-8")
            stats[split] += 1
            stats["total_chars"] += len(label_lines)

            # Save augmented copies
            if augment and split == "train":
                for copy_idx in range(augment_copies):
                    aug_img = augment_plate_image(plate_img, rng)
                    aug_name = f"{global_index:08d}_{source.stem}_aug{copy_idx}{source.suffix}"
                    aug_out = output_dir / "images" / split / aug_name
                    aug_label_out = output_dir / "labels" / split / f"{Path(aug_name).stem}.txt"
                    aug_img.save(aug_out)
                    aug_label_out.write_text("".join(label_lines), encoding="utf-8")
                    stats["augmented"] += 1
                    stats["total_chars"] += len(label_lines)

    # Write chars.yaml
    yaml_path = output_dir / "chars.yaml"
    lines = [
        f"path: {output_dir.resolve()}\n",
        "train: images/train\n",
        "val: images/val\n",
        "test: images/test\n",
        f"nc: {NUM_CLASSES}\n",
        "names:\n",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}\n")
    yaml_path.write_text("".join(lines), encoding="utf-8")

    print(f"YOLO char dataset v2 saved to: {output_dir}")
    print(f"chars.yaml saved to: {yaml_path}")
    print(f"train samples: {stats['train']}")
    print(f"val samples:   {stats['val']}")
    print(f"test samples:  {stats['test']}")
    print(f"augmented:     {stats['augmented']}")
    print(f"warped plates: {stats['warped']}")
    print(f"total char labels: {stats['total_chars']}")
    print(f"skipped rows: {stats['skipped']}")
    print(f"character classes (nc): {NUM_CLASSES}")


def main() -> None:
    """Command-line entry point for YOLO character dataset generation."""

    parser = argparse.ArgumentParser(
        description="Convert CCPD labels.csv into a YOLO character-detection dataset v2. "
        "Uses perspective correction from CCPD points, improved pseudo char boxes, and optional augmentation."
    )
    parser.add_argument("--labels", type=Path, default=CCPD_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit train+val samples for quick debug.")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation for training images.")
    parser.add_argument("--augment-copies", type=int, default=1, help="Number of augmented copies per image.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_path = resolve_project_path(args.labels)
    output_dir = resolve_project_path(args.output)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    prepare_dataset(labels_path, output_dir, args.limit, args.augment, args.augment_copies, args.seed)


if __name__ == "__main__":
    main()
