from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from plate_chars import DIGITS, LETTERS, PROVINCES, is_valid_plate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "license_plates"
CROPPED_DIR = OUTPUT_DIR / "cropped"
LABELS_PATH = OUTPUT_DIR / "labels.csv"
SAMPLES_FIGURE = PROJECT_ROOT / "results" / "license_plate_preprocess_samples.png"

CCPD_PROVINCES = list(PROVINCES)
CCPD_LETTERS = list(LETTERS)
CCPD_ADS = list(LETTERS + DIGITS)


@dataclass
class ParsedCCPD:
    label: str
    bbox: tuple[int, int, int, int]
    points: list[tuple[int, int]]


def parse_pair(text: str) -> tuple[int, int]:
    x, y = text.split("&")
    return int(x), int(y)


def parse_ccpd_filename(path: Path) -> ParsedCCPD:
    parts = path.stem.split("-")
    if len(parts) < 5:
        raise ValueError("CCPD filename has too few dash-separated fields")

    left_top, right_bottom = parts[2].split("_")
    x1, y1 = parse_pair(left_top)
    x2, y2 = parse_pair(right_bottom)
    bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    points = [parse_pair(item) for item in parts[3].split("_")]
    indices = [int(item) for item in parts[4].split("_")]
    if len(indices) < 7:
        raise ValueError("CCPD plate label contains fewer than 7 indices")

    province = CCPD_PROVINCES[indices[0]]
    letter = CCPD_LETTERS[indices[1]]
    tail = "".join(CCPD_ADS[index] for index in indices[2:])
    label = province + letter + tail
    if not is_valid_plate(label):
        raise ValueError(f"Illegal parsed plate label: {label}")
    return ParsedCCPD(label=label, bbox=bbox, points=points)


def crop_plate(image: Image.Image, parsed: ParsedCCPD, padding: int = 4) -> Image.Image:
    width, height = image.size
    xs = [x for x, _ in parsed.points] or [parsed.bbox[0], parsed.bbox[2]]
    ys = [y for _, y in parsed.points] or [parsed.bbox[1], parsed.bbox[3]]
    left = max(0, min(xs + [parsed.bbox[0]]) - padding)
    top = max(0, min(ys + [parsed.bbox[1]]) - padding)
    right = min(width, max(xs + [parsed.bbox[2]]) + padding)
    bottom = min(height, max(ys + [parsed.bbox[3]]) + padding)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box: {(left, top, right, bottom)}")
    return image.crop((left, top, right, bottom))


def collect_images(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def choose_split(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    value = (index + 0.5) / max(1, total)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def save_sample_figure(rows: list[dict[str, str]], max_items: int = 12) -> None:
    if not rows:
        return
    cache_dir = PROJECT_ROOT / "results" / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    selected = rows[:max_items]
    cols = min(4, len(selected))
    grid_rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3.4, grid_rows * 1.7))
    axes = [axes] if len(selected) == 1 else axes.reshape(-1)
    for axis, row in zip(axes, selected):
        image = Image.open(row["image_path"]).convert("RGB")
        axis.imshow(image)
        axis.set_title(row["label"], fontsize=9)
        axis.axis("off")
    for axis in axes[len(selected):]:
        axis.axis("off")
    SAMPLES_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(SAMPLES_FIGURE, dpi=150)
    plt.close()


def write_debug_overlay(source: Path, parsed: ParsedCCPD, output_path: Path) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(parsed.bbox, outline="red", width=3)
    if parsed.points:
        draw.line(parsed.points + [parsed.points[0]], fill="yellow", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cropped CCPD license plate images.")
    parser.add_argument("--ccpd-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    cropped_dir = args.output_dir / "cropped"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args.ccpd_root)
    random.shuffle(image_paths)
    if args.max_samples:
        image_paths = image_paths[: args.max_samples]

    rows: list[dict[str, str]] = []
    stats = {"ok": 0, "parse_error": 0, "image_error": 0, "crop_error": 0}

    for index, image_path in enumerate(image_paths):
        try:
            parsed = parse_ccpd_filename(image_path)
        except Exception as exc:
            stats["parse_error"] += 1
            print(f"[parse skip] {image_path.name}: {exc}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            stats["image_error"] += 1
            print(f"[image skip] {image_path}: {exc}")
            continue

        try:
            cropped = crop_plate(image, parsed, padding=args.padding)
        except Exception as exc:
            stats["crop_error"] += 1
            print(f"[crop skip] {image_path.name}: {exc}")
            continue

        split = choose_split(index, len(image_paths), args.train_ratio, args.val_ratio)
        out_name = f"{split}_{stats['ok']:06d}_{parsed.label}.jpg"
        out_path = cropped_dir / out_name
        cropped.save(out_path)
        if args.save_overlays and len(rows) < 20:
            write_debug_overlay(image_path, parsed, PROJECT_ROOT / "results" / "overlays" / out_name)
        rows.append({"image_path": str(out_path), "label": parsed.label, "split": split})
        stats["ok"] += 1

    labels_path = args.output_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    save_sample_figure(rows)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]}
    print(f"Processed images: {len(image_paths)}")
    print(f"Saved cropped plates: {stats['ok']}")
    print(f"Skipped: parse={stats['parse_error']}, image={stats['image_error']}, crop={stats['crop_error']}")
    print(f"Split counts: {split_counts}")
    print(f"Labels saved to: {labels_path}")
    print(f"Sample figure saved to: {SAMPLES_FIGURE}")


if __name__ == "__main__":
    main()
