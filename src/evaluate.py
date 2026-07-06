from __future__ import annotations
"""Evaluate the trained CRNN recognizer on the CCPD test split."""

import argparse
import csv
from pathlib import Path

import torch

from load_data import build_dataloader
from metrics import greedy_ctc_decode, summarize_predictions
from model_crnn import CRNNLicensePlate
from plate_chars import NUM_CLASSES
from utils import CCPD_LABELS_PATH, DEFAULT_MODEL_PATH, RESULTS_DIR, get_device, missing_labels_message, resolve_project_path


def main() -> None:
    """Load a checkpoint, run test inference, and save per-image predictions."""

    parser = argparse.ArgumentParser(description="Evaluate CRNN on the real CCPD test split.")
    parser.add_argument("--labels", type=Path, default=CCPD_LABELS_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--color-mode", choices=["grayscale", "rgb"], default=None)
    parser.add_argument("--no-crop", action="store_true", help="Evaluate on full images instead of filename-derived plate crop.")
    args = parser.parse_args()

    args.labels = resolve_project_path(args.labels)
    args.model = resolve_project_path(args.model)
    if not args.labels.exists():
        raise FileNotFoundError(missing_labels_message(args.labels))
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}. Run python src/train.py first.")

    device = get_device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    color_mode = args.color_mode or checkpoint.get("color_mode", "grayscale")
    crop_plate = checkpoint.get("crop_plate", True) and not args.no_crop
    input_channels = 1 if color_mode == "grayscale" else 3

    loader = build_dataloader(args.labels, "test", args.batch_size, False, 0, color_mode, None, crop_plate)
    model = CRNNLicensePlate(num_classes=NUM_CLASSES, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, str]] = []
    labels: list[str] = []
    predictions: list[str] = []
    empty_count = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device))
            batch_predictions = greedy_ctc_decode(logits)
            for filename, true_plate, pred_plate in zip(batch["filenames"], batch["plate_numbers"], batch_predictions):
                # Keep row-level predictions so later scripts can inspect
                # exactly which plates failed, not only aggregate metrics.
                labels.append(true_plate)
                predictions.append(pred_plate)
                empty_count += int(pred_plate == "")
                rows.append(
                    {
                        "filename": filename,
                        "true_plate": true_plate,
                        "pred_plate": pred_plate,
                        "is_correct": "1" if true_plate == pred_plate else "0",
                    }
                )

    metrics = summarize_predictions(labels, predictions)
    print(f"character_accuracy: {metrics['character_accuracy']:.4f}")
    print(f"full_plate_accuracy: {metrics['full_plate_accuracy']:.4f}")
    print(f"avg_edit_distance: {metrics['avg_edit_distance']:.4f}")
    print(f"normalized_edit_distance: {metrics['normalized_edit_distance']:.4f}")
    print(f"empty_predictions: {empty_count}/{len(predictions)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ccpd_predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_plate", "pred_plate", "is_correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
