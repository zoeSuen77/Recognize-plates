from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

from license_plate_dataset import DEFAULT_LABELS, LicensePlateDataset, license_plate_collate_fn
from metrics import confusion_pairs, greedy_ctc_decode, summarize_predictions
from model_crnn import CRNNLicensePlate
from plate_chars import NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "license_plate_crnn_best.pth"


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_prediction_samples(rows: list[dict[str, str]]) -> None:
    cache_dir = RESULTS_DIR / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return
    for kind, selected in {
        "correct": [row for row in rows if row["correct"] == "1"][:12],
        "wrong": [row for row in rows if row["correct"] == "0"][:12],
    }.items():
        if not selected:
            continue
        cols = min(4, len(selected))
        grid_rows = (len(selected) + cols - 1) // cols
        fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3.4, grid_rows * 1.8))
        axes = [axes] if len(selected) == 1 else axes.reshape(-1)
        for axis, row in zip(axes, selected):
            axis.imshow(Image.open(row["image_path"]).convert("RGB"))
            axis.set_title(f"T:{row['label']}\nP:{row['prediction']}", fontsize=8)
            axis.axis("off")
        for axis in axes[len(selected):]:
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"license_plate_{kind}_samples.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRNN license plate recognition.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--color-mode", choices=["grayscale", "rgb"], default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    device = get_device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    color_mode = args.color_mode or checkpoint.get("color_mode", "grayscale")
    input_channels = 1 if color_mode == "grayscale" else 3

    dataset = LicensePlateDataset(args.labels, split=args.split, color_mode=color_mode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=license_plate_collate_fn)
    model = CRNNLicensePlate(num_classes=NUM_CLASSES, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, str]] = []
    labels: list[str] = []
    predictions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device))
            batch_predictions = greedy_ctc_decode(logits)
            for path, label, prediction in zip(batch["image_paths"], batch["labels"], batch_predictions):
                labels.append(label)
                predictions.append(prediction)
                correct = label == prediction
                rows.append({"image_path": path, "label": label, "prediction": prediction, "correct": "1" if correct else "0"})
                print(f"true={label} | pred={prediction} | correct={correct}")

    metrics = summarize_predictions(labels, predictions)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wrong_path = RESULTS_DIR / "license_plate_errors.csv"
    with wrong_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "prediction", "correct"])
        writer.writeheader()
        writer.writerows([row for row in rows if row["correct"] == "0"])
    save_prediction_samples(rows)

    pairs = confusion_pairs(labels, predictions)
    print("Common confusion pairs:")
    for (truth, pred), count in pairs.most_common(20):
        mark = " key" if (truth, pred) in {("0", "O"), ("O", "0"), ("1", "I"), ("I", "1"), ("5", "S"), ("S", "5"), ("8", "B"), ("B", "8")} else ""
        print(f"  {truth}->{pred}: {count}{mark}")
    print(f"Wrong predictions saved to: {wrong_path}")


if __name__ == "__main__":
    main()
