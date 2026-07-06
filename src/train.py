from __future__ import annotations
"""Train the CRNN + CTC recognizer on CCPD plate crops.

The training loop reports both character accuracy and full-plate accuracy.
Full-plate accuracy is the key business metric here because one wrong
character makes the final license plate string incorrect.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch import nn

from load_data import build_dataloader
from metrics import greedy_ctc_decode, summarize_predictions
from model_crnn import CRNNLicensePlate
from plate_chars import BLANK_INDEX, NUM_CLASSES
from utils import CCPD_LABELS_PATH, DEFAULT_MODEL_PATH, MODELS_DIR, RESULTS_DIR, get_device, missing_labels_message, resolve_project_path, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, dict[str, float]]:
    """Run one train or validation epoch and return loss plus metrics."""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    labels: list[str] = []
    predictions: list[str] = []
    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            if logits.dim() != 3 or logits.size(1) != images.size(0) or logits.size(2) != NUM_CLASSES:
                raise RuntimeError(f"CRNN output must be [T, N, C], got {tuple(logits.shape)}")
            log_probs = logits.log_softmax(dim=2)
            # CTC needs the available time-step count for each sample. The CRNN
            # produces the same sequence length for every image in a batch after
            # resize/pad preprocessing, so one repeated value is sufficient.
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        labels.extend(batch["plate_numbers"])
        predictions.extend(greedy_ctc_decode(logits))
    return total_loss / len(loader.dataset), summarize_predictions(labels, predictions)


def save_history(history: list[dict[str, float]]) -> None:
    """Persist epoch metrics for later report figures and comparison."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ccpd_train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS_DIR / "ccpd_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]]) -> None:
    """Save loss and accuracy curves when matplotlib is available."""

    cache_dir = RESULTS_DIR / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("CTC loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ccpd_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["val_character_accuracy"] for row in history], label="character accuracy")
    plt.plot(epochs, [row["val_full_plate_accuracy"] for row in history], label="full plate accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ccpd_accuracy_curve.png", dpi=150)
    plt.close()


def main() -> None:
    """Command-line entry point for CRNN training and checkpointing."""

    parser = argparse.ArgumentParser(description="Train CRNN + CTC on the real CCPD dataset.")
    parser.add_argument("--labels", type=Path, default=CCPD_LABELS_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--color-mode", choices=["grayscale", "rgb"], default="grayscale")
    parser.add_argument("--no-crop", action="store_true", help="Train on full CCPD images instead of filename-derived plate crop.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    args.labels = resolve_project_path(args.labels)
    args.model_path = resolve_project_path(args.model_path)
    if args.resume is not None:
        args.resume = resolve_project_path(args.resume)
    if not args.labels.exists():
        raise FileNotFoundError(missing_labels_message(args.labels))

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    crop_plate = not args.no_crop
    train_loader = build_dataloader(args.labels, "train", args.batch_size, True, args.num_workers, args.color_mode, args.max_samples, crop_plate)
    val_loader = build_dataloader(args.labels, "val", args.batch_size, False, args.num_workers, args.color_mode, args.max_samples, crop_plate)

    input_channels = 1 if args.color_mode == "grayscale" else 3
    model = CRNNLicensePlate(input_channels=input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    start_epoch = 1
    best_full_acc = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_full_acc = float(checkpoint.get("full_plate_accuracy", -1.0))

    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_character_accuracy": train_metrics["character_accuracy"],
            "train_full_plate_accuracy": train_metrics["full_plate_accuracy"],
            "val_character_accuracy": val_metrics["character_accuracy"],
            "val_full_plate_accuracy": val_metrics["full_plate_accuracy"],
            "val_normalized_edit_distance": val_metrics["normalized_edit_distance"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} | lr={current_lr:.6f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"char_acc={val_metrics['character_accuracy']:.4f} | full_acc={val_metrics['full_plate_accuracy']:.4f} | "
            f"ned={val_metrics['normalized_edit_distance']:.4f}"
        )
        save_history(history)
        plot_history(history)
        scheduler.step()
        if val_metrics["full_plate_accuracy"] > best_full_acc:
            # Save by full-plate accuracy, not loss, because deployment cares
            # about exact final plate strings.
            best_full_acc = val_metrics["full_plate_accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "full_plate_accuracy": best_full_acc,
                    "blank_index": BLANK_INDEX,
                    "num_classes": NUM_CLASSES,
                    "color_mode": args.color_mode,
                    "crop_plate": crop_plate,
                },
                args.model_path,
            )
            print(f"Saved best model to {args.model_path}")


if __name__ == "__main__":
    main()
