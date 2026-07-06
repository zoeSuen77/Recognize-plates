from __future__ import annotations
"""Train the YOLO character detector used as a comparison recognizer.

The main production-like route uses CRNN for full-plate recognition. This
script trains the alternative character-detection model, then copies the best
checkpoint to ``models/char_detector.pt`` for evaluation and visualization.
"""

import argparse
import shutil
import signal
import sys
from pathlib import Path

from ultralytics import YOLO

from utils import MODELS_DIR, PROJECT_ROOT, resolve_project_path


DEFAULT_DATA = PROJECT_ROOT / "data" / "yolo_chars" / "chars.yaml"
DEFAULT_PROJECT = str((PROJECT_ROOT / "runs" / "detect").resolve())
DEFAULT_RUN_NAME = "yolo_chars"
CHAR_MODEL_PATH = MODELS_DIR / "char_detector.pt"
DEFAULT_CHECKPOINT = str((PROJECT_ROOT / "runs" / "detect" / "yolo_chars" / "weights" / "last.pt").resolve())


def graceful_exit(*args: object) -> None:
    """Handle Ctrl+C by telling the user how to resume from last.pt."""

    print("\n[info] Training interrupted by user.")
    print(f"[info] Checkpoint saved at last.pt — resume later with:")
    print(f"  python src/train_yolo_char.py --resume")
    print(f"  python src/train_yolo_char.py --resume --checkpoint {DEFAULT_CHECKPOINT}")
    sys.exit(130)


def main() -> None:
    """Command-line entry point for fresh or resumed YOLO character training."""

    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 character detector on the cropped-plate char dataset. "
        "Supports resume from checkpoint. After training, best.pt is copied to models/char_detector.pt"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to chars.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO model (ignored when --resume)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="YOLO project directory (runs/detect)")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Run name (saved to <project>/<name>)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 = disable)")
    parser.add_argument("--save-period", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to last.pt checkpoint for resume. Defaults to <project>/<name>/weights/last.pt",
    )
    args = parser.parse_args()

    # --- Resume mode ---
    if args.resume:
        ckpt_path = args.checkpoint or DEFAULT_CHECKPOINT
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            # Also try resolving relative to project root
            alt = resolve_project_path(ckpt_path)
            if alt.exists():
                ckpt = alt
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {ckpt_path}\n"
                    f"Checked: {ckpt.resolve()}\n"
                    f"Checked: {alt.resolve()}\n"
                    "Train from scratch first or provide a valid --checkpoint path."
                )

        print(f"Resuming training from checkpoint: {ckpt}")
        model = YOLO(str(ckpt))

        # Register Ctrl+C handler before training starts
        signal.signal(signal.SIGINT, graceful_exit)

        model.train(
            resume=True,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
        )
    else:
        # --- Fresh training ---
        data_path = resolve_project_path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(
                f"chars.yaml not found: {data_path}. Run src/prepare_yolo_char_dataset.py first."
            )

        model = YOLO(args.model)

        # Register Ctrl+C handler before training starts
        signal.signal(signal.SIGINT, graceful_exit)

        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            lr0=args.lr0,
            patience=args.patience,
            save_period=args.save_period,
            exist_ok=args.exist_ok,
        )

    # --- Copy best.pt to models/char_detector.pt ---
    # Keep a stable model path so evaluation scripts do not need to know the
    # exact Ultralytics run directory name.
    run_dir = Path(args.project) / args.name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"Expected best.pt at {best_pt.resolve()}, but it was not found. "
            "Training may have been interrupted before completion."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_pt), str(CHAR_MODEL_PATH))
    print(f"char detector saved to: {CHAR_MODEL_PATH}")


if __name__ == "__main__":
    main()
