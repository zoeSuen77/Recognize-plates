from __future__ import annotations
"""Summarize common failure modes in YOLO character recognition outputs.

The character detector may add, miss, or misclassify individual characters.
This script reads the prediction CSV, groups errors by type, optionally copies
sample images, and writes a short summary for reports.
"""

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from plate_chars import DIGITS, LETTERS
from utils import RESULTS_DIR, resolve_project_path


DEFAULT_PREDICTIONS = RESULTS_DIR / "yolo_char_predictions.csv"
DEFAULT_ERROR_DIR = RESULTS_DIR / "yolo_char_errors"
DEFAULT_SUMMARY = RESULTS_DIR / "yolo_char_error_summary.txt"


def get_prediction(row: dict[str, str]) -> str:
    """Read the best available prediction column from a result row."""

    return row.get("postprocessed_prediction") or row.get("prediction") or ""


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two plate strings."""

    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[-1]


def character_accuracy(truth: str, prediction: str) -> float:
    """Approximate character accuracy using normalized edit distance."""

    max_len = max(len(truth), len(prediction), 1)
    distance = edit_distance(truth, prediction)
    return max(0.0, 1.0 - distance / max_len)


def classify_error(label: str, prediction: str) -> str:
    """Classify a prediction as missing, extra, or same-length wrong chars."""

    if label == prediction:
        return "correct"
    if len(prediction) < len(label):
        return "missing_chars"
    if len(prediction) > len(label):
        return "extra_chars"
    return "same_length_classification"


def is_digit_letter_confusion(truth: str, pred: str) -> bool:
    """Return whether an error swaps a digit with a letter or vice versa."""

    return (truth in DIGITS and pred in LETTERS) or (truth in LETTERS and pred in DIGITS)


def safe_copy_name(index: int, label: str, prediction: str, source: Path) -> str:
    """Create an informative filename for copied error samples."""

    suffix = source.suffix or ".jpg"
    return f"{index:03d}_{label}_pred_{prediction}{suffix}"


def copy_error_samples(error_rows: list[dict[str, str]], output_dir: Path, sample_count: int, seed: int) -> int:
    """Copy a reproducible sample of failed images into an error folder."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sampled = rng.sample(error_rows, min(sample_count, len(error_rows)))
    copied = 0
    for index, row in enumerate(sampled, start=1):
        image_text = row.get("image_path") or row.get("filename") or ""
        source = Path(image_text)
        if not source.exists():
            continue
        label = row.get("label") or row.get("true_plate") or ""
        prediction = get_prediction(row)
        target = output_dir / safe_copy_name(index, label, prediction, source)
        shutil.copy2(source, target)
        copied += 1
    return copied


def analyze(rows: list[dict[str, str]]) -> dict[str, object]:
    """Compute aggregate failure statistics from prediction rows."""

    total = len(rows)
    errors = [row for row in rows if (row.get("label") or row.get("true_plate") or "") != get_prediction(row)]
    char_acc_values = []
    missing_chars = 0
    extra_chars = 0
    same_length_classification = 0
    province_errors = 0
    second_letter_errors = 0
    digit_letter_confusions: Counter[str] = Counter()

    for row in rows:
        label = row.get("label") or row.get("true_plate") or ""
        prediction = get_prediction(row)
        char_acc_values.append(character_accuracy(label, prediction))
        error_type = classify_error(label, prediction)
        if error_type == "missing_chars":
            missing_chars += 1
        elif error_type == "extra_chars":
            extra_chars += 1
        elif error_type == "same_length_classification":
            same_length_classification += 1

        if label != prediction and label[:1] != prediction[:1]:
            province_errors += 1
        if len(label) >= 2 and (len(prediction) < 2 or label[1] != prediction[1]):
            second_letter_errors += 1

        for truth_char, pred_char in zip(label, prediction):
            if truth_char != pred_char and is_digit_letter_confusion(truth_char, pred_char):
                digit_letter_confusions[f"{truth_char}->{pred_char}"] += 1

    full_plate_acc = 1.0 - (len(errors) / total) if total else 0.0
    char_acc = sum(char_acc_values) / total if total else 0.0

    return {
        "total_samples": total,
        "plate_error_count": len(errors),
        "character_accuracy": char_acc,
        "full_plate_accuracy": full_plate_acc,
        "missing_chars": missing_chars,
        "extra_chars": extra_chars,
        "same_length_classification": same_length_classification,
        "province_errors": province_errors,
        "second_letter_errors": second_letter_errors,
        "digit_letter_confusions": digit_letter_confusions,
        "error_rows": errors,
    }


def format_summary(stats: dict[str, object], copied_count: int, error_dir: Path) -> str:
    """Format error statistics as a plain-text report."""

    confusion_counter = stats["digit_letter_confusions"]
    assert isinstance(confusion_counter, Counter)
    lines = [
        "YOLO Character Error Summary",
        "=" * 34,
        f"总样本数: {stats['total_samples']}",
        f"整牌错误数量: {stats['plate_error_count']}",
        f"字符准确率: {stats['character_accuracy']:.4f}",
        f"整牌准确率: {stats['full_plate_accuracy']:.4f}",
        f"少字符错误数量: {stats['missing_chars']}",
        f"多字符错误数量: {stats['extra_chars']}",
        f"等长但字符分类错误数量: {stats['same_length_classification']}",
        f"省份位错误数量: {stats['province_errors']}",
        f"第二位字母错误数量: {stats['second_letter_errors']}",
        "",
        "数字/字母混淆统计:",
    ]
    if confusion_counter:
        for pair, count in confusion_counter.most_common(30):
            lines.append(f"  {pair}: {count}")
    else:
        lines.append("  无")
    lines.extend(
        [
            "",
            f"错误样本保存目录: {error_dir}",
            f"已复制错误样本数: {copied_count}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command-line entry point for YOLO character error analysis."""

    parser = argparse.ArgumentParser(description="Analyze YOLO character detection prediction errors.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--error-dir", type=Path, default=DEFAULT_ERROR_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    predictions_path = resolve_project_path(args.predictions)
    error_dir = resolve_project_path(args.error_dir)
    summary_path = resolve_project_path(args.summary)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    with predictions_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats = analyze(rows)
    error_rows = stats["error_rows"]
    assert isinstance(error_rows, list)
    copied_count = copy_error_samples(error_rows, error_dir, args.sample_count, args.seed)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = format_summary(stats, copied_count, error_dir)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
