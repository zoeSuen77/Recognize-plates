from __future__ import annotations
"""Decoding and evaluation metrics shared by training and evaluation scripts."""

from collections import Counter

import torch

from plate_chars import BLANK_INDEX, INDEX_TO_CHAR


def greedy_ctc_decode(logits: torch.Tensor, blank_index: int = BLANK_INDEX) -> list[str]:
    """Greedily decode CRNN logits into plate strings."""

    indices = logits.detach().cpu().argmax(dim=2).transpose(0, 1)
    predictions: list[str] = []
    for sequence in indices:
        chars: list[str] = []
        previous = blank_index
        for index in sequence.tolist():
            if index != blank_index and index != previous:
                chars.append(INDEX_TO_CHAR.get(index, ""))
            previous = index
        predictions.append("".join(chars))
    return predictions


def greedy_decode_with_confidence(logits: torch.Tensor, blank_index: int = BLANK_INDEX) -> list[dict[str, object]]:
    """Decode logits and keep a confidence score for each emitted character."""

    probs = logits.detach().softmax(dim=2).cpu()
    pred_indices = probs.argmax(dim=2).transpose(0, 1)
    pred_probs = probs.max(dim=2).values.transpose(0, 1)
    decoded: list[dict[str, object]] = []
    for indices, confidences in zip(pred_indices, pred_probs):
        chars: list[str] = []
        char_confidences: list[float] = []
        previous = blank_index
        running_conf: list[float] = []
        for index, confidence in zip(indices.tolist(), confidences.tolist()):
            if index == previous and index != blank_index:
                running_conf.append(confidence)
            elif index != blank_index:
                chars.append(INDEX_TO_CHAR.get(index, ""))
                char_confidences.append(max([confidence] + running_conf))
                running_conf = []
            else:
                running_conf = []
            previous = index
        decoded.append({"text": "".join(chars), "confidences": char_confidences})
    return decoded


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
    """Approximate character-level accuracy from normalized edit distance."""

    max_len = max(len(truth), len(prediction), 1)
    distance = edit_distance(truth, prediction)
    return max(0.0, 1.0 - distance / max_len)


def summarize_predictions(labels: list[str], predictions: list[str]) -> dict[str, float]:
    """Aggregate character accuracy, full-plate accuracy, and edit-distance metrics."""

    if not labels:
        return {"character_accuracy": 0.0, "full_plate_accuracy": 0.0, "avg_edit_distance": 0.0, "normalized_edit_distance": 0.0}
    distances = [edit_distance(label, pred) for label, pred in zip(labels, predictions)]
    normalizers = [max(len(label), len(pred), 1) for label, pred in zip(labels, predictions)]
    return {
        "character_accuracy": sum(character_accuracy(label, pred) for label, pred in zip(labels, predictions)) / len(labels),
        "full_plate_accuracy": sum(label == pred for label, pred in zip(labels, predictions)) / len(labels),
        "avg_edit_distance": sum(distances) / len(distances),
        "normalized_edit_distance": sum(distance / norm for distance, norm in zip(distances, normalizers)) / len(distances),
    }


def confusion_pairs(labels: list[str], predictions: list[str]) -> Counter[tuple[str, str]]:
    """Count aligned character substitutions for quick error analysis."""

    counter: Counter[tuple[str, str]] = Counter()
    for label, prediction in zip(labels, predictions):
        for truth_char, pred_char in zip(label, prediction):
            if truth_char != pred_char:
                counter[(truth_char, pred_char)] += 1
    return counter
