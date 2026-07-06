from __future__ import annotations
"""Character vocabulary and CTC label helpers for Chinese license plates."""

PROVINCES = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
DIGITS = "0123456789"

PLATE_CHARS = PROVINCES + LETTERS + DIGITS
CHAR_TO_INDEX = {char: index for index, char in enumerate(PLATE_CHARS)}
INDEX_TO_CHAR = {index: char for char, index in CHAR_TO_INDEX.items()}
BLANK_INDEX = len(PLATE_CHARS)
NUM_CLASSES = len(PLATE_CHARS) + 1


def encode_plate(label: str) -> list[int]:
    """Convert a plate string into class indices used by CTC targets."""

    unknown = [char for char in label if char not in CHAR_TO_INDEX]
    if unknown:
        raise ValueError(f"Illegal license plate characters in {label!r}: {unknown}")
    return [CHAR_TO_INDEX[char] for char in label]


def decode_indices(indices: list[int], blank_index: int = BLANK_INDEX) -> str:
    """Collapse CTC indices by removing blanks and repeated labels."""

    chars: list[str] = []
    previous = blank_index
    for index in indices:
        if index != blank_index and index != previous:
            chars.append(INDEX_TO_CHAR.get(index, ""))
        previous = index
    return "".join(chars)


def is_valid_plate(label: str) -> bool:
    """Return whether all characters are covered by the project vocabulary."""

    return bool(label) and all(char in CHAR_TO_INDEX for char in label)
