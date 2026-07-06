from __future__ import annotations

from plate_chars import DIGITS, LETTERS, PROVINCES

# Plate type constants
PLATE_STANDARD = "standard"  # 7-char blue plate
PLATE_NEW_ENERGY = "new_energy"  # 8-char green plate
PLATE_POLICE = "police"  # 警
PLATE_TRAILER = "trailer"  # 挂
PLATE_STUDENT = "student"  # 学
PLATE_HK_MACAU = "hk_macau"  # 港澳
PLATE_UNKNOWN = "unknown"

# Special characters by position (0-indexed tail char)
PLATE_TYPE_MAP: dict[str, str] = {
    "警": PLATE_POLICE,
    "挂": PLATE_TRAILER,
    "学": PLATE_STUDENT,
    "港": PLATE_HK_MACAU,
    "澳": PLATE_HK_MACAU,
}

# Plate lengths by type
PLATE_LENGTHS: dict[str, list[int]] = {
    PLATE_STANDARD: [7],
    PLATE_NEW_ENERGY: [8],
    PLATE_POLICE: [7],
    PLATE_TRAILER: [7],
    PLATE_STUDENT: [7],
    PLATE_HK_MACAU: [7, 8],
    PLATE_UNKNOWN: [7, 8],
}


def infer_plate_type(plate_text: str) -> str:
    """Infer plate type from a partial or complete plate string.

    Checks last character for special types,
    then checks length for new-energy vs standard.
    """
    if not plate_text:
        return PLATE_UNKNOWN

    last_char = plate_text[-1] if len(plate_text) >= 7 else ""

    # Check if last char indicates a special type
    if last_char in PLATE_TYPE_MAP:
        return PLATE_TYPE_MAP[last_char]

    # 8-char plates are new energy (if not HK/Macau)
    if len(plate_text) >= 8:
        return PLATE_NEW_ENERGY

    return PLATE_STANDARD


def allowed_chars_for_position(plate_type: str, position: int) -> set[str]:
    """Return the set of allowed characters for a given (type, position).

    Position is 0-indexed (0 = province, 1 = letter, 2+ = alphanumeric).
    """
    all_alnum = set(LETTERS) | set(DIGITS)

    if position == 0:
        return set(PROVINCES)

    if plate_type == PLATE_STANDARD:
        if position == 1:
            return set(LETTERS)
        return all_alnum

    if plate_type == PLATE_NEW_ENERGY:
        if position == 1:
            return set(LETTERS)
        return all_alnum

    if plate_type == PLATE_POLICE:
        if position == 1:
            return set(LETTERS)
        return all_alnum

    if plate_type == PLATE_TRAILER:
        if position == 1:
            return set(LETTERS)
        return all_alnum

    if plate_type == PLATE_STUDENT:
        if position == 1:
            return set(LETTERS)
        return all_alnum

    if plate_type == PLATE_HK_MACAU:
        if position == 1:
            return set(LETTERS)
        return all_alnum

    # Unknown – be permissive
    return set(PROVINCES) | set(LETTERS) | set(DIGITS)


def expected_plate_lengths(plate_type: str) -> list[int]:
    """Return the expected length(s) for the given plate type."""
    return PLATE_LENGTHS.get(plate_type, [7, 8])


def score_plate_candidate(candidate_text: str, detections: list | None = None) -> float:
    """Score a candidate plate string for plausibility.

    Returns a float where higher = more plausible.
    Considers: correct length for type, positional legality, confidence.
    """
    if not candidate_text:
        return -1.0

    plate_type = infer_plate_type(candidate_text)
    expected = expected_plate_lengths(plate_type)

    score = 0.0

    # Length match
    if len(candidate_text) in expected:
        score += 1.0
    else:
        score -= 0.5

    # Positional legality
    valid_count = 0
    for pos, ch in enumerate(candidate_text):
        if ch in allowed_chars_for_position(plate_type, pos):
            valid_count += 1
        else:
            score -= 0.3
    score += 0.1 * valid_count

    # If detections available, factor in confidence
    if detections and len(detections) == len(candidate_text):
        confs = [d.get("confidence", 0.5) for d in detections]
        score += 0.3 * (sum(confs) / len(confs))

    return score
