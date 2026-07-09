"""Preprocessing pipeline for CIC-ToN-IoT data."""


def normalize_attack_label(value: object, aliases: dict[str, str]) -> str:
    """Normalize raw attack labels to the canonical proposal label names."""
    key = str(value).strip().lower()
    return aliases.get(key, str(value).strip())
