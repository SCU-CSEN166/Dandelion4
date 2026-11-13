"""Clean Whisper transcripts into canonical v2 Message JSON files.

Reads a v2 transcript ``Message`` array via :func:`contracts.load_messages`,
preserves Whisper's raw output in ``"asr_text"``, normalizes spoken English
number words and common Kenyan place-name ASR misspellings in ``"message"``,
sets ``"source": "audio"``, and writes the result with :func:`contracts.write_json`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import contracts

_UNITS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

_TEENS: dict[str, int] = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

PLACE_NAME_CORRECTIONS: dict[str, str] = {
    # Narok
    "narak": "Narok",
    "naraq": "Narok",
    "narock": "Narok",
    "narook": "Narok",
    "naruk": "Narok",
    "narrok": "Narok",
    # Kajiado
    "kajiadu": "Kajiado",
    "kajado": "Kajiado",
    "kajiando": "Kajiado",
    "kajadu": "Kajiado",
    "kajiyado": "Kajiado",
    "kajiadoo": "Kajiado",
    "kajyado": "Kajiado",
    # Nakuru
    "nakurro": "Nakuru",
    "nakulu": "Nakuru",
    "nakuro": "Nakuru",
    "nakoro": "Nakuru",
    "nakurru": "Nakuru",
    # Baringo
    "balingo": "Baringo",
    "baringoo": "Baringo",
    "barango": "Baringo",
    "baringgo": "Baringo",
    "beringo": "Baringo",
    # Other Kenyan locations
    "nairobbi": "Nairobi",
    "narobi": "Nairobi",
    "nairobe": "Nairobi",
    "kisumo": "Kisumu",
    "mombassa": "Mombasa",
    "mombaza": "Mombasa",
    "eldorett": "Eldoret",
    "eldorret": "Eldoret",
    "kiambo": "Kiambu",
    "mogotiyo": "Mogotio",
    "mogoteo": "Mogotio",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_PLACE_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(variant) for variant in sorted(PLACE_NAME_CORRECTIONS)) + r")\b",
    re.IGNORECASE,
)


def _is_joiner(gap: str) -> bool:
    """Return True if *gap* can join words within a single number phrase."""
    return gap == "-" or (bool(gap) and gap.strip(" \t") == "" and "\n" not in gap)


def _parse_below_hundred(
    text: str, tokens: list[re.Match[str]], index: int
) -> tuple[int, int] | None:
    """Parse a 1..99 number phrase starting at *index*, or return None."""
    word = tokens[index].group(0).lower()
    if word in _TENS:
        value = _TENS[word]
        if index + 1 < len(tokens) and _is_joiner(
            text[tokens[index].end() : tokens[index + 1].start()]
        ):
            next_word = tokens[index + 1].group(0).lower()
            if next_word in _UNITS:
                return value + _UNITS[next_word], index + 2
        return value, index + 1
    if word in _TEENS:
        return _TEENS[word], index + 1
    if word in _UNITS:
        return _UNITS[word], index + 1
    return None


def _parse_number_phrase(
    text: str, tokens: list[re.Match[str]], index: int
) -> tuple[int, int] | None:
    """Parse a standalone English number phrase (0..999) starting at *index*."""
    word = tokens[index].group(0).lower()
    if word == "zero":
        return 0, index + 1

    hundreds: int | None = None
    after_hundred = index
    if word == "hundred":
        hundreds = 100
        after_hundred = index + 1
    elif (
        word in _UNITS
        and index + 1 < len(tokens)
        and _is_joiner(text[tokens[index].end() : tokens[index + 1].start()])
        and tokens[index + 1].group(0).lower() == "hundred"
    ):
        hundreds = _UNITS[word] * 100
        after_hundred = index + 2

    if hundreds is not None:
        if after_hundred < len(tokens) and _is_joiner(
            text[tokens[after_hundred - 1].end() : tokens[after_hundred].start()]
        ):
            if tokens[after_hundred].group(0).lower() == "and":
                if after_hundred + 1 < len(tokens) and _is_joiner(
                    text[tokens[after_hundred].end() : tokens[after_hundred + 1].start()]
                ):
                    sub = _parse_below_hundred(text, tokens, after_hundred + 1)
                    if sub is not None:
                        return hundreds + sub[0], sub[1]
            else:
                sub = _parse_below_hundred(text, tokens, after_hundred)
                if sub is not None:
                    return hundreds + sub[0], sub[1]
        return hundreds, after_hundred

    return _parse_below_hundred(text, tokens, index)


def convert_number_words(text: str) -> str:
    """Convert standalone spoken English number words (0..999) into digits."""
    tokens = list(_TOKEN_RE.finditer(text))
    if not tokens:
        return text

    parts: list[str] = []
    cursor = 0
    index = 0
    while index < len(tokens):
        parsed = _parse_number_phrase(text, tokens, index)
        if parsed is None:
            index += 1
            continue
        value, next_index = parsed
        start = tokens[index].start()
        end = tokens[next_index - 1].end()
        parts.append(text[cursor:start])
        parts.append(str(value))
        cursor = end
        index = next_index

    parts.append(text[cursor:])
    return "".join(parts)


def normalize_place_names(text: str) -> str:
    """Replace common ASR misspellings of Kenyan place names with canonical names."""
    return _PLACE_NAME_RE.sub(
        lambda match: PLACE_NAME_CORRECTIONS[match.group(0).lower()],
        text,
    )


def clean_text(text: str) -> str:
    """Apply number-word conversion and place-name normalization to *text*."""
    return normalize_place_names(convert_number_words(text))


def clean_message(message: contracts.Message | dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned transcript dict preserving raw ASR output in ``asr_text``."""
    data = message.to_dict() if isinstance(message, contracts.Message) else dict(message)
    raw_asr = data["asr_text"] if "asr_text" in data else data["message"]
    return {
        "id": data["id"],
        "sender_id": data["sender_id"],
        "source": "audio",
        "message": clean_text(data["message"]),
        "asr_text": raw_asr,
    }


def clean_transcripts(input_path: Path, out_path: Path | None = None) -> list[dict[str, Any]]:
    """Load transcript Messages, clean them, validate as transcripts, and write JSON."""
    messages = contracts.load_messages(input_path)
    cleaned = [clean_message(msg) for msg in messages]
    problems = contracts.validate_file("transcripts", cleaned)
    if problems:
        raise ValueError("\n".join(problems))
    target = out_path if out_path is not None else input_path
    contracts.write_json(target, cleaned)
    return cleaned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for transcript cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean ASR transcripts: convert number words to digits and fix place names."
    )
    parser.add_argument("input", type=Path, help="Input v2 transcript Message JSON file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (defaults to overwriting INPUT in place).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    args = parse_args(argv)
    try:
        clean_transcripts(args.input, args.out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
